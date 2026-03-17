"""
Pinecone vector store client.

Handles chunking, embedding (text-embedding-3-large), upsert, and retrieval
of content chunks. Every chunk is stored under a session-scoped namespace.

Key constants (per architecture spec):
- Embedding model : text-embedding-3-large (OpenAI)
- Chunk size      : 512 tokens, overlap 64 tokens
- top_k           : 5 (all callers must use this default)
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric, VectorType

from schemas.findings import Chunk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 512        # tokens (approximate; splitter uses chars)
CHUNK_OVERLAP = 64      # tokens
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5               # fixed per architecture spec — do not override


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------

_pc: Pinecone | None = None
_embedder: OpenAIEmbeddings | None = None
_splitter: RecursiveCharacterTextSplitter | None = None


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY environment variable is not set.")
        _pc = Pinecone(api_key=api_key)
    return _pc


_index = None


async def _get_index():
    global _index
    if _index is not None:
        return _index

    pc = _get_pinecone()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "growth-intel")

    # Create index if it doesn't exist (text-embedding-3-large → 3072 dims)
    existing = await asyncio.to_thread(lambda: list(pc.list_indexes().names()))
    if index_name not in existing:
        await asyncio.to_thread(
            pc.create_index,
            name=index_name,
            dimension=3072,
            metric=Metric.COSINE,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1,
            ),
            vector_type=VectorType.DENSE,
        )

    # Fetch the host via describe_index, then connect using host= to skip any
    # further describe_index calls on subsequent Index method usage.
    index_info = await asyncio.to_thread(pc.describe_index, index_name)
    _index = await asyncio.to_thread(pc.Index, host=index_info.host)
    return _index


def _get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        _embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
    return _embedder


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        # Approximate: 1 token ≈ 4 chars for English text
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE * 4,
            chunk_overlap=CHUNK_OVERLAP * 4,
            length_function=len,
        )
    return _splitter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def make_chunks(
    text: str,
    source_url: str,
    domain_tag: str,
    agent_id: str = "",
) -> list[Chunk]:
    """Split text into Chunk objects ready for upsert."""
    splitter = _get_splitter()
    pieces = splitter.split_text(text)
    now = datetime.utcnow()
    return [
        Chunk(
            id=str(uuid.uuid4()),
            text=piece,
            source_url=source_url,
            domain_tag=domain_tag,
            timestamp=now,
            agent_id=agent_id,
        )
        for piece in pieces
        if piece.strip()
    ]


async def upsert_chunks(session_id: str, chunks: list[Chunk]) -> None:
    """Embed and upsert chunks into Pinecone under the session namespace."""
    if not chunks:
        logger.info("[Pinecone] No chunks to upsert, skipping")
        return

    logger.info("[Pinecone] Upserting %d chunks for session %s", len(chunks), session_id[:12])
    embedder = _get_embedder()
    index = await _get_index()

    texts = [c.text for c in chunks]
    vectors = await asyncio.to_thread(embedder.embed_documents, texts)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "id": chunk.id,
                "values": vector,
                "metadata": {
                    "text": chunk.text,
                    "source_url": chunk.source_url,
                    "domain_tag": chunk.domain_tag,
                    "timestamp": chunk.timestamp.isoformat(),
                    "agent_id": chunk.agent_id,
                },
            }
        )

    # Upsert in batches of 100 to respect Pinecone limits
    batch_size = 100
    for i in range(0, len(records), batch_size):
        await asyncio.to_thread(
            index.upsert,
            vectors=records[i : i + batch_size],
            namespace=session_id,
        )


async def query_chunks(
    session_id: str,
    query: str,
    domain_tag: str | None = None,
    top_k: int = TOP_K,
) -> list[Chunk]:
    """Semantic search over chunks in the session namespace.

    Args:
        session_id: Pinecone namespace to search within.
        query: Natural-language query string.
        domain_tag: Optional filter — only return chunks from this domain.
        top_k: Number of results. Defaults to architecture-spec value of 5.
    """
    embedder = _get_embedder()
    index = await _get_index()
    logger.info("[Pinecone] Querying top-%d chunks for domain=%s", top_k, domain_tag)

    query_vector: list[float] = await asyncio.to_thread(embedder.embed_query, query)

    filter_expr: dict[str, Any] | None = None
    if domain_tag:
        filter_expr = {"domain_tag": {"$eq": domain_tag}}

    response = await asyncio.to_thread(
        index.query,
        vector=query_vector,
        top_k=top_k,
        namespace=session_id,
        filter=filter_expr,
        include_metadata=True,
    )

    results: list[Chunk] = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        results.append(
            Chunk(
                id=match["id"],
                text=meta.get("text", ""),
                source_url=meta.get("source_url", ""),
                domain_tag=meta.get("domain_tag", ""),
                timestamp=datetime.fromisoformat(
                    meta.get("timestamp", datetime.utcnow().isoformat())
                ),
                agent_id=meta.get("agent_id", ""),
            )
        )

    return results
