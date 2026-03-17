"""
BaseDomainAgent — abstract base class for all six domain intelligence agents.

Every domain agent:
1. Checks Mem0 for prior results to avoid redundant API calls
2. Forms domain-specific sub-queries
3. Fetches signals (search + scrape + social, sequentially)
4. Chunks and upserts all content to Pinecone
5. Retrieves top-5 relevant chunks
6. Synthesises a DomainFinding via the Pydantic few-shot loop
7. Persists the finding to Mem0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import memory.mem0_client as mem0_client
import memory.pinecone_client as pinecone_client
from schemas.findings import Chunk, DomainFinding

logger = logging.getLogger(__name__)


class BaseDomainAgent(ABC):
    """Abstract base for all domain intelligence agents."""

    DOMAIN_TAG: str  # Override in each subclass

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    @abstractmethod
    async def run(self, query: str) -> DomainFinding:
        """Run the full domain intelligence pipeline for the given query."""
        ...

    async def recall(self, query: str) -> list[Chunk]:
        """Retrieve the top-5 most relevant chunks from Pinecone for this domain."""
        return await pinecone_client.query_chunks(
            self.session_id,
            query,
            domain_tag=self.DOMAIN_TAG,
            top_k=5,
        )

    async def recall_or_fallback(
        self, query: str, local_chunks: list[Chunk]
    ) -> list[Chunk]:
        """Retrieve from Pinecone. Fall back to local chunks if Pinecone is empty.

        Pinecone Serverless has eventual consistency — freshly upserted vectors
        may not be queryable for a few seconds.  When that happens, fallback to
        the chunks produced locally so synthesis always has something to work with.
        """
        top_chunks = await self.recall(query)
        if top_chunks:
            logger.info(
                "[%s] Pinecone recall returned %d chunks",
                self.DOMAIN_TAG, len(top_chunks),
            )
            return top_chunks

        if local_chunks:
            fallback = local_chunks[:5]
            logger.warning(
                "[%s] Pinecone recall returned 0 chunks — using %d local chunks as fallback",
                self.DOMAIN_TAG, len(fallback),
            )
            return fallback

        logger.warning("[%s] No chunks available (Pinecone empty AND no local chunks)", self.DOMAIN_TAG)
        return []

    async def remember(self, finding: dict[str, Any]) -> None:
        """Persist a domain finding to Mem0 thread memory."""
        await mem0_client.append_domain_finding(
            self.session_id, self.DOMAIN_TAG, finding
        )

    async def get_prior(self) -> list[dict[str, Any]]:
        """Fetch prior Mem0 thread for this domain."""
        return await mem0_client.get_domain_thread(self.session_id, self.DOMAIN_TAG)

    async def store_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert a list of chunks to Pinecone."""
        await pinecone_client.upsert_chunks(self.session_id, chunks)
