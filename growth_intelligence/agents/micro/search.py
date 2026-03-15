"""
Search micro-agent — Gemini Flash 2.0 with Google Search grounding.

Uses Gemini's built-in google_search tool to retrieve grounded, real-time
search results with source attribution via grounding_metadata.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from schemas.findings import SearchResult


_client: ChatGoogleGenerativeAI | None = None


def _get_client() -> ChatGoogleGenerativeAI:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _client = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
        )
    return _client


_URL_RE = re.compile(r"https?://[^\s\]\)>\"]+")


def _extract_urls_from_text(text: str) -> list[str]:
    urls: list[str] = []
    for match in _URL_RE.findall(text or ""):
        url = match.rstrip(").,]}")
        if url and url not in urls:
            urls.append(url)
    return urls


def _parse_grounding_results(response: Any, query: str) -> list[SearchResult]:
    """Extract structured SearchResult objects from Gemini grounding metadata."""
    results: list[SearchResult] = []

    main_text = response.content if isinstance(response.content, str) else ""

    # Try to extract grounding chunks from response metadata
    meta_sources: list[dict] = []
    if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
        meta_sources.append(response.response_metadata)
    if hasattr(response, "additional_kwargs") and isinstance(response.additional_kwargs, dict):
        meta_sources.append(response.additional_kwargs)

    for meta in meta_sources:
        grounding_meta = (
            meta.get("grounding_metadata")
            or meta.get("groundingMetadata")
            or {}
        )
        grounding_chunks = (
            grounding_meta.get("grounding_chunks")
            or grounding_meta.get("groundingChunks")
            or meta.get("grounding_chunks")
            or meta.get("groundingChunks")
            or []
        )
        for chunk in grounding_chunks:
            web = chunk.get("web", {}) if isinstance(chunk, dict) else {}
            if web:
                results.append(
                    SearchResult(
                        query=query,
                        title=web.get("title", ""),
                        url=web.get("uri", "") or web.get("url", ""),
                        snippet=main_text[:500] if not results else "",
                        raw_content=main_text if not results else "",
                    )
                )

        # Try citation-based URLs if present
        citations = meta.get("citations") or meta.get("citation_metadata", {}).get("citations", [])
        for citation in citations or []:
            url = ""
            if isinstance(citation, dict):
                url = citation.get("uri") or citation.get("url") or ""
            if url:
                results.append(
                    SearchResult(
                        query=query,
                        title=citation.get("title", "") if isinstance(citation, dict) else "",
                        url=url,
                        snippet=main_text[:500] if not results else "",
                        raw_content=main_text if not results else "",
                    )
                )

    # Fallback: extract URLs from text if metadata was empty
    if not results and main_text:
        urls = _extract_urls_from_text(main_text)
        if urls:
            for i, url in enumerate(urls):
                results.append(
                    SearchResult(
                        query=query,
                        title="Gemini Search Result",
                        url=url,
                        snippet=main_text[:500] if i == 0 else "",
                        raw_content=main_text if i == 0 else "",
                    )
                )

    # Final fallback to a single result containing the full response text
    if not results and main_text:
        results.append(
            SearchResult(
                query=query,
                title="Gemini Search Result",
                url="",
                snippet=main_text[:500],
                raw_content=main_text,
            )
        )

    return results


async def execute(query: str) -> list[SearchResult]:
    """Execute a grounded Gemini search and return structured results.

    Args:
        query: Natural-language search query.

    Returns:
        List of SearchResult objects with URLs and content.
    """
    client = _get_client()

    response = await asyncio.to_thread(
        lambda: client.invoke(
            [HumanMessage(content=query)],
            tools=[{"google_search": {}}],
        )
    )

    return _parse_grounding_results(response, query)
