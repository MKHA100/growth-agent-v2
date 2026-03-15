"""
Search micro-agent — Gemini Flash 2.0 with Google Search grounding.

Uses Gemini's built-in google_search tool to retrieve grounded, real-time
search results with source attribution via grounding_metadata.
"""

from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai

from schemas.findings import SearchResult


_client: genai.GenerativeModel | None = None


def _get_client() -> genai.GenerativeModel:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        _client = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")
    return _client


def _parse_grounding_results(response: Any, query: str) -> list[SearchResult]:
    """Extract structured SearchResult objects from Gemini grounding metadata."""
    results: list[SearchResult] = []

    # Extract main response text
    main_text = ""
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text"):
            main_text += part.text

    # Try to extract grounding chunks
    grounding_meta = getattr(response.candidates[0], "grounding_metadata", None)
    if grounding_meta and hasattr(grounding_meta, "grounding_chunks"):
        for chunk in grounding_meta.grounding_chunks:
            web = getattr(chunk, "web", None)
            if web:
                results.append(
                    SearchResult(
                        query=query,
                        title=getattr(web, "title", ""),
                        url=getattr(web, "uri", ""),
                        snippet=main_text[:500] if not results else "",
                        raw_content=main_text if not results else "",
                    )
                )
    
    # Fall back to a single result containing the full response text
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

    response = client.generate_content(
        contents=query,
        tools=[{"google_search": {}}],
    )

    return _parse_grounding_results(response, query)
