"""
Scrape micro-agent — Firecrawl only.

Converts a web page to clean markdown. For PDF URLs, sets the is_pdf flag
so the caller can route to LlamaParse instead of continuing with markdown.
"""

from __future__ import annotations

import asyncio
import logging
import os

from firecrawl import Firecrawl

from schemas.findings import ScrapeResult

logger = logging.getLogger(__name__)


_client: Firecrawl | None = None

# File extensions that should be parsed by LlamaParse instead of Firecrawl
_PDF_EXTENSIONS = (".pdf",)


def _get_client() -> Firecrawl:
    global _client
    if _client is None:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise RuntimeError("FIRECRAWL_API_KEY environment variable is not set.")
        _client = Firecrawl(api_key=api_key)
    return _client


def _is_pdf_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(_PDF_EXTENSIONS)


async def execute(url: str) -> ScrapeResult:
    """Scrape a URL and return its markdown content.

    If the URL points to a PDF, returns a ScrapeResult with is_pdf=True
    and an empty markdown so the caller can route to LlamaParse.

    Args:
        url: The URL to scrape.

    Returns:
        ScrapeResult with markdown content or is_pdf=True flag.
    """
    if _is_pdf_url(url):
        logger.info("[ScrapeAgent] PDF detected, skipping Firecrawl: %s", url)
        return ScrapeResult(url=url, markdown="", is_pdf=True)

    client = _get_client()
    logger.info("[ScrapeAgent] Scraping: %s", url)

    try:
        result = await asyncio.to_thread(
            client.scrape,
            url,
            formats=["markdown"],
            only_main_content=True,
        )
        # Firecrawl v4+ returns a Document object with .markdown attribute
        markdown = getattr(result, "markdown", None) or ""
        # Check if the returned content is actually a PDF indicator
        metadata = getattr(result, "metadata", None)
        content_type = getattr(metadata, "content_type", "") if metadata else ""
        is_pdf = markdown == "" and content_type.startswith("application/pdf")
        logger.info(
            "[ScrapeAgent] Scraped %d chars from %s", len(markdown), url
        )
        return ScrapeResult(url=url, markdown=markdown, is_pdf=is_pdf)
    except Exception as exc:  # noqa: BLE001
        # Return empty result on scrape failure — let caller degrade gracefully
        logger.warning("[ScrapeAgent] Scrape failed for %s: %s", url, exc)
        return ScrapeResult(url=url, markdown=f"[Scrape failed: {exc}]", is_pdf=False)
