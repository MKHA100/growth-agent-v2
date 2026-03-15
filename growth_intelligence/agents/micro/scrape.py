"""
Scrape micro-agent — Firecrawl only.

Converts a web page to clean markdown. For PDF URLs, sets the is_pdf flag
so the caller can route to LlamaParse instead of continuing with markdown.
"""

from __future__ import annotations

import os

from firecrawl import FirecrawlApp

from schemas.findings import ScrapeResult


_client: FirecrawlApp | None = None

# File extensions that should be parsed by LlamaParse instead of Firecrawl
_PDF_EXTENSIONS = (".pdf",)


def _get_client() -> FirecrawlApp:
    global _client
    if _client is None:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            raise RuntimeError("FIRECRAWL_API_KEY environment variable is not set.")
        _client = FirecrawlApp(api_key=api_key)
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
        return ScrapeResult(url=url, markdown="", is_pdf=True)

    client = _get_client()

    try:
        result = client.scrape_url(
            url=url,
            params={
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        )
        markdown = result.get("markdown", "") if isinstance(result, dict) else ""
        # Check if the returned content is actually a PDF indicator
        is_pdf = markdown == "" and (
            result.get("metadata", {}).get("contentType", "").startswith("application/pdf")
            if isinstance(result, dict)
            else False
        )
        return ScrapeResult(url=url, markdown=markdown, is_pdf=is_pdf)
    except Exception as exc:  # noqa: BLE001
        # Return empty result on scrape failure — let caller degrade gracefully
        return ScrapeResult(url=url, markdown=f"[Scrape failed: {exc}]", is_pdf=False)
