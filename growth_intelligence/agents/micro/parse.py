"""
Deep parse micro-agent — LlamaParse.

Only used for PDF URLs or research reports. Do NOT call this on every URL —
it consumes LlamaParse API credits. The scrape agent sets is_pdf=True when
Firecrawl detects a PDF, which is the trigger to route here instead.
"""

from __future__ import annotations

import os
import tempfile

import httpx
from llama_parse import LlamaParse

from schemas.findings import ParseResult


_LLAMAPARSE_API_KEY_ENV = "LLAMAPARSE_API_KEY"


async def execute(url: str) -> ParseResult:
    """Download a PDF from a URL and parse it to markdown via LlamaParse.

    Args:
        url: URL of the PDF to parse. Must be a direct link to a PDF file.

    Returns:
        ParseResult with the extracted markdown content.
    """
    api_key = os.environ.get(_LLAMAPARSE_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{_LLAMAPARSE_API_KEY_ENV} environment variable is not set.")

    # Download the raw bytes
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        raw_bytes = response.content

    # Write to a temp file so LlamaParse can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        verbose=False,
    )

    try:
        docs = await parser.aload_data(tmp_path, extra_info={"file_name": url})
        content = " ".join(d.text for d in docs)
        return ParseResult(url=url, content=content)
    finally:
        import os as _os
        _os.unlink(tmp_path)
