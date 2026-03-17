"""
PDF export — WeasyPrint + Jinja2.

Renders the FinalReport Pydantic model into an HTML template, converts to
PDF via WeasyPrint, saves it locally (or uploads to S3 in production), and
returns a download URL.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from schemas.findings import FinalReport


# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_jinja_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)

# Register a custom filter for percentage formatting
_jinja_env.filters["pct"] = lambda v: f"{float(v):.0%}"
_jinja_env.filters["domain_title"] = lambda s: s.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def export_pdf(session_id: str, report: FinalReport) -> str:
    """Render the report to PDF and return a base64 data URL for download.

    Saves the PDF to PDF_OUTPUT_DIR as a local backup, and returns a
    base64-encoded data URL that works directly in the chat UI without
    needing a static file server.

    Args:
        session_id: Used to name the output file.
        report: The fully populated FinalReport model.

    Returns:
        A base64 data URL string like 'data:application/pdf;base64,...'.
    """
    output_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "/tmp/reports"))
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)

    filename = f"report_{session_id[:8]}_{int(time.time())}.pdf"
    output_path = output_dir / filename

    template = _jinja_env.get_template("report.html")
    html_content = template.render(
        report=report,
        generated_at=datetime.utcnow(),
    )

    # WeasyPrint is synchronous — run in thread pool to avoid blocking the event loop
    await asyncio.to_thread(
        _write_pdf,
        html_content,
        str(output_path),
    )

    # Read the PDF and encode as base64 data URL
    pdf_bytes = await asyncio.to_thread(output_path.read_bytes)
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return f"data:application/pdf;base64,{b64}"


def _write_pdf(html: str, output_path: str) -> None:
    """Synchronous WeasyPrint call — runs in executor thread."""
    import weasyprint  # lazy import: system libs only needed at PDF render time
    weasyprint.HTML(string=html).write_pdf(output_path)
