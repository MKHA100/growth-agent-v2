"""
PDF export — WeasyPrint + Jinja2.

Renders the FinalReport Pydantic model into an HTML template, converts to
PDF via WeasyPrint, saves it locally (or uploads to S3 in production), and
returns a download URL.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path

import weasyprint
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
    """Render the report to PDF and return a download URL.

    For development: saves to PDF_OUTPUT_DIR (default /tmp/reports).
    For production: upload to S3 and return a presigned URL.

    Args:
        session_id: Used to name the output file.
        report: The fully populated FinalReport model.

    Returns:
        A relative URL path like '/reports/report_abc12345_1710000000.pdf'.
    """
    output_dir = Path(os.environ.get("PDF_OUTPUT_DIR", "/tmp/reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

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

    return f"/reports/{filename}"


def _write_pdf(html: str, output_path: str) -> None:
    """Synchronous WeasyPrint call — runs in executor thread."""
    weasyprint.HTML(string=html).write_pdf(output_path)
