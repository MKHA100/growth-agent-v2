"""
Synthesis micro-agent — Pydantic few-shot retry loop via Claude Sonnet 4.5.

Feeds the model:
1. The target Pydantic schema as a JSON schema string
2. Three few-shot examples (raw text → valid JSON) per domain
3. The retrieved Pinecone chunks

Retries up to 3 times on Pydantic validation failure, passing the error
message back to the model for self-correction.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from schemas.findings import Chunk, DomainFinding

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 3
_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"


class SynthesisFailedError(Exception):
    """Raised when all synthesis retry attempts are exhausted."""


_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


def _build_prompt(
    chunks: list[Chunk],
    schema: Type[T],
    few_shots: list[dict],
) -> str:
    """Build the initial synthesis prompt."""
    schema_str = json.dumps(schema.model_json_schema(), indent=2)
    chunks_text = "\n\n---\n\n".join(
        f"[Source: {c.source_url}]\n{c.text}" for c in chunks
    )

    shots_text = ""
    for i, shot in enumerate(few_shots, 1):
        shots_text += (
            f"\n### Example {i}\n"
            f"**Input:**\n{shot['raw_input']}\n\n"
            f"**Expected JSON output:**\n```json\n"
            f"{json.dumps(shot['expected_output'], indent=2)}\n```\n"
        )

    return (
        f"You are a market intelligence analyst. "
        f"Extract structured intelligence from the source material below and return ONLY valid JSON "
        f"matching the provided schema. No markdown fences, no explanation.\n\n"
        f"## Target JSON Schema\n```json\n{schema_str}\n```\n\n"
        f"## Few-shot Examples{shots_text}\n\n"
        f"## Source Material\n{chunks_text}\n\n"
        f"Return ONLY the JSON object. No other text."
    )


def _build_retry_prompt(
    bad_output: str,
    validation_error: str,
    schema: Type[T],
    few_shots: list[dict],
    chunks: list[Chunk],
) -> str:
    """Build a retry prompt that includes the previous failed output and the error."""
    base = _build_prompt(chunks, schema, few_shots)
    return (
        f"{base}\n\n"
        f"## Previous Attempt (FAILED)\n"
        f"Your previous response was:\n```\n{bad_output}\n```\n\n"
        f"It failed Pydantic validation with this error:\n```\n{validation_error}\n```\n\n"
        f"Fix the JSON and return ONLY the corrected JSON object."
    )


async def execute(
    chunks: list[Chunk],
    schema: Type[T],
    few_shots: list[dict],
) -> T:
    """Synthesise retrieved chunks into a structured Pydantic model.

    Args:
        chunks: Pinecone chunks retrieved for context.
        schema: The Pydantic model class to validate the output against.
        few_shots: List of {raw_input, expected_output} dicts for this domain.

    Returns:
        A validated instance of `schema`.

    Raises:
        SynthesisFailedError: If all MAX_RETRIES attempts fail validation.
    """
    client = _get_client()
    prompt = _build_prompt(chunks, schema, few_shots)
    last_raw = ""
    last_error = ""

    logger.info(
        "[Synthesis] Starting synthesis with %d chunks, schema=%s",
        len(chunks), schema.__name__,
    )

    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            logger.warning(
                "[Synthesis] Retry %d/%d — last error: %s",
                attempt + 1, MAX_RETRIES, last_error[:200],
            )
            prompt = _build_retry_prompt(last_raw, last_error, schema, few_shots, chunks)

        response = await client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=4096,
            system="Respond ONLY with valid JSON matching the schema. No markdown fences, no explanation.",
            messages=[{"role": "user", "content": prompt}],
        )

        last_raw = response.content[0].text.strip()

        # Strip accidental markdown fences if present
        if last_raw.startswith("```"):
            lines = last_raw.split("\n")
            last_raw = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            return schema.model_validate_json(last_raw)
        except ValidationError as exc:
            last_error = str(exc)
            logger.warning(
                "[Synthesis] Validation failed attempt %d: %s", attempt + 1, last_error[:200]
            )
            continue

    logger.error(
        "[Synthesis] All %d attempts exhausted. Last error: %s",
        MAX_RETRIES, last_error[:300],
    )
    raise SynthesisFailedError(
        f"Synthesis failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}\nLast raw output: {last_raw}"
    )
