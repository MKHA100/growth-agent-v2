"""
Mem0 thread memory client.

Provides two scopes:
- Global thread: session-wide context (product, user role, overall conclusions)
- Domain thread: per-domain findings history

Uses Mem0's in-place update semantics — conflicting facts are mutated,
never appended as duplicates.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from mem0 import MemoryClient

logger = logging.getLogger(__name__)


_client: MemoryClient | None = None


def _get_client() -> MemoryClient:
    global _client
    if _client is None:
        api_key = os.environ.get("MEM0_API_KEY")
        if not api_key:
            raise RuntimeError("MEM0_API_KEY environment variable is not set.")
        _client = MemoryClient(api_key=api_key)
    return _client


async def _ensure_client() -> MemoryClient:
    """Ensure the MemoryClient singleton is initialized (thread-safe for async)."""
    global _client
    if _client is None:
        api_key = os.environ.get("MEM0_API_KEY")
        if not api_key:
            raise RuntimeError("MEM0_API_KEY environment variable is not set.")
        _client = await asyncio.to_thread(MemoryClient, api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Global context helpers
# ---------------------------------------------------------------------------


async def get_global_context(session_id: str) -> dict[str, Any]:
    """Return the stored global context for this session as a flat dict."""
    logger.info("[Mem0] Fetching global context for session %s", session_id[:12])
    client = await _ensure_client()
    memories = await asyncio.to_thread(
        lambda: client.get_all(filters={"user_id": f"global:{session_id}"})
    )
    if not memories:
        return {}
    # Mem0 v2 returns {"results": [...], "count": n}
    results = memories.get("results", memories) if isinstance(memories, dict) else memories
    if not results:
        return {}
    # Mem0 returns a list of memory objects; collapse to dict
    context: dict[str, Any] = {}
    for mem in results:
        # Each memory has a 'memory' text field and optional metadata
        context[mem.get("id", str(len(context)))] = mem.get("memory", "")
    return context


async def set_global_context(session_id: str, updates: dict[str, Any]) -> None:
    """Write or update global context facts for this session.

    Mem0 will match semantically similar existing memories and update them
    in place rather than creating duplicates.
    """
    client = await _ensure_client()
    # Convert the updates dict to a list of natural-language messages
    messages = [
        {"role": "user", "content": f"{key}: {value}"}
        for key, value in updates.items()
    ]
    await asyncio.to_thread(
        lambda: client.add(messages=messages, user_id=f"global:{session_id}")
    )


# ---------------------------------------------------------------------------
# Domain thread helpers
# ---------------------------------------------------------------------------


async def get_domain_thread(session_id: str, domain: str) -> list[dict[str, Any]]:
    """Return all stored findings for a given domain in this session."""
    logger.info("[Mem0] Fetching domain thread: %s for session %s", domain, session_id[:12])
    client = await _ensure_client()
    memories = await asyncio.to_thread(
        lambda: client.get_all(filters={"user_id": f"domain:{session_id}:{domain}"})
    )
    results = memories.get("results", memories) if isinstance(memories, dict) else memories
    return [
        {"id": m.get("id"), "memory": m.get("memory", ""), "metadata": m.get("metadata", {})}
        for m in (results or [])
    ]


async def append_domain_finding(
    session_id: str, domain: str, finding: dict[str, Any]
) -> None:
    """Persist a domain finding to Mem0 for this session.

    Mem0 performs semantic deduplication — if the finding is substantively
    the same as an existing one, it updates in place.
    """
    logger.info("[Mem0] Appending finding for domain=%s session=%s", domain, session_id[:12])
    client = await _ensure_client()

    # Build a rich message body with the full finding content so Mem0 can
    # semantically deduplicate on it.  Keep metadata small (< 2000 chars)
    # to stay within Mem0's metadata size limit.
    import json as _json

    finding_json = _json.dumps(finding, default=str)
    messages = [
        {
            "role": "assistant",
            "content": (
                f"Domain '{domain}' finding — "
                f"status: {finding.get('status', 'unknown')}, "
                f"confidence: {finding.get('confidence', 0)}. "
                f"Summary: {finding.get('summary', '')}. "
                f"Full finding JSON: {finding_json}"
            ),
        }
    ]
    await asyncio.to_thread(
        lambda: client.add(
            messages=messages,
            user_id=f"domain:{session_id}:{domain}",
            metadata={
                "domain": domain,
                "session_id": session_id,
                "status": finding.get("status", "unknown"),
            },
        )
    )
