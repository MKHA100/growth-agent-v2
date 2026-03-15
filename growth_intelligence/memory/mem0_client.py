"""
Mem0 thread memory client.

Provides two scopes:
- Global thread: session-wide context (product, user role, overall conclusions)
- Domain thread: per-domain findings history

Uses Mem0's in-place update semantics — conflicting facts are mutated,
never appended as duplicates.
"""

from __future__ import annotations

import os
from typing import Any

from mem0 import AsyncMemoryClient


_client: AsyncMemoryClient | None = None


def _get_client() -> AsyncMemoryClient:
    global _client
    if _client is None:
        api_key = os.environ.get("MEM0_API_KEY")
        if not api_key:
            raise RuntimeError("MEM0_API_KEY environment variable is not set.")
        _client = AsyncMemoryClient(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Global context helpers
# ---------------------------------------------------------------------------


async def get_global_context(session_id: str) -> dict[str, Any]:
    """Return the stored global context for this session as a flat dict."""
    client = _get_client()
    memories = await client.get_all(user_id=f"global:{session_id}")
    if not memories:
        return {}
    # Mem0 returns a list of memory objects; collapse to dict
    context: dict[str, Any] = {}
    for mem in memories:
        # Each memory has a 'memory' text field and optional metadata
        context[mem.get("id", str(len(context)))] = mem.get("memory", "")
    return context


async def set_global_context(session_id: str, updates: dict[str, Any]) -> None:
    """Write or update global context facts for this session.

    Mem0 will match semantically similar existing memories and update them
    in place rather than creating duplicates.
    """
    client = _get_client()
    # Convert the updates dict to a list of natural-language messages
    messages = [
        {"role": "user", "content": f"{key}: {value}"}
        for key, value in updates.items()
    ]
    await client.add(messages=messages, user_id=f"global:{session_id}")


# ---------------------------------------------------------------------------
# Domain thread helpers
# ---------------------------------------------------------------------------


async def get_domain_thread(session_id: str, domain: str) -> list[dict[str, Any]]:
    """Return all stored findings for a given domain in this session."""
    client = _get_client()
    memories = await client.get_all(user_id=f"domain:{session_id}:{domain}")
    return [
        {"id": m.get("id"), "memory": m.get("memory", ""), "metadata": m.get("metadata", {})}
        for m in memories
    ]


async def append_domain_finding(
    session_id: str, domain: str, finding: dict[str, Any]
) -> None:
    """Persist a domain finding to Mem0 for this session.

    Mem0 performs semantic deduplication — if the finding is substantively
    the same as an existing one, it updates in place.
    """
    client = _get_client()
    messages = [
        {
            "role": "assistant",
            "content": (
                f"Domain '{domain}' finding — "
                f"status: {finding.get('status', 'unknown')}, "
                f"summary: {finding.get('summary', '')}, "
                f"confidence: {finding.get('confidence', 0)}"
            ),
        }
    ]
    await client.add(
        messages=messages,
        user_id=f"domain:{session_id}:{domain}",
        metadata={"domain": domain, "session_id": session_id},
    )
