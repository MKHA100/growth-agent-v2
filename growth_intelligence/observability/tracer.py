"""
LangFuse observability — trace logging for all LLM calls, tool calls,
Pinecone queries, and Mem0 writes.

Attach the handler to any LangChain/LangGraph invocation via:
    config={"callbacks": [get_handler(session_id)]}

No per-function instrumentation needed — the callback handler captures
everything automatically via LangChain's callback system.
"""

from __future__ import annotations

import os

from langfuse.langchain import CallbackHandler


def get_handler(session_id: str) -> CallbackHandler:
    """Return a configured LangFuse CallbackHandler for a given session.

    Args:
        session_id: Unique identifier for the current session.
                    Used to group all traces for a single user interaction.

    Returns:
        A LangFuse CallbackHandler ready to attach to LangGraph invocations.
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        # Return a no-op handler stub if LangFuse is not configured.
        # This prevents crashes in environments without observability set up.
        return _NoOpHandler()

    # langfuse v3: CallbackHandler reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
    # and LANGFUSE_HOST from environment variables automatically.
    # session_id, trace_name, and tags are no longer constructor arguments.
    return CallbackHandler()


class _NoOpHandler:
    """A minimal no-op stub that satisfies the callback contract.

    Used when LangFuse keys are not configured so that the rest of the
    system doesn't need to check for None handlers everywhere.
    """

    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None
