"""
OrchestratorState — the shared LangGraph state for the Growth Intelligence Agent.

The `messages` key is required by agent-chat-ui's useStream hook.
All domain agents write their findings into `domain_findings` keyed by domain tag.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from schemas.findings import DomainFinding


class OrchestratorState(BaseModel):
    """Full shared state for the Growth Intelligence graph."""

    # Required by agent-chat-ui — useStream hooks into this key
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Session identity
    session_id: str = ""

    # User's original question
    user_query: str = ""

    # Set by classify_domains node — which of the 6 domains to run
    relevant_domains: list[str] = Field(default_factory=list)

    # Populated sequentially by each domain node
    domain_findings: dict[str, DomainFinding] = Field(default_factory=dict)

    # Domains that failed — used by synthesise node for caveats
    error_domains: list[str] = Field(default_factory=list)

    # Set by export_pdf node — the download URL
    pdf_url: str | None = None
