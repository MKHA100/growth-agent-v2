"""
LangGraph StateGraph — Growth Intelligence Orchestrator.

Sequential execution flow:
  receive_message
    → classify_domains
    → [run_market | skip] → [run_competitive | skip] → [run_win_loss | skip]
    → [run_pricing | skip] → [run_positioning | skip] → [run_adjacent | skip]
    → synthesise
    → export_pdf
    → stream_response
    → update_memory

All domain nodes are wrapped in try/except for graceful degradation.
Intermediate tool messages use the 'do-not-render-' prefix to be hidden from
the agent-chat-ui frontend.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

import memory.mem0_client as mem0_client
from agents.domain.adjacent import AdjacentAgent
from agents.domain.competitive import CompetitiveAgent
from agents.domain.market import MarketAgent
from agents.domain.positioning import PositioningAgent
from agents.domain.pricing import PricingAgent
from agents.domain.win_loss import WinLossAgent
from observability.tracer import get_handler
from orchestrator.state import OrchestratorState
from pdf_export.generator import export_pdf
from schemas.findings import DomainFinding, FinalReport, PartialFinding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM for classify_domains and synthesise
# ---------------------------------------------------------------------------

_ALL_DOMAINS = ["market", "competitive", "win_loss", "pricing", "positioning", "adjacent"]
_DOMAIN_ORDER = _ALL_DOMAINS


def _get_llm() -> ChatAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, max_tokens=1024)


# ---------------------------------------------------------------------------
# Node: receive_message
# ---------------------------------------------------------------------------


async def receive_message(state: OrchestratorState) -> dict:
    """Extract user query from the latest human message and bootstrap session."""
    session_id = state.session_id or str(uuid.uuid4())
    user_query = ""

    # Find the latest human message
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    logger.info("[Orchestrator] Received query: %s (session=%s)", user_query[:100], session_id[:12])

    # Reset per-turn state so a second question in the same thread
    # does not inherit stale findings, errors, or domains from turn 1.
    turn_reset: dict = {
        "session_id": session_id,
        "user_query": user_query,
        "domain_findings": {},
        "error_domains": [],
        "relevant_domains": [],
        "pdf_url": None,
    }

    # Load existing global context from Mem0 (non-blocking; ignore errors)
    try:
        global_ctx = await mem0_client.get_global_context(session_id)
        if global_ctx:
            ctx_str = json.dumps(global_ctx, default=str)
            # Append as a hidden context note (not rendered to user)
            ctx_msg = AIMessage(content=f"[Session context loaded: {ctx_str[:200]}...]")
            ctx_msg.id = f"do-not-render-{ctx_msg.id}"
            return {**turn_reset, "messages": [ctx_msg]}
    except Exception:  # noqa: BLE001
        pass

    return turn_reset


# ---------------------------------------------------------------------------
# Node: classify_domains
# ---------------------------------------------------------------------------


async def classify_domains(state: OrchestratorState) -> dict:
    """Use the LLM to determine which intelligence domains are relevant."""
    llm = _get_llm()
    handler = get_handler(state.session_id)

    prompt = (
        f"Given the following product/market intelligence question, determine which of these "
        f"six analysis domains are most relevant. Return ONLY a JSON array of domain tags "
        f"from this list: {_ALL_DOMAINS}\n\n"
        f"Question: {state.user_query}\n\n"
        f"Return a JSON array like: [\"market\", \"competitive\"]\n"
        f"Include all highly relevant domains. Minimum 1, maximum 6."
    )

    response = await llm.ainvoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [handler]},
    )
    raw = response.content if isinstance(response.content, str) else str(response.content)

    # Parse domain list robustly
    try:
        # Strip markdown fences if present
        cleaned = raw.strip()
        if "```" in cleaned:
            lines = cleaned.split("\n")
            cleaned = "\n".join(l for l in lines if not l.startswith("```")).strip()
        domains = json.loads(cleaned)
        valid_domains = [d for d in domains if d in _ALL_DOMAINS]
        if not valid_domains:
            valid_domains = _ALL_DOMAINS  # fallback: run all
    except (json.JSONDecodeError, TypeError):
        valid_domains = _ALL_DOMAINS  # fallback: run all

    logger.info("[Orchestrator] Classified domains: %s", valid_domains)

    # Hide the classification message from UI
    classify_msg = AIMessage(content=f"Domains selected: {valid_domains}")
    classify_msg.id = f"do-not-render-{classify_msg.id}"

    return {
        "relevant_domains": valid_domains,
        "messages": [classify_msg],
    }


# ---------------------------------------------------------------------------
# Domain node factory
# ---------------------------------------------------------------------------


def _next_relevant_domain(relevant_domains: list[str], after: str | None = None) -> str | None:
    """Return the next relevant domain after `after`, or None if none remain."""
    start_idx = 0
    if after:
        try:
            start_idx = _DOMAIN_ORDER.index(after) + 1
        except ValueError:
            start_idx = 0

    for domain in _DOMAIN_ORDER[start_idx:]:
        if domain in relevant_domains:
            return domain
    return None


def _make_domain_node(domain_tag: str, agent_class):
    """Create a domain node function with graceful degradation."""

    async def run_domain(state: OrchestratorState) -> dict:
        session_id = state.session_id
        logger.info("[Orchestrator] Running %s agent", domain_tag)

        try:
            agent = agent_class(session_id)
            finding = await agent.run(state.user_query)
        except Exception as exc:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            logger.error(
                "[Orchestrator] %s agent FAILED: %s\n%s", domain_tag, exc, tb
            )
            finding = PartialFinding(
                domain=domain_tag,
                status="failed",
                error_reason=f"{exc}",
            )
            error_domains = state.error_domains + [domain_tag]

            # Visible failure message so user sees what went wrong
            fail_msg = AIMessage(
                content=f"⚠️ **{domain_tag.replace('_', ' ').title()}** analysis failed: {exc}"
            )

            return {
                "domain_findings": {**state.domain_findings, domain_tag: finding},
                "error_domains": error_domains,
                "messages": [fail_msg],
            }

        confidence_pct = f"{finding.confidence:.0%}"
        next_domain = _next_relevant_domain(state.relevant_domains, domain_tag)
        next_label = next_domain.replace("_", " ").title() if next_domain else "synthesis"

        # Visible progress message to user
        progress_msg = AIMessage(
            content=f"{domain_tag.replace('_', ' ').title()} analysis complete "
                    f"({confidence_pct} confidence). "
                    f"{'Running ' + next_label + ' analysis...' if next_domain else 'Synthesising findings...'}"
        )

        return {
            "domain_findings": {**state.domain_findings, domain_tag: finding},
            "messages": [progress_msg],
        }

    run_domain.__name__ = f"run_{domain_tag}"
    return run_domain


# ---------------------------------------------------------------------------
# Node: synthesise
# ---------------------------------------------------------------------------


async def synthesise(state: OrchestratorState) -> dict:
    """Cross-domain narrative synthesis using Claude."""
    llm = _get_llm()
    handler = get_handler(state.session_id)
    logger.info("[Orchestrator] Starting synthesis across %d domains", len(state.domain_findings))

    findings_summary = ""
    for domain, finding in state.domain_findings.items():
        status_note = f" [FAILED: {finding.error_reason}]" if finding.status == "failed" else ""
        findings_summary += (
            f"\n### {domain.replace('_', ' ').title()}{status_note}\n"
            f"{finding.summary}\n"
            f"Confidence: {finding.confidence:.0%}\n"
        )

    failed_note = ""
    if state.error_domains:
        failed_note = (
            f"\n**Note:** The following domains failed to return results: "
            f"{', '.join(state.error_domains)}. "
            f"The synthesis below is based on available data only."
        )

    prompt = (
        f"You are a senior market intelligence analyst. "
        f"Based on the following domain findings, write a concise executive summary "
        f"(3-5 sentences) that synthesises the key insights and strategic recommendations. "
        f"Be direct and actionable.\n\n"
        f"Product/Market Question: {state.user_query}\n\n"
        f"Domain Findings:{findings_summary}{failed_note}\n\n"
        f"Return ONLY the executive summary text. No headers, no JSON."
    )

    response = await llm.ainvoke(
        [HumanMessage(content=prompt)],
        config={"callbacks": [handler]},
    )
    exec_summary = response.content if isinstance(response.content, str) else str(response.content)

    # Store synthesis as a hidden intermediate
    synth_msg = AIMessage(content=exec_summary)
    synth_msg.id = f"do-not-render-{synth_msg.id}"

    # Also store synthesis finding for PDF
    synthesis_finding = DomainFinding(
        domain="synthesis",
        status="complete",
        summary=exec_summary,
        facts=[],
        interpretations=[],
        confidence=sum(
            f.confidence for f in state.domain_findings.values() if f.status != "failed"
        ) / max(len([f for f in state.domain_findings.values() if f.status != "failed"]), 1),
    )

    return {
        "domain_findings": {**state.domain_findings, "synthesis": synthesis_finding},
        "messages": [synth_msg],
    }


# ---------------------------------------------------------------------------
# Node: export_pdf
# ---------------------------------------------------------------------------


async def run_export_pdf(state: OrchestratorState) -> dict:
    """Generate the PDF report and store the download URL."""
    # Extract product name from query (use first 5 words as product name heuristic)
    words = state.user_query.split()
    product_name = " ".join(words[:5]) if words else "Intelligence Report"

    report = FinalReport(
        product_name=product_name,
        domains={k: v for k, v in state.domain_findings.items() if k != "synthesis"},
        executive_summary=state.domain_findings.get("synthesis", DomainFinding(
            domain="synthesis", status="complete", summary="", facts=[], interpretations=[], confidence=0.0
        )).summary,
        failed_domains=state.error_domains,
    )

    try:
        pdf_url = await export_pdf(state.session_id, report)
    except Exception as exc:  # noqa: BLE001
        pdf_url = None
        fail_msg = AIMessage(content=f"PDF export failed: {exc}")
        fail_msg.id = f"do-not-render-{fail_msg.id}"
        return {"pdf_url": pdf_url, "messages": [fail_msg]}

    export_msg = AIMessage(content=f"Report exported to {pdf_url}")
    export_msg.id = f"do-not-render-{export_msg.id}"

    return {"pdf_url": pdf_url, "messages": [export_msg]}


# ---------------------------------------------------------------------------
# Node: stream_response
# ---------------------------------------------------------------------------


def stream_response(state: OrchestratorState) -> dict:
    """Emit the final visible AI message with synthesis + PDF download link."""
    synthesis = state.domain_findings.get("synthesis")
    summary = synthesis.summary if synthesis else "Intelligence analysis complete."

    if state.pdf_url:
        pdf_link = f"\n\n[Download Intelligence Report (PDF)]({state.pdf_url})"
    else:
        pdf_link = "\n\n_(PDF export unavailable — see logs for details.)_"

    domains_run = [d for d in state.domain_findings if d != "synthesis"]
    domains_ok = [d for d in domains_run if d not in state.error_domains]
    domains_failed = state.error_domains

    domain_status = f"\n\n**Domains analysed:** {', '.join(domains_ok)}"
    if domains_failed:
        domain_status += f"\n**Domains failed:** {', '.join(domains_failed)}"

    content = f"{summary}{domain_status}{pdf_link}"
    final_message = AIMessage(content=content)

    return {"messages": [final_message]}


# ---------------------------------------------------------------------------
# Node: update_memory
# ---------------------------------------------------------------------------


async def update_memory(state: OrchestratorState) -> dict:
    """Write session conclusions to Mem0 global thread."""
    synthesis = state.domain_findings.get("synthesis")
    try:
        await mem0_client.set_global_context(
            state.session_id,
            {
                "query": state.user_query,
                "domains_run": state.relevant_domains,
                "failed_domains": state.error_domains,
                "executive_summary": synthesis.summary if synthesis else "",
                "pdf_url": state.pdf_url or "",
            },
        )
    except Exception:  # noqa: BLE001
        pass

    return {}


# ---------------------------------------------------------------------------
# Conditional routing helpers
# ---------------------------------------------------------------------------

def _route_after_classify(state: OrchestratorState) -> str:
    """Route to the first relevant domain or synthesise if none are relevant."""
    next_domain = _next_relevant_domain(state.relevant_domains, None)
    return f"run_{next_domain}" if next_domain else "synthesise"


def _make_route_after(domain: str):
    """Return a routing function that skips to the next relevant domain."""
    def route(state: OrchestratorState) -> str:
        next_domain = _next_relevant_domain(state.relevant_domains, domain)
        return f"run_{next_domain}" if next_domain else "synthesise"

    route.__name__ = f"route_after_{domain}"
    return route


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """Assemble and compile the full Growth Intelligence StateGraph."""

    builder = StateGraph(OrchestratorState)

    # Add static nodes
    builder.add_node("receive_message", receive_message)
    builder.add_node("classify_domains", classify_domains)
    builder.add_node("synthesise", synthesise)
    builder.add_node("export_pdf", run_export_pdf)
    builder.add_node("stream_response", stream_response)
    builder.add_node("update_memory", update_memory)

    # Add domain nodes
    _domain_classes = {
        "market": MarketAgent,
        "competitive": CompetitiveAgent,
        "win_loss": WinLossAgent,
        "pricing": PricingAgent,
        "positioning": PositioningAgent,
        "adjacent": AdjacentAgent,
    }

    for domain, klass in _domain_classes.items():
        builder.add_node(f"run_{domain}", _make_domain_node(domain, klass))

    # Entry point
    builder.set_entry_point("receive_message")

    # Linear flow: receive → classify
    builder.add_edge("receive_message", "classify_domains")

    # Conditional routing: classify → first relevant domain (or synthesise)
    route_map = {f"run_{d}": f"run_{d}" for d in _DOMAIN_ORDER}
    route_map["synthesise"] = "synthesise"
    builder.add_conditional_edges(
        "classify_domains",
        _route_after_classify,
        route_map,
    )

    # Conditional routing between domain nodes: always jump to next relevant domain
    for domain in _DOMAIN_ORDER:
        builder.add_conditional_edges(
            f"run_{domain}",
            _make_route_after(domain),
            route_map,
        )

    # Linear tail: synthesise → export_pdf → stream_response → update_memory → END
    builder.add_edge("synthesise", "export_pdf")
    builder.add_edge("export_pdf", "stream_response")
    builder.add_edge("stream_response", "update_memory")
    builder.add_edge("update_memory", END)

    return builder.compile()
