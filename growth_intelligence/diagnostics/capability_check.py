"""Offline capability checks with stubbed external dependencies.

Runs lightweight tests to verify:
- Mem0 prior reconstruction
- Domain skip routing
- Search URL extraction fallback
- Signal gathering (search/scrape/parse/social)
- Domain agent pipeline wiring (search -> chunk -> recall -> synthesis)

No external APIs or packages are required.
"""
from __future__ import annotations

import asyncio
import json
import sys
import traceback
import types
from pathlib import Path
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so local packages resolve.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Stub external modules so imports don't fail in offline environments.
# ---------------------------------------------------------------------------

def _stub_module(name: str, attrs: dict[str, Any] | None = None) -> types.ModuleType:
    if name in sys.modules:
        mod = sys.modules[name]
    else:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    if attrs:
        for key, val in attrs.items():
            setattr(mod, key, val)
    return mod


def install_stubs() -> None:
    # langchain_core.messages
    @dataclass
    class BaseMessage:
        content: Any = ""
        id: str | None = None

    class HumanMessage(BaseMessage):
        pass

    class AIMessage(BaseMessage):
        pass

    core = _stub_module("langchain_core")
    messages = _stub_module(
        "langchain_core.messages",
        {
            "BaseMessage": BaseMessage,
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
        },
    )
    core.messages = messages

    # langgraph.graph + message
    def add_messages(existing, new):
        return (existing or []) + (new or [])

    class StateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            pass

        def add_edge(self, *args, **kwargs):
            pass

        def add_conditional_edges(self, *args, **kwargs):
            pass

        def set_entry_point(self, *args, **kwargs):
            pass

        def compile(self):
            return self

    graph = _stub_module("langgraph.graph", {"END": "__end__", "StateGraph": StateGraph})
    graph_message = _stub_module("langgraph.graph.message", {"add_messages": add_messages})
    langgraph = _stub_module("langgraph")
    langgraph.graph = graph
    langgraph.graph.message = graph_message

    # langchain_anthropic
    class ChatAnthropic:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, *args, **kwargs):
            return AIMessage(content="[]")

    _stub_module("langchain_anthropic", {"ChatAnthropic": ChatAnthropic})

    # langchain_google_genai
    class ChatGoogleGenerativeAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            class Resp:
                content = ""
                response_metadata = {}
                additional_kwargs = {}

            return Resp()

    _stub_module("langchain_google_genai", {"ChatGoogleGenerativeAI": ChatGoogleGenerativeAI})

    # langchain_text_splitters
    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_text(self, text: str):
            return [text]

    _stub_module("langchain_text_splitters", {"RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter})

    # langchain_openai
    class OpenAIEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

        def embed_documents(self, texts):
            return [[0.0] * 3 for _ in texts]

        def embed_query(self, text):
            return [0.0] * 3

    _stub_module("langchain_openai", {"OpenAIEmbeddings": OpenAIEmbeddings})

    # pinecone
    class _Index:
        def upsert(self, *args, **kwargs):
            return None

        def query(self, *args, **kwargs):
            return {"matches": []}

    class Pinecone:
        def __init__(self, *args, **kwargs):
            pass

        def list_indexes(self):
            return []

        def create_index(self, *args, **kwargs):
            return None

        def Index(self, *args, **kwargs):
            return _Index()

    class ServerlessSpec:
        def __init__(self, *args, **kwargs):
            pass

    _stub_module("pinecone", {"Pinecone": Pinecone, "ServerlessSpec": ServerlessSpec})

    # mem0
    class MemoryClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_all(self, *args, **kwargs):
            return []

        def add(self, *args, **kwargs):
            return None

    _stub_module("mem0", {"MemoryClient": MemoryClient})

    # firecrawl
    class FirecrawlApp:
        def __init__(self, *args, **kwargs):
            pass

        def scrape_url(self, *args, **kwargs):
            return {"markdown": ""}

    _stub_module("firecrawl", {"FirecrawlApp": FirecrawlApp})

    # asyncpraw
    class Reddit:
        def __init__(self, *args, **kwargs):
            pass

    _stub_module("asyncpraw", {"Reddit": Reddit})

    # httpx
    class AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *args, **kwargs):
            class Resp:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {"hits": []}

            return Resp()

    _stub_module("httpx", {"AsyncClient": AsyncClient})

    # llama_parse
    class LlamaParse:
        def __init__(self, *args, **kwargs):
            pass

        async def aload_data(self, *args, **kwargs):
            return []

    _stub_module("llama_parse", {"LlamaParse": LlamaParse})

    # anthropic
    class Anthropic:
        def __init__(self, *args, **kwargs):
            pass

        class messages:
            @staticmethod
            def create(*args, **kwargs):
                class Resp:
                    content = []

                return Resp()

    _stub_module("anthropic", {"Anthropic": Anthropic})

    # langfuse.langchain
    class CallbackHandler:
        def __init__(self, *args, **kwargs):
            pass

    lf = _stub_module("langfuse")
    lf_lc = _stub_module("langfuse.langchain", {"CallbackHandler": CallbackHandler})
    lf.langchain = lf_lc

    # jinja2
    class Environment:
        def __init__(self, *args, **kwargs):
            self.filters = {}

        def get_template(self, *args, **kwargs):
            class T:
                def render(self, *args, **kwargs):
                    return ""

            return T()

    class FileSystemLoader:
        def __init__(self, *args, **kwargs):
            pass

    def select_autoescape(*args, **kwargs):
        return None

    _stub_module(
        "jinja2",
        {
            "Environment": Environment,
            "FileSystemLoader": FileSystemLoader,
            "select_autoescape": select_autoescape,
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_from_prior_roundtrip() -> None:
    from schemas.findings import DomainFinding

    finding = {
        "domain": "market",
        "status": "complete",
        "summary": "Test summary",
        "facts": [],
        "interpretations": [],
        "confidence": 0.5,
        "error_reason": None,
    }
    prior = [{"metadata": {"finding": finding}}]
    restored = DomainFinding.from_prior(prior)
    assert restored.domain == "market"
    assert restored.summary == "Test summary"


def test_is_already_answered_metadata() -> None:
    from agents._domain_utils import is_already_answered

    prior = [{"metadata": {"finding": {"status": "complete"}}}]
    assert is_already_answered(prior, "q") is True


def test_search_url_fallback() -> None:
    from agents.micro import search as search_agent

    class DummyResponse:
        content = "See https://example.com/report.pdf for details."
        response_metadata = {}
        additional_kwargs = {}

    results = search_agent._parse_grounding_results(DummyResponse(), "q")
    assert results, "Expected at least one SearchResult"
    assert results[0].url.startswith("https://example.com"), "URL extraction fallback failed"


def test_routing_next_relevant() -> None:
    from orchestrator import graph as og

    assert og._next_relevant_domain(["adjacent"], None) == "adjacent"
    assert og._next_relevant_domain(["market", "adjacent"], "market") == "adjacent"
    assert og._next_relevant_domain(["market"], "market") is None


async def test_gather_signals_pipeline() -> None:
    from agents import _domain_utils as du
    from schemas.findings import SearchResult, ScrapeResult, ParseResult, Post

    calls = {"search": 0, "scrape": 0, "parse": 0, "reddit": 0, "hn": 0}

    async def fake_search(q: str):
        calls["search"] += 1
        return [SearchResult(query=q, url="https://example.com/doc.pdf", raw_content="r")]

    async def fake_scrape(url: str):
        calls["scrape"] += 1
        return ScrapeResult(url=url, markdown="", is_pdf=True)

    async def fake_parse(url: str):
        calls["parse"] += 1
        return ParseResult(url=url, content="parsed pdf")

    async def fake_reddit(query: str, limit: int = 15):
        calls["reddit"] += 1
        return [Post(title="t", body="b", url="u", comments=[], source="reddit")]

    async def fake_hn(query: str, limit: int = 15):
        calls["hn"] += 1
        return [Post(title="t", body="b", url="u", comments=[], source="hackernews")]

    du.search_agent.execute = fake_search
    du.scrape_agent.execute = fake_scrape
    du.parse_agent.execute = fake_parse
    du.social_agent.search_reddit = fake_reddit
    du.social_agent.search_hn = fake_hn

    search_results, scrape_results, social_posts = await du.gather_signals(
        ["query"], "market", include_social=True
    )

    assert calls["search"] == 1
    assert calls["scrape"] == 1
    assert calls["parse"] == 1
    assert calls["reddit"] == 1
    assert calls["hn"] == 1
    assert search_results
    assert scrape_results
    assert social_posts


async def test_domain_agent_pipeline() -> None:
    import agents.domain.competitive as comp
    import memory.mem0_client as mem0_client
    import memory.pinecone_client as pinecone_client
    from schemas.findings import Chunk, DomainFinding

    # Patch memory + pinecone to avoid external calls
    async def fake_get_domain_thread(session_id: str, domain: str):
        return []

    async def fake_append_domain_finding(session_id: str, domain: str, finding: dict[str, Any]):
        return None

    async def fake_upsert_chunks(session_id: str, chunks: list[Chunk]):
        return None

    async def fake_query_chunks(session_id: str, query: str, domain_tag: str | None = None, top_k: int = 5):
        return [
            Chunk(
                id="c1",
                text="chunk",
                source_url="https://example.com",
                domain_tag=domain_tag or "competitive",
                timestamp=__import__("datetime").datetime.utcnow(),
                agent_id="CompetitiveAgent",
            )
        ]

    mem0_client.get_domain_thread = fake_get_domain_thread
    mem0_client.append_domain_finding = fake_append_domain_finding
    pinecone_client.upsert_chunks = fake_upsert_chunks
    pinecone_client.query_chunks = fake_query_chunks

    # Patch domain-local helpers
    async def fake_gather_signals(queries, domain_tag, include_social=True):
        return [], [], []

    def fake_chunk_all(*args, **kwargs):
        return []

    async def fake_synthesis_execute(chunks, schema, few_shots):
        return DomainFinding(
            domain="competitive",
            status="complete",
            summary="ok",
            facts=[],
            interpretations=[],
            confidence=0.7,
            error_reason=None,
        )

    comp.gather_signals = fake_gather_signals
    comp.chunk_all = fake_chunk_all
    comp.synthesis_agent.execute = fake_synthesis_execute
    comp.is_already_answered = lambda prior, query: False

    agent = comp.CompetitiveAgent("session")
    finding = await agent.run("query")
    assert finding.domain == "competitive"
    assert finding.summary == "ok"


def run() -> int:
    failures: list[str] = []

    tests = [
        ("from_prior_roundtrip", test_from_prior_roundtrip),
        ("is_already_answered_metadata", test_is_already_answered_metadata),
        ("search_url_fallback", test_search_url_fallback),
        ("routing_next_relevant", test_routing_next_relevant),
        ("gather_signals_pipeline", lambda: asyncio.run(test_gather_signals_pipeline())),
        ("domain_agent_pipeline", lambda: asyncio.run(test_domain_agent_pipeline())),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append(name)
            print(f"FAIL {name}: {exc}")
            print(traceback.format_exc())

    if failures:
        print("\nFailures:")
        for name in failures:
            print(f"- {name}")
        return 1

    print("\nAll capability checks passed.")
    return 0


if __name__ == "__main__":
    install_stubs()
    raise SystemExit(run())
