"""
Shared pipeline utilities used by all domain agents.

Provides:
- is_already_answered: checks if Mem0 prior covers the query
- form_queries: generates domain-scoped sub-queries
- gather_signals: orchestrates search + scrape + social for a set of queries
- chunk_all: converts raw signal content into Chunk objects
"""

from __future__ import annotations

import re
from typing import Any

import agents.micro.scrape as scrape_agent
import agents.micro.search as search_agent
import agents.micro.social as social_agent
import agents.micro.parse as parse_agent
import memory.pinecone_client as pinecone_client
from schemas.findings import Chunk, Post, ScrapeResult, SearchResult


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


def is_already_answered(prior: list[dict[str, Any]], query: str) -> bool:
    """Heuristic check: are the prior findings still relevant for this query?

    Returns True only when the prior contains at least one complete finding.
    Always returns False so the agent reruns on new sessions (prior may be stale).
    In a production system this could do semantic similarity against the query.
    """
    if not prior:
        return False
    # Check if any prior entry has status=complete
    return any(
        entry.get("memory", "").find("status: complete") != -1
        for entry in prior
    )


# ---------------------------------------------------------------------------
# Sub-query formation
# ---------------------------------------------------------------------------

_DOMAIN_QUERY_TEMPLATES: dict[str, list[str]] = {
    "market": [
        "{query} market size trends 2026",
        "{query} industry growth forecast leading indicators",
        "{query} venture capital funding investment trends",
    ],
    "competitive": [
        "{query} top competitors comparison 2026",
        "{query} competitor pricing features",
        "{query} competitive landscape analysis",
    ],
    "win_loss": [
        "{query} customer reviews complaints why switched",
        "{query} win loss analysis buyer reasons",
        "{query} product failures churn reasons Reddit",
    ],
    "pricing": [
        "{query} pricing model comparison willingness to pay",
        "{query} pricing strategy SaaS benchmark",
        "{query} pricing page analysis competitors",
    ],
    "positioning": [
        "{query} messaging positioning unique value proposition",
        "{query} marketing copy differentiation analysis",
        "{query} brand positioning competitor messaging gap",
    ],
    "adjacent": [
        "{query} adjacent market disruption threat 2026",
        "{query} platform encroachment substitute products",
        "{query} emerging competitors adjacent category",
    ],
}


def form_queries(query: str, domain_tag: str) -> list[str]:
    """Generate 3 domain-specific sub-queries from the user query."""
    templates = _DOMAIN_QUERY_TEMPLATES.get(domain_tag, ["{query} {domain_tag} analysis"])
    return [t.format(query=query, domain_tag=domain_tag) for t in templates]


# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------


def extract_urls(results: list[SearchResult]) -> list[str]:
    """Extract unique, non-empty URLs from a list of SearchResult objects."""
    seen: set[str] = set()
    urls: list[str] = []
    for r in results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            urls.append(r.url)
    return urls


# ---------------------------------------------------------------------------
# Signal gathering
# ---------------------------------------------------------------------------


async def gather_signals(
    queries: list[str],
    domain_tag: str,
    include_social: bool = True,
) -> tuple[list[SearchResult], list[ScrapeResult], list[Post]]:
    """Run search → scrape → social for a list of sub-queries, sequentially.

    Args:
        queries: Sub-queries to execute.
        domain_tag: Used for social search queries.
        include_social: Whether to include Reddit + HN results.

    Returns:
        Tuple of (search_results, scrape_results, social_posts).
    """
    search_results: list[SearchResult] = []
    scrape_results: list[ScrapeResult] = []
    social_posts: list[Post] = []

    # 1. Search
    for q in queries:
        try:
            results = await search_agent.execute(q)
            search_results.extend(results)
        except Exception:  # noqa: BLE001
            pass

    # 2. Scrape URLs found in search results (deduplicated)
    urls = extract_urls(search_results)
    for url in urls[:10]:  # cap at 10 scrapes per domain to control costs
        try:
            scraped = await scrape_agent.execute(url)
            if scraped.is_pdf:
                # Route PDF to LlamaParse
                try:
                    parsed = await parse_agent.execute(url)
                    scrape_results.append(
                        ScrapeResult(url=url, markdown=parsed.content, is_pdf=False)
                    )
                except Exception:  # noqa: BLE001
                    scrape_results.append(scraped)
            else:
                scrape_results.append(scraped)
        except Exception:  # noqa: BLE001
            pass

    # 3. Social signals
    if include_social and queries:
        social_query = queries[0]  # use primary query for social search
        try:
            reddit_posts = await social_agent.search_reddit(social_query, limit=15)
            social_posts.extend(reddit_posts)
        except Exception:  # noqa: BLE001
            pass
        try:
            hn_posts = await social_agent.search_hn(social_query, limit=15)
            social_posts.extend(hn_posts)
        except Exception:  # noqa: BLE001
            pass

    return search_results, scrape_results, social_posts


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_all(
    search_results: list[SearchResult],
    scrape_results: list[ScrapeResult],
    social_posts: list[Post],
    domain_tag: str,
    agent_id: str = "",
) -> list[Chunk]:
    """Convert all raw signal content into Chunk objects for Pinecone upsert."""
    chunks: list[Chunk] = []

    for r in search_results:
        if r.raw_content:
            chunks.extend(
                pinecone_client.make_chunks(
                    text=r.raw_content,
                    source_url=r.url or "gemini-search",
                    domain_tag=domain_tag,
                    agent_id=agent_id,
                )
            )

    for s in scrape_results:
        if s.markdown and not s.markdown.startswith("[Scrape failed"):
            chunks.extend(
                pinecone_client.make_chunks(
                    text=s.markdown,
                    source_url=s.url,
                    domain_tag=domain_tag,
                    agent_id=agent_id,
                )
            )

    for p in social_posts:
        text = f"{p.title}\n{p.body}\n" + "\n".join(p.comments)
        if text.strip():
            chunks.extend(
                pinecone_client.make_chunks(
                    text=text,
                    source_url=p.url or f"social:{p.source}",
                    domain_tag=domain_tag,
                    agent_id=agent_id,
                )
            )

    return chunks
