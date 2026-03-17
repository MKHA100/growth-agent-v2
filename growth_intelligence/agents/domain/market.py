"""
Market & Trend Sensing domain agent.

Core question: Where is the category heading? What are the leading indicators?

Primary sources: Gemini Search, SerpAPI Trends, Gemini USPTO patent queries.
PDF artifact: Trend summary + leading indicators section.
"""

from __future__ import annotations

from agents._domain_utils import (
    chunk_all,
    form_queries,
    gather_signals,
    is_already_answered,
)
from agents.base import BaseDomainAgent
from agents.micro import synthesis as synthesis_agent
from schemas.findings import DomainFinding, FEW_SHOTS, PartialFinding


class MarketAgent(BaseDomainAgent):
    DOMAIN_TAG = "market"

    async def run(self, query: str) -> DomainFinding:
        """Run market & trend intelligence pipeline."""
        # 1. Check for prior work
        prior = await self.get_prior()
        if is_already_answered(prior, query):
            return DomainFinding.from_prior(prior)

        # 2. Form sub-queries specific to market & trend sensing
        sub_queries = form_queries(query, self.DOMAIN_TAG)
        # Add patent-specific query for market signals
        sub_queries.append(f"{query} patent filings innovation trends 2025 2026")

        # 3. Gather signals (search + scrape + social)
        search_results, scrape_results, social_posts = await gather_signals(
            sub_queries, self.DOMAIN_TAG, include_social=True
        )

        # 4. Chunk and upsert to Pinecone
        chunks = chunk_all(
            search_results, scrape_results, social_posts,
            domain_tag=self.DOMAIN_TAG,
            agent_id="MarketAgent",
        )
        await self.store_chunks(chunks)

        # 5. Retrieve top-5 relevant chunks (with local fallback)
        top_chunks = await self.recall_or_fallback(query, chunks)

        # 6. Synthesise via Pydantic few-shot loop
        finding = await synthesis_agent.execute(
            top_chunks, DomainFinding, FEW_SHOTS["market"]
        )

        # Ensure domain tag is set correctly
        finding = finding.model_copy(update={"domain": self.DOMAIN_TAG})

        # 7. Persist to Mem0
        await self.remember(finding.model_dump(mode="json"))

        return finding
