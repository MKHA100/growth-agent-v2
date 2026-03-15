"""
Positioning & Messaging domain agent.

Core question: Not what to build — how to talk about what already exists.

Primary sources: Meta Ad Library, Firecrawl (landing pages), Gemini Search.
PDF artifact: Messaging gap summary.
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
from schemas.findings import DomainFinding, FEW_SHOTS


class PositioningAgent(BaseDomainAgent):
    DOMAIN_TAG = "positioning"

    async def run(self, query: str) -> DomainFinding:
        """Run positioning & messaging intelligence pipeline."""
        # 1. Check for prior work
        prior = await self.get_prior()
        if is_already_answered(prior, query):
            return DomainFinding.from_prior(prior)

        # 2. Form sub-queries focused on messaging and positioning
        sub_queries = form_queries(query, self.DOMAIN_TAG)
        sub_queries.append(f"{query} homepage value proposition tagline messaging")
        sub_queries.append(f"{query} marketing copy ads creative 2026 how they talk about it")

        # 3. Gather signals — landing page scraping is key for this domain
        search_results, scrape_results, social_posts = await gather_signals(
            sub_queries, self.DOMAIN_TAG, include_social=False  # less social for messaging
        )

        # 4. Chunk and upsert to Pinecone
        chunks = chunk_all(
            search_results, scrape_results, social_posts,
            domain_tag=self.DOMAIN_TAG,
            agent_id="PositioningAgent",
        )
        await self.store_chunks(chunks)

        # 5. Retrieve top-5 relevant chunks
        top_chunks = await self.recall(query)

        # 6. Synthesise
        finding = await synthesis_agent.execute(
            top_chunks, DomainFinding, FEW_SHOTS["positioning"]
        )

        finding = finding.model_copy(update={"domain": self.DOMAIN_TAG})

        # 7. Persist to Mem0
        await self.remember(finding.model_dump(mode="json"))

        return finding
