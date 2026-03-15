"""
Adjacent Market Collision domain agent.

Core question: What is coming from outside your category that you are not watching?

Primary sources: USPTO Patents, HN Algolia, SerpAPI News.
PDF artifact: Adjacent threat list.
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


class AdjacentAgent(BaseDomainAgent):
    DOMAIN_TAG = "adjacent"

    async def run(self, query: str) -> DomainFinding:
        """Run adjacent market collision intelligence pipeline."""
        # 1. Check for prior work
        prior = await self.get_prior()
        if is_already_answered(prior, query):
            return DomainFinding.from_prior(prior)

        # 2. Form sub-queries focused on adjacency threats
        sub_queries = form_queries(query, self.DOMAIN_TAG)
        sub_queries.append(f"{query} platform entering market disruption threat big tech")
        sub_queries.append(f"{query} adjacent market patents 2025 2026 new entrant")

        # 3. Gather signals — HN is particularly rich for adjacent threat signals
        search_results, scrape_results, social_posts = await gather_signals(
            sub_queries, self.DOMAIN_TAG, include_social=True
        )

        # 4. Chunk and upsert to Pinecone
        chunks = chunk_all(
            search_results, scrape_results, social_posts,
            domain_tag=self.DOMAIN_TAG,
            agent_id="AdjacentAgent",
        )
        await self.store_chunks(chunks)

        # 5. Retrieve top-5 relevant chunks
        top_chunks = await self.recall(query)

        # 6. Synthesise
        finding = await synthesis_agent.execute(
            top_chunks, DomainFinding, FEW_SHOTS["adjacent"]
        )

        finding = finding.model_copy(update={"domain": self.DOMAIN_TAG})

        # 7. Persist to Mem0
        await self.remember(finding.model_dump(mode="json"))

        return finding
