"""
Competitive Landscape domain agent.

Core question: Who is doing what? Is a given bet worth making?

Primary sources: Firecrawl (competitor pages), Meta Ad Library, Gemini Search.
PDF artifact: Competitor scorecard.
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


class CompetitiveAgent(BaseDomainAgent):
    DOMAIN_TAG = "competitive"

    async def run(self, query: str) -> DomainFinding:
        """Run competitive landscape intelligence pipeline."""
        # 1. Check for prior work
        prior = await self.get_prior()
        if is_already_answered(prior, query):
            return DomainFinding.from_prior(prior)

        # 2. Form sub-queries specific to competitive analysis
        sub_queries = form_queries(query, self.DOMAIN_TAG)
        sub_queries.append(f"{query} G2 Capterra reviews competitor comparison")
        sub_queries.append(f"{query} funding rounds investors Series A B 2025 2026")

        # 3. Gather signals
        search_results, scrape_results, social_posts = await gather_signals(
            sub_queries, self.DOMAIN_TAG, include_social=True
        )

        # 4. Chunk and upsert to Pinecone
        chunks = chunk_all(
            search_results, scrape_results, social_posts,
            domain_tag=self.DOMAIN_TAG,
            agent_id="CompetitiveAgent",
        )
        await self.store_chunks(chunks)

        # 5. Retrieve top-5 relevant chunks
        top_chunks = await self.recall(query)

        # 6. Synthesise
        finding = await synthesis_agent.execute(
            top_chunks, DomainFinding, FEW_SHOTS["competitive"]
        )

        finding = finding.model_copy(update={"domain": self.DOMAIN_TAG})

        # 7. Persist to Mem0
        await self.remember(finding.model_dump(mode="json"))

        return finding
