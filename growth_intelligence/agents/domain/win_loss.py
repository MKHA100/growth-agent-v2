"""
Win / Loss Intelligence domain agent.

Core question: Why are deals being lost? What does the market look like from the buyer?

Primary sources: Reddit OAuth2, HN Algolia, Gemini Search.
PDF artifact: Win/loss reason breakdown.
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


class WinLossAgent(BaseDomainAgent):
    DOMAIN_TAG = "win_loss"

    async def run(self, query: str) -> DomainFinding:
        """Run win/loss intelligence pipeline."""
        # 1. Check for prior work
        prior = await self.get_prior()
        if is_already_answered(prior, query):
            return DomainFinding.from_prior(prior)

        # 2. Form sub-queries focused on buyer perspective
        sub_queries = form_queries(query, self.DOMAIN_TAG)
        sub_queries.append(f"{query} customer success stories case studies")
        sub_queries.append(f"{query} churn reasons buyer feedback reddit site:reddit.com")

        # 3. Gather signals — social is especially important for win/loss
        search_results, scrape_results, social_posts = await gather_signals(
            sub_queries, self.DOMAIN_TAG, include_social=True
        )

        # 4. Chunk and upsert to Pinecone
        chunks = chunk_all(
            search_results, scrape_results, social_posts,
            domain_tag=self.DOMAIN_TAG,
            agent_id="WinLossAgent",
        )
        await self.store_chunks(chunks)

        # 5. Retrieve top-5 relevant chunks (with local fallback)
        top_chunks = await self.recall_or_fallback(query, chunks)

        # 6. Synthesise
        finding = await synthesis_agent.execute(
            top_chunks, DomainFinding, FEW_SHOTS["win_loss"]
        )

        finding = finding.model_copy(update={"domain": self.DOMAIN_TAG})

        # 7. Persist to Mem0
        await self.remember(finding.model_dump(mode="json"))

        return finding
