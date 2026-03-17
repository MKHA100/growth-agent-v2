"""
Pricing & Packaging domain agent.

Core question: Is the pricing model right? Where is WTP shifting?

Primary sources: Firecrawl (pricing pages), Reddit pricing threads.
PDF artifact: Pricing matrix.
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


class PricingAgent(BaseDomainAgent):
    DOMAIN_TAG = "pricing"

    async def run(self, query: str) -> DomainFinding:
        """Run pricing & packaging intelligence pipeline."""
        # 1. Check for prior work
        prior = await self.get_prior()
        if is_already_answered(prior, query):
            return DomainFinding.from_prior(prior)

        # 2. Form sub-queries focused on pricing signals
        sub_queries = form_queries(query, self.DOMAIN_TAG)
        sub_queries.append(f"{query} pricing page tiers monthly annual plans")
        sub_queries.append(f"site:reddit.com {query} pricing too expensive worth it")

        # 3. Gather signals
        search_results, scrape_results, social_posts = await gather_signals(
            sub_queries, self.DOMAIN_TAG, include_social=True
        )

        # 4. Chunk and upsert to Pinecone
        chunks = chunk_all(
            search_results, scrape_results, social_posts,
            domain_tag=self.DOMAIN_TAG,
            agent_id="PricingAgent",
        )
        await self.store_chunks(chunks)

        # 5. Retrieve top-5 relevant chunks (with local fallback)
        top_chunks = await self.recall_or_fallback(query, chunks)

        # 6. Synthesise
        finding = await synthesis_agent.execute(
            top_chunks, DomainFinding, FEW_SHOTS["pricing"]
        )

        finding = finding.model_copy(update={"domain": self.DOMAIN_TAG})

        # 7. Persist to Mem0
        await self.remember(finding.model_dump(mode="json"))

        return finding
