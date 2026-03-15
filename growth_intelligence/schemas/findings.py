"""
Pydantic schemas for all Growth Intelligence Agent data models.

All finding schemas, report schemas, few-shot examples, and chunk models
are defined here and shared across agents, orchestrator, and PDF export.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------


class Fact(BaseModel):
    """A single grounded, source-attributed claim."""

    claim: str = Field(..., description="The factual claim extracted from source material.")
    source_url: str = Field(..., description="URL of the source this claim was retrieved from.")
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0.")


class Chunk(BaseModel):
    """A single text chunk stored in Pinecone."""

    id: str = Field(..., description="Unique chunk ID (uuid4 or hash).")
    text: str = Field(..., description="The chunk text content.")
    source_url: str
    domain_tag: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = Field(default="", description="Which agent produced this chunk.")
    embedding: list[float] | None = Field(default=None, exclude=True)


# ---------------------------------------------------------------------------
# Domain finding schemas
# ---------------------------------------------------------------------------


class DomainFinding(BaseModel):
    """Structured output from a single domain agent."""

    domain: str
    status: Literal["complete", "partial", "failed"] = "complete"
    summary: str = Field(..., description="2–3 sentence synthesis of what was found.")
    facts: list[Fact] = Field(default_factory=list, description="Grounded, source-attributed facts.")
    interpretations: list[str] = Field(
        default_factory=list,
        description="Agent inferences — clearly labelled as interpretations, not facts.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0.")
    error_reason: str | None = None

    @classmethod
    def from_prior(cls, prior: list[dict]) -> "DomainFinding":
        """Reconstruct a DomainFinding from Mem0 prior thread data."""
        if not prior:
            raise ValueError("Cannot reconstruct from empty prior.")
        latest = prior[-1]

        # If the latest record already looks like a DomainFinding dict, use it directly.
        if isinstance(latest, dict) and "domain" in latest and "summary" in latest:
            return cls(**latest)

        # Mem0 stores structured data in metadata when available.
        metadata = latest.get("metadata", {}) if isinstance(latest, dict) else {}
        finding = metadata.get("finding") or metadata.get("finding_json")

        if isinstance(finding, str):
            try:
                finding = json.loads(finding)
            except json.JSONDecodeError:
                finding = None

        if isinstance(finding, dict):
            return cls(**finding)

        raise ValueError("Mem0 prior does not contain a reconstructable DomainFinding.")


class PartialFinding(DomainFinding):
    """A finding produced when a domain agent fails entirely."""

    status: Literal["failed"] = "failed"  # type: ignore[assignment]
    summary: str = "Domain agent failed — see error_reason for details."
    facts: list[Fact] = Field(default_factory=list)
    interpretations: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class FinalReport(BaseModel):
    """The complete cross-domain intelligence report."""

    product_name: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    domains: dict[str, DomainFinding] = Field(default_factory=dict)
    executive_summary: str = ""
    failed_domains: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Search / Scrape micro-agent outputs
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """Output from the Gemini grounded search micro-agent."""

    query: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    raw_content: str = ""


class ScrapeResult(BaseModel):
    """Output from the Firecrawl scrape micro-agent."""

    url: str
    markdown: str = ""
    is_pdf: bool = False


class ParseResult(BaseModel):
    """Output from the LlamaParse deep-parse micro-agent."""

    url: str
    content: str = ""


class Post(BaseModel):
    """A social media post from Reddit or HN."""

    title: str
    body: str = ""
    url: str = ""
    comments: list[str] = Field(default_factory=list)
    source: Literal["reddit", "hackernews"] = "reddit"


# ---------------------------------------------------------------------------
# Few-shot examples for synthesis agent
# ---------------------------------------------------------------------------

FEW_SHOTS: dict[str, list[dict]] = {
    "market": [
        {
            "raw_input": (
                "Gartner 2026 Hype Cycle: AI SDR tools at peak of inflated expectations. "
                "Market size $2.1B in 2025, projected $8.4B by 2028 (CAGR 58%). "
                "Leading indicators: VC funding in category up 3x YoY, 47 new entrants since Q3 2024."
            ),
            "expected_output": {
                "domain": "market",
                "status": "complete",
                "summary": (
                    "The AI SDR category is at peak hype with explosive growth projected. "
                    "VC activity and new entrants signal a highly competitive window. "
                    "Leading indicators suggest the category will consolidate in 2026-2027."
                ),
                "facts": [
                    {
                        "claim": "AI SDR market projected to reach $8.4B by 2028 at 58% CAGR",
                        "source_url": "https://www.gartner.com/en/research",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.85,
                    }
                ],
                "interpretations": [
                    "Category likely to see price compression as 47 new entrants compete for share",
                    "First-mover advantage window closing; differentiation on quality over quantity",
                ],
                "confidence": 0.82,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "USPTO patent filings for 'conversational sales AI' up 210% YoY. "
                "Google Trends: 'AI sales agent' search volume +340% in 12 months. "
                "ProductHunt: 14 AI SDR launches in February 2026 alone."
            ),
            "expected_output": {
                "domain": "market",
                "status": "complete",
                "summary": (
                    "Patent activity and search trends confirm the AI sales agent category "
                    "is in rapid expansion. ProductHunt launch velocity indicates strong "
                    "builder interest but also commoditisation risk."
                ),
                "facts": [
                    {
                        "claim": "USPTO filings for 'conversational sales AI' increased 210% YoY",
                        "source_url": "https://patents.google.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.90,
                    }
                ],
                "interpretations": [
                    "High patent activity suggests large players building IP moats",
                    "14 ProductHunt launches in a single month indicates commoditisation pressure",
                ],
                "confidence": 0.88,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Forrester Wave Q1 2026: No clear leader segment yet in AI SDR. "
                "3 vendors in Strong Performers quadrant. "
                "Enterprise buyers report avg 18-month evaluation cycles."
            ),
            "expected_output": {
                "domain": "market",
                "status": "complete",
                "summary": (
                    "No dominant vendor has emerged in the AI SDR space as of Q1 2026. "
                    "Long enterprise evaluation cycles create a window for mid-market focus. "
                    "The market remains fragmented across Strong Performers."
                ),
                "facts": [
                    {
                        "claim": "Forrester Wave Q1 2026 shows no leader segment in AI SDR category",
                        "source_url": "https://www.forrester.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.92,
                    }
                ],
                "interpretations": [
                    "No leader = window for category-defining positioning by a new entrant",
                    "18-month enterprise eval cycles suggest PLG mid-market is faster path to revenue",
                ],
                "confidence": 0.85,
                "error_reason": None,
            },
        },
    ],
    "competitive": [
        {
            "raw_input": (
                "Lilian AI SDR pricing page: Starter $499/mo, 500 sequences. "
                "Pro $999/mo, unlimited sequences + CRM sync. "
                "Enterprise custom. Founded 2024, $12M Series A."
            ),
            "expected_output": {
                "domain": "competitive",
                "status": "complete",
                "summary": (
                    "Lilian offers a three-tier subscription starting at $499/month. "
                    "The Pro tier at $999 targets mid-market with CRM integration. "
                    "Recent Series A indicates they are scaling sales capacity."
                ),
                "facts": [
                    {
                        "claim": "Lilian AI SDR Starter tier is priced at $499/month with 500 sequences",
                        "source_url": "https://lilian.ai/pricing",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.95,
                    }
                ],
                "interpretations": [
                    "Lilian is targeting mid-market with the Pro tier at $999",
                    "Custom enterprise tier suggests they are pursuing upmarket expansion post-Series A",
                ],
                "confidence": 0.90,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Outreach.io homepage: 'The #1 Sales Execution Platform'. "
                "Features: sequences, deal intelligence, forecasting. "
                "Pricing: not public. G2 rating: 4.3/5 from 3,400 reviews."
            ),
            "expected_output": {
                "domain": "competitive",
                "status": "complete",
                "summary": (
                    "Outreach positions as a full sales execution platform, not a point AI SDR solution. "
                    "High G2 review volume indicates strong market penetration. "
                    "Opaque pricing is typical for enterprise-first vendors."
                ),
                "facts": [
                    {
                        "claim": "Outreach.io has 4.3/5 rating from 3,400 G2 reviews as of March 2026",
                        "source_url": "https://www.g2.com/products/outreach/reviews",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.93,
                    }
                ],
                "interpretations": [
                    "Outreach's breadth makes them a platform competitor, not a direct AI SDR substitute",
                    "Lack of public pricing signals enterprise-only go-to-market",
                ],
                "confidence": 0.87,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Apollo.io changelog March 2026: released AI-written personalised emails beta. "
                "Pricing: Free tier + $49/mo individual + $99/mo team. "
                "240M contact database."
            ),
            "expected_output": {
                "domain": "competitive",
                "status": "complete",
                "summary": (
                    "Apollo is adding AI writing features to its database-first model at aggressive prices. "
                    "The free tier creates a large funnel that converts to paid AI features. "
                    "240M contacts gives them a defensible data moat."
                ),
                "facts": [
                    {
                        "claim": "Apollo.io launched AI personalised email beta in March 2026 at $49/mo individual",
                        "source_url": "https://apollo.io/changelog",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.91,
                    }
                ],
                "interpretations": [
                    "Apollo's data moat + AI layer is a dangerous combination for pure-play AI SDR vendors",
                    "Aggressive free tier will suppress WTP for standalone AI SDR tools",
                ],
                "confidence": 0.89,
                "error_reason": None,
            },
        },
    ],
    "win_loss": [
        {
            "raw_input": (
                "r/sales thread: 'We switched from [AI SDR vendor] because the emails felt robotic. "
                "Prospects kept replying saying they knew it was AI.' 47 upvotes. "
                "Top comment: 'Quality > quantity. One good email beats 100 AI blasts.'"
            ),
            "expected_output": {
                "domain": "win_loss",
                "status": "complete",
                "summary": (
                    "Buyers are losing trust in AI-generated outreach as detection improves. "
                    "The dominant loss reason is perceived inauthenticity. "
                    "Sellers gravitate toward quality-over-quantity messaging strategies."
                ),
                "facts": [
                    {
                        "claim": "Reddit/sales community reports AI SDR email detection as primary rejection reason",
                        "source_url": "https://reddit.com/r/sales",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.80,
                    }
                ],
                "interpretations": [
                    "Authenticity is now a key WTP driver — vendors claiming 'human-like' outreach will win",
                    "Volume-based pricing models may be losing appeal vs quality-based SLAs",
                ],
                "confidence": 0.78,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "HN thread 'Ask HN: Why did your company stop using AI SDRs?': "
                "Top response: 'CRM integration was broken, leads disappeared.' "
                "Second: 'Too expensive for the reply rate we got.' "
                "Third: 'Our AEs hated the handoff quality.'"
            ),
            "expected_output": {
                "domain": "win_loss",
                "status": "complete",
                "summary": (
                    "HackerNews buyers cite CRM integration failures and poor AE handoff quality as "
                    "primary churn drivers. Cost-to-reply-rate ROI concerns are the second most common "
                    "reason for discontinuing AI SDR tools."
                ),
                "facts": [
                    {
                        "claim": "HN community identifies broken CRM integration as top AI SDR churn reason",
                        "source_url": "https://news.ycombinator.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.76,
                    }
                ],
                "interpretations": [
                    "CRM-native or deeply integrated AI SDR tools have significant retention advantage",
                    "AE handoff quality is an underrated differentiation vector in vendor selection",
                ],
                "confidence": 0.75,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "G2 reviews for competitor X: 'The tool books meetings but our AEs say the leads "
                "are not qualified.' 'Set-up took 3 weeks and required a dedicated CSM.' "
                "'Pricing is fair but the ROI is unclear after 6 months.'"
            ),
            "expected_output": {
                "domain": "win_loss",
                "status": "complete",
                "summary": (
                    "Lead quality and long onboarding time are the dominant loss signals in G2 reviews. "
                    "Buyers want pre-qualified meetings, not just booked slots. "
                    "ROI ambiguity at 6 months is a retention risk."
                ),
                "facts": [
                    {
                        "claim": "G2 reviews highlight 3-week onboarding and unqualified leads as top complaints",
                        "source_url": "https://www.g2.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.83,
                    }
                ],
                "interpretations": [
                    "Fast time-to-value (<1 week onboarding) is a strong differentiator in this market",
                    "Vendors offering meeting quality guarantees or SLAs will see lower churn",
                ],
                "confidence": 0.81,
                "error_reason": None,
            },
        },
    ],
    "pricing": [
        {
            "raw_input": (
                "Competitor pricing comparison: Tool A $299/mo (50 sequences), "
                "Tool B $799/mo (unlimited + analytics), Tool C $0 freemium + $149 AI add-on. "
                "Reddit pricing thread: 'I'd pay $500/mo if it actually booked qualified meetings.'"
            ),
            "expected_output": {
                "domain": "pricing",
                "status": "complete",
                "summary": (
                    "AI SDR pricing spans $149 add-ons to $800/month subscriptions with no clear consensus model. "
                    "Buyer WTP is outcome-anchored — qualified meetings, not sequence volume. "
                    "Freemium entry is becoming table stakes in the category."
                ),
                "facts": [
                    {
                        "claim": "AI SDR tools range from $149/mo add-on to $799/mo full platform in March 2026",
                        "source_url": "https://reddit.com/r/sales",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.88,
                    }
                ],
                "interpretations": [
                    "Outcome-based pricing (per qualified meeting) may command premium over seat-based models",
                    "Freemium with AI upsell (Tool C model) creates sticky user base before conversion",
                ],
                "confidence": 0.83,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "SaaS pricing research 2026: Usage-based pricing adoption in AI tooling up to 62% of new products. "
                "Annual contracts down 8% YoY as buyers prefer monthly flexibility. "
                "AI SDR churn at 6 months: 34% on monthly, 12% on annual."
            ),
            "expected_output": {
                "domain": "pricing",
                "status": "complete",
                "summary": (
                    "Usage-based pricing is becoming dominant in AI tooling, with 62% adoption for new products. "
                    "Monthly flexibility preference is rising but annual contracts dramatically reduce churn. "
                    "Hybrid models offering monthly with annual incentive may optimise both."
                ),
                "facts": [
                    {
                        "claim": "Usage-based pricing adopted by 62% of new AI tool products in 2026",
                        "source_url": "https://openviewpartners.com/saas-pricing-2026",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.87,
                    }
                ],
                "interpretations": [
                    "Usage-based models align incentives but create revenue unpredictability for vendors",
                    "Annual plan churn at 12% vs 34% monthly justifies significant annual discount to buyers",
                ],
                "confidence": 0.84,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Gartner buyer survey 2026: 71% of AI tool buyers say 'price transparency' affects vendor selection. "
                "42% abandoned evaluation because pricing required a sales call. "
                "'Self-serve trial with clear upgrade path' rated #1 buying preference."
            ),
            "expected_output": {
                "domain": "pricing",
                "status": "complete",
                "summary": (
                    "Price transparency is now a selection criterion for 71% of AI tool buyers. "
                    "Sales-gated pricing creates abandonment: 42% walk away. "
                    "Self-serve trials with clear upgrade paths are the highest-rated buying experience."
                ),
                "facts": [
                    {
                        "claim": "71% of AI tool buyers cite price transparency as a vendor selection factor",
                        "source_url": "https://www.gartner.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.91,
                    }
                ],
                "interpretations": [
                    "Hiding pricing behind sales calls is a conversion killer for buyers who have already researched",
                    "Public pricing page with clear tier differentiation is a competitive advantage",
                ],
                "confidence": 0.89,
                "error_reason": None,
            },
        },
    ],
    "positioning": [
        {
            "raw_input": (
                "Competitor homepage headlines: 'AI That Books Meetings For You', "
                "'The SDR That Never Sleeps', '10x Your Pipeline With AI'. "
                "Meta ads: mostly performance claims, reply rate stats, case study videos."
            ),
            "expected_output": {
                "domain": "positioning",
                "status": "complete",
                "summary": (
                    "Competitors rely on volume and performance claim messaging. "
                    "No vendor owns the 'quality outreach' or 'buyer-centric AI' positioning space. "
                    "Meta ad creative is saturated with similar stat-based proof points."
                ),
                "facts": [
                    {
                        "claim": "Top 5 AI SDR competitors use volume/performance headlines with no quality differentiation",
                        "source_url": "https://facebook.com/ads/library",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.86,
                    }
                ],
                "interpretations": [
                    "'Human-quality AI outreach' or 'The SDR that buyers actually respond to' is an ownable gap",
                    "Buyer-centric messaging (benefits to the prospect, not just the seller) is entirely absent",
                ],
                "confidence": 0.84,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Landing page analysis: 3 competitors use 'pipeline' as primary value prop. "
                "2 use 'book more meetings'. "
                "None mention integration depth, data privacy, or compliance as headline claims."
            ),
            "expected_output": {
                "domain": "positioning",
                "status": "complete",
                "summary": (
                    "Pipeline growth and meeting booking are the saturated category claims. "
                    "Integration depth, data privacy, and compliance are unowned positioning territory. "
                    "Enterprise buyers increasingly care about these secondary claims."
                ),
                "facts": [
                    {
                        "claim": "Zero top AI SDR competitors use integration depth or compliance as headline positioning",
                        "source_url": "https://firecrawl.dev",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.88,
                    }
                ],
                "interpretations": [
                    "Security/compliance-first positioning could differentiate strongly for enterprise segment",
                    "Deep integration narrative ('the SDR that lives inside your CRM') is entirely unowned",
                ],
                "confidence": 0.85,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Gemini search: top content ranking for 'best AI SDR 2026' is dominated by listicles. "
                "Vendor blogs focus on feature announcements. "
                "No vendor owns a consistent thought leadership category around 'the future of sales'."
            ),
            "expected_output": {
                "domain": "positioning",
                "status": "complete",
                "summary": (
                    "SEO positioning is commoditised via listicle dominance. "
                    "No AI SDR vendor owns a thought leadership narrative about the future of sales. "
                    "Feature-announcement content is the dominant vendor blog strategy."
                ),
                "facts": [
                    {
                        "claim": "Listicle content dominates Google SERPs for 'best AI SDR 2026' with no vendor thought leadership",
                        "source_url": "https://google.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.82,
                    }
                ],
                "interpretations": [
                    "Publishing original research on 'AI SDR effectiveness benchmarks' could own a search category",
                    "Category-creation content strategy is available and uncontested",
                ],
                "confidence": 0.80,
                "error_reason": None,
            },
        },
    ],
    "adjacent": [
        {
            "raw_input": (
                "HN Show: 'We built an AI that writes entire sales playbooks from CRM data.' "
                "USPTO: 12 patents filed by Salesforce for 'autonomous revenue generation agent' in 2025-2026. "
                "News: Microsoft Copilot for Sales v3 announced at Ignite 2025."
            ),
            "expected_output": {
                "domain": "adjacent",
                "status": "complete",
                "summary": (
                    "Platform giants (Salesforce, Microsoft) are building autonomous revenue agents that will "
                    "commoditise point AI SDR tools. Salesforce's 12 patent filings signal serious IP investment. "
                    "Microsoft Copilot v3 brings AI SDR capabilities to the entire Office 365 install base."
                ),
                "facts": [
                    {
                        "claim": "Salesforce filed 12 patents for 'autonomous revenue generation agent' in 2025-2026",
                        "source_url": "https://patents.google.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.92,
                    }
                ],
                "interpretations": [
                    "CRM-native AI SDR from Salesforce/HubSpot is the existential threat to standalone vendors",
                    "Niche differentiation (e.g., specific vertical expertise) is the only durable moat",
                ],
                "confidence": 0.88,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "HN thread: 'Voice AI cold calling is replacing email SDRs'. "
                "Startup Bland.ai raised $22M for AI phone agents. "
                "Tavus raised $18M for AI video personalisation at scale."
            ),
            "expected_output": {
                "domain": "adjacent",
                "status": "complete",
                "summary": (
                    "Voice AI and personalised video are the adjacent modalities threatening email-first AI SDR. "
                    "Bland.ai ($22M) and Tavus ($18M) represent well-funded alternatives to text outreach. "
                    "Multi-modal outreach platforms may commoditise single-channel AI SDR tools."
                ),
                "facts": [
                    {
                        "claim": "Bland.ai raised $22M for AI phone agent platform in 2025-2026",
                        "source_url": "https://techcrunch.com",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.90,
                    }
                ],
                "interpretations": [
                    "Email-only AI SDR tools are vulnerable to multi-channel platform consolidation",
                    "Adding voice or video modality is a strategic hedge against channel commoditisation",
                ],
                "confidence": 0.86,
                "error_reason": None,
            },
        },
        {
            "raw_input": (
                "Research: 'Buyer intent data platforms' (G2, Bombora, 6sense) adding AI outreach natively. "
                "6sense Q4 2025 update: one-click AI sequence from intent signal. "
                "Bombora partnership with Clay for enrichment-triggered AI outbound."
            ),
            "expected_output": {
                "domain": "adjacent",
                "status": "complete",
                "summary": (
                    "Intent data platforms are closing the loop to outreach, bypassing standalone AI SDR tools. "
                    "6sense's one-click AI sequence and Bombora-Clay partnership reduce the need for a separate "
                    "AI SDR tool for intent-driven outbound."
                ),
                "facts": [
                    {
                        "claim": "6sense added one-click AI sequence generation from intent signals in Q4 2025",
                        "source_url": "https://6sense.com/blog",
                        "retrieved_at": "2026-03-15T09:00:00Z",
                        "confidence": 0.89,
                    }
                ],
                "interpretations": [
                    "Intent-to-outreach platforms collapse the AI SDR value chain into existing buyer workflows",
                    "Differentiation via proprietary intent signal integration may be more defensible than AI writing quality",
                ],
                "confidence": 0.85,
                "error_reason": None,
            },
        },
    ],
}
