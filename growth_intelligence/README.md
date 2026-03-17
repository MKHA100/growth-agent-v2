# Growth Intelligence Agent

A sequential multi-agent system for product and market intelligence, built on LangGraph, Gemini, Claude, Firecrawl, Mem0, and Pinecone. Delivers a PDF intelligence report via a downloadable link in the chat interface.

## Stack

| Layer          | Technology                                                               |
| -------------- | ------------------------------------------------------------------------ |
| Chat UI        | [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) (Next.js) |
| Orchestration  | LangGraph `StateGraph`                                                   |
| Primary LLM    | Claude Sonnet 4.5 (Anthropic)                                            |
| Search LLM     | Gemini Flash 2.0 with Google Search grounding                            |
| Embeddings     | `text-embedding-3-large` (OpenAI)                                        |
| Vector store   | Pinecone (`top_k=5`)                                                     |
| Thread memory  | Mem0 (in-place updates)                                                  |
| Web scraping   | Firecrawl                                                                |
| Social signals | Reddit OAuth2 (`asyncpraw`) + HN Algolia                                 |
| Deep parse     | LlamaParse                                                               |
| PDF export     | WeasyPrint + Jinja2                                                      |
| Observability  | LangFuse                                                                 |

## Six Intelligence Domains

1. **Market & Trend Sensing** — Where is the category heading?
2. **Competitive Landscape** — Who is doing what?
3. **Win / Loss Intelligence** — Why are deals being lost?
4. **Pricing & Packaging** — Is the pricing model right?
5. **Positioning & Messaging** — How to talk about what exists?
6. **Adjacent Market Collision** — What is coming from outside?

## Quick Start

### Backend

```bash
cd growth_intelligence

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# → fill in all API keys in .env

# Start LangGraph dev server
langgraph dev
# → serves at http://localhost:2024
```

### Frontend

```bash
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install

# Create .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:2024" > .env.local
echo "NEXT_PUBLIC_ASSISTANT_ID=growth_intelligence" >> .env.local

pnpm dev
# → http://localhost:3000
```

## Project Structure

```
growth_intelligence/
├── langgraph.json             # LangGraph server config
├── main.py                    # graph export
├── orchestrator/
│   ├── graph.py               # StateGraph definition
│   └── state.py               # OrchestratorState Pydantic model
├── agents/
│   ├── base.py                # BaseDomainAgent ABC
│   ├── domain/
│   │   ├── market.py
│   │   ├── competitive.py
│   │   ├── win_loss.py
│   │   ├── pricing.py
│   │   ├── positioning.py
│   │   └── adjacent.py
│   └── micro/
│       ├── search.py          # Gemini grounded search
│       ├── scrape.py          # Firecrawl
│       ├── social.py          # Reddit + HN Algolia
│       ├── parse.py           # LlamaParse
│       └── synthesis.py       # Pydantic few-shot retry loop
├── memory/
│   ├── mem0_client.py
│   └── pinecone_client.py
├── schemas/
│   └── findings.py            # All Pydantic models
├── pdf_export/
│   ├── generator.py
│   └── templates/report.html
└── observability/
    └── tracer.py
```

## Environment Variables

See `.env.example` for all required keys.

## Key Contracts

- Backend state **must** have a `messages` key — `Stream.tsx` uses `useStream` which expects this
- Prefix intermediate message IDs with `do-not-render-` to hide them from the chat
- `NEXT_PUBLIC_ASSISTANT_ID` must match the graph key in `langgraph.json`
- `NEXT_PUBLIC_API_URL` must point to the LangGraph server (`http://localhost:2024` locally)
- The final AI message contains the synthesis summary + markdown PDF download link
- `text-embedding-3-large` requires `OPENAI_API_KEY` even when Claude is the generation LLM
- `top_k=5` in every Pinecone `query_chunks` call

## Production Deployment

Deploy the frontend to Vercel and point it at LangGraph Cloud:

```
NEXT_PUBLIC_ASSISTANT_ID=growth_intelligence
LANGGRAPH_API_URL=https://your-langgraph-cloud-url.langgraph.app
NEXT_PUBLIC_API_URL=https://your-vercel-app.vercel.app/api
LANGSMITH_API_KEY=lsv2_...
```
