# Growth Intelligence Agent

A **sophisticated multi-agent system** for comprehensive product and market intelligence gathering. This LangGraph-powered orchestrator combines AI reasoning, web scraping, vector search, and threaded memory to deliver structured business intelligence reports in real-time through a chat interface.

**Key Capability:** Analyzes markets across six distinct intelligence domains, synthesizes findings into a coherent narrative, and exports a professional PDF intelligence report—all accessible via a conversational chat interface.

---

## 🎯 Overview

### The Problem
Traditional market research requires manual data collection across multiple sources, synthesis by multiple analysts, and weeks of turnaround time. This system automates that entire workflow.

### The Solution
A **sequential multi-agent orchestrator** that:
- ✅ Runs 6 specialized domain agents in sequence
- ✅ Gathers intelligence from search engines, web scrapers, social media, and document parsers
- ✅ Stores findings in a vector database for semantic retrieval
- ✅ Maintains thread memory across conversations
- ✅ Synthesizes findings into a professional PDF report
- ✅ Serves results through a chat interface in real-time

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Chat Interface (Next.js)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │      LangGraph Orchestrator State       │
    │  (Sequential execution, error handling) │
    └────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ Classify │ │ Domain │ │Synthesize│
    │ Domains  │ │Agents  │ │& Export  │
    └────────┘ └────────┘ └────────┘
        │            │            │
        └────────────┼────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ┌─────────────┐         ┌──────────────┐
   │Micro-Agents │         │Memory & Store│
   │ (Search, etc)         │ (Pinecone, Memory)
   └─────────────┘         └──────────────┘
```

### Sequential Pipeline

1. **receive_message** → Bootstrap session and extract user query
2. **classify_domains** → Claude determines which domains are relevant
3. **Domain Agents (Sequential):**
   - Market & Trend Sensing
   - Competitive Landscape
   - Win / Loss Intelligence
   - Pricing & Packaging
   - Positioning & Messaging
   - Adjacent Market Collision
4. **synthesise** → Cross-domain analysis and key insights
5. **export_pdf** → Generate professional PDF report
6. **stream_response** → Return PDF link and findings to UI
7. **update_memory** → Persist findings to Mem0 for future reference

---

## 📊 The Six Intelligence Domains

Each domain agent gathers specialized signals to answer a critical business question:

| Domain | Question | Key Signals |
|--------|----------|------------|
| **Market & Trend** | Where is the category heading? | Industry reports, analyst predictions, growth trends |
| **Competitive** | Who is doing what? | Competitor positioning, recent moves, market share |
| **Win / Loss** | Why are deals being lost? | Customer feedback, competitive wins/losses, churn reasons |
| **Pricing** | Is the pricing model right? | Industry pricing benchmarks, value perception, margins |
| **Positioning** | How should we talk about what exists? | Messaging frameworks, market narratives, differentiation |
| **Adjacent** | What is coming from outside? | Adjacent market trends, potential disruption, new players |

---

## 🛠️ Tech Stack

### Core Orchestration
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Graph Engine** | LangGraph `StateGraph` | Multi-agent orchestration & state management |
| **Primary LLM** | Claude Sonnet 4.5 (Anthropic) | Complex reasoning, synthesis, domain classification |
| **Search LLM** | Gemini Flash 2.0 + Google Search | Web-grounded search with real-time results |

### Data & Memory
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Store** | Pinecone | Semantic search & RAG for domain findings (`top_k=5`) |
| **Thread Memory** | Mem0 | Persistent session memory with in-place updates |
| **Embeddings** | OpenAI `text-embedding-3-large` | Dense vector representations for similarity search |

### Data Collection
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Scraping** | Firecrawl | Extract HTML from any web page |
| **Document Parse** | LlamaParse | Deep parsing of PDFs and documents |
| **Social Signals** | asyncpraw (Reddit) + Algolia (HN) | Reddit posts/comments + Hacker News discussions |

### Output & Observability
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Export** | WeasyPrint + Jinja2 templates | Generate professional PDF reports |
| **Chat UI** | Next.js + [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) | Real-time conversational interface |
| **Observability** | LangFuse | Trace LLM calls, debug agents, monitor performance |

---

## 📁 Project Structure

```
growth_intelligence/
│
├── 📄 main.py                    # LangGraph entry point (graph export)
├── 📄 langgraph.json             # LangGraph server configuration
├── 📄 requirements.txt            # Python dependencies
├── 🔧 diagnos.py                 # Capability check / debugging utilities
│
├── orchestrator/
│   ├── graph.py                  # StateGraph definition & all node functions
│   └── state.py                  # OrchestratorState Pydantic model
│
├── agents/
│   ├── base.py                   # BaseDomainAgent (abstract base class)
│   ├── _domain_utils.py          # Shared utilities for domain agents
│   │
│   ├── domain/
│   │   ├── market.py             # Market & Trend Sensing agent
│   │   ├── competitive.py        # Competitive Landscape agent
│   │   ├── win_loss.py           # Win / Loss Intelligence agent
│   │   ├── pricing.py            # Pricing & Packaging agent
│   │   ├── positioning.py        # Positioning & Messaging agent
│   │   └── adjacent.py           # Adjacent Market Collision agent
│   │
│   └── micro/
│       ├── search.py             # Gemini Flash grounded search
│       ├── scrape.py             # Firecrawl web scraping
│       ├── social.py             # Reddit + HN Algolia social signals
│       ├── parse.py              # LlamaParse document parsing
│       └── synthesis.py          # Pydantic few-shot retry loop (Claude)
│
├── memory/
│   ├── mem0_client.py            # Mem0 thread memory integration
│   └── pinecone_client.py        # Pinecone vector store operations
│
├── schemas/
│   └── findings.py               # All Pydantic models:
│                                   # - Fact, Chunk, DomainFinding
│                                   # - PartialFinding, FinalReport
│
├── pdf_export/
│   ├── generator.py              # PDF generation pipeline
│   └── templates/
│       └── report.html           # Jinja2 HTML template for PDF
│
└── observability/
    └── tracer.py                 # LangFuse tracing configuration
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- API keys for:
  - **Anthropic** (Claude Sonnet 4.5)
  - **Google** (Gemini Flash 2.0)
  - **OpenAI** (embeddings)
  - **Pinecone** (vector database)
  - **Mem0** (thread memory)
  - **Firecrawl** (web scraping)
  - **LlamaParse** (document parsing)
  - **LangFuse** (optional, for observability)

### Backend Setup

```bash
# Clone and enter the project directory
cd growth_intelligence

# Install dependencies
pip install -r requirements.txt

# Create .env file with required API keys
cat > .env << 'EOF'
# LLMs
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Memory & Vector Store
MEM0_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=growth_intelligence
PINECONE_ENVIRONMENT=us-east-1

# Web Services
FIRECRAWL_API_KEY=your_key_here
LLAMA_PARSE_API_KEY=your_key_here

# Social Signals
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=growth_intelligence/1.0

# Observability (optional)
LANGFUSE_PUBLIC_KEY=your_key_here
LANGFUSE_SECRET_KEY=your_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
EOF

# Start the LangGraph dev server
langgraph dev
# Serves at http://localhost:2024
```

### Frontend Setup

```bash
# In a separate terminal, clone and set up the chat UI
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui
pnpm install

# Create .env.local
cat > .env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=growth_intelligence
EOF

# Start dev server
pnpm dev
# Accessible at http://localhost:3000
```

### Testing

```bash
# Run end-to-end test suite
python -m pytest test_e2e.py -v

# Run constructor/initialization tests
python -m pytest test_constructors.py -v

# Diagnostic capability check
python diagnostics/capability_check.py
```

---

## 🔄 How It Works

### 1. **User Query** → Chat Interface
A user asks a market intelligence question (e.g., "What are the key trends in AI infrastructure startups?")

### 2. **Orchestrator receives and classifies**
- `receive_message`: Extracts query and bootstraps session
- `classify_domains`: Claude determines which of the 6 domains are relevant

### 3. **Domain Agents run sequentially**
For each active domain:
- ✅ **Check memory** → Mem0 for prior results (avoid API redundancy)
- ✅ **Form sub-queries** → Domain-specific reformulation
- ✅ **Gather signals** → Parallel micro-agent execution:
  - Search (Gemini + Google)
  - Scrape (Firecrawl)
  - Social (Reddit + HN)
  - Parse (LlamaParse)
- ✅ **Chunk & store** → All content indexed in Pinecone
- ✅ **Semantic retrieval** → Top-5 relevant chunks from vector store
- ✅ **Synthesize** → Claude generates structured DomainFinding
  - Facts (grounded, source-attributed)
  - Interpretations (clearly labeled as inferences)
  - Confidence scores
- ✅ **Persist** → Save to Mem0 for future reference

### 4. **Cross-domain synthesis**
`synthesise` node:
- Analyzes relationships between domain findings
- Identifies key insights
- Flags contradictions or gaps

### 5. **PDF Export**
`export_pdf`:
- Renders findings to HTML using Jinja2 template
- Converts to PDF via WeasyPrint
- Uploads to cloud storage with download link

### 6. **Stream to UI**
`stream_response`:
- Returns findings incrementally to chat interface
- Provides downloadable PDF link

### 7. **Update Memory**
`update_memory`:
- Persists findings to Mem0
- Associates with session thread
- Enables future reference & memory continuity

---

## 🎓 Key Design Patterns

### Error Handling & Graceful Degradation
All domain nodes are wrapped in `try/except`:
- If a domain fails, execution continues with other domains
- Failed domains produce `PartialFinding` objects
- User receives partial intelligence rather than a complete failure

### Tool Message Visibility
- Intermediate tool messages use `do-not-render-` prefix
- The frontend hides these from the UI
- Only final findings and PDF link are visible to users

### Memory Continuity
- Mem0 stores findings with structured metadata
- Future queries check memory before running expensive operations
- Results are reconstructable from prior thread data

### Vector Store Semantics
- All content is chunked and embedded
- `top_k=5` retrieval provides relevant context to each domain agent
- Semantic search enables cross-domain pattern discovery

---

## 📊 Data Models & Schemas

### Core Structures (in `schemas/findings.py`)

**Fact**: A single, grounded, source-attributed claim
```python
class Fact(BaseModel):
    claim: str                      # The factual assertion
    source_url: str                 # Where this came from
    retrieved_at: datetime          # When it was retrieved
    confidence: float (0.0-1.0)     # Confidence score
```

**DomainFinding**: Output from a single domain agent
```python
class DomainFinding(BaseModel):
    domain: str                     # "market", "competitive", etc.
    status: Literal["complete", "partial", "failed"]
    summary: str                    # 2–3 sentence synthesis
    facts: list[Fact]               # Grounded, source-attributed claims
    interpretations: list[str]      # Clearly labeled inferences
    confidence: float               # Overall confidence 0.0-1.0
    error_reason: str | None        # If status == "failed"
```

**Chunk**: A text segment stored in Pinecone
```python
class Chunk(BaseModel):
    id: str                         # Unique identifier
    text: str                       # Content
    source_url: str
    domain_tag: str                 # Which domain produced it
    timestamp: datetime
    agent_id: str                   # Which agent
    embedding: list[float] | None   # Vector representation
```

**FinalReport**: The complete cross-domain intelligence report
- Aggregates findings from all domains
- Contains execution summary
- Includes PDF export link

---

## 🧪 Testing

### Unit Tests
```bash
# Test agent constructors and initialization
python -m pytest test_constructors.py -v
```

### End-to-End Tests
```bash
# Test full orchestrator pipeline
python -m pytest test_e2e.py -v
python -m pytest test_e2e2.py -v
```

### Capability Check
```bash
# Verify all API keys and dependencies are configured
python diagnostics/capability_check.py
```

---

## 📈 Observability

### LangFuse Integration
All LLM calls are traced:
- **Traces** of full orchestrator runs
- **Spans** for each domain agent
- **Events** for each micro-agent signal
- **Metrics** for token usage, latency, errors

### View Traces
1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Sign in with your project
3. See all orchestrator runs with full call stack

### Local Logging
```bash
# Console output configured in main.py shows:
# - Timestamp
# - Log level (INFO, WARNING, ERROR)
# - Logger name (e.g., "agents.domain.market")
# - Message with query context
```

---

## 🔌 Environment Variables

Create `.env` file in project root with:

```bash
# === LLMs ===
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-proj-...

# === Memory & Vector Store ===
MEM0_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX=growth_intelligence
PINECONE_ENVIRONMENT=us-east-1

# === Web Services ===
FIRECRAWL_API_KEY=...
LLAMA_PARSE_API_KEY=...

# === Social Signals ===
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=growth_intelligence/1.0

# === Observability (Optional) ===
LANGFUSE_PUBLIC_KEY=pk_...
LANGFUSE_SECRET_KEY=sk_...
LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## 🐛 Troubleshooting

### Issue: "API key not found"
**Solution:** Verify all required keys are in `.env` and run `python diagnostics/capability_check.py`

### Issue: Micro-agent timeouts
**Solution:** Increase `asyncio-throttle` limits in `requirements.txt` or reduce concurrent parallel calls in micro-agents

### Issue: Pinecone query returns no results
**Solution:** 
- Verify Pinecone index name matches config
- Check that embeddings are being upserted correctly
- Run test: `python -m pytest test_e2e.py::test_pinecone_operations -v`

### Issue: PDF export fails
**Solution:** 
- Ensure WeasyPrint system dependencies are installed
- On macOS: `brew install cairo pango gdk-pixbuf libffi`
- On Ubuntu: `apt-get install libpango-1.0-0 libpango-gobject-0`

### Issue: Memory not persisting across conversations
**Solution:** Verify Mem0 credentials and check that session_id is being passed correctly in the state

---

## 🚀 Deployment

### Deploy to LangGraph Cloud

```bash
# Install LangGraph CLI
pip install langgraph-cli

# Deploy
langgraph deploy

# This creates a production endpoint:
# https://your-org-abc123.api.stack.langserve.com/growth_intelligence/
```

### Deploy Frontend (Next.js)

```bash
cd agent-chat-ui
pnpm build
pnpm start
```

Or deploy to Vercel:
```bash
vercel deploy
```

---

## 📚 API Reference

### LangGraph Endpoint

**POST** `/growth_intelligence/invoke`
```json
{
  "input": {
    "messages": [
      {
        "type": "human",
        "content": "What are the latest trends in AI infrastructure?"
      }
    ]
  }
}
```

**Returns:**
```json
{
  "output": {
    "messages": [...],
    "findings": [{
      "domain": "market",
      "status": "complete",
      "summary": "...",
      "facts": [...],
      "interpretations": [...],
      "confidence": 0.85
    }],
    "report_pdf_url": "https://..."
  }
}
```

---

## 🤝 Contributing

### Adding a New Domain Agent

1. Create `agents/domain/your_domain.py`
2. Inherit from `BaseDomainAgent`
3. Implement `async def run(self, query: str) -> DomainFinding:`
4. Register in `orchestrator/graph.py`

### Adding a New Micro-Agent Signal

1. Create `agents/micro/your_signal.py`
2. Implement async function returning `list[Chunk]`
3. Call from domain agents in `agents/domain/*.py`

### Updating the PDF Template

1. Edit `pdf_export/templates/report.html`
2. Test with `weasyprint your_template.html output.pdf`

---

## 📄 License

This project is proprietary. All rights reserved.

---

## 👥 Support

For issues, questions, or feature requests:
- Check the **Troubleshooting** section above
- Review **Test Files**: `test_e2e.py`, `test_e2e2.py`
- Run **Diagnostics**: `python diagnostics/capability_check.py`

---

## 🎯 Roadmap

- [ ] Multi-turn refinement within a domain
- [ ] Custom domain agent definitions via prompt injection
- [ ] Real-time streaming of domain findings as they complete
- [ ] Competitor watch—scheduled automated updates
- [ ] Integration with CRM for deal intelligence
- [ ] Mobile app for report viewing
- [ ] Multi-language support

---

**Built with ❤️ using LangGraph, Claude, and Pinecone**

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
