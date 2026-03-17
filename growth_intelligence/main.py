"""
Growth Intelligence Agent — LangGraph entry point.

Usage:
    langgraph dev   # serves at http://localhost:2024
"""

import logging

# Configure logging so all agents emit visible output to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce noise from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("asyncpraw").setLevel(logging.WARNING)
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

from orchestrator.graph import build_graph

graph = build_graph()
