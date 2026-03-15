"""
Growth Intelligence Agent — LangGraph entry point.

Usage:
    langgraph dev   # serves at http://localhost:2024
"""

from orchestrator.graph import build_graph

graph = build_graph()
