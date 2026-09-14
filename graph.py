from __future__ import annotations

import os
from typing import List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from retriever import RetrievedChunk, retrieve, QueryEmbedder
from generator import generate_answer
from grader import check_groundedness


DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hukuk:hukuk@localhost:5432/hukuk_db"
)

DISTANCE_THRESHOLD = 0.40


class GraphState(TypedDict):
    query: str
    kanun_adi: Optional[str]
    top_k: int
    embedder: Optional[QueryEmbedder]  # API'den gelen, önceden yüklenmiş model

    results: List[RetrievedChunk]
    answer: str
    is_grounded: bool
    grading_reason: str
    retry_count: int
    max_retries: int

def retrieve_node(state: GraphState) -> dict:
    results = retrieve(
        state["query"],
        DB_URL,
        top_k=state["top_k"],
        kanun_adi=state.get("kanun_adi"),
        embedder=state.get("embedder"),
    )
    return {"results": results}


def generate_node(state: GraphState) -> dict:
    query = state["query"]
    if state.get("retry_count", 0) > 0:
        query = query + (
            "\n\n[SİSTEM NOTU: Önceki cevabın verilen madde metinlerinde "
            "karşılığı olmayan iddialar içeriyordu. Bu sefer SADECE aşağıda "
            "verilen madde metinlerinde birebir yer alan bilgileri kullan, "
            "hiçbir çıkarım veya varsayım ekleme.]"
        )
    answer = generate_answer(query, state["results"])
    return {"answer": answer}


def grade_node(state: GraphState) -> dict:
    is_grounded, reason = check_groundedness(
        state["query"], state["results"], state["answer"]
    )
    return {
        "is_grounded": is_grounded,
        "grading_reason": reason,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def no_context_node(state: GraphState) -> dict:
    return {
        "answer": "Bu soruyla ilgili elimdeki kanun maddeleri arasında bir sonuç bulamadım.",
    }


def fallback_node(state: GraphState) -> dict:
    return {
        "answer": (
            "Bu soruya, elimdeki kanun maddelerine dayanarak güvenilir bir "
            "cevap üretemedim. Lütfen bir hukuk uzmanına danışın ya da "
            "soruyu farklı bir şekilde tekrar sorun."
        ),
    }

def decide_after_retrieve(state: GraphState) -> str:
    if not state["results"]:
        return "no_context"

    best_distance = min(r.distance for r in state["results"])
    if best_distance > DISTANCE_THRESHOLD:
        return "no_context"

    return "generate"


def decide_after_grading(state: GraphState) -> str:
    if state["is_grounded"]:
        return "end"
    if state["retry_count"] > state.get("max_retries", 2):
        return "fallback"
    return "retry"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    graph.add_node("no_context", no_context_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        decide_after_retrieve,
        {"generate": "generate", "no_context": "no_context"},
    )
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges(
        "grade",
        decide_after_grading,
        {"end": END, "retry": "generate", "fallback": "fallback"},
    )
    graph.add_edge("no_context", END)
    graph.add_edge("fallback", END)

    return graph.compile()

compiled_graph = build_graph()