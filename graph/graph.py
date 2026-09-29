from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    answer_node,
    check_retrieval_node,
    retrieve_node,
    rewrite_query_node,
)
from graph.state import AgentState
from retrieval.retriever import Retriever


def build_graph(
    retriever=None,
    llm=None,
):
    if retriever is None:
        retriever = Retriever()

    if llm is None:
        raise ValueError("LLM is required.")

    graph = StateGraph(AgentState)

    graph.add_node(
        "rewrite_query",
        lambda state: rewrite_query_node(
            state,
            llm,
        ),
    )

    graph.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            retriever,
        ),
    )

    graph.add_node(
        "check_retrieval",
        check_retrieval_node,
    )

    graph.add_node(
        "answer",
        lambda state: answer_node(
            state,
            llm,
        ),
    )

    graph.add_edge(
        START,
        "rewrite_query",
    )

    graph.add_edge(
        "rewrite_query",
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "check_retrieval",
    )

    graph.add_edge(
        "check_retrieval",
        "answer",
    )

    graph.add_edge(
        "answer",
        END,
    )

    return graph.compile()