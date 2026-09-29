from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    rewritten_query: str
    retrieved_documents: list[dict[str, Any]]
    retrieval_sufficient: bool
    answer: str