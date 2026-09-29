from typing import Any

from retrieval.retriever import Retriever


DEFAULT_TOP_K = 5


def retrieve_node(
    state: dict[str, Any],
    retriever: Retriever,
) -> dict[str, Any]:
    """Retrieve relevant documents for the current query."""

    query = state.get("rewritten_query") or state["query"]

    results = retriever.search(
        query=query,
        top_k=DEFAULT_TOP_K,
    )

    return {
        "retrieved_documents": results,
    }


def check_retrieval_node(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Check whether retrieval returned usable context.

    This is intentionally a simple structural check for now.
    Later, this can be replaced with a more meaningful
    relevance check.
    """

    documents = state.get("retrieved_documents", [])

    if not documents:
        return {
            "retrieval_sufficient": False,
        }

    sufficient = len(documents) >= 2

    return {
        "retrieval_sufficient": sufficient,
    }


def answer_node(
    state: dict[str, Any],
    llm,
) -> dict[str, Any]:
    """Generate a grounded answer from retrieved documents."""

    documents = state.get("retrieved_documents", [])

    context_parts = []

    for index, document in enumerate(documents, start=1):
        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {document['source']}\n"
            f"Page: {document['page']}\n"
            f"Section: {document.get('section', 'Unknown')}\n"
            f"Content:\n{document['content']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a question-answering assistant for the
NIST AI Risk Management Framework.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information to answer
the question, say that the available context is insufficient.

Do not invent facts or use information outside the context.

User question:
{state["query"]}

Retrieved context:
{context}

Provide a concise answer.

For factual claims, cite the relevant source using:
[Document, Page X]
""".strip()

    answer = llm.generate(prompt)

    return {
        "answer": answer,
    }