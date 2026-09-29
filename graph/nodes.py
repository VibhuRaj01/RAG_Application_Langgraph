from typing import Any

from conversation.history import format_history
from retrieval.retriever import Retriever


DEFAULT_TOP_K = 5


def rewrite_query_node(
    state: dict[str, Any],
    llm,
) -> dict[str, str]:
    """
    Convert a conversational follow-up into a standalone query.

    If there is no previous conversation, the original query
    is already standalone and does not need rewriting.
    """

    query = state["query"]
    history = state.get("history", [])

    if not history:
        return {
            "rewritten_query": query
        }

    prompt = f"""
You rewrite conversational questions into standalone questions.

Use the previous conversation only to resolve references
such as "it", "they", "this", "that", "which one", etc.

Do not answer the question.

Do not add information that is not present in the conversation.

Previous conversation:
{format_history(history)}

Current question:
{query}

Return only the standalone question.
""".strip()

    rewritten_query = llm.generate(prompt)

    return {
        "rewritten_query": rewritten_query
    }


def retrieve_node(
    state: dict[str, Any],
    retriever: Retriever,
) -> dict[str, Any]:
    query = state.get("rewritten_query") or state["query"]

    results = retriever.search(
        query=query,
        top_k=DEFAULT_TOP_K,
    )

    return {
        "retrieved_documents": results
    }


def check_retrieval_node(
    state: dict[str, Any],
) -> dict[str, bool]:
    documents = state.get(
        "retrieved_documents",
        [],
    )

    if not documents:
        return {
            "retrieval_sufficient": False
        }

    sufficient = len(documents) >= 2

    return {
        "retrieval_sufficient": sufficient
    }


def answer_node(
    state: dict[str, Any],
    llm,
) -> dict[str, str]:
    documents = state.get(
        "retrieved_documents",
        [],
    )

    if not documents:
        return {
            "answer": (
                "The available NIST documents do not contain "
                "enough information to answer this question."
            )
        }

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
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
[Document, Page X, Section Y]
""".strip()

    answer = llm.generate(prompt)

    return {
        "answer": answer
    }