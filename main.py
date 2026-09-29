from conversation.history import (
    add_message,
    load_history,
    save_history,
)
from graph.graph import build_graph
from llm.llm import GroqLLM
from retrieval.retriever import Retriever


def main():
    retriever = Retriever()
    llm = GroqLLM()

    graph = build_graph(
        retriever=retriever,
        llm=llm,
    )

    history = load_history()

    print("NIST AI RMF RAG Assistant")
    print("Type 'exit' to quit.")
    print("Type 'clear' to clear conversation history.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            save_history(history)
            print("Goodbye.")
            break

        if query.lower() == "clear":
            history = []
            save_history(history)
            print("Conversation history cleared.")
            continue

        if not query:
            continue

        result = graph.invoke(
            {
                "query": query,
                "history": history,
            }
        )

        answer = result["answer"]

        print("\nAssistant:")
        print(answer)

        add_message(
            history,
            "user",
            query,
        )

        add_message(
            history,
            "assistant",
            answer,
        )

        save_history(history)


if __name__ == "__main__":
    main()