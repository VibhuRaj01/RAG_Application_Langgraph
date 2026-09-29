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

    print("NIST AI RMF RAG Assistant")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if not query:
            continue

        result = graph.invoke({
            "query": query,
        })

        print("\nAssistant:")
        print(result["answer"])


if __name__ == "__main__":
    main()