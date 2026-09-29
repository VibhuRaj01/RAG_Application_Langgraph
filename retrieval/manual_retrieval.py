from retrieval.retriever import Retriever


def main():
    retriever = Retriever()

    queries = [
        "What are the four core functions of the NIST AI RMF?",
        "What is the purpose of the NIST AI RMF?",
        "What are the risks associated with generative AI?",
        "What is the GOVERN function?",
    ]

    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = retriever.search(query, top_k=5)

        for i, result in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Distance : {result['distance']:.4f}")
            print(f"Source   : {result['source']}")
            print(f"Page     : {result['page']}")
            print(f"Section  : {result['section']}")
            print(f"\n{result['content'][:1000]}")


if __name__ == "__main__":
    main()