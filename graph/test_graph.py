import unittest

from graph.graph import build_graph


class FakeRetriever:

    def search(self, query, top_k):
        return [
            {
                "content": (
                    "The four functions are GOVERN, MAP, "
                    "MEASURE, and MANAGE."
                ),
                "source": "nist.ai.100-1.pdf",
                "document": "NIST AI RMF",
                "page": 25,
                "section": "AI RMF Core",
                "distance": 0.3,
            },
            {
                "content": (
                    "Governance is a cross-cutting function."
                ),
                "source": "nist.ai.100-1.pdf",
                "document": "NIST AI RMF",
                "page": 25,
                "section": "AI RMF Core",
                "distance": 0.35,
            },
        ]


class FakeLLM:

    def generate(self, prompt):
        return (
            "The four core functions are GOVERN, MAP, "
            "MEASURE, and MANAGE. "
            "[nist.ai.100-1.pdf, Page 25]"
        )


class TestGraph(unittest.TestCase):

    def test_graph_runs(self):
        graph = build_graph(
            retriever=FakeRetriever(),
            llm=FakeLLM(),
        )

        result = graph.invoke({
            "query": "What are the four core functions?",
        })

        self.assertEqual(
            result["query"],
            "What are the four core functions?",
        )

        self.assertEqual(
            len(result["retrieved_documents"]),
            2,
        )

        self.assertTrue(
            result["retrieval_sufficient"]
        )

        self.assertIn(
            "GOVERN",
            result["answer"],
        )


if __name__ == "__main__":
    unittest.main()