import unittest

from graph.graph import build_graph


class FakeRetriever:

    def __init__(self):
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k

        return [
            {
                "content": "GOVERN is the cross-cutting function.",
                "source": "AI_RMF_1.0.pdf",
                "page": 25,
                "section": "2. Core and Profiles",
                "distance": 0.2,
            },
            {
                "content": (
                    "The NIST AI RMF has four core functions: "
                    "GOVERN, MAP, MEASURE, and MANAGE."
                ),
                "source": "AI_RMF_1.0.pdf",
                "page": 25,
                "section": "2. Core and Profiles",
                "distance": 0.3,
            },
        ]


class FakeLLM:

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)

        # Query-rewriting prompt
        if "rewrite conversational questions" in prompt.lower():
            return (
                "Which NIST AI RMF function is cross-cutting?"
            )

        # Answer prompt
        return (
            "GOVERN is the cross-cutting function."
        )


class TestGraph(unittest.TestCase):

    def test_graph_with_follow_up_question(self):
        retriever = FakeRetriever()
        llm = FakeLLM()

        graph = build_graph(
            retriever=retriever,
            llm=llm,
        )

        result = graph.invoke(
            {
                "query": "Which one is cross-cutting?",
                "history": [
                    {
                        "role": "user",
                        "content": (
                            "What are the four functions "
                            "of the NIST AI RMF?"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "Govern, Map, Measure, and Manage."
                        ),
                    },
                ],
            }
        )

        # The follow-up should have been rewritten
        # before retrieval.
        self.assertEqual(
            retriever.last_query,
            "Which NIST AI RMF function is cross-cutting?",
        )

        # Retrieval should have happened.
        self.assertEqual(
            retriever.last_top_k,
            5,
        )

        # Two documents means retrieval is considered sufficient
        # by the current retrieval check.
        self.assertTrue(
            result["retrieval_sufficient"]
        )

        # The answer node should have executed.
        self.assertIn(
            "GOVERN",
            result["answer"],
        )

        # Both query rewriting and answering should have
        # called the LLM.
        self.assertEqual(
            len(llm.prompts),
            2,
        )

    def test_graph_without_history(self):
        retriever = FakeRetriever()
        llm = FakeLLM()

        graph = build_graph(
            retriever=retriever,
            llm=llm,
        )

        result = graph.invoke(
            {
                "query": "What are the four functions?",
                "history": [],
            }
        )

        self.assertEqual(
            retriever.last_query,
            "What are the four functions?",
        )

        self.assertTrue(
            result["retrieval_sufficient"]
        )

        self.assertIn(
            "GOVERN",
            result["answer"],
        )

        # With no history, rewriting should not call the LLM.
        # Only the answer node should call it.
        self.assertEqual(
            len(llm.prompts),
            1,
        )

    def test_graph_preserves_retrieved_documents(self):
        retriever = FakeRetriever()
        llm = FakeLLM()

        graph = build_graph(
            retriever=retriever,
            llm=llm,
        )

        result = graph.invoke(
            {
                "query": "What are the four functions?",
                "history": [],
            }
        )

        self.assertEqual(
            len(result["retrieved_documents"]),
            2,
        )

        self.assertEqual(
            result["retrieved_documents"][0]["source"],
            "AI_RMF_1.0.pdf",
        )

    def test_graph_handles_insufficient_retrieval(self):
        class EmptyRetriever:

            def search(self, query, top_k=5):
                return []

        retriever = EmptyRetriever()
        llm = FakeLLM()

        graph = build_graph(
            retriever=retriever,
            llm=llm,
        )

        result = graph.invoke(
            {
                "query": "What is something unrelated?",
                "history": [],
            }
        )

        self.assertFalse(
            result["retrieval_sufficient"]
        )

        self.assertIn(
            "not contain enough information",
            result["answer"],
        )

        # Answer node is still called with the insufficient
        # retrieval result, but it should not call the LLM.
        self.assertEqual(
            len(llm.prompts),
            0,
        )


if __name__ == "__main__":
    unittest.main()