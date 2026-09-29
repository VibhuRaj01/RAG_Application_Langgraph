import unittest

from graph.nodes import (
    answer_node,
    check_retrieval_node,
    retrieve_node,
    rewrite_query_node,
)


class FakeRetriever:
    def __init__(self, results=None):
        self.results = results or []
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k=5):
        self.last_query = query
        self.last_top_k = top_k
        return self.results


class FakeLLM:
    def __init__(self, response="Fake answer"):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


class TestRewriteQueryNode(unittest.TestCase):

    def test_uses_original_query_without_history(self):
        llm = FakeLLM()

        state = {
            "query": "What are the four functions of the NIST AI RMF?",
            "history": [],
        }

        result = rewrite_query_node(state, llm)

        self.assertEqual(
            result["rewritten_query"],
            state["query"],
        )

        self.assertEqual(
            len(llm.prompts),
            0,
        )

    def test_rewrites_follow_up_query(self):
        llm = FakeLLM(
            "Which NIST AI RMF function is cross-cutting?"
        )

        state = {
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

        result = rewrite_query_node(state, llm)

        self.assertEqual(
            result["rewritten_query"],
            "Which NIST AI RMF function is cross-cutting?",
        )

        self.assertEqual(len(llm.prompts), 1)

        prompt = llm.prompts[0]

        self.assertIn(
            "Which one is cross-cutting?",
            prompt,
        )

        self.assertIn(
            "Govern, Map, Measure, and Manage.",
            prompt,
        )


class TestRetrieveNode(unittest.TestCase):

    def test_uses_rewritten_query(self):
        retriever = FakeRetriever(
            [
                {
                    "content": "GOVERN is cross-cutting.",
                    "source": "AI_RMF_1.0.pdf",
                    "page": 25,
                    "section": "2. Core",
                    "distance": 0.2,
                }
            ]
        )

        state = {
            "query": "Which one is cross-cutting?",
            "rewritten_query": (
                "Which NIST AI RMF function is cross-cutting?"
            ),
        }

        result = retrieve_node(
            state,
            retriever,
        )

        self.assertEqual(
            retriever.last_query,
            "Which NIST AI RMF function is cross-cutting?",
        )

        self.assertEqual(
            retriever.last_top_k,
            5,
        )

        self.assertEqual(
            len(result["retrieved_documents"]),
            1,
        )

    def test_falls_back_to_original_query(self):
        retriever = FakeRetriever([])

        state = {
            "query": "What is GOVERN?",
        }

        retrieve_node(
            state,
            retriever,
        )

        self.assertEqual(
            retriever.last_query,
            "What is GOVERN?",
        )


class TestCheckRetrievalNode(unittest.TestCase):

    def test_no_documents_is_insufficient(self):
        state = {
            "retrieved_documents": [],
        }

        result = check_retrieval_node(state)

        self.assertFalse(
            result["retrieval_sufficient"]
        )

    def test_one_document_is_insufficient(self):
        state = {
            "retrieved_documents": [
                {
                    "content": "GOVERN is cross-cutting.",
                }
            ],
        }

        result = check_retrieval_node(state)

        self.assertFalse(
            result["retrieval_sufficient"]
        )

    def test_multiple_documents_are_sufficient(self):
        state = {
            "retrieved_documents": [
                {
                    "content": "Document 1",
                },
                {
                    "content": "Document 2",
                },
            ],
        }

        result = check_retrieval_node(state)

        self.assertTrue(
            result["retrieval_sufficient"]
        )


class TestAnswerNode(unittest.TestCase):

    def test_returns_answer_from_llm(self):
        llm = FakeLLM(
            "GOVERN is the cross-cutting function."
        )

        state = {
            "query": "Which function is cross-cutting?",
            "retrieved_documents": [
                {
                    "content": "GOVERN is cross-cutting.",
                    "source": "AI_RMF_1.0.pdf",
                    "page": 25,
                    "section": "2. Core",
                    "distance": 0.2,
                },
                {
                    "content": "The AI RMF has four functions.",
                    "source": "AI_RMF_1.0.pdf",
                    "page": 25,
                    "section": "2. Core",
                    "distance": 0.3,
                },
            ],
        }

        result = answer_node(
            state,
            llm,
        )

        self.assertEqual(
            result["answer"],
            "GOVERN is the cross-cutting function.",
        )

        self.assertEqual(len(llm.prompts), 1)

        prompt = llm.prompts[0]

        self.assertIn(
            "Which function is cross-cutting?",
            prompt,
        )

        self.assertIn(
            "GOVERN is cross-cutting.",
            prompt,
        )

        self.assertIn(
            "AI_RMF_1.0.pdf",
            prompt,
        )

    def test_handles_empty_documents(self):
        llm = FakeLLM()

        state = {
            "query": "What is unrelated information?",
            "retrieved_documents": [],
        }

        result = answer_node(
            state,
            llm,
        )

        self.assertIn(
            "not contain enough information",
            result["answer"],
        )

        self.assertEqual(
            len(llm.prompts),
            0,
        )


if __name__ == "__main__":
    unittest.main()