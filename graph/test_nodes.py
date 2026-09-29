import unittest

from graph.nodes import (
    answer_node,
    check_retrieval_node,
    retrieve_node,
)


class FakeRetriever:

    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_top_k = None

    def search(self, query, top_k):
        self.last_query = query
        self.last_top_k = top_k

        return self.results


class FakeLLM:

    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt):
        self.last_prompt = prompt

        return (
            "The four core functions are GOVERN, MAP, "
            "MEASURE, and MANAGE. "
            "[nist.ai.100-1.pdf, Page 25]"
        )


class TestRetrieveNode(unittest.TestCase):

    def test_uses_rewritten_query(self):
        fake_results = [
            {
                "content": "GOVERN is a core function.",
                "source": "nist.ai.100-1.pdf",
                "page": 25,
                "section": "AI RMF Core",
                "distance": 0.3,
            }
        ]

        retriever = FakeRetriever(fake_results)

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
            result["retrieved_documents"],
            fake_results,
        )

    def test_falls_back_to_original_query(self):
        fake_results = [
            {
                "content": (
                    "The AI RMF Core has four functions."
                ),
                "source": "nist.ai.100-1.pdf",
                "page": 25,
                "section": "AI RMF Core",
                "distance": 0.3,
            }
        ]

        retriever = FakeRetriever(fake_results)

        state = {
            "query": "What are the four functions?",
        }

        result = retrieve_node(
            state,
            retriever,
        )

        self.assertEqual(
            retriever.last_query,
            "What are the four functions?",
        )

        self.assertEqual(
            result["retrieved_documents"],
            fake_results,
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

    def test_single_document_is_insufficient(self):
        state = {
            "retrieved_documents": [
                {"content": "chunk 1"},
            ],
        }

        result = check_retrieval_node(state)

        self.assertFalse(
            result["retrieval_sufficient"]
        )

    def test_multiple_documents_are_sufficient(self):
        state = {
            "retrieved_documents": [
                {"content": "chunk 1"},
                {"content": "chunk 2"},
            ],
        }

        result = check_retrieval_node(state)

        self.assertTrue(
            result["retrieval_sufficient"]
        )


class TestAnswerNode(unittest.TestCase):

    def test_generates_grounded_answer(self):
        llm = FakeLLM()

        state = {
            "query": (
                "What are the four core functions?"
            ),
            "retrieved_documents": [
                {
                    "content": (
                        "The AI RMF Core is composed of "
                        "four functions: GOVERN, MAP, "
                        "MEASURE, and MANAGE."
                    ),
                    "source": "nist.ai.100-1.pdf",
                    "page": 25,
                    "section": "AI RMF Core",
                    "distance": 0.3291,
                }
            ],
        }

        result = answer_node(
            state,
            llm,
        )

        self.assertIn(
            "GOVERN",
            result["answer"],
        )

        self.assertIn(
            "MAP",
            result["answer"],
        )

        self.assertIn(
            "MEASURE",
            result["answer"],
        )

        self.assertIn(
            "MANAGE",
            result["answer"],
        )

    def test_prompt_contains_query(self):
        llm = FakeLLM()

        state = {
            "query": "What is the GOVERN function?",
            "retrieved_documents": [
                {
                    "content": (
                        "The GOVERN function establishes "
                        "organizational processes."
                    ),
                    "source": "nist.ai.100-1.pdf",
                    "page": 26,
                    "section": "5.1 Govern",
                    "distance": 0.4,
                }
            ],
        }

        answer_node(
            state,
            llm,
        )

        self.assertIn(
            "What is the GOVERN function?",
            llm.last_prompt,
        )

    def test_prompt_contains_source_metadata(self):
        llm = FakeLLM()

        state = {
            "query": "What is the GOVERN function?",
            "retrieved_documents": [
                {
                    "content": "GOVERN manages AI risks.",
                    "source": "nist.ai.100-1.pdf",
                    "page": 26,
                    "section": "5.1 Govern",
                    "distance": 0.4,
                }
            ],
        }

        answer_node(
            state,
            llm,
        )

        self.assertIn(
            "nist.ai.100-1.pdf",
            llm.last_prompt,
        )

        self.assertIn(
            "Page: 26",
            llm.last_prompt,
        )

        self.assertIn(
            "5.1 Govern",
            llm.last_prompt,
        )


if __name__ == "__main__":
    unittest.main()