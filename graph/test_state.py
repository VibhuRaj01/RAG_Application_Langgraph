import unittest

from graph.state import AgentState


class TestAgentState(unittest.TestCase):

    def test_state_supports_conversation_fields(self):
        state: AgentState = {
            "query": "Which one is cross-cutting?",
            "history": [
                {
                    "role": "user",
                    "content": "What are the four functions?",
                },
                {
                    "role": "assistant",
                    "content": "Govern, Map, Measure, and Manage.",
                },
            ],
            "rewritten_query": (
                "Which NIST AI RMF function is cross-cutting?"
            ),
        }

        self.assertEqual(
            state["query"],
            "Which one is cross-cutting?",
        )

        self.assertEqual(
            len(state["history"]),
            2,
        )

        self.assertEqual(
            state["rewritten_query"],
            "Which NIST AI RMF function is cross-cutting?",
        )


if __name__ == "__main__":
    unittest.main()