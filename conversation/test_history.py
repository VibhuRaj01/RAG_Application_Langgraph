import tempfile
import unittest
from pathlib import Path

from conversation.history import (
    add_message,
    format_history,
    load_history,
    save_history,
)


class TestHistory(unittest.TestCase):

    def test_load_missing_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "history.json"

            history = load_history(history_file)

            self.assertEqual(history, [])

    def test_save_and_load_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "history.json"

            history = [
                {
                    "role": "user",
                    "content": "What are the four functions?",
                },
                {
                    "role": "assistant",
                    "content": "Govern, Map, Measure, and Manage.",
                },
            ]

            save_history(history, history_file)

            loaded_history = load_history(history_file)

            self.assertEqual(loaded_history, history)

    def test_add_message(self):
        history = []

        add_message(
            history,
            "user",
            "What is GOVERN?",
        )

        self.assertEqual(
            history,
            [
                {
                    "role": "user",
                    "content": "What is GOVERN?",
                }
            ],
        )

    def test_format_history(self):
        history = [
            {
                "role": "user",
                "content": "What are the four functions?",
            },
            {
                "role": "assistant",
                "content": "Govern, Map, Measure, and Manage.",
            },
        ]

        formatted = format_history(history)

        self.assertIn(
            "User: What are the four functions?",
            formatted,
        )

        self.assertIn(
            "Assistant: Govern, Map, Measure, and Manage.",
            formatted,
        )

    def test_format_history_limits_messages(self):
        history = [
            {
                "role": "user",
                "content": f"Question {i}",
            }
            for i in range(10)
        ]

        formatted = format_history(
            history,
            max_messages=4,
        )

        self.assertNotIn("Question 0", formatted)
        self.assertIn("Question 6", formatted)
        self.assertIn("Question 9", formatted)


if __name__ == "__main__":
    unittest.main()