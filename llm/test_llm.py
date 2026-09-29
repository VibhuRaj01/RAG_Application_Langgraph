import unittest
from unittest.mock import MagicMock, patch

from llm.llm import GroqLLM


class TestGroqLLM(unittest.TestCase):

    @patch("llm.llm.Groq")
    def test_generate(self, mock_groq):
        mock_response = MagicMock()

        mock_response.choices[0].message.content = (
            "GOVERN, MAP, MEASURE, and MANAGE."
        )

        mock_groq.return_value.chat.completions.create.return_value = (
            mock_response
        )

        llm = GroqLLM(api_key="test-key")

        result = llm.generate(
            "What are the four core functions?"
        )

        self.assertEqual(
            result,
            "GOVERN, MAP, MEASURE, and MANAGE.",
        )

        mock_groq.return_value.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()