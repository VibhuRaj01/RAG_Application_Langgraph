import unittest
from unittest.mock import MagicMock, patch

import torch

from ingestion.embeddings import TransformerEmbedder


class TestTransformerEmbedder(unittest.TestCase):

    def test_mean_pooling(self):
        model_output = MagicMock()

        model_output.last_hidden_state = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ]
        )

        attention_mask = torch.tensor(
            [[1, 1, 0]]
        )

        result = TransformerEmbedder.mean_pooling(
            model_output,
            attention_mask,
        )

        expected = torch.tensor(
            [[2.0, 3.0]]
        )

        torch.testing.assert_close(
            result,
            expected,
        )

    @patch("ingestion.embeddings.AutoModel.from_pretrained")
    @patch("ingestion.embeddings.AutoTokenizer.from_pretrained")
    def test_encode(
        self,
        mock_tokenizer,
        mock_model,
    ):
        tokenizer = MagicMock()

        tokenizer.return_value = {
            "input_ids": torch.tensor(
                [[1, 2, 3]]
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1]]
            ),
        }

        mock_tokenizer.return_value = tokenizer

        model = MagicMock()

        model_output = MagicMock()

        model_output.last_hidden_state = torch.tensor(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ]
        )

        model.return_value = model_output

        mock_model.return_value = model

        embedder = TransformerEmbedder(
            model_name="test-model",
            device="cpu",
        )

        result = embedder.encode(
            ["hello"]
        )

        self.assertEqual(
            len(result),
            1,
        )

        self.assertEqual(
            len(result[0]),
            2,
        )

        # Embeddings should be normalized.
        magnitude = sum(
            value ** 2
            for value in result[0]
        ) ** 0.5

        self.assertAlmostEqual(
            magnitude,
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()