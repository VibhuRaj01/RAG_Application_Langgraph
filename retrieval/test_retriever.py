import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from retrieval.retriever import Retriever


class TestRetriever(unittest.TestCase):

    @patch("retrieval.retriever.TransformerEmbedder")
    @patch("retrieval.retriever.chromadb.PersistentClient")
    def test_search(
        self,
        mock_client,
        mock_embedder_class,
    ):
        collection = MagicMock()

        collection.query.return_value = {
            "documents": [
                [
                    "Govern, Map, Measure, and Manage."
                ]
            ],
            "metadatas": [
                [
                    {
                        "source": "nist_ai_rmf_1.0.pdf",
                        "document": "nist_ai_rmf_1.0",
                        "page": 25,
                        "section": "1.3. AI RMF Core",
                    }
                ]
            ],
            "distances": [
                [0.15]
            ],
        }

        client = MagicMock()
        client.get_collection.return_value = (
            collection
        )

        mock_client.return_value = client

        embedder = MagicMock()
        embedder.encode.return_value = [
            [0.1, 0.2, 0.3]
        ]

        mock_embedder_class.return_value = (
            embedder
        )

        with tempfile.TemporaryDirectory() as temp_dir:

            retriever = Retriever(
                chroma_dir=Path(temp_dir),
                embedder=embedder,
            )

            results = retriever.search(
                "What are the four core functions?",
                top_k=1,
            )

        embedder.encode.assert_called_once_with(
            ["What are the four core functions?"]
        )

        collection.query.assert_called_once_with(
            query_embeddings=[
                [0.1, 0.2, 0.3]
            ],
            n_results=1,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0]["content"],
            "Govern, Map, Measure, and Manage.",
        )

        self.assertEqual(
            results[0]["source"],
            "nist_ai_rmf_1.0.pdf",
        )

        self.assertEqual(
            results[0]["document"],
            "nist_ai_rmf_1.0",
        )

        self.assertEqual(
            results[0]["page"],
            25,
        )

        self.assertEqual(
            results[0]["section"],
            "1.3. AI RMF Core",
        )

        self.assertEqual(
            results[0]["distance"],
            0.15,
        )

    @patch("retrieval.retriever.TransformerEmbedder")
    @patch("retrieval.retriever.chromadb.PersistentClient")
    def test_empty_query(
        self,
        mock_client,
        mock_embedder_class,
    ):
        embedder = MagicMock()

        retriever = Retriever(
            embedder=embedder
        )

        with self.assertRaises(ValueError):
            retriever.search("")

        embedder.encode.assert_not_called()

    @patch("retrieval.retriever.TransformerEmbedder")
    @patch("retrieval.retriever.chromadb.PersistentClient")
    def test_whitespace_query(
        self,
        mock_client,
        mock_embedder_class,
    ):
        embedder = MagicMock()

        retriever = Retriever(
            embedder=embedder
        )

        with self.assertRaises(ValueError):
            retriever.search("   ")

        embedder.encode.assert_not_called()

    @patch("retrieval.retriever.TransformerEmbedder")
    @patch("retrieval.retriever.chromadb.PersistentClient")
    def test_invalid_top_k(
        self,
        mock_client,
        mock_embedder_class,
    ):
        embedder = MagicMock()

        retriever = Retriever(
            embedder=embedder
        )

        with self.assertRaises(ValueError):
            retriever.search(
                "What is AI RMF?",
                top_k=0,
            )

        with self.assertRaises(ValueError):
            retriever.search(
                "What is AI RMF?",
                top_k=-1,
            )

        embedder.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()