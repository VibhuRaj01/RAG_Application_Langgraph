import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from ingestion.ingest import (
    create_chunk_id,
    load_documents,
    split_documents,
    store_chunks,
)


class TestIngestion(unittest.TestCase):

    def test_create_chunk_id_is_deterministic(self):
        document = Document(
            page_content="Test content",
            metadata={
                "source": "test.pdf",
                "page": 1,
                "document": "test",
                "chunk_id": 0,
            },
        )

        first_id = create_chunk_id(document)
        second_id = create_chunk_id(document)

        self.assertEqual(
            first_id,
            second_id,
        )

    def test_create_chunk_id_changes_when_content_changes(self):
        document_1 = Document(
            page_content="Content A",
            metadata={
                "source": "test.pdf",
                "page": 1,
            },
        )

        document_2 = Document(
            page_content="Content B",
            metadata={
                "source": "test.pdf",
                "page": 1,
            },
        )

        self.assertNotEqual(
            create_chunk_id(document_1),
            create_chunk_id(document_2),
        )

    def test_split_documents(self):
        document = Document(
            page_content="This is a test. " * 200,
            metadata={
                "source": "test.pdf",
                "page": 1,
                "document": "test",
                "section": "1. Introduction",
            },
        )

        chunks = split_documents([document])

        self.assertGreater(
            len(chunks),
            1,
        )

        for index, chunk in enumerate(chunks):
            self.assertIn(
                "chunk_id",
                chunk.metadata,
            )

            self.assertEqual(
                chunk.metadata["chunk_id"],
                index,
            )

    @patch(
        "ingestion.ingest.assign_sections"
    )
    @patch(
        "ingestion.ingest.PyMuPDFLoader"
    )
    def test_load_documents(
        self,
        mock_loader,
        mock_assign_sections,
    ):
        page = Document(
            page_content="Test page",
            metadata={
                "page": 0,
            },
        )

        loader_instance = MagicMock()
        loader_instance.load.return_value = [page]

        mock_loader.return_value = loader_instance

        mock_assign_sections.side_effect = (
            lambda documents: documents
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = (
                Path(temp_dir)
                / "test.pdf"
            )

            pdf_path.touch()

            documents = load_documents(
                Path(temp_dir)
            )

        self.assertEqual(
            len(documents),
            1,
        )

        self.assertEqual(
            documents[0].metadata["page"],
            1,
        )

        self.assertEqual(
            documents[0].metadata["source"],
            "test.pdf",
        )

        self.assertEqual(
            documents[0].metadata["document"],
            "test",
        )

        mock_assign_sections.assert_called_once()

    def test_store_chunks(self):
        collection = MagicMock()

        chunks = [
            Document(
                page_content="Chunk one",
                metadata={
                    "source": "test.pdf",
                    "page": 1,
                    "document": "test",
                    "section": "1. Introduction",
                    "chunk_id": 0,
                },
            ),
            Document(
                page_content="Chunk two",
                metadata={
                    "source": "test.pdf",
                    "page": 2,
                    "document": "test",
                    "section": "1.1. Scope",
                    "chunk_id": 1,
                },
            ),
        ]

        embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        store_chunks(
            chunks,
            embeddings,
            collection,
        )

        collection.upsert.assert_called_once()

        call_kwargs = (
            collection
            .upsert
            .call_args.kwargs
        )

        self.assertEqual(
            len(call_kwargs["ids"]),
            2,
        )

        self.assertEqual(
            call_kwargs["documents"],
            [
                "Chunk one",
                "Chunk two",
            ],
        )

        self.assertEqual(
            call_kwargs["embeddings"],
            embeddings,
        )

        self.assertEqual(
            call_kwargs["metadatas"],
            [
                {
                    "source": "test.pdf",
                    "document": "test",
                    "page": 1,
                    "section": "1. Introduction",
                    "chunk_id": 0,
                },
                {
                    "source": "test.pdf",
                    "document": "test",
                    "page": 2,
                    "section": "1.1. Scope",
                    "chunk_id": 1,
                },
            ],
        )

    def test_store_empty_chunks(self):
        collection = MagicMock()

        store_chunks(
            [],
            [],
            collection,
        )

        collection.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()