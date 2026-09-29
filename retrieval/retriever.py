from pathlib import Path
from typing import Any

import chromadb

from ingestion.embeddings import TransformerEmbedder


DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "nist_ai_rmf"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class Retriever:
    """Retrieve relevant chunks from the NIST Chroma collection."""

    def __init__(
        self,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedder: TransformerEmbedder | None = None,
    ):
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir)
        )

        self.collection = (
            self.client.get_collection(
                name=collection_name
            )
        )

        self.embedder = embedder or TransformerEmbedder(
            model_name=embedding_model
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k chunks relevant to the query.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_embedding = self.embedder.encode(
            [query]
        )[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved.append({
                "content": document,
                "source": metadata.get("source"),
                "document": metadata.get("document"),
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "distance": distance,
            })

        return retrieved