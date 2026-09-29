from pathlib import Path
import hashlib

import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.embeddings import TransformerEmbedder
from ingestion.section_detector import assign_sections


DATA_DIR = Path("data/raw")
CHROMA_DIR = Path("data/chroma")

COLLECTION_NAME = "nist_ai_rmf"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_documents(data_dir: Path = DATA_DIR):
    """Load all PDF pages from the raw data directory."""

    documents = []

    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}"
        )

    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["document"] = pdf_path.stem
            page.metadata["source"] = pdf_path.name

            # PyMuPDFLoader uses zero-based page numbers.
            page.metadata["page"] = (
                page.metadata.get("page", 0) + 1
            )

        documents.extend(pages)

    # Detect and propagate section information.
    documents = assign_sections(documents)

    return documents


def split_documents(documents):
    """Split loaded pages into searchable chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def create_chunk_id(chunk) -> str:
    """Create a deterministic ID for a chunk."""

    raw = (
        f"{chunk.metadata['source']}:"
        f"{chunk.metadata['page']}:"
        f"{chunk.page_content}"
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def create_chroma_collection(
    chroma_dir: Path = CHROMA_DIR,
):
    """Create or load the persistent Chroma collection."""

    client = chromadb.PersistentClient(
        path=str(chroma_dir)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "NIST AI RMF knowledge base"
        },
    )


def store_chunks(
    chunks,
    embeddings,
    collection,
):
    """Store chunks, embeddings and metadata in Chroma."""

    ids = []
    documents = []
    metadatas = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):
        ids.append(
            create_chunk_id(chunk)
        )

        documents.append(
            chunk.page_content
        )

        metadatas.append({
            "source": chunk.metadata["source"],
            "document": chunk.metadata["document"],
            "page": chunk.metadata["page"],
            "section": chunk.metadata.get(
                "section",
                "Unknown",
            ),
            "chunk_id": chunk.metadata["chunk_id"],
        })

    if not ids:
        return

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def ingest(
    data_dir: Path = DATA_DIR,
    chroma_dir: Path = CHROMA_DIR,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Run the complete ingestion pipeline."""

    documents = load_documents(data_dir)

    chunks = split_documents(documents)

    embedder = TransformerEmbedder(
        model_name=embedding_model
    )

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedder.encode(texts)

    collection = create_chroma_collection(
        chroma_dir
    )

    store_chunks(
        chunks,
        embeddings,
        collection,
    )

    return {
        "pages": len(documents),
        "chunks": len(chunks),
        "collection_count": collection.count(),
    }


def main():
    print("=== NIST AI RMF Ingestion ===")

    result = ingest()

    print(
        f"Loaded pages: {result['pages']}"
    )

    print(
        f"Created chunks: {result['chunks']}"
    )

    print(
        f"Chroma documents: "
        f"{result['collection_count']}"
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    main()