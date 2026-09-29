# NIST AI RMF Agentic RAG System

A small **Agentic RAG (Retrieval-Augmented Generation) system** built with **LangGraph** for answering questions about:

* NIST AI Risk Management Framework (AI RMF) 1.0
* NIST AI RMF Generative AI Profile

The system retrieves relevant content from the NIST documents and uses an LLM to generate grounded answers with source and page references.

The project is designed to demonstrate:

* Document ingestion and chunking
* Embedding generation
* Vector search with ChromaDB
* Retrieval-Augmented Generation
* LangGraph-based orchestration
* Conversational follow-up handling
* Source attribution
* Insufficient-context handling
* Unit testing

---

## 1. Architecture

```text
                         User Question
                              │
                              ▼
                    ┌──────────────────┐
                    │  Conversation    │
                    │     History      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Query Rewriter  │
                    │    (LLM)         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Retriever     │
                    │                  │
                    │   ChromaDB       │
                    │ + BGE Embeddings │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Retrieval Check  │
                    └────────┬─────────┘
                             │
                       Sufficient?
                       /          \
                     Yes           No
                      │             │
                      ▼             ▼
              ┌─────────────┐   Insufficient
              │ Answer Node │   Context
              │    (LLM)    │
              └──────┬──────┘
                     │
                     ▼
              Grounded Answer
              + Citations
                     │
                     ▼
             Conversation History
```

The workflow is implemented using LangGraph as required by the assignment. The assignment specifically asks the graph to define nodes/state and handle retrieval, insufficient context, query modification/retry decisions, and conversation context.

---

## 2. Project Structure

```text
Himanshu/
│
├── ingestion/
│   ├── __init__.py
│   ├── ingest.py
│   ├── embeddings.py
│   ├── section_detector.py
│   ├── test_ingest.py
│   ├── test_embeddings.py
│   └── test_section_detector.py
│
├── retrieval/
│   ├── __init__.py
│   ├── retriever.py
│   ├── manual_retrieval.py
│   └── test_retriever.py
│
├── graph/
│   ├── __init__.py
│   ├── state.py
│   ├── nodes.py
│   ├── graph.py
│   ├── test_nodes.py
│   ├── test_graph.py
│   └── test_state.py
│
├── conversation/
│   ├── __init__.py
│   ├── history.py
│   └── test_history.py
│
├── llm/
│   ├── __init__.py
│   ├── llm.py
│   └── test_llm.py
│
├── data/
│   ├── raw/
│   │   ├── AI_RMF_1.0.pdf
│   │   └── Generative_AI_Profile.pdf
│   │
│   └── chroma/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

# 3. Knowledge Base

The knowledge base contains the two NIST documents specified by the assignment:

1. NIST AI Risk Management Framework (AI RMF) 1.0
2. NIST AI RMF Generative AI Profile

The assignment requires these official NIST documents to be used as the system's knowledge base.

The documents are stored under:

```text
data/raw/
```

---

# 4. Document Ingestion

The ingestion pipeline performs the following steps:

```text
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Page Documents
 │
 ▼
Section Detection
 │
 ▼
Text Chunking
 │
 ▼
Embedding Generation
 │
 ▼
ChromaDB
```

## PDF Loading

PDFs are loaded using PyMuPDF.

Each page retains metadata such as:

```text
document
source
page
section
```

Page numbers are stored using **1-based indexing** so that citations match the page numbers visible to users.

---

## Chunking

The project uses `RecursiveCharacterTextSplitter`.

Current configuration:

```python
chunk_size = 900
chunk_overlap = 150
```

Separators:

```python
[
    "\n\n",
    "\n",
    ". ",
    " ",
    ""
]
```

The overlap helps preserve context when an important statement crosses a chunk boundary.

Each chunk receives a unique `chunk_id`.

---

# 5. Section Detection

The ingestion pipeline attempts to identify numbered headings such as:

```text
1. Introduction
1.1. Purpose and Scope
2. Core and Profiles
```

The detected section is stored as metadata with each chunk.

This allows retrieved information to contain:

```text
Document
Page
Section
```

which can then be used when generating citations.

---

# 6. Embeddings

The project uses:

```text
BAAI/bge-small-en-v1.5
```

through Hugging Face Transformers.

Embeddings are generated using mean pooling over the model's token embeddings with the attention mask applied.

The resulting vectors are L2-normalized before being stored in ChromaDB.

The same embedding model is used for:

* Document embeddings during ingestion
* Query embeddings during retrieval

Using the same embedding space for both is important because the query vector must be comparable with the stored document vectors.

---

# 7. Vector Database

The project uses **ChromaDB** as the vector database.

The database is persisted locally:

```text
data/chroma/
```

The collection is:

```text
nist_ai_rmf
```

Each stored chunk contains:

```text
content
embedding
document
source
page
section
chunk_id
```

The current knowledge base contains approximately:

```text
112 pages
380 chunks
```

after ingestion.

---

# 8. Retrieval

The retriever:

1. Receives a natural-language query.
2. Generates an embedding using BGE.
3. Searches ChromaDB.
4. Returns the top 5 matching chunks.
5. Preserves source metadata.

Example retrieved result:

```python
{
    "content": "...",
    "source": "AI_RMF_1.0.pdf",
    "document": "AI_RMF_1.0.pdf",
    "page": 25,
    "section": "2. Core and Profiles",
    "distance": 0.2
}
```

The current implementation intentionally keeps retrieval simple and transparent rather than introducing a more complicated reranking pipeline.

---

# 9. LangGraph Workflow

The LangGraph workflow contains four main nodes.

```text
START
  │
  ▼
rewrite_query
  │
  ▼
retrieve
  │
  ▼
check_retrieval
  │
  ▼
answer
  │
  ▼
END
```

## `rewrite_query`

This node uses conversation history to convert follow-up questions into standalone questions.

For example:

```text
User:
What are the four core functions of the NIST AI RMF?

User:
Which one is cross-cutting?
```

The second question can be rewritten to something similar to:

```text
Which NIST AI RMF function is cross-cutting?
```

This makes the query more suitable for vector retrieval.

If there is no conversation history, the original question is used directly.

---

## `retrieve`

The retriever searches ChromaDB using the rewritten query.

If no rewritten query exists, the original query is used.

---

## `check_retrieval`

The graph checks whether retrieval produced enough context to continue.

The current simple rule is:

```text
0 documents → insufficient
1 document  → insufficient
2+ documents → sufficient
```

This deliberately keeps the implementation simple and understandable.

---

## `answer`

The answer node sends the retrieved context to the Groq LLM.

The prompt explicitly instructs the model to:

* Answer using only retrieved context
* Avoid unsupported facts
* State when the context is insufficient
* Provide source/page information

This is intended to keep the generated answer grounded in the NIST knowledge base.

---

# 10. LLM

The project currently uses Groq for LLM inference.

The configured model is:

```text
openai/gpt-oss-120b
```

The model is used for two tasks:

### Query rewriting

```text
Conversation history
        +
Current question
        ↓
Standalone question
```

### Answer generation

```text
User question
      +
Retrieved documents
      ↓
Grounded answer
```

The LLM is not responsible for retrieving information from the NIST documents. Retrieval is handled separately through the embedding model and ChromaDB.

---

# 11. Conversational History

Conversation history is stored in:

```text
conversation_history.json
```

The history contains messages in this format:

```json
[
  {
    "role": "user",
    "content": "What are the four core functions?"
  },
  {
    "role": "assistant",
    "content": "GOVERN, MAP, MEASURE, and MANAGE."
  }
]
```

History is loaded when the application starts and saved after each interaction.

The command:

```text
clear
```

clears the current conversation history.

The file is excluded from Git because it contains user-specific conversation data.

---

# 12. Follow-up Questions

The system supports conversational questions such as:

```text
User:
What are the four core functions of the NIST AI RMF?

Assistant:
GOVERN, MAP, MEASURE, and MANAGE.

User:
Which one is cross-cutting?

Assistant:
GOVERN is the cross-cutting function.

User:
What is the purpose of the framework?

Assistant:
...
```

The second question does not explicitly mention the NIST AI RMF or the four functions.

The query rewriting node uses the previous conversation to resolve this context before retrieval.

This satisfies the assignment requirement that follow-up questions use prior conversational context rather than treating every question as completely independent.

---

# 13. Source Attribution

Retrieved chunks contain:

```text
Document
Page
Section
```

This metadata is passed to the LLM along with the retrieved content.

The intended citation format is:

```text
[Document: <document name>, Page: <page number>, Section: <section>]
```

For example:

```text
[Document: AI_RMF_1.0.pdf, Page: 25, Section: 2. Core and Profiles]
```

Source attribution is included because the assignment requires generated answers to identify where information came from.

---

# 14. Handling Insufficient Context

The system should not answer using general model knowledge when the retrieved NIST content does not provide enough information.

If retrieval returns no documents, the system responds with an insufficient-context message rather than generating an unsupported answer.

For example:

```text
The available NIST documents do not contain enough information
to answer this question.
```

This is preferable to allowing the LLM to hallucinate an answer outside the knowledge base.

The assignment explicitly evaluates handling of insufficient or irrelevant retrieval results.

---

# 15. Installation

Create a virtual environment:

```bash
python -m venv llmenv
```

Activate it.

Linux/macOS:

```bash
source llmenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 16. Environment Variables

Create a `.env` file or export the Groq API key:

```bash
export GROQ_API_KEY="your_api_key"
```

The API key is required for answer generation and conversational query rewriting.

Do not commit the API key to Git.

---

# 17. Ingest the Documents

Place the NIST PDFs in:

```text
data/raw/
```

Then run:

```bash
python -m ingestion.ingest
```

This will:

1. Load the PDFs
2. Extract pages
3. Detect sections
4. Split pages into chunks
5. Generate embeddings
6. Store the chunks in ChromaDB

The resulting vector database is stored under:

```text
data/chroma/
```

---

# 18. Run the Application

From the project root:

```bash
python main.py
```

You should see:

```text
NIST AI RMF RAG Assistant
Type 'exit' to quit.
Type 'clear' to clear conversation history.
```

Example:

```text
You: What are the four core functions of the NIST AI RMF?

Assistant:
The four core functions are GOVERN, MAP, MEASURE, and MANAGE...
```

Then:

```text
You: Which one is cross-cutting?
```

The system uses the previous conversation to understand the question.

---

# 19. Testing

The project uses Python's built-in `unittest` framework.

Run all tests from the project root:

```bash
python -m unittest discover -v
```

Or run individual test modules:

```bash
python -m unittest ingestion.test_ingest -v
python -m unittest ingestion.test_embeddings -v
python -m unittest ingestion.test_section_detector -v
python -m unittest retrieval.test_retriever -v
python -m unittest graph.test_state -v
python -m unittest graph.test_nodes -v
python -m unittest graph.test_graph -v
python -m unittest conversation.test_history -v
python -m unittest llm.test_llm -v
```

Most tests use fake retrievers and fake LLMs so that graph and node behavior can be tested without making real API calls.

---

# 20. Docker

The project also includes a simple Docker setup.

Build:

```bash
docker compose build
```

Run:

```bash
docker compose run --rm nist-rag
```

The ChromaDB directory is included in the project so the existing vector database can be used by the container.

The Groq API key is passed through the environment:

```bash
GROQ_API_KEY
```

---

# 21. Design Choices

## Why ChromaDB?

ChromaDB is lightweight and easy to run locally.

For this small knowledge base, a local persistent vector database is sufficient and avoids unnecessary infrastructure.

## Why BGE-small?

`BAAI/bge-small-en-v1.5` provides a relatively lightweight embedding model while producing useful semantic embeddings.

It also avoids requiring a separate hosted embedding API.

## Why 900-character chunks?

The documents contain structured technical material.

A moderate chunk size keeps enough surrounding context while avoiding unnecessarily large retrieval results.

An overlap of 150 characters helps preserve information across chunk boundaries.

## Why LangGraph?

The assignment explicitly requires LangGraph for orchestration.

LangGraph also makes the individual stages explicit:

```text
Query rewriting
      ↓
Retrieval
      ↓
Retrieval check
      ↓
Answer generation
```

This makes the workflow easier to extend later.

## Why JSON for conversation history?

The assignment only requires conversational context and at least two follow-up questions.

A small JSON file is sufficient for this use case and avoids adding a database or external memory service.

---

# 22. Limitations and Possible Improvements

The current implementation intentionally favors simplicity.

Possible future improvements include:

### Better retrieval evaluation

Currently retrieval sufficiency is based primarily on the number of retrieved documents.

A more advanced implementation could evaluate:

* Similarity distance
* Minimum relevance score
* LLM-based relevance checking
* Reranking

### Query retry

If retrieval is poor, the graph could:

```text
Retrieve
   ↓
Poor results?
   ↓
Rewrite query
   ↓
Retrieve again
```

### Better citation control

The answer-generation prompt could require every factual statement to contain a citation and validate that cited pages actually exist in the retrieved metadata.

### Persistent conversation sessions

The current implementation stores one conversation in a JSON file.

A production system could use a database or session identifier.

### Hybrid retrieval

Keyword search could be combined with semantic vector search to improve retrieval for exact terminology and section names.

---

# 23. Example Questions

### Basic

```text
What are the four core functions of the NIST AI RMF?
```

```text
What is the purpose of the NIST AI RMF?
```

```text
What is the Generative AI Profile?
```

### Conversational

```text
What are the four core functions?

Which one is cross-cutting?

What does it mean by cross-cutting?
```

### Context switching

```text
What is the NIST AI RMF Generative AI Profile?

What risks does it address?

Does it replace the original AI RMF?
```

### Insufficient-context testing

```text
What is the population of India?
```

The system should not use the LLM's general knowledge to answer this because the answer is outside the intended NIST knowledge base.

---

# 24. Assignment Coverage

| Requirement                | Implementation                      |
| -------------------------- | ----------------------------------- |
| NIST AI RMF knowledge base | `data/raw/`                         |
| PDF ingestion              | `ingestion/ingest.py`               |
| Chunking                   | Recursive character splitter        |
| Metadata                   | Document, page, section, chunk ID   |
| Embeddings                 | BGE-small via Transformers          |
| Vector store               | ChromaDB                            |
| Retrieval                  | `retrieval/retriever.py`            |
| LangGraph                  | `graph/graph.py`                    |
| Query rewriting            | `graph/nodes.py`                    |
| Conversation history       | `conversation/history.py`           |
| Source attribution         | Answer prompt + retrieved metadata  |
| Insufficient context       | Retrieval check + fallback          |
| LLM                        | Groq                                |
| Testing                    | `unittest` test files               |
| Docker                     | `Dockerfile` + `docker-compose.yml` |

---

# 25. Summary

This project implements a deliberately small Agentic RAG pipeline:

```text
NIST PDFs
   ↓
Ingestion
   ↓
Chunking + Metadata
   ↓
BGE Embeddings
   ↓
ChromaDB
   ↓
User Question
   ↓
Conversation-aware Query Rewriting
   ↓
Vector Retrieval
   ↓
Retrieval Check
   ↓
Groq LLM
   ↓
Grounded Answer + Sources
```

The implementation focuses on making the RAG and LangGraph concepts clear rather than introducing unnecessary infrastructure or abstractions.
