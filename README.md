# RAG System (HotpotQA)

A Retrieval-Augmented Generation (RAG) pipeline built on:

- HotpotQA (fullwiki)
- FAISS vector search
- SentenceTransformers embeddings
- FLAN-T5 grounded generation

This system indexes:

- Wikipedia article chunks
- Question–Answer pairs

and performs grounded answer generation using retrieved evidence.

---

## Architecture

Ingestion (`ingest.py`)
- Load HotpotQA fullwiki
- Extract unique Wikipedia documents
- Chunk using sliding window tokenization
- Add QA pairs as memory chunks
- Generate embeddings
- Normalize vectors (L2)
- Build FAISS IndexFlatIP
- Save index + metadata

Retrieval (`retrieval.py`)
- Embed query
- Perform similarity search
- Build structured context
- Generate grounded answer with FLAN-T5

---

## Installation

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Build the Index
```
python ingest.py
```

This will create:

- faiss_index.bin

- metadata.pkl

First run will download:

- HotpotQA dataset

- Embedding model

- Tokenizer

### 3. Run Retrival and Generation
```
python retrieval.py
```

Then ask questions.
