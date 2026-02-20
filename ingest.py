import os
import pickle
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import faiss
import numpy as np

# =============================
# Configuration
# =============================
MAX_TOKENS = 300
STRIDE = 150

EMBED_MODEL = "all-MiniLM-L6-v2"
TOKENIZER_MODEL = "google/flan-t5-small"

INDEX_PATH = "faiss_index.bin"
META_PATH = "metadata.pkl"

# =============================
# 1. Load Dataset
# =============================
print("Loading HotpotQA dataset...")
ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

# =============================
# 2. Extract Unique Wiki Articles
# =============================
print("Extracting wiki docs...")
docs = {}

for split in ds.keys():
    for example in ds[split]:
        context = example["context"]

        titles = context["title"]
        sentences = context["sentences"]

        for title, sent_list in zip(titles, sentences):
            if title not in docs:
                docs[title] = " ".join(sent_list)

docs = dict(list(docs.items())[:3000])
print(f"Collected {len(docs)} unique articles")

# =============================
# 3. Chunking
# =============================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

def chunk_text(text, max_tokens=MAX_TOKENS, stride=STRIDE):
    input_ids = tokenizer(text)["input_ids"]
    chunks = []

    i = 0
    while i < len(input_ids):
        chunk_ids = input_ids[i:i+max_tokens]

        if not chunk_ids:
            break

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        chunks.append(chunk_text)

        if i + max_tokens >= len(input_ids):
            break

        i += stride

    return chunks

print("Chunking wiki docs...")
all_chunks = []
metadata = []

# -----------------------------
# Add wiki chunks
# -----------------------------
for title, text in docs.items():
    chunks = chunk_text(text)

    for ind, chunk in enumerate(chunks):
        chunk_id = f"{title}__{ind}"

        all_chunks.append(chunk)
        metadata.append({
            "doc_title": title,
            "chunk_id": chunk_id,
            "text": chunk,
            "type": "wiki"
        })

print(f"Created {len(all_chunks)} wiki chunks")

# =============================
# 3.5 Add QA Chunks
# =============================
print("Extracting QA pairs...")
qa_pairs = []

for split in ds.keys():
    for example in ds[split]:
        question = example["question"]
        answer = example["answer"]
        qa_pairs.append((question, answer))

qa_pairs = qa_pairs[:3000]

print("Adding QA chunks...")

for i, (question, answer) in enumerate(qa_pairs):
    qa_text = f"Question: {question}\nAnswer: {answer}"
    chunk_id = f"QA__{i}"

    all_chunks.append(qa_text)
    metadata.append({
        "doc_title": "HotpotQA",
        "chunk_id": chunk_id,
        "text": qa_text,
        "type": "qa"
    })

print(f"Total chunks after QA addition: {len(all_chunks)}")

# =============================
# 4. Generate Embeddings
# =============================
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Generating embeddings...")
embeddings = embed_model.encode(
    all_chunks,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")
faiss.normalize_L2(embeddings)

# =============================
# 5. Build FAISS Index
# =============================
print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("Saving index...")
faiss.write_index(index, INDEX_PATH)

print("Saving metadata...")
with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("\nIngestion complete.")
print(f"FAISS index saved to: {INDEX_PATH}")
print(f"Metadata saved to: {META_PATH}")
