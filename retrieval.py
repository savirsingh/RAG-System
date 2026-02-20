import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =============================
# Config
# =============================
INDEX_PATH = "faiss_index.bin"
META_PATH = "metadata.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

TOP_K = 5

# =============================
# Load Index + Metadata
# =============================
print("Loading index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

#print(metadata[0])

# =============================
# Load Models
# =============================
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading generator model...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)

# =============================
# Retrieval
# =============================
def retrieve(query, k=TOP_K):
    query_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_emb)

    scores, ind = index.search(query_emb, k)

    results = []
    for score, idx in zip(scores[0], ind[0]):
        meta = metadata[idx]

        results.append({
            "score": float(score),
            "chunk_id": meta["chunk_id"],
            "doc_title": meta["doc_title"],
            "text": meta["text"],
            "type": meta["type"]
        })

    return results

# =============================
# Context Builder
# =============================
def build_context(retrieved):
    context_parts = []

    for r in retrieved:
        block = (
            f"[{r['chunk_id']} | {r['doc_title']} | {r['type']}]\n"
            f"{r['text']}"
        )
        context_parts.append(block)

    return "\n\n".join(context_parts)

# =============================
# Grounded Generation
# =============================
def generate_answer(query, retrieved):
    context = build_context(retrieved)

    prompt = f"""
You are reading evidence documents to answer a question.

Rules:
1) Only use information explicitly written in the context
2) Combine multiple sentences if necessary
3) Do NOT use outside knowledge
4) If the answer is missing, say exactly: I don't know

Return a short answer only.

CONTEXT:
{context}

QUESTION: {query}

FINAL ANSWER:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = generator.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return answer

# =============================
# Public Query Function
# =============================
def query_rag(question, k=TOP_K):
    retrieved = retrieve(question, k)
    answer = generate_answer(question, retrieved)

    return {
        "question": question,
        "answer": answer,
        "evidence": retrieved
    }

# =============================
# cli test
# =============================
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower()=="exit":
            break

        result = query_rag(q)

        print("\nANSWER:\n", result["answer"])
        print("\nEVIDENCE:")
        for e in result["evidence"]:
            print(
                f"- {e['chunk_id']} ({e['doc_title']}) "
                f"[{e['type']}] score={e['score']:.3f}"
            )
