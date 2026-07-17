import pdfplumber
import re
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv
import os

# --- Step 1: Extract PDF text ---
pdf_path = "harrypotter pdf.pdf"

pages_text = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

print(f"Pages extracted: {len(pages_text)}")

# --- Step 2: Clean text ---
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

# --- Step 3: Chunk text ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []
all_metadatas = []

for page_num, page_text in enumerate(pages_text):
    page_clean = clean_text(page_text)
    if not page_clean.strip():
        continue
    page_chunks = splitter.split_text(page_clean)
    for chunk in page_chunks:
        all_chunks.append(chunk)
        all_metadatas.append({"page": page_num + 1})

print(f"Total chunks: {len(all_chunks)}")

lengths = [len(c) for c in all_chunks]
print(f"Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.0f}")
print(f"Unique chunks: {len(set(all_chunks))} / {len(all_chunks)}")

# --- Step 4: Generate embeddings ---
print("Generating embeddings...")
embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = list(embed_model.embed(all_chunks))
embedding_matrix = np.array(embeddings).astype("float32")
print(f"Embeddings shape: {embedding_matrix.shape}")

# --- Step 5: Build FAISS index ---
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)
print(f"Vectors in index: {index.ntotal}")

# --- Step 6: Build BM25 index ---
tokenized_chunks = [c.lower().split() for c in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# --- Step 7: Hybrid search function ---
def hybrid_search(question, n_results=8, bm25_weight=0.5):
    q_embedding = np.array(list(embed_model.embed([question]))).astype("float32")
    distances, indices = index.search(q_embedding, n_results * 2)

    tokenized_q = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_q)

    combined_scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        combined_scores[idx] = (1 - bm25_weight) * (1 / (1 + dist))

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for idx, score in enumerate(bm25_scores):
        if idx in combined_scores:
            combined_scores[idx] += bm25_weight * (score / max_bm25)

    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
    return [{"chunk": all_chunks[idx], "page": all_metadatas[idx]["page"], "score": score} for idx, score in ranked]

# --- Step 8: Groq client ---
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# --- Step 9: Query expansion + ask_question ---
def expand_query(question):
    aliases = {
        "harry's uncle": "Mr. Dursley Uncle Vernon Grunnings drills",
        "harry's aunt": "Mrs. Dursley Aunt Petunia",
        "harry's school": "Hogwarts",
        "harry's best friend": "Ron Weasley Hermione Granger",
        "harry's house": "Sorting Hat Gryffindor sorted",
    }
    expanded = question
    for phrase, exp in aliases.items():
        if phrase.lower() in question.lower():
            expanded += " " + exp
    return expanded

def ask_question(question, n_results=8):
    expanded_q = expand_query(question)
    context_chunks = hybrid_search(expanded_q, n_results=n_results)

    existing = {c["chunk"] for c in context_chunks}
    for idx in range(3):
        if all_chunks[idx] not in existing:
            context_chunks.insert(0, {"chunk": all_chunks[idx], "page": all_metadatas[idx]["page"], "score": 1.0})

    context = "\n\n---\n\n".join([f"[Page {c['page']}]: {c['chunk']}" for c in context_chunks])

    system_prompt = """You are answering questions about a book using only the provided excerpts.
Read ALL excerpts carefully before answering. If the excerpts don't contain the answer, say so.
Cite the page number(s) you used."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# --- Step 10: Test it ---
print("\n--- Test question ---")
print(ask_question("What does Harry's uncle do for work?"))
