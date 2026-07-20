import pickle
import re
import numpy as np
import faiss
import pdfplumber
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv
import os

_embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def build_pipeline(index_path="book_index.faiss", data_path="book_data.pkl"):
    """Loads the pre-built index for the default book (Harry Potter)."""
    index = faiss.read_index(index_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    all_chunks = data["chunks"]
    all_metadatas = data["metadatas"]

    tokenized_chunks = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    return {
        "all_chunks": all_chunks,
        "all_metadatas": all_metadatas,
        "index": index,
        "bm25": bm25,
        "client": client,
    }


def _clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def build_pipeline_from_pdf(pdf_path):
    """Builds a pipeline on the fly from ANY uploaded PDF (no pre-built index needed)."""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=75,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks, all_metadatas = [], []
    for page_num, page_text in enumerate(pages_text):
        page_clean = _clean_text(page_text)
        if not page_clean.strip():
            continue
        for chunk in splitter.split_text(page_clean):
            all_chunks.append(chunk)
            all_metadatas.append({"page": page_num + 1})

    embeddings = list(_embed_model.embed(all_chunks))
    embedding_matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    tokenized_chunks = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    return {
        "all_chunks": all_chunks,
        "all_metadatas": all_metadatas,
        "index": index,
        "bm25": bm25,
        "client": client,
    }


def hybrid_search(pipeline, question, n_results=8, bm25_weight=0.5):
    all_chunks = pipeline["all_chunks"]
    all_metadatas = pipeline["all_metadatas"]
    index = pipeline["index"]
    bm25 = pipeline["bm25"]

    q_embedding = np.array(list(_embed_model.embed([question]))).astype("float32")
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


def ask_question(pipeline, question, n_results=8):
    all_chunks = pipeline["all_chunks"]
    all_metadatas = pipeline["all_metadatas"]
    client = pipeline["client"]

    context_chunks = hybrid_search(pipeline, question, n_results=n_results)

    existing = {c["chunk"] for c in context_chunks}
    for idx in range(3):
        if all_chunks[idx] not in existing:
            context_chunks.insert(0, {"chunk": all_chunks[idx], "page": all_metadatas[idx]["page"], "score": 1.0})

    context_chunks_sorted = sorted(context_chunks, key=lambda c: c["page"])
    context = "\n\n---\n\n".join([f"[Page {c['page']}]: {c['chunk']}" for c in context_chunks_sorted])

    system_prompt = """You are answering questions about a book using only the provided excerpts.
Read ALL excerpts carefully before answering. Excerpts are presented in page order (chronological order in the book).
If events change or outcomes are revealed later in the excerpts, the LATER page's information is the final/correct outcome.
Answer in exactly ONE concise sentence. Include a page citation in parentheses at the end, like (Page 42).
If the excerpts don't contain the answer, say so in one sentence."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content, context_chunks_sorted