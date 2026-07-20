# Harry Potter RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about *Harry Potter , grounded entirely in the actual book text.

**Live demo:** https://harry-potter-rag-chatbot-z8yk5z7pugkb5urgxsbjsy.streamlit.app/

---

##  Overview

This project extracts text from a PDF of the book, splits it into searchable chunks, retrieves the most relevant passages for any question using a hybrid search strategy, and generates a grounded, cited answer using an LLM — refusing to guess when the answer isn't in the retrieved text.

```
PDF (Harry Potter Book 1)
        │
        ▼
Text Extraction (pdfplumber)
        │
        ▼
Cleaning & Chunking (LangChain RecursiveCharacterTextSplitter)
        │
        ▼
Embeddings (fastembed / BGE-small)  ──▶  Pre-built once, saved to disk
        │
        ▼
Vector Index (FAISS) + Keyword Index (BM25)
        │
        ▼
   User Question
        │
        ▼
Hybrid Search + Query Expansion
        │
        ▼
Relevant Chunks (sorted chronologically)
        │
        ▼
LLM Answer Generation (Groq / Llama 3.3)
        │
        ▼
Streamlit UI

- Support multiple books / a full series
- Add conversation memory for follow-up questions
- Add a cross-encoder re-ranking step on top of hybrid search results
