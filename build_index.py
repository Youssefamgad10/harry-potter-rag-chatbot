import pdfplumber
import re
import pickle
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
import faiss

pdf_path = "harrypotter pdf.pdf"

pages_text = []
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

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

embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = list(embed_model.embed(all_chunks))
embedding_matrix = np.array(embeddings).astype("float32")

# --- Save FAISS index as a binary file ---
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)
faiss.write_index(index, "book_index.faiss")

# --- Save chunks + metadata as a pickle file ---
with open("book_data.pkl", "wb") as f:
    pickle.dump({"chunks": all_chunks, "metadatas": all_metadatas}, f)

print("Saved book_index.faiss and book_data.pkl")