import os
import json
import shutil
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === CONFIG ===
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_MODEL = os.getenv(
    "BGE_EMBED_DIR",
    "/app/embeddings/bge-large-en-v1.5"
)
SAVE_DIR = "output"

# === MAIN FUNCTIONS ===

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def split_text_to_documents(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={
            "heading": f"Chunk {i+1}",
            "content": chunk,
            "url": "uploaded_pdf"
        }) for i, chunk in enumerate(chunks)
    ]
    return docs

def save_json(docs: List[Document], output_file: str):
    json_data = []
    for doc in docs:
        json_data.append({
            "heading": doc.metadata.get("heading", ""),
            "text": doc.page_content,
            "content": doc.metadata.get("content", ""),
            "url": doc.metadata.get("url", "")
        })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

def save_vector_store(docs: List[Document], output_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(output_dir)

def process_pdfs_and_create_vector_store(uploaded_file_path: str):
    Path(SAVE_DIR).mkdir(exist_ok=True)

    print("[INFO] Extracting and splitting PDF...")
    text = extract_text_from_pdf(uploaded_file_path)
    docs = split_text_to_documents(text)

    print(f"[INFO] Saving {len(docs)} chunks to JSON...")
    save_json(docs, os.path.join(SAVE_DIR, "combined_pdf_data_tagged.json"))

    print("[INFO] Creating vector store...")
    save_vector_store(docs, os.path.join(SAVE_DIR, "vector_store_chunks"))

    print("[SUCCESS] Done generating vector store and JSON.")


