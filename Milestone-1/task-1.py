import os
import json
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Input and Output directories
INPUT_DIR = "data"
OUTPUT_DIR = "output_json"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "all_chunks.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_file(file_path):
    """Load file based on extension using LangChain loaders"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        try:
            return TextLoader(file_path, encoding="utf-8").load()
        except Exception:
            return TextLoader(file_path, encoding="latin-1").load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def chunk_documents(docs, chunk_size=2000, chunk_overlap=200):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def process_files():
    """Load all files, split into chunks, and save to one JSON"""
    all_chunks = []

    if not os.path.exists(INPUT_DIR):
        print(f" Input folder not found: {INPUT_DIR}")
        return

    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]
    if not files:
        print(" No files found in 'data' folder.")
        return

    for file_path in files:
        if not os.path.isfile(file_path):
            continue

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()

        print(f"\n Processing file: {file_name} ({ext})")

        try:
            docs = load_file(file_path)
            chunks = chunk_documents(docs)

            for i, chunk in enumerate(chunks, start=1):
                all_chunks.append({
                    "file_name": file_name,
                    "extension": ext,
                    "chunk_id": i,
                    "content": chunk.page_content.strip()
                })

            print(f" {file_name}: {len(chunks)} chunks created.")

        except Exception as e:
            print(f" Error with {file_name}: {str(e)}")

    # Save all chunks in one JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n All chunks saved → {OUTPUT_FILE}")
    print(f" Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    process_files()
