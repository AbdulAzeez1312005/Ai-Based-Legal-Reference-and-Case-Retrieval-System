import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader

def load_file(file_path):
    """Load document based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def main():
    # Change this to your file name
    file_path = "sample.pdf"  # can be .txt / .pdf / .docx

    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return

    docs = load_file(file_path)

    print(f"\n Loaded {len(docs)} document(s) from {file_path}\n")
    for i, doc in enumerate(docs, start=1):
        print(f"--- Document {i} ---\n")
        print(doc.page_content[:500])  # print first 500 chars
        print("\n")

if __name__ == "__main__":
    main()
