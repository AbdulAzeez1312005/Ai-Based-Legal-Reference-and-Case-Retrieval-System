import os
import re
import json
from collections import defaultdict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader
)

# ---------- CONFIG ----------
OUTPUT_DIR = "output_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_DIR = "data"
TASK2_FILE = os.path.join(OUTPUT_DIR, "ipc_laws_QA_chunks.json")

# ---------- CLEAN OUTPUT (OPTIONAL) ----------
for f in os.listdir(OUTPUT_DIR):
    if f.endswith("_embeddings.json") or f == "all_embeddings.json":
        os.remove(os.path.join(OUTPUT_DIR, f))

# ---------- HELPER FUNCTIONS ----------
def clean_filename(name: str) -> str:
    """Normalize filename: lowercase, remove duplicates (1), underscores, remove special chars."""
    name = name.lower()
    name = os.path.splitext(name)[0]  # remove extension
    name = re.sub(r"\s*\(\d+\)", "", name)  # remove (1), (2), etc.
    name = re.sub(r"\s+", "_", name)        # spaces → underscores
    name = re.sub(r"[^\w_]+", "", name)     # remove non-alphanumeric/underscore
    return name

def load_documents(file_path: str):
    """Load documents depending on file extension."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith((".ppt", ".pptx")):
        loader = UnstructuredPowerPointLoader(file_path)
    elif file_path.endswith((".html", ".htm")):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        print(f"⚠️ Skipping unsupported file: {file_path}")
        return []
    return loader.load()

# ---------- LOAD TASK 1 FILES ----------
task1_chunks = []
for file_name in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file_name)
    if os.path.isfile(file_path):
        docs = load_documents(file_path)
        for doc in docs:
            task1_chunks.append({
                "file_name": file_name,
                "content": doc.page_content
            })

print(f"📄 Loaded {len(task1_chunks)} chunks from {len(os.listdir(DATA_DIR))} files")

# ---------- LOAD TASK 2 CHUNKS ----------
with open(TASK2_FILE, "r", encoding="utf-8") as f:
    task2_chunks = json.load(f)

# ---------- EMBEDDING MODEL ----------
print("📥 Loading Hugging Face model: sentence-transformers/all-MiniLM-L6-v2 ...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------- TEXT SPLITTER ----------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

# ---------- TASK 1: SPLIT + EMBED BY FILE ----------
docs_by_file = defaultdict(list)
for chunk in task1_chunks:
    docs_by_file[chunk["file_name"]].append(chunk)

all_embeddings = []

for file_name, chunks in docs_by_file.items():
    output_data = []
    id_counter = 1
    for chunk in chunks:
        sub_chunks = splitter.split_text(chunk["content"].strip())
        for sub_text in sub_chunks:
            embedding = embeddings.embed_query(sub_text)
            record = {
                "id": id_counter,
                "source": file_name,
                "text": sub_text,
                "embedding": embedding
            }
            output_data.append(record)
            all_embeddings.append(record)
            id_counter += 1

    # Save per-document
    clean_name = clean_filename(file_name)
    output_file = os.path.join(OUTPUT_DIR, f"{clean_name}_embeddings.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(output_data)} embeddings → {output_file}")

# ---------- TASK 2: IPC DATASET ----------
ipc_output = []
id_counter = 1
for chunk in task2_chunks:
    sub_chunks = splitter.split_text(chunk["answer_chunk"].strip())
    for sub_text in sub_chunks:
        embedding = embeddings.embed_query(sub_text)
        record = {
            "id": id_counter,
            "source": "ipc_laws_QA_chunks",
            "text": sub_text,
            "embedding": embedding
        }
        ipc_output.append(record)
        all_embeddings.append(record)
        id_counter += 1

ipc_file = os.path.join(OUTPUT_DIR, "ipc_laws_embeddings.json")
with open(ipc_file, "w", encoding="utf-8") as f:
    json.dump(ipc_output, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(ipc_output)} embeddings → {ipc_file}")

# ---------- MASTER MERGED FILE ----------
master_file = os.path.join(OUTPUT_DIR, "all_embeddings.json")
with open(master_file, "w", encoding="utf-8") as f:
    json.dump(all_embeddings, f, indent=2, ensure_ascii=False)

print(f"🎯 Final merged embeddings saved → {master_file}")
