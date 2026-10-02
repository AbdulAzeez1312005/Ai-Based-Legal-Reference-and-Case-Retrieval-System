import os
import time
import traceback
import hashlib
import re
from datasets import load_dataset
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredHTMLLoader,
)
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# --- CONFIGURATION ---
load_dotenv()
INDEX_NAME = "ipc-embeddings"
DATASET_NAME = "NahOR102/Indian-IPC-Laws"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
UPSERT_BATCH_SIZE = 100
DATA_DIR = "data"

# --- INITIALIZATION ---
def initialize_services():
    """Initializes and returns Pinecone index and the embedding model."""
    # Initialize Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in the .env file.")

    print("🚀 Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index exists, create it if it doesn't
    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        print(f"🆕 Creating new Pinecone index: '{INDEX_NAME}'")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Dimension for sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for the index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            
    index = pc.Index(INDEX_NAME)
    print("✅ Pinecone is ready.")

    # Initialize Embedder
    print("🚀 Initializing sentence-transformer model...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedder is ready.")
    
    return index, embedder

# --- STEP 1: COMBINE ALL DATA ---
def load_and_normalize_data():
    """Loads and normalizes texts from Hugging Face dataset and local files."""
    all_texts_with_metadata = []
    
    # Load from Hugging Face Dataset
    print(f"📥 Loading dataset '{DATASET_NAME}' from Hugging Face...")
    try:
        dataset = load_dataset(DATASET_NAME, split="train")
        for item in dataset:
            messages = item.get("messages", [])
            assistant_content = next((msg.get("content") for msg in messages if msg.get("role") == "assistant"), None)
            if assistant_content:
                all_texts_with_metadata.append({"text": assistant_content})
        print(f"✅ Loaded {len(all_texts_with_metadata)} records from dataset.")
    except Exception as e:
        print(f"⚠️ Could not load Hugging Face dataset: {e}")

    # Load from local files
    print(f"📂 Loading local documents from '{DATA_DIR}' folder...")
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ '{DATA_DIR}' directory not found. Please create it and add files.")
        return all_texts_with_metadata

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        try:
            if filename.lower().endswith(".pdf"): loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".docx"): loader = Docx2txtLoader(file_path)
            elif filename.lower().endswith(".txt"): loader = TextLoader(file_path, encoding="utf-8")
            elif filename.lower().endswith((".html", ".htm")): loader = UnstructuredHTMLLoader(file_path)
            else: continue
            
            docs = loader.load()
            for doc in docs:
                if doc.page_content:
                    all_texts_with_metadata.append({"text": doc.page_content})
            print(f"   - Loaded '{filename}'")
        except Exception as e:
            print(f"❌ Failed to load '{filename}': {e}")
            
    return all_texts_with_metadata

# --- STEP 2: CHUNK, DEDUPLICATE, AND PUSH TO PINECONE ---
def process_and_upsert_data(index, embedder, texts_with_metadata):
    """Chunks, embeds, deduplicates based on content, and upserts vectors into Pinecone."""
    if not texts_with_metadata:
        print("🤷 No data to process.")
        return

    print("\nSplitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    all_chunks = []
    for item in texts_with_metadata:
        chunks = text_splitter.split_text(item["text"])
        for chunk in chunks:
            all_chunks.append({"text": chunk})
    print(f"✅ Created {len(all_chunks)} total text chunks.")

    print("\nRemoving duplicate chunks...")
    unique_chunks_dict = {}
    for chunk_item in all_chunks:
        chunk_hash = hashlib.sha256(chunk_item["text"].encode('utf-8')).hexdigest()
        if chunk_hash not in unique_chunks_dict:
            unique_chunks_dict[chunk_hash] = chunk_item
    
    unique_chunks = list(unique_chunks_dict.values())
    print(f"✅ Found {len(unique_chunks)} unique chunks to process.")

    if not unique_chunks:
        print("🤷 No new unique content to upsert.")
        return

    print(f"\nEmbedding {len(unique_chunks)} unique chunks...")
    texts_to_embed = [chunk["text"] for chunk in unique_chunks]
    embeddings = embedder.embed_documents(texts_to_embed)

    print(f"📤 Upserting {len(unique_chunks)} vectors in batches...")
    
    vectors_to_upsert = []
    chunk_counters = {}

    for i, chunk_item in enumerate(unique_chunks):
        text = chunk_item["text"]
        vector_id = ""

        # Attempt to create a specific ID from text content based on IPC section
        ipc_match = re.match(r"IPC\s*(\d+[A-Z]?)", text, re.IGNORECASE)
        if ipc_match:
            ipc_code = ipc_match.group(1).upper()
            base_id = f"IPC-{ipc_code}"
            
            chunk_index = chunk_counters.get(base_id, 0)
            vector_id = f"{base_id}-{chunk_index}"
            chunk_counters[base_id] = chunk_index + 1
        else:
            # Fallback to a hash-based ID for non-IPC content
            vector_id = hashlib.sha256(text.encode('utf-8')).hexdigest()
            
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embeddings[i],
            "metadata": {"text": text}
        })

    # Upsert in batches
    for i in tqdm(range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE)):
        batch = vectors_to_upsert[i:i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
    
    print("✅ Bulk ingest process complete.")
    print("Index stats:", index.describe_index_stats())


def main():
    """Main function to run the bulk ingestion pipeline."""
    confirm = input("This script will load ALL data, deduplicate it, and push it to Pinecone. This is for initial setup. Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return
        
    try:
        index, embedder = initialize_services()
        data = load_and_normalize_data()
        process_and_upsert_data(index, embedder, data)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

