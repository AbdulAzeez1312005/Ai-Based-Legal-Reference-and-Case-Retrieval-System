import os
import time
import traceback
import hashlib
import re # Import the regular expression library
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
INDEX_NAME = "legal-document-index"
DATA_DIR = "data2"  # Folder where you'll store your documents
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
UPSERT_BATCH_SIZE = 100

# --- INITIALIZATION ---
def initialize_services():
    """Initializes and returns Pinecone index and the embedding model."""
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in the .env file.")

    print("🚀 Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
        print(f"🆕 Creating new Pinecone index: '{INDEX_NAME}'")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Dimension for sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)

    index = pc.Index(INDEX_NAME)
    print("✅ Pinecone is ready.")

    print("🚀 Initializing sentence-transformer model...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedder is ready.")

    return index, embedder

# --- STEP 1: LOAD ALL DATA FROM THE 'data' FOLDER ---
def load_data_from_folder():
    """Loads texts from all supported files in the specified data directory."""
    all_docs_with_metadata = []

    print(f"📂 Loading local documents from '{DATA_DIR}' folder...")
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ '{DATA_DIR}' directory not found. Please create it and add your files.")
        return []

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            elif filename.lower().endswith((".html", ".htm")):
                loader = UnstructuredHTMLLoader(file_path)
            else:
                continue  # Skip unsupported file types

            docs = loader.load()
            for doc in docs:
                # Add the source filename to the metadata
                doc.metadata['source'] = filename
                all_docs_with_metadata.append(doc)
            print(f"  - Loaded '{filename}'")
        except Exception as e:
            print(f"❌ Failed to load '{filename}': {e}")

    return all_docs_with_metadata

# --- ✨ NEW: Function to extract a section ID from text ---
def extract_section_id(text):
    """
    Uses regex to find patterns like 'IPC 28' or 'Article 21A' at the start of a text chunk.
    This improves accuracy by only tagging chunks that are explicitly about a section.
    """
    # This regex looks for common legal prefixes at the beginning of the text (^)
    match = re.search(r"^(IPC\s*\d+[A-Z]?|Section\s*\d+[A-Z]?|Article\s*\d+[A-Z]?)", text.strip(), re.IGNORECASE)
    if match:
        # Normalize the found ID: uppercase and remove all spaces
        return re.sub(r'\s+', '', match.group(0)).upper()
    return None

# --- STEP 2: CHUNK, DEDUPLICATE, AND PUSH TO PINECONE (MODIFIED) ---
def process_and_upsert_data(index, embedder, documents):
    """Chunks, embeds, adds section_id metadata, and upserts vectors into Pinecone."""
    if not documents:
        print("🤷 No data to process.")
        return

    print("\nSplitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    all_chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(all_chunks)} total text chunks.")

    print("\nRemoving duplicate chunks (based on content)...")
    unique_chunks_dict = {}
    for chunk in all_chunks:
        chunk_hash = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        if chunk_hash not in unique_chunks_dict:
            unique_chunks_dict[chunk_hash] = chunk
    
    unique_chunks = list(unique_chunks_dict.values())
    print(f"✅ Found {len(unique_chunks)} unique chunks to process.")

    if not unique_chunks:
        print("🤷 No new unique content to upsert.")
        return

    print(f"\nEmbedding {len(unique_chunks)} unique chunks...")
    texts_to_embed = [chunk.page_content for chunk in unique_chunks]
    embeddings = embedder.embed_documents(texts_to_embed)

    print(f"📤 Preparing and upserting {len(unique_chunks)} vectors in batches...")
    
    vectors_to_upsert = []
    for i, chunk in enumerate(unique_chunks):
        vector_id = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        
        metadata = chunk.metadata.copy()
        metadata['text'] = chunk.page_content

        # ✨ NEW: Extract section ID and add it to metadata if found
        section_id = extract_section_id(chunk.page_content)
        if section_id:
            metadata['section_id'] = section_id

        vectors_to_upsert.append({
            "id": vector_id,
            "values": embeddings[i],
            "metadata": metadata
        })

    # Upsert in batches
    for i in tqdm(range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE), desc="Upserting to Pinecone"):
        batch = vectors_to_upsert[i:i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
    
    print("\n✅ Bulk ingest process complete.")
    print("Index stats:", index.describe_index_stats())


def main():
    """Main function to run the ingestion pipeline."""
    confirm = input(f"This script will load all files from '{DATA_DIR}' and push them to the '{INDEX_NAME}' Pinecone index. This may overwrite data. Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return
        
    try:
        index, embedder = initialize_services()
        data = load_data_from_folder()
        process_and_upsert_data(index, embedder, data)
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()