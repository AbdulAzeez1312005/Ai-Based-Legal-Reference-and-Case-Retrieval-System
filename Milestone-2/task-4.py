import os
import time
import traceback
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

# --- CONFIGURATION ---
load_dotenv()
INDEX_NAME = "ipc-embeddings"
DATASET_NAME = "NahOR102/Indian-IPC-Laws"
DATA_DIR = "data"  # Folder for local files like PDF, DOCX, TXT
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
UPSERT_BATCH_SIZE = 100

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
        print("⏳ Waiting for index to be ready...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            
    index = pc.Index(INDEX_NAME)
    print("✅ Pinecone is ready.")
    print(f"   Index stats: {index.describe_index_stats()}")

    # Initialize Embedder
    print("🚀 Initializing sentence-transformer model...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedder is ready.")
    
    return index, embedder

# --- STEP 1 (Optional): BULK DATA PUSH ---
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
                all_texts_with_metadata.append({"text": assistant_content, "source": "dataset", "filename": DATASET_NAME})
        print(f"✅ Loaded {len(all_texts_with_metadata)} records from dataset.")
    except Exception as e:
        print(f"⚠️ Could not load Hugging Face dataset: {e}")

    # Load from local files
    print(f"📂 Loading local documents from '{DATA_DIR}' folder...")
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ '{DATA_DIR}' directory not found. Please create it and add files to it for local ingestion.")
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
                    all_texts_with_metadata.append({"text": doc.page_content, "source": "document", "filename": filename})
            print(f"   - Loaded '{filename}'")
        except Exception as e:
            print(f"❌ Failed to load '{filename}': {e}")
            
    return all_texts_with_metadata

def chunk_and_upsert_data(index, embedder, texts_with_metadata):
    """Chunks texts and upserts them as vectors into Pinecone."""
    if not texts_with_metadata:
        print("🤷 No data to process for bulk ingest.")
        return

    print("\nSplitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    vectors_to_upsert = []
    total_chunks = 0
    for i, item in enumerate(texts_with_metadata):
        chunks = text_splitter.split_text(item["text"])
        for j, chunk in enumerate(chunks):
            clean_filename = os.path.splitext(item['filename'])[0].replace(" ", "_")
            vector_id = f"{item['source']}-{clean_filename}-{i}-{j}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedder.embed_query(chunk),
                "metadata": {"text": chunk, "source": item["source"], "filename": item["filename"]}
            })
            total_chunks += 1
    print(f"✅ Created {total_chunks} text chunks.")

    print(f"📤 Upserting {len(vectors_to_upsert)} vectors in batches...")
    for i in range(0, len(vectors_to_upsert), UPSERT_BATCH_SIZE):
        batch = vectors_to_upsert[i:i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"   - Upserted batch {i//UPSERT_BATCH_SIZE + 1}/{(len(vectors_to_upsert) + UPSERT_BATCH_SIZE - 1)//UPSERT_BATCH_SIZE}")
    
    print("✅ Bulk ingest process complete.")
    print("   Index stats:", index.describe_index_stats())

# --- STEP 2: CRUD OPERATIONS ---

def create_operation(index, embedder):
    """(C)reates new vectors from user input text."""
    print("\n--- CREATE (Add Text) Operation ---")
    input_text = input("Enter the text you want to add (this will be marked as 'test_data'):\n> ").strip()

    if not input_text:
        print("🤷 No text provided.")
        return

    print("📝 Processing and embedding text...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_text(input_text)
        
        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            vector_id = f"test_data-{time.time()}-{i}" # Unique ID for test data
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedder.embed_query(chunk),
                "metadata": { "text": chunk, "source": "test_data", "filename": "manual_input" }
            })
        
        if vectors_to_upsert:
            print(f"📤 Upserting {len(vectors_to_upsert)} vector(s)...")
            index.upsert(vectors=vectors_to_upsert)
            print("✅ Text processed and added to Pinecone as 'test_data'.")
        else:
            print("🤷 No content found to process.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

def read_operation(index, embedder):
    """(R)eads/queries data from Pinecone by semantic query or ID."""
    print("\n--- READ Operation ---")
    search_by = input("Search by [query/id]: ").strip().lower()

    if search_by == "query":
        query_text = input("Enter search query: ")
        top_k = int(input("How many results to fetch? (e.g., 5): ") or "5")
        
        print("\n🔎 Searching...")
        query_embedding = embedder.embed_query(query_text)
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        
        if results.get("matches"):
            for match in results["matches"]:
                print("-" * 40)
                print(f"Score: {match['score']:.4f}")
                print(f"ID: {match['id']}")
                print(f"Source: {match['metadata'].get('filename', 'N/A')}")
                print(f"Text: {match['metadata'].get('text', 'N/A')[:500]}...")
        else:
            print("No results found.")

    elif search_by == "id":
        vector_id = input("Enter vector ID: ").strip()
        result = index.fetch(ids=[vector_id])
        print("\n🔎 Fetching result...")
        if result.get('vectors') and vector_id in result['vectors']:
            print(result['vectors'][vector_id])
        else:
             print(f"No vector found with ID '{vector_id}'.")
    else:
        print("⚠️ Invalid choice.")

def update_operation(index, embedder):
    """(U)pdates an existing vector's text and embedding."""
    print("\n--- UPDATE Operation ---")
    vector_id = input("Enter the ID of the vector to update: ").strip()
    new_text = input("Enter the new text for this vector: ")
    
    if not vector_id or not new_text:
        print("🤷 Vector ID and new text cannot be empty.")
        return

    print(f"🔄 Updating vector '{vector_id}'...")
    new_embedding = embedder.embed_query(new_text)
    
    try:
        # Fetch existing metadata to preserve it
        existing_vector = index.fetch(ids=[vector_id])
        metadata = existing_vector.get('vectors', {}).get(vector_id, {}).get('metadata', {})
        metadata['text'] = new_text # Update the text in the metadata
    except Exception:
        # If fetch fails, create a basic metadata structure
        metadata = {"text": new_text, "source": "manual_update"}
        
    index.upsert(vectors=[{"id": vector_id, "values": new_embedding, "metadata": metadata}])
    print("✅ Vector updated successfully.")

def delete_operation(index):
    """(D)eletes one or more vectors by ID."""
    print("\n--- DELETE Operation ---")
    vector_ids_str = input("Enter vector ID(s) to delete (comma-separated): ").strip()
    vector_ids = [vid.strip() for vid in vector_ids_str.split(",") if vid.strip()]
    
    if not vector_ids:
        print("🤷 No IDs provided.")
        return
        
    confirm = input(f"Are you sure you want to delete {len(vector_ids)} vector(s)? (y/n): ").lower()
    if confirm == 'y':
        print(f"🗑️ Deleting {len(vector_ids)} vector(s)...")
        index.delete(ids=vector_ids)
        print("✅ Vector(s) deleted successfully.")
    else:
        print("Deletion cancelled.")

# --- MAIN EXECUTION ---
def main():
    """Main function to run the entire pipeline."""
    try:
        index, embedder = initialize_services()
    except Exception as e:
        print(f"❌ Critical Error: Could not initialize services. Exiting. Details: {e}")
        return

    # --- Optional Bulk Push ---
    confirm_bulk = input("\nPerform initial bulk data push from Hugging Face and local files? (y/n): ").strip().lower()
    if confirm_bulk == 'y':
        print("\n--- Starting Bulk Data Ingestion ---")
        try:
            data = load_and_normalize_data()
            chunk_and_upsert_data(index, embedder, data)
        except Exception as e:
            print(f"\n❌ An error occurred during bulk ingestion: {e}")
            traceback.print_exc()
        print("\n--- Bulk Ingestion Complete ---")
    else:
        print("\nSkipping bulk data push.")

    # --- Interactive CRUD Menu ---
    while True:
        print("\n" + "="*50)
        print("   Pinecone CRUD Operations for Indian Law")
        print("="*50)
        choice = input("Choose operation [create / read / update / delete / exit]: ").strip().lower()
        
        try:
            if choice == "create":
                create_operation(index, embedder)
            elif choice == "read":
                read_operation(index, embedder)
            elif choice == "update":
                update_operation(index, embedder)
            elif choice == "delete":
                delete_operation(index)
            elif choice == "exit":
                print("👋 Exiting program.")
                break
            else:
                print("⚠️ Invalid choice. Please try again.")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()