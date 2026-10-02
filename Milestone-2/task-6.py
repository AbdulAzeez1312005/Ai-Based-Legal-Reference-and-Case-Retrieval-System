import os
import time
import traceback
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# --- CONFIGURATION ---
load_dotenv()
INDEX_NAME = "ipc-embeddings"

# --- INITIALIZATION ---
def initialize_services():
    """Initializes Pinecone and returns the embedding model."""
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
            
    print("✅ Pinecone is ready.")

    # Initialize Embedder
    print("🚀 Initializing sentence-transformer model...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedder is ready.")
    
    return embedding

# --- MAIN EXECUTION ---
def main():
    """Connects to Pinecone and retrieves documents based on a query."""
    try:
        embedding = initialize_services()

        print("\n--- Connecting to Existing Pinecone Index ---")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embedding
        )
        print("✅ Connected to vectorstore successfully.")

        # Set up the retriever to fetch the top 4 results
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Get user query
        query = input("\nEnter your search query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return

        # Retrieve and print results
        print("\n🔎 Retrieving top 4 results...")
        results = retriever.invoke(query)

        if not results:
            print("🤷 No results found.")
        else:
            for i, doc in enumerate(results):
                print("-" * 60)
                print(f"RESULT {i+1}")
                source = doc.metadata.get('filename', 'N/A')
                print(f"Source: {source}")
                print(f"Content: {doc.page_content}")
        print("-" * 60)

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()