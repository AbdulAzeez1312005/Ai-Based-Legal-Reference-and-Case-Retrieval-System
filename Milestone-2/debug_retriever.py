# Save this as debug_retriever.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- CONFIGURATION ---
load_dotenv()
INDEX_NAME = "ipc-embeddings"

def test_retriever(query: str):
    """Initializes a retriever and fetches documents for a given query."""
    print("--- Initializing Retriever ---")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key='text'  # Ensure this is here
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ Retriever initialized successfully.")

        print(f"\n--- Retrieving documents for query: '{query}' ---")
        docs = retriever.invoke(query)

        if not docs:
            print("\n❌ No documents were found. The index might be empty or the query is not matching.")
            return

        print(f"\n✅ Found {len(docs)} documents. Here is their content:\n")
        for i, doc in enumerate(docs):
            print(f"--- DOCUMENT {i+1} ---")
            print(f"Content: {doc.page_content}\n")
            print(f"Metadata: {doc.metadata}\n")
            print("-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    user_query = input("Enter the query to test: ")
    test_retriever(user_query)