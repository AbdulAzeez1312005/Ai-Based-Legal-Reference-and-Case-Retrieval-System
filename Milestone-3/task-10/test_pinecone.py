import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# --- SETUP ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ipc-embeddings"  # << MAKE SURE THIS MATCHES YOUR INDEX

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# --- A SIMPLE QUERY ---
# Use a query you know should have a match in your documents
query_text = "What is the law regarding theft?" 

print(f"Connecting to Pinecone index: '{INDEX_NAME}'...")
try:
    # 1. Initialize Pinecone connection
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    # 2. Initialize the same embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 3. Convert your text query into a vector
    print(f"Embedding query: '{query_text}'")
    query_vector = model.encode(query_text).tolist()
    
    # 4. Query the Pinecone index
    print("Searching for similar documents in Pinecone...")
    results = index.query(
        vector=query_vector,
        top_k=3,  # Ask for the top 3 results
        include_metadata=True  # We MUST get the metadata back
    )
    
    # 5. Print the results
    print("\n--- QUERY RESULTS ---")
    if results['matches']:
        for match in results['matches']:
            print(f"Score: {match['score']:.4f}")
            # IMPORTANT: Check if the 'text' key exists and has content
            if 'metadata' in match and 'text' in match['metadata']:
                print(f"Text: {match['metadata']['text'][:200]}...") # Print first 200 chars
            else:
                print("!!! ERROR: Found a match, but its metadata does not contain a 'text' field.")
                print(f"Available metadata: {match.get('metadata', 'None')}")
            print("-" * 20)
    else:
        print("!!! No matches found in the index for this query.")

except Exception as e:
    print(f"An error occurred: {e}")