import os
import traceback
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Load Environment Variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ipc-embeddings"  # << IMPORTANT: Use your Pinecone index name here

# --- System Template for Output Formatting ---
SYSTEM_TEMPLATE = """
You are a specialized legal research assistant. Your task is to analyze the provided legal text snippets (context) and generate a structured response based *exclusively* on that text. Do not invent information or use external knowledge.

Your response must follow this exact format:

**Legal Disclaimer**: This information is for research purposes only. I am not a licensed attorney, and this does not constitute legal advice. For matters affecting your legal rights, please consult a qualified attorney.

**RELEVANT LEGAL PROVISIONS:**
- [Identify the source and specific section/article from the context. Format as: "Source | Article/Section number"]
  Excerpt: "[Quote the most relevant sentence or short passage directly from the provided context.]"
(If multiple provisions are found in the context, list each one in this format.)
(If no provisions are found, state "N/A")

**LEGAL SUMMARY:**
[Write a brief, neutral summary of the core legal rule or principle described in the provided context.]
(If context is insufficient for a summary, state "N/A")

**CITATIONS & SOURCES:**
- [List all sources mentioned in the context, like "Constitution of India | Article 15".]
(If no sources are identifiable in the context, state "N/A")
"""

def create_rag_chain():
    """
    Initializes and creates a conversational Retrieval-Augmented Generation (RAG) chain.
    """
    print("🚀 Initializing RAG chain resources...")
    
    try:
        # 1. Initialize Embeddings Model (to understand query semantics)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Initialize Language Model (to generate answers)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

        # 3. Connect to Pinecone Vector Store and create a retriever
        print(f"🔗 Connecting to Pinecone index: '{INDEX_NAME}'...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key='text'  # <--- THIS IS THE CRITICAL FIX
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ Connected to vectorstore and created retriever successfully.")

        # --- CONSTRUCT THE RAG CHAIN ---

        # Prompt for rephrasing a follow-up question to be standalone
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a concise, standalone search query to look up in order to get information relevant to the conversation.")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, rephrase_prompt
        )

        # Final prompt to answer the user's question using the retrieved context
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE + "\n\n**Context from retrieved documents:**\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        # Chain to "stuff" the retrieved documents into the final prompt
        question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
        
        # The complete RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        print("✅ RAG chain loaded successfully.")
        return rag_chain

    except Exception as e:
        print(f"❌ An error occurred during chain creation: {e}")
        traceback.print_exc()
        return None

def main():
    """
    Main function to run the interactive chat session.
    """
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        print("❌ Error: Make sure OPENAI_API_KEY and PINECONE_API_KEY are set in your .env file.")
        return

    rag_chain = create_rag_chain()
    if not rag_chain:
        print("Could not initialize the RAG chain. Exiting.")
        return
        
    chat_history = []
    print("\n--- AI Legal Research Assistant ---")
    print("Enter your query below. Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            
            print("🧠 Thinking...")
            # Invoke the chain with the user's input and conversation history
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            # Print the formatted answer from the LLM
            print("\n" + response["answer"])

            # Update chat history for the next turn
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response["answer"]))

        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()