# app.py

import os
import traceback
import streamlit as st
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
# Make sure to create a .env file with your API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ipc-embeddings" # << IMPORTANT: Use your Pinecone index name here

# --- System Template for Output Formatting (Copied from your original script) ---
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

# --- Core Chatbot Logic (from your Task 8) ---
# Use st.cache_resource to initialize the RAG chain only once.
@st.cache_resource
def get_rag_chain():
    """
    Initializes and creates a conversational RAG chain.
    The @st.cache_resource decorator ensures this heavy function runs only once.
    """
    try:
        # 1. Initialize Embeddings Model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Initialize Language Model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key=OPENAI_API_KEY)

        # 3. Connect to Pinecone Vector Store and create a retriever
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key='text'
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # --- CONSTRUCT THE RAG CHAIN ---
        rephrase_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a concise, standalone search query to look up in order to get information relevant to the conversation.")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, rephrase_prompt
        )

        final_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE + "\n\n**Context from retrieved documents:**\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    except Exception as e:
        st.error(f"An error occurred during chain creation: {e}")
        st.error(traceback.format_exc())
        return None

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Legal Research Assistant", page_icon="⚖️")
st.title("⚖️ AI Legal Research Assistant")

# Check for API keys and display a warning if they're missing
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    st.warning(
        "Please add your OpenAI and Pinecone API keys to a `.env` file in the same directory."
    )
    st.stop()

# Initialize the RAG chain
with st.spinner("Initializing assistant... This may take a moment."):
    rag_chain = get_rag_chain()

if not rag_chain:
    st.error("Failed to initialize the RAG chain. Please check the console for errors.")
    st.stop()
    
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you with your legal research today?"}
    ]

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about the Indian Penal Code..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the chat history for the LangChain model
    # Exclude the last message (the user's current prompt) from history
    langchain_chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=msg["content"]))
        else:
            langchain_chat_history.append(AIMessage(content=msg["content"]))

    # Get the response from the chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": langchain_chat_history
                })
                
                answer = response["answer"]
                st.markdown(answer)
                
                # Add assistant's response to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"An error occurred while getting the response: {e}")
                st.error(traceback.format_exc())