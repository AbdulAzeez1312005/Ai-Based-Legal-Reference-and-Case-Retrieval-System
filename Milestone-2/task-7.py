import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# --- SETUP (runs only once) ---
try:
    # 1. Initialize the model
    model = ChatOpenAI(model="gpt-3.5-turbo")

    # 2. Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's question and then stop."),
        ("user", "{input}")
    ])

    # 3. Create the output parser
    output_parser = StrOutputParser()

    # 4. Create the chain
    chain = prompt | model | output_parser

    print(" Model initialized. You can now start asking questions.")
except Exception as e:
    print(f" Error initializing model: {e}")
    exit()
# --------------------------------

# --- INTERACTIVE LOOP (runs continuously) ---
while True:
    # 5. Get query from user input
    my_query = input("\nPlease enter your query (or type 'exit' to quit): ")

    # 6. Check for exit condition
    if my_query.lower() == 'exit':
        print("Goodbye!")
        break

    # 7. Invoke the chain and print the response
    try:
        print(" Thinking...")
        response = chain.invoke({"input": my_query})
        print("\n Response:")
        print(response)
    except Exception as e:
        print(f"\n An error occurred: {e}")