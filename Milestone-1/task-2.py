import os
import json
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- CONFIG ----------
OUTPUT_DIR = "output_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the dataset from Hugging Face
print("📥 Downloading dataset from Hugging Face...")
dataset = load_dataset("NahOR102/Indian-IPC-Laws")

# Initialize text splitter (for answers/outputs)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""]
)

all_chunks = []

# Loop through dataset
for split in dataset.keys():
    for row in dataset[split]:
        messages = row.get("messages", [])

        # Extract user question and assistant answer
        question = ""
        answer = ""
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"].strip()
            elif msg["role"] == "assistant":
                answer = msg["content"].strip()

        if not question or not answer:
            continue

        # Split only the answer into chunks
        answer_chunks = splitter.split_text(answer)

        for chunk in answer_chunks:
            all_chunks.append({
                "question": question,
                "answer_chunk": chunk.strip()
            })

# Save into JSON
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ipc_laws_QA_chunks.json")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(all_chunks)} question-answer chunks → {OUTPUT_FILE}")

# Show one example
if all_chunks:
    print("\n🔍 Example QA chunk:")
    print(all_chunks[0])