# AI-Based Legal Reference and Case Retrieval System — Milestone 4

Milestone 4 rebuilds the earlier document-processing and RAG work as a React and Flask application. It searches Indian legal statutes and case documents in Pinecone, then drafts a source-grounded answer using the local Ollama model Qwen2.5 7B.

## Technologies

- React 18 and Vite for the web interface.
- Flask, Flask-Cors, Flask-SQLAlchemy, and SQLite for the application API, login, and saved conversations.
- Account profile settings for changing a display name and private PNG, JPEG, or WebP photo; chat history supports deleting individual conversations and their messages.
- Pinecone for hosted vector retrieval.
- Sentence Transformers, using sentence-transformers/all-MiniLM-L6-v2 (384 dimensions), for document/query embeddings.
- Ollama and qwen2.5:7b for local answer generation; no OpenAI key is used.
- pypdf, docx2txt, and BeautifulSoup for supported source files.

The generic ingestion pipeline recognizes section, article, rule, regulation, clause, and order headings from the documents themselves. It does not contain a hard-coded list of provisions for one Act. PDFs and text files remain in backend/data2.

## Setup on Windows

1. Install Python 3.11 or newer, Node.js 20 or newer, and VS Code.
2. Keep the existing backend/.env. If setting up a fresh copy, make backend/.env from backend/.env.example and provide a Pinecone API key locally. Do not paste credentials into chat or commit the .env file.
3. From this Milestone-4 directory, create and install the Python virtual environment:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
~~~

The backend reads its configuration from `backend/.env`:

| Variable | Purpose | Typical value |
| --- | --- | --- |
| `PINECONE_API_KEY` | Pinecone credential; keep it private | Set locally |
| `PINECONE_INDEX` | Pinecone index name | `legal-reference-m4` |
| `OLLAMA_BASE_URL` | URL reachable by Flask for the Ollama API | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local model used for answers | `qwen2.5:7b` |
| `FRONTEND_ORIGIN` | Allowed frontend origin for credentialed API requests | `http://localhost:5173` |
| `PORT` | Flask API port | `5000` |

The template also lists optional Flask, Pinecone, embedding, retrieval, and Ollama tuning settings.

4. Ensure the Ollama model is installed. If it is not already in the model store used by Ollama, run:

~~~powershell
$env:OLLAMA_MODELS = Join-Path (Get-Location) "ollama-models"
ollama pull qwen2.5:7b
~~~

5. In a separate VS Code terminal, start the Ollama copy and models stored in this project:

~~~powershell
.\start-ollama.ps1
~~~

The launcher sets `OLLAMA_MODELS` to the project-local `ollama-models` folder as well.

6. In another terminal, start Flask:

~~~powershell
.\.venv\Scripts\python.exe backend\app.py
~~~

7. In a third terminal, install and start the React app:

~~~powershell
cd frontend
npm install
npm run dev
~~~

8. Open http://127.0.0.1:5173, create an account, and start a research chat.

The Flask API health endpoint is http://127.0.0.1:5000/api/health. It reports whether Ollama responds, and which Pinecone index/model settings are selected.

Use the profile settings control at the bottom of the sidebar to update the display name or photo. Profile photos are limited to 1 MB and stored with the account. Use the trash control beside a conversation to delete that chat after confirmation.

## Ingest and refresh Pinecone

First parse the local source files without changing Pinecone:

~~~powershell
.\.venv\Scripts\python.exe backend\ingest.py --dry-run
~~~

When the report looks reasonable, upsert the vectors:

~~~powershell
.\.venv\Scripts\python.exe backend\ingest.py
~~~

The uploader confirms before sending. It writes versioned records with source, physical PDF page, provision label/title, document id, and chunk order. It uses Pinecone upsert with deterministic ids, does not delete vectors, and can be rerun safely. Ensure the configured index exists and its dimension matches the selected embedding model (384 for MiniLM).

## Verification

Run offline parser and retrieval regressions:

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend -p "test_*.py"
~~~

After ingestion, optional live retrieval checks can be run with:

~~~powershell
.\.venv\Scripts\python.exe backend\test_rag_pipeline.py --live
~~~

The live checks require Pinecone access and a populated index. Live answer generation also requires Ollama to be running and the configured model to be available.

## Deployment

Ollama runs locally for development. A deployed Flask server must be able to reach an Ollama service and Pinecone; the browser does not receive either service's credentials. The host running Ollama needs sufficient RAM or GPU memory for the model (about 5 GB for Qwen2.5 7B, plus runtime overhead). A serverless platform that cannot run Ollama or reach a private Ollama service cannot use this local setup directly.
