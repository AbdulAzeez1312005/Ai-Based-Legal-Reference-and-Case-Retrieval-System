# AI-Based Legal Reference and Case Retrieval System — Milestone 4

Milestone 4 is a full-stack retrieval-augmented generation (RAG) application for asking questions about Indian legal material. Users can ask about Acts, provisions, and judgments; the system retrieves relevant passages, generates an answer grounded in those passages, and cites the source file and page. It also provides sign-in, saved conversations, profile settings, and per-chat deletion.

The application searches the corpus in `backend/data2/`. It does not provide an end-user document upload or document-library management screen.

## Architecture

```text
Browser (React + Vite)
        │  /api requests; Vite proxies to Flask during local development
        ▼
Flask API (backend/app.py) ───────────────► SQLite
        │                                    users, profile photos,
        │                                    conversations, messages
        ├── Question ─► Sentence Transformers ─► Pinecone vector search
        │                                      │
        │◄──────── retrieved passages, source files and page metadata
        │
        └── grounded prompt + passages ───────► Ollama (qwen2.5:7b)
                    ◄────────────────────────── generated answer
        │
        └── answer and verified citations ────► Browser

backend/data2/ ─► parser/chunker ─► MiniLM embeddings ─► Pinecone
                   (backend/ingest.py)
```

- **Frontend:** React 18 and Vite; presents the sign-in screen, chat workspace, saved chat history, profile controls, and errors.
- **API and persistence:** Flask serves the API. SQLite stores user accounts, profile photos, conversations, and messages in `backend/app.db`.
- **Ingestion and retrieval:** `backend/legal_rag.py` parses supported legal source files, identifies provision headings generically, creates chunks, and retrieves relevant passages. It does not hard-code provision numbers for particular Acts.
- **Vector search:** Pinecone stores 384-dimensional vectors from `sentence-transformers/all-MiniLM-L6-v2`, along with source, page, and provision metadata.
- **Answer generation:** Flask sends retrieved excerpts to the configured Ollama model. Answers are instructed to stay within the retrieved evidence and use citations checked against retrieval metadata.
- **Corpus:** The included PDFs and text documents are under `backend/data2/`. Ingestion upserts vectors to Pinecone; it does not delete existing vectors.

## Requirements

- Windows 10/11, PowerShell, and VS Code.
- Python 3.11.
- Node.js 20 or newer and npm.
- [Ollama for Windows](https://ollama.com/download).
- A Pinecone account and API key.
- Internet access for package installation and the first download of the embedding model.
- Enough memory for Ollama to run Qwen2.5 7B (about 5 GB for the model, plus runtime overhead).

## Download and open in VS Code

Open VS Code, choose **Terminal → New Terminal**, and run these commands. Replace the example parent directory with the location where you want the project cloned:

```powershell
Set-Location "D:\Projects"
git clone https://github.com/AbdulAzeez1312005/Ai-Based-Legal-Reference-and-Case-Retrieval-System.git
Set-Location ".\Ai-Based-Legal-Reference-and-Case-Retrieval-System\Milestone-4"
code .
```

Alternatively, use the repository's **Code → Download ZIP** option on GitHub, extract it, open the extracted `Milestone-4` folder in VS Code, and continue with the setup below.

## Configure Pinecone and the application

From the `Milestone-4` directory in the VS Code terminal:

```powershell
Copy-Item backend\.env.example backend\.env
code backend\.env
```

Set `PINECONE_API_KEY` to your own Pinecone key in `backend/.env`. Keep this file private; it is ignored by Git. Do not paste the key into source files or commit it.

The main environment variables are:

| Variable | Purpose | Default/example |
| --- | --- | --- |
| `PINECONE_API_KEY` | Private Pinecone API credential | Set this yourself |
| `PINECONE_INDEX` | Pinecone index used by the application | `legal-reference-m4` |
| `PINECONE_NAMESPACE` | Pinecone namespace | Empty (default namespace) |
| `OLLAMA_BASE_URL` | Ollama API URL reachable from Flask | `http://localhost:11434` |
| `OLLAMA_MODEL` | Model used to generate answers | `qwen2.5:7b` |
| `FRONTEND_ORIGIN` | Allowed frontend origin | `http://localhost:5173` |
| `PORT` | Flask API port | `5000` |

The template also includes optional Flask, Pinecone region/cloud, embedding, retrieval, and Ollama tuning settings. The ingestion script creates the configured Pinecone index if it does not exist, using 384 dimensions and cosine similarity. An existing index must have the expected dimension.

## Install Python and JavaScript dependencies

In the VS Code terminal with the working directory set to `Milestone-4`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Install the frontend dependencies:

```powershell
Set-Location frontend
npm install
Set-Location ..
```

## Install Ollama and its model

Install Ollama for Windows, then in a VS Code terminal run:

```powershell
ollama pull qwen2.5:7b
```

Ollama commonly starts as a background service after installation. If the local API is not already running, start it in its own terminal:

```powershell
ollama serve
```

Leave that terminal open. The app expects the Ollama API at `http://localhost:11434` unless `OLLAMA_BASE_URL` is changed in `backend/.env`.

## Ingest the legal corpus

Run ingestion from the `Milestone-4` directory. First, optionally parse the corpus without contacting or changing Pinecone:

```powershell
.\.venv\Scripts\python.exe backend\ingest.py --dry-run
```

To create/populate the configured index, run:

```powershell
.\.venv\Scripts\python.exe backend\ingest.py
```

Review the source and chunk report, then enter `y` at the confirmation prompt to upsert. PDF parsing and embedding can take a while. Re-running ingestion upserts deterministic record IDs; it does not delete Pinecone vectors. Do not use a different Pinecone index from the one configured for the app.

## Run the app from VS Code terminals

Keep the Ollama, Flask, and frontend terminals open while using the app.

**Terminal 1 — Ollama:** run `ollama serve` if Ollama is not already running. Keep this terminal open.

**Terminal 2 — Flask API** (from the `Milestone-4` directory):

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

The API runs at `http://127.0.0.1:5000` by default. Check its health endpoint at `http://127.0.0.1:5000/api/health`.

**Terminal 3 — React/Vite frontend:**

```powershell
Set-Location ".\frontend"
npm run dev
```

Open the URL printed by Vite, normally `http://127.0.0.1:5173`. Create an account or sign in, then ask a question in a new research chat. Use the profile control at the bottom of the sidebar to change your display name or photo. Use the trash icon beside a conversation to delete it after confirmation.

To stop a foreground server, focus its terminal and press `Ctrl+C`. If the frontend or API ports are changed, update the Vite proxy and the relevant `.env` settings consistently.

## Tests and production build

Run offline backend tests from the `Milestone-4` directory:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s backend -p "test_*.py"
```

Run live Pinecone retrieval checks after ingestion:

```powershell
.\.venv\Scripts\python.exe backend\test_rag_pipeline.py --live
```

The live retrieval checks need a working Pinecone key and an ingested index. Live answer generation through the UI also needs Ollama and the selected model available.

Build the frontend for production:

```powershell
Set-Location frontend
npm run build
```

## Deployment notes

The Flask server must be able to reach both Pinecone and an Ollama service. For deployment, provide Ollama on a reachable self-hosted VM or container with sufficient RAM/GPU memory; the browser must not connect directly to Ollama or receive Pinecone credentials. Serverless hosts that cannot run or reach Ollama will not work with this setup. Set a strong `FLASK_SECRET_KEY`, configure `FRONTEND_ORIGIN` for the deployed frontend, and keep `.env` and user data out of source control.
