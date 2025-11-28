# Multi-Source Chatbot with Docker Deployment

This system is a Retrieval-Augmented Generation (RAG) chatbot that processes multi-source documents (PDF, URL, JSON) and exposes a RESTful API deployed via Docker. It returns well-formatted answers with citations, persists its indexes across restarts, and emits application-level logs for observability.

Tech Stack: Python, FastAPI/Uvicorn, LangGraph, LangChain Core, Sentence-Transformers, scikit-learn (TF‑IDF), ChromaDB, Docker, Groq

## Setup and Prerequisites 🐳

- Prerequisites:
  - Git: https://git-scm.com/downloads
  - Docker Desktop: https://www.docker.com/products/docker-desktop
    - Linux users: install Docker Engine and the Compose plugin via your distribution
- Cloning the Repository:

```bash
git clone YOUR_GITHUB_REPO_URL
cd YOUR_PROJECT_FOLDER
```

- Configuration (.env):
  - Create a `.env` file in the project root. Do not commit this file to Git.
  - Required and optional variables:

```env
GROQ_API_KEY="<YOUR_GROQ_API_KEY>"
API_BASE_URL="http://127.0.0.1:5000"
USE_DENSE="1"                # enable sentence-transformer embeddings (1/0)
EMBED_MODEL="all-MiniLM-L6-v2"  # sentence-transformer model name
MAX_UPLOAD_BYTES="20000000"   # max upload size in bytes
MAX_PDF_PAGES="200"          # PDF page processing limit
TESSERACT_CMD=""             # path to Tesseract (optional)
TAVILY_API_KEY=""            # optional external web search key
SERPER_API_KEY=""            # optional external web search key
```

## Docker Deployment and Run Instructions 🚀

- Build the image (optional but recommended):

```bash
docker compose build
```

- Start the system:

```bash
docker compose up -d
```

- The API will be accessible at `http://127.0.0.1:5000`.

- Shut down:

```bash
docker compose down
```

This stops and removes containers but persists data via the Compose volume.

- View logs (application-level logging for queries and responses):

```bash
docker compose logs -f chatbot
```

## API Endpoints and Usage Examples 🔗

Base URL: `http://127.0.0.1:5000`

- `/health` (GET) — readiness and internal component status

```bash
curl http://127.0.0.1:5000/health
```

- `/ingest/doc` (POST) — upload a document (PDF/TXT/MD/DOCX) for ingestion

```bash
curl -X POST \
  -F "file=@./path/to/sample.pdf;type=application/pdf" \
  -F "name=sample.pdf" \
  http://127.0.0.1:5000/ingest/doc
```

Response:

```json
{"doc_id":"<uuid>","chunks":<count>}
```

- `/ingest/url` (POST) — ingest content from a public URL

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/data"}' \
  http://127.0.0.1:5000/ingest/url
```

Response:

```json
{"page_id":"<uuid>","chunks":<count>}
```

- `/query` (POST) — send a user question and get a RAG response with citations

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}' \
  http://127.0.0.1:5000/query
```

Response (example shape):

```json
{
  "answer": "<well-formatted answer>\n\nReferences:\n* Source: ...",
  "citations": [ {"type":"document","id":"...","name":"...","url":null,"chunk_id":"..."} ],
  "origin": "local" | "web" | "general"
}
```

## System Architecture and Design Notes 💡

- Persistence: Achieved via the Docker Compose volume `./data:/app/data`, which stores ingested files, chunked text, and vector indexes; this ensures data survives container restarts.
- Logging: Implemented with a FastAPI `BaseHTTPMiddleware` plus structured `/query` endpoint logging (user query, answer snippet, citations count, origin) emitted to `stdout` at INFO level.
- Fallback Logic: If `TAVILY_API_KEY` or `SERPER_API_KEY` is unset, the system gracefully falls back to Wikipedia-based web context; when OCR is unavailable, primary and fallback PDF extractors are used.

## Frontend (Optional but Recommended)

- Run the Streamlit frontend against the Dockerized backend:

```bash
streamlit run streamlit_frontend.py
```

The frontend reads `API_BASE_URL` from `.env` and targets the backend at `http://127.0.0.1:5000` by default.
