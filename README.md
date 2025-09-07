# RAG ChatBot — Streamlit + LangChain + AstraDB

Chat with your own PDF/TXT knowledge base using a hybrid retriever (Multi-Query + BM25), chat memory, multi-session UI (new chat, auto-title, reset, delete), and an optional best-effort answering mode when your documents don’t fully cover a question.

[Optional screenshot placeholder: docs/screenshot_ui.png]

-------------------------------------------------------------------------------

## Table of Contents
- Highlights
- Architecture
- Prerequisites
- Quickstart
- Configuration
- How it Works
- Project Structure
- Deployment
- Troubleshooting
- FAQ
- Performance & Cost Tips
- Roadmap

-------------------------------------------------------------------------------

## Highlights
- Bring your own knowledge: Upload one or more PDFs/TXTs. The app chunks, embeds, and upserts to AstraDB (serverless vector store).
- Hybrid retrieval:
  • Multi-Query Retriever (MQR) expands your question into diverse semantic queries.
  • BM25 (lexical) catches exact strings (names, domains, typos).
  • Ensemble scoring balances both for robust recall.
- Chat memory & multi-session: Per-chat history persisted in SQLite with UI for new chat, auto-title after first turn, reset memory, and delete chat.
- Grounded or best-effort: Toggle between grounded-only answers (strictly from your files) and best-effort mode (clearly labeled, no fake citations).
- Modern LangChain patterns: LCEL pipeline, RunnableWithMessageHistory, templated prompts, transparent sources panel.

-------------------------------------------------------------------------------

## Architecture

User uploads PDFs/TXTs
 → Chunk & metadata
 → Embeddings (text-embedding-3-large)
 → AstraDB Vector Store

User question
 → Standalone question rewriter (gpt-4o-mini)
 → Hybrid Retriever:
      - MQR → vector search (AstraDB)
      - BM25 → lexical search (in-memory over current docs)
 → Top-k relevant chunks
 → Prompt to gpt-4o
 → Answer + Sources
 → SQLite (messages, sessions)

Key components:
- Answer LLM: gpt-4o
- Rewrite/title LLM: gpt-4o-mini
- Embeddings: text-embedding-3-large
- Vector store: AstraDB (DataStax)
- Lexical: BM25
- UI/runtime: Streamlit

-------------------------------------------------------------------------------

## Prerequisites
- Python 3.10+
- OpenAI API key
- AstraDB serverless DB (Application Token, API Endpoint, Keyspace)

-------------------------------------------------------------------------------

## Quickstart

1) Create and activate a virtual env
   Windows PowerShell:
       python -m venv .venv
       . .venv/Scripts/Activate.ps1
   macOS/Linux:
       python -m venv .venv
       source .venv/bin/activate

2) Install dependencies
       pip install -r requirements.txt

3) Configure environment
       copy .env.example to .env
       edit .env with your keys/endpoints

4) Run the app
       streamlit run app.py

Open the printed local URL. In the sidebar:
1) Paste credentials (or load from .env),
2) Upload files,
3) Click “Connect / Initialize”,
4) Click “Index uploaded files to Astra”,
5) Start chatting.

-------------------------------------------------------------------------------

## Configuration

Create a .env file (or use the sidebar inputs):

OPENAI_API_KEY=sk-xxxx
ASTRA_DB_APPLICATION_TOKEN=token-xxxx
ASTRA_DB_API_ENDPOINT=https://xxxx.apps.astra.datastax.com
ASTRA_DB_KEYSPACE=langchain_db
ASTRA_DB_COLLECTION_NAME=pdf_query

Note: .env is git-ignored. Never commit secrets.

-------------------------------------------------------------------------------

## How it Works

1) Upload & Index
   - Files are split with RecursiveCharacterTextSplitter (configurable chunk size/overlap).
   - Chunks are embedded via text-embedding-3-large and upserted to AstraDB with stable IDs.
   - BM25 is built in memory for lexical matching.

2) Question Understanding
   - The user’s message is rewritten into a standalone question using chat history to resolve pronouns/typos/domains (gpt-4o-mini).

3) Hybrid Retrieval
   - MQR generates multiple semantic queries for vector search in AstraDB.
   - BM25 searches the same corpus lexically.
   - EnsembleRetriever merges both (default weights 0.7/0.3).

4) Answering
   - Retrieved excerpts + chat history are injected into a prompt for gpt-4o.
   - Grounded mode: answer strictly from excerpts or say “not found”.
   - Best-effort mode: if retrieved chunk count < threshold, the model can answer from general knowledge with a clear disclaimer (never fabricates citations).

5) Persistence & Sessions
   - Messages and sessions are stored in SQLite (rag_chat.db).
   - After the first full turn, the app auto-generates a concise chat title.

-------------------------------------------------------------------------------

## Project Structure

.
├─ app.py                   # Streamlit app (UI + RAG pipeline)
├─ requirements.txt
├─ .env.example             # Template for environment variables
├─ .gitignore
├─ rag_chat.db              # (runtime) SQLite for sessions/messages
└─ docs/
   └─ screenshot_ui.png     # (optional) UI screenshot

-------------------------------------------------------------------------------

## Deployment

Streamlit Community Cloud
- Connect this GitHub repo.
- Add secrets (env vars) from .env to the app settings.
- Set the entry point to app.py.

Docker (optional)
- Example Dockerfile:

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

-------------------------------------------------------------------------------

## Troubleshooting

- AstraDB connection issues:
  • Ensure endpoint has no trailing slash.
  • Verify token permissions and keyspace spelling.

- OpenAI auth errors:
  • Check OPENAI_API_KEY and quota/rate limits.

- “No relevant excerpts found”:
  • Increase chunk size/overlap.
  • Confirm the phrase actually exists (use Smoke checks).
  • Try more/less specific phrasing.

- Memory not updating:
  • Ensure rag_chat.db is writable.
  • Use “Reset chat memory”.

- Large/complex PDFs:
  • Consider chunk_size ~800–1200 and overlap ~150–250.
  • Pre-OCR/clean PDFs if extraction is noisy.

-------------------------------------------------------------------------------

## FAQ

Q: Can it answer outside of the uploaded files?
A: Yes—enable “best-effort answers” in the sidebar. The model will explicitly disclaim when the info wasn’t found and will not fabricate citations.

Q: Why hybrid retrieval?
A: Vector search captures semantics; BM25 catches exact strings (names, emails, domains, typos). Ensemble improves recall.

Q: Where are chats stored?
A: Locally in SQLite (rag_chat.db). You can migrate to Postgres later.

Q: Multiple files?
A: Yes—upload many PDFs/TXTs; all are indexed together.

-------------------------------------------------------------------------------

## Performance & Cost Tips

- Keep gpt-4o-mini for rewriting/titles (already configured) to reduce cost.
- Balance chunk size: too small = fragmentation; too large = token bloat.
- Adjust retriever k / weights and consider MMR for different recall/precision tradeoffs.
- Cache embeddings for static corpora to avoid re-embedding.

-------------------------------------------------------------------------------

## Roadmap

- ✅ Multi-session UI (new, select, auto-title, reset, delete)
- ✅ Best-effort answering with explicit disclaimer
- ⏩ Per-chat corpora & file-scoped collections
- ⏩ Export transcripts (HTML/Markdown)
- ⏩ Retrieval analytics (hit rate, avg chunks, latency)
- ⏩ Optional figure/table extraction from PDFs

-------------------------------------------------------------------------------

MIT — see LICENSE.

