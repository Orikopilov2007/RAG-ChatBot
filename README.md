# RAG ChatBot — Streamlit + LangChain + Pinecone

A production-ready **RAG (Retrieval-Augmented Generation) chat application** that lets you chat with your own **PDF/TXT knowledge base**.

The app uses **hybrid retrieval (Vector + BM25)**, **multi-session chat memory**, automatic indexing, **Hebrew & English support**, and smart routing between **document-grounded answers**, **general knowledge**, and **optional web search**.

## Key Features

- 📄 **Bring your own knowledge**  
  Upload PDF/TXT files. Files are chunked, embedded, and indexed automatically.

- 🔎 **Hybrid Retrieval (high accuracy)**  
  - Vector search via **Pinecone** (MMR-based)  
  - **BM25** lexical retrieval for exact matches  
  - Ensemble scoring for better recall & precision  

- 💬 **Multi-session chat with memory**  
  - New chat / delete chat / reset memory  
  - Chat history persisted in **SQLite**  
  - Automatic chat title after first question  

- 🌍 **Hebrew & English support**  
  - Automatic language detection  
  - Explicit language requests supported  
  - Per-session language preference  

- 🧠 **Smart answer routing**  
  - Uses documents when there is enough evidence (RAG)  
  - Falls back to general knowledge if not  
  - Can escalate to web search when explicitly requested  

## Tech Stack

- **UI**: Streamlit  
- **RAG framework**: LangChain (LCEL)  
- **LLMs**:  
  - Answering: `gpt-4o`  
  - Rewriting / logic: `gpt-4o-mini`  
- **Embeddings**: `text-embedding-3-large`  
- **Vector DB**: Pinecone  
- **Lexical search**: BM25  
- **Persistence**: SQLite  

## Prerequisites

- Python 3.10+  
- OpenAI API key  
- Pinecone account with an existing index  

## Quickstart

### 1. Create virtual environment

**Windows (PowerShell)**
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file (do not commit it):

```env
OPENAI_API_KEY=sk-xxxx
PINECONE_API_KEY=pcsk-xxxx
PINECONE_INDEX_NAME=pdfquery
```

### 4. Run the app
```bash
streamlit run app.py
```

Upload PDF/TXT files and start chatting.  
Indexing happens automatically on the first question.

## How It Works (High Level)

### 1. Upload & Index
- Files are split into chunks  
- Chunks are embedded and stored in Pinecone  
- BM25 index is built in memory  

### 2. Question Understanding
- User question is rewritten into a standalone question using chat history  

### 3. Retrieval
- Vector + BM25 retrieval  
- Results are merged and filtered  

### 4. Answering
- If enough document evidence exists → **RAG answer**  
- Otherwise → **best-effort general answer**  
- Optional web search if explicitly requested  

### 5. Persistence
- Sessions and messages are stored in SQLite  
- Chat titles are auto-generated  

## Project Structure

```
.
├─ app.py
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ rag_chat.db   # runtime (SQLite)
```

## Notes

- The Pinecone index must already exist.  
- The index dimension must match `text-embedding-3-large`.  
- Secrets should always be provided via `.env` or environment variables.  
