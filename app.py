# app.py
# =========================================
# Streamlit RAG app (Pinecone + OpenAI + LangChain)
# - Upload PDF/TXT knowledge base
# - Auto-index on first question (no buttons)
# - Hybrid retrieval: MultiQueryRetriever (vector) + BM25 (local)
# - Multi-session chat with SQLite persistence
# - Optional "best-effort" mode when files don't cover the answer
# =========================================

import os
import tempfile
import hashlib
import uuid
from datetime import datetime
from operator import itemgetter
from typing import List, Optional
from openai import OpenAI
import re
import json

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ---- LangChain core / OpenAI ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ---- Loaders & splitters ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document

# ---- Hybrid retrieval extras ----
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ---- Pinecone vector store ----
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# --- SQLite persistence (sessions + messages) ---
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from langchain.retrievers.document_compressors import EmbeddingsFilter

# ======================
# Config / Secrets
# ======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdfquery")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    # we'll show a UI error later; don't crash import-time
    pass

# ======================
# DB (SQLite)
# ======================
engine = create_engine("sqlite:///rag_chat.db", connect_args={"check_same_thread": False})
Base = declarative_base()
DB = sessionmaker(bind=engine)

class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True)
    role = Column(String)  # "user" / "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

def db_get_or_create_session(sid: str, name: str):
    db = DB()
    obj = db.get(ChatSession, sid)
    if not obj:
        db.add(ChatSession(id=sid, name=name))
        db.commit()
    db.close()

def db_load_history(sid: str) -> InMemoryChatMessageHistory:
    db = DB()
    rows = db.query(Message).filter_by(session_id=sid).order_by(Message.id).all()
    hist = InMemoryChatMessageHistory()
    for r in rows:
        hist.add_user_message(r.content) if r.role == "user" else hist.add_ai_message(r.content)
    db.close()
    return hist

def db_save_message(sid: str, role: str, content: str):
    db = DB()
    db.add(Message(session_id=sid, role=role, content=content))
    db.commit()
    db.close()

def db_delete_session(sid: str):
    db = DB()
    db.query(Message).filter_by(session_id=sid).delete()
    db.query(ChatSession).filter_by(id=sid).delete()
    db.commit()
    db.close()

def db_list_sessions():
    db = DB()
    rows = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
    db.close()
    return rows

def db_clear_messages(sid: str):
    db = DB()
    db.query(Message).filter_by(session_id=sid).delete()
    db.commit()
    db.close()

def db_update_session_name(sid: str, new_name: str):
    db = DB()
    obj = db.get(ChatSession, sid)
    if obj:
        obj.name = new_name
        db.commit()
    db.close()

# ======================
# Language (sticky override)
# ======================
def detect_lang(text: str) -> str:
    if re.search(r'[\u0590-\u05FF]', text or ""):
        return "he"
    return "en"

def parse_explicit_lang_request(text: str) -> Optional[str]:
    t = text or ""
    # explicit, sticky
    if re.search(r"(?:answer|respond)\s+in\s+english|(?:ענה|תענה)\s+באנגלית|באנגלית\s+בלבד", t, flags=re.IGNORECASE):
        return "en"
    if re.search(r"(?:answer|respond)\s+in\s+hebrew|(?:ענה|תענה)\s+בעברית|בעברית\s+בלבד", t, flags=re.IGNORECASE):
        return "he"
    return None

def lang_instruction(lang: str) -> str:
    return "ענה בעברית." if lang == "he" else "Answer in English."

def update_lang_pref(session_id: str, user_text: str) -> str:
    """
    Rule:
    - Default: reply in the language of the user's last message
    - If user explicitly requests a language => sticky until changed by explicit request
    """
    st.session_state.setdefault("lang_prefs", {})  # {sid: {"lang": "...", "sticky": bool}}
    pref = st.session_state["lang_prefs"].get(session_id, {"lang": "he", "sticky": False})

    explicit = parse_explicit_lang_request(user_text)
    if explicit:
        pref = {"lang": explicit, "sticky": True}
    elif not pref.get("sticky", False):
        pref["lang"] = detect_lang(user_text)

    st.session_state["lang_prefs"][session_id] = pref
    return pref["lang"]

# ======================
# Session State
# ======================
def init_session_state():
    st.session_state.setdefault("connected", False)
    st.session_state.setdefault("vstore", None)
    st.session_state.setdefault("bm25", None)
    st.session_state.setdefault("docs", [])

    # ✅ קריטי: חסר אצלך
    st.session_state.setdefault("history_store", {})  # {session_id: InMemoryChatMessageHistory}

    st.session_state.setdefault("session_index", {s.id: {"name": s.name} for s in db_list_sessions()})
    st.session_state.setdefault("current_session", next(iter(st.session_state.session_index.keys()), None))
    st.session_state.setdefault("chain", None)

    # answering mode (kept as defaults; not user-facing)
    st.session_state.setdefault("allow_guess", True)   # ✅ תמיד דלוק
    st.session_state.setdefault("auto_web", True)      # ✅ תמיד דלוק (allow web)
    st.session_state.setdefault("force_web", False)    # ✅ תמיד כבוי
    st.session_state.setdefault("show_debug", True)    # ✅ תמיד דלוק

    # indexing guards
    st.session_state.setdefault("indexed_fp", None)
    st.session_state.setdefault("index_namespace", None)

    # retriever pieces
    st.session_state.setdefault("base_retriever", None)
    st.session_state.setdefault("compressor", None)

init_session_state()

# ======================
# Chat/session helpers
# ======================
def create_chat(name: str = "Untitled") -> str:
    sid = uuid.uuid4().hex[:8]
    db_get_or_create_session(sid, name)
    st.session_state.session_index[sid] = {"name": name}
    st.session_state.history_store[sid] = InMemoryChatMessageHistory()
    st.session_state.current_session = sid
    return sid

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    hist = st.session_state.history_store.get(session_id)
    if hist is not None:
        return hist
    hist = db_load_history(session_id)
    st.session_state.history_store[session_id] = hist
    return hist

# ======================
# Connection (no UI keys)
# ======================
def ensure_connected() -> bool:
    try:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        pinecone_key = os.getenv("PINECONE_API_KEY", "")
        index_name = os.getenv("PINECONE_INDEX_NAME", "pdfquery")

        if not all([openai_key, pinecone_key, index_name]):
            st.session_state.connected = False
            return False

        os.environ["OPENAI_API_KEY"] = openai_key

        st.session_state.qa_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        st.session_state.rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=openai_key,
        )

        # ✅ חשוב: נשמר לשימוש ב-EmbeddingsFilter
        st.session_state.embedding = embedding

        pc = Pinecone(api_key=pinecone_key)
        index = pc.Index(index_name)

        st.session_state.vstore = PineconeVectorStore(index=index, embedding=embedding)
        st.session_state.connected = True
        return True

    except Exception as e:
        st.session_state.connected = False
        st.error(f"Connection failed: {e}")
        return False

def ensure_indexed(uploaded_files, chunk_size: int, chunk_overlap: int):
    """
    Auto-index only when needed:
    - new file set -> index
    - same file set -> no-op
    """
    if not uploaded_files:
        st.warning("Upload at least one PDF/TXT file.")
        st.stop()

    fp = files_fingerprint(uploaded_files)
    namespace = f"kb-{fp}"

    # already indexed this exact file set
    if st.session_state.get("indexed_fp") == fp and st.session_state.get("chain") is not None:
        return

    with st.spinner("Indexing uploaded files (auto)…"):
        new_docs = load_uploaded_files_to_docs(uploaded_files, chunk_size, chunk_overlap)
        if not new_docs:
            st.warning("No documents loaded.")
            st.stop()

        st.session_state.docs = new_docs

        ids = []
        for i, d in enumerate(new_docs):
            fn = d.metadata.get("filename", "doc")
            cid = d.metadata.get("chunk_id", i)
            ids.append(f"{fp}:{fn}:{cid}")

        st.session_state.vstore.add_documents(new_docs, ids=ids, namespace=namespace)

        st.session_state.indexed_fp = fp
        st.session_state.index_namespace = namespace

        build_hybrid_retriever(st.session_state.docs)
        build_chain()

# ======================
# Files -> Documents
# ======================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
VEC_K = 12
VEC_FETCH_K = 40

def load_uploaded_files_to_docs(uploaded_files, csize: int, coverlap: int) -> List[Document]:
    all_docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=csize, chunk_overlap=coverlap)

    for uf in uploaded_files or []:
        fname = uf.name
        data = uf.getvalue()  # safe to call multiple times

        if fname.lower().endswith(".txt"):
            text = data.decode("utf-8", errors="ignore")
            base_doc = Document(page_content=text, metadata={"filename": fname, "source": "upload"})
            chunks = splitter.split_documents([base_doc])
            for i, d in enumerate(chunks):
                d.metadata.update({"chunk_id": i})
            all_docs.extend(chunks)

        elif fname.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                pdf_pages = loader.load()
            finally:
                # ✅ תמיד מנקים את הקובץ הזמני
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            chunks = splitter.split_documents(pdf_pages)
            for i, d in enumerate(chunks):
                d.metadata.update({"filename": fname, "source": "upload", "chunk_id": i})
            all_docs.extend(chunks)

    return all_docs

def files_fingerprint(uploaded_files) -> str:
    """Stable fingerprint for the current set of uploaded files."""
    h = hashlib.sha256()
    for uf in uploaded_files or []:
        data = uf.getvalue()
        h.update(uf.name.encode("utf-8"))
        h.update(str(len(data)).encode("utf-8"))
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()[:16]

# ======================
# Retrieval: Hybrid (Vector MQR + BM25)
# ======================
def build_hybrid_retriever(docs: List[Document]):
    vstore = st.session_state.vstore
    namespace = st.session_state.get("index_namespace")

    search_kwargs = {
        "k": 12,
        "fetch_k": 80,
        "lambda_mult": 0.5,
    }
    if namespace:
        search_kwargs["namespace"] = namespace

    vec = vstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 10

    base = EnsembleRetriever(retrievers=[vec, bm25], weights=[0.8, 0.2])

    # שמור גם את ה-base וגם את הקומפרסור כדי שלא "ייאבד" כיסוי בגלל סינון אגרסיבי
    st.session_state.base_retriever = base
    st.session_state.compressor = EmbeddingsFilter(
        embeddings=st.session_state.embedding,
        similarity_threshold=0.60,
    )
    return base

def openai_web_search_answer(question: str, lang: str) -> tuple[str, list[str]]:
    """
    Uses OpenAI Responses API web_search tool to answer with sources.
    Ensures answer language follows the user's language preference.
    """
    try:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            return "(Web search failed: missing OPENAI_API_KEY)", []

        client = OpenAI(api_key=openai_key)

        model = os.getenv("OPENAI_WEB_MODEL", "gpt-4o-mini")

        # Keep the web answer in the desired language
        web_input = (
            f"{lang_instruction(lang)}\n"
            "If you use web information, keep it concise and factual.\n"
            "Do NOT omit sources.\n\n"
            f"{question}"
        )

        resp = client.responses.create(
            model=model,
            tools=[{"type": "web_search"}],
            input=web_input,
            include=["web_search_call.action.sources"],
        )

        text = getattr(resp, "output_text", None) or ""
        sources: list[str] = []

        # best-effort extraction
        try:
            data = resp.model_dump()
            for item in data.get("output", []):
                if item.get("type") == "web_search_call":
                    action = item.get("action", {}) or {}
                    for s in action.get("sources", []) or []:
                        url = s.get("url")
                        if url:
                            sources.append(url)
        except Exception:
            pass

        return (text.strip() or "(No web answer returned.)"), sources

    except Exception as e:
        return f"(Web search failed: {e})", []

def user_requested_web(q: str) -> bool:
    if not q:
        return False
    return bool(re.search(
        r"(?:\bweb\b|\bgoogle\b|אינטרנט|בדוק באינטרנט|חפש|חיפוש|תבדוק|check online|search online|requirements)",
        q,
        flags=re.IGNORECASE
    ))

def decide_min_evidence(question: str, lang: str) -> int:
    """
    Decided by GPT (per user requirement).
    Returns an integer threshold for "covered" before using RAG-only answers.
    """
    try:
        llm = st.session_state.rewriter_llm
        prompt = (
            "Decide how many retrieved chunks are minimally needed to consider the KB 'covered'.\n"
            "Return ONLY a JSON object like: {\"min_evidence\": 2}\n"
            "Allowed min_evidence values: 1, 2, 3, 4, 6\n"
            "Rules:\n"
            "- Very narrow factual question => 1\n"
            "- Typical question => 2\n"
            "- Multi-part / compare / analysis => 3\n"
            "- Broad research / policy / long explanation => 4 or 6\n"
            f"User language: {lang}\n"
            f"Question: {question}"
        )
        raw = llm.invoke(prompt)
        txt = getattr(raw, "content", str(raw)).strip()
        data = json.loads(txt)
        v = int(data.get("min_evidence", 2))
        return v if v in (1, 2, 3, 4, 6) else 2
    except Exception:
        # safe fallback (minimal change): heuristic
        q = (question or "").strip()
        if len(q) > 220 or re.search(r"(השווה|השווא|compare|pros|cons|יתרונות|חסרונות|מיפוי|תוכנית|אסטרטגיה|policy)", q, re.IGNORECASE):
            return 3
        return 2

def build_chain():
    qa_llm = st.session_state.qa_llm

    base_retriever = st.session_state.get("base_retriever")
    compressor = st.session_state.get("compressor")

    if base_retriever is None or compressor is None:
        return None

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's message into a fully self-contained question. "
         "Return only the rewritten question. "
         "Keep the same language as the user's message."),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ])

    standalone_question = (
        {"question": itemgetter("question"), "history": itemgetter("history")}
        | contextualize_prompt
        | st.session_state.rewriter_llm
        | StrOutputParser()
    )

    def format_docs(docs_):
        if not docs_:
            return "(no relevant excerpts found)"
        out = []
        for i, d in enumerate(docs_):
            fn = d.metadata.get("filename", "source")
            cid = d.metadata.get("chunk_id", "?")
            out.append(f"[{i+1}] ({fn}#{cid})\n{d.page_content}")
        return "\n\n".join(out)

    # 1) Pull docs BEFORE compression (so doc_count reflects real recall)
    qa_inputs = (
        RunnablePassthrough
        .assign(
            history=itemgetter("history"),
            question=itemgetter("question"),
            lang=itemgetter("lang"),
            rewritten=standalone_question
        )
        .assign(raw_docs=itemgetter("rewritten") | base_retriever | RunnableLambda(lambda d: d or []))
        .assign(doc_count=RunnableLambda(lambda x: len(x["raw_docs"])))
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "{lang_instr}\n"
         "You are a rigorous analyst.\n"
         "Use ONLY the provided excerpts as evidence.\n"
         "Never invent facts not supported by the excerpts.\n"
         "Always separate: (A) Evidence vs (B) Inference.\n\n"
         "If the user asks for an evaluation / recommendation / comparison / conclusion that is not explicitly stated, you MUST:\n"
         "1) Evidence\n2) Inference\n3) Gaps/Unknowns\n4) Conclusion/Recommendation\n5) Confidence: High/Medium/Low\n"
        ),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\n\nRelevant excerpts:\n{context}")
    ])

    general_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "{lang_instr}\n"
         "You are a helpful assistant.\n"
         "The uploaded files did not provide enough evidence.\n"
         "Answer from general knowledge as best-effort.\n\n"
         "IMPORTANT: Append exactly ONE final line (in English token, do not translate):\n"
         "NEEDS_WEB: YES\n"
         "or\n"
         "NEEDS_WEB: NO\n"
        ),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}")
    ])

    def parse_needs_web(text: str) -> tuple[str, bool]:
        m = re.search(r"\nNEEDS_WEB:\s*(YES|NO)\s*$", text.strip(), flags=re.IGNORECASE)
        if not m:
            return text.strip(), False
        needs = (m.group(1).upper() == "YES")
        cleaned = re.sub(r"\nNEEDS_WEB:\s*(YES|NO)\s*$", "", text.strip(), flags=re.IGNORECASE).strip()
        return cleaned, needs

    def answer_router(x: dict):
        docs = x.get("raw_docs") or []
        doc_count = int(x.get("doc_count", 0))

        force_web = bool(st.session_state.get("force_web", False))  # default False
        auto_web = bool(st.session_state.get("auto_web", True))     # default True

        # language
        lang = x.get("lang", "he")
        lang_instr = lang_instruction(lang)

        # ✅ decide threshold via GPT (not user-facing)
        min_ev = decide_min_evidence(x.get("rewritten", "") or x.get("question", ""), lang)

        # ✅ detect explicit user intent to use the internet
        raw_user_q = x.get("question", "")  # original user input (not rewritten)
        explicit_web = user_requested_web(raw_user_q)

        # ✅ If user explicitly asked for web -> go web (even if docs exist)
        if explicit_web and auto_web:
            web_text, web_sources = openai_web_search_answer(x["rewritten"], lang)
            return {
                "answer": web_text,
                "sources": [],
                "web_sources": web_sources,
                "mode": "web",
                "rewritten": x["rewritten"],
                "doc_count": doc_count,
                "explicit_web": True,
                "min_evidence_used": min_ev,
                "lang": lang,
            }

        # A) Covered by docs => RAG
        if doc_count >= min_ev and doc_count > 0:
            try:
                compressed_docs = compressor.compress_documents(docs, x["rewritten"]) or docs
            except Exception:
                compressed_docs = docs

            context = format_docs(compressed_docs)

            ans = (rag_prompt | qa_llm | StrOutputParser()).invoke({
                "question": x["rewritten"],
                "context": context,
                "history": x["history"],
                "lang_instr": lang_instr,
            })

            return {
                "answer": ans,
                "sources": compressed_docs,
                "web_sources": [],
                "mode": "rag",
                "rewritten": x["rewritten"],
                "doc_count": doc_count,
                "explicit_web": False,
                "min_evidence_used": min_ev,
                "lang": lang,
            }

        # B) Not covered => General best-effort
        gen = (general_prompt | qa_llm | StrOutputParser()).invoke({
            "question": x["rewritten"],
            "history": x["history"],
            "lang_instr": lang_instr,
        })
        gen_clean, needs_web = parse_needs_web(gen)

        # If force_web is on -> always web when not covered
        if not force_web and (not needs_web or not auto_web):
            return {
                "answer": gen_clean,
                "sources": [],
                "web_sources": [],
                "mode": "general",
                "rewritten": x["rewritten"],
                "doc_count": doc_count,
                "explicit_web": False,
                "min_evidence_used": min_ev,
                "lang": lang,
            }

        # C) Web search
        web_text, web_sources = openai_web_search_answer(x["rewritten"], lang)
        final = (
            f"{gen_clean}\n\n---\n\n{web_text}"
        )
        return {
            "answer": final,
            "sources": [],
            "web_sources": web_sources,
            "mode": "web",
            "rewritten": x["rewritten"],
            "doc_count": doc_count,
            "explicit_web": False,
            "min_evidence_used": min_ev,
            "lang": lang,
        }

    # diagnostics in sidebar (optional) - kept ON by default, not user-facing
    def debug_print_fn(x: dict):
        if st.session_state.get("show_debug", True):
            st.sidebar.caption(
                f"🧭 Standalone: {x.get('rewritten','')}\n\n"
                f"📄 docs(before-compress): {len(x.get('raw_docs') or [])}"
            )
        return x

    chain = RunnableWithMessageHistory(
        qa_inputs
        | RunnableLambda(debug_print_fn)
        | RunnableLambda(answer_router),
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )

    st.session_state.chain = chain
    return chain

# ======================
# Auto title
# ======================
title_prompt = ChatPromptTemplate.from_template(
    "Write a concise 3–8 word title for this chat based on the user's first question and the assistant's first answer. "
    "No punctuation at the end. Be specific.\n\nUser: {q}\nAssistant: {a}"
)

def auto_title_if_needed(session_id: str, question: str, answer_text: str):
    # ensure the session exists in index
    if session_id not in st.session_state.session_index:
        db_get_or_create_session(session_id, "Untitled")
        st.session_state.session_index[session_id] = {"name": "Untitled"}

    name = st.session_state.session_index.get(session_id, {}).get("name", "")
    is_default = (not name) or (name.lower() in ("untitled", "chat")) or name.startswith("Chat ")

    db = DB()
    msg_count = db.query(Message).filter_by(session_id=session_id, role="user").count()
    db.close()

    if is_default and msg_count == 1:
        title = (title_prompt | st.session_state.rewriter_llm | StrOutputParser()).invoke(
            {"q": question, "a": answer_text}
        ).strip().strip('"').strip("'")
        if title:
            db_update_session_name(session_id, title)
            st.session_state.session_index[session_id]["name"] = title
            st.rerun()

# ======================
# UI
# ======================
st.set_page_config(page_title="PDF/TXT RAG Chat (Pinecone)", page_icon="💬", layout="wide")
st.title("💬 Chat Bot")

with st.sidebar:
    st.header("🔐 Credentials")
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        st.error("Missing OPENAI_API_KEY / PINECONE_API_KEY in .env (or Streamlit secrets).")
    else:
        st.success(f"Keys loaded from environment ✅\nIndex: {PINECONE_INDEX_NAME}")

    st.markdown("---")
    st.header("📄 Upload Knowledge Base")
    files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    st.markdown("---")
    st.header("🧠 Answering mode")
    # ✅ user requested these to be defaults, not user-facing toggles
    st.success("✅ Best-effort is always ON (Docs → General → Web)")
    st.caption("Defaults: Web allowed ✅ | Diagnostics ✅ | Force web ❌")
    st.caption("Coverage threshold is decided by GPT per question (not user-controlled).")

# --- Sessions management ---
st.header("💬 Sessions")

if st.button("➕ New chat", use_container_width=True):
    create_chat()
    st.rerun()

rows = db_list_sessions()

if rows:
    labels = [r.name for r in rows]
    ids_by_label = {lab: r.id for lab, r in zip(labels, rows)}
    current_label = next((r.name for r in rows if r.id == st.session_state.current_session), labels[0])
    chosen_label = st.selectbox("Select a chat", labels, index=labels.index(current_label))
    st.session_state.current_session = ids_by_label[chosen_label]

    st.markdown("**Delete a chat**")
    for r in rows:
        col1, col2 = st.columns([0.82, 0.18])
        with col1:
            prefix = "✅ " if r.id == st.session_state.current_session else "  "
            st.write(prefix + r.name)
        with col2:
            if st.button("🗑️", key=f"del_{r.id}", help=f"Delete '{r.name}'"):
                db_delete_session(r.id)
                st.session_state.history_store.pop(r.id, None)
                st.session_state.session_index.pop(r.id, None)
                remaining = db_list_sessions()
                st.session_state.current_session = remaining[0].id if remaining else None
                st.success(f"Deleted '{r.name}'")
                st.rerun()

if st.session_state.current_session and st.button("🧹 Reset chat memory", use_container_width=True):
    sid_ = st.session_state.current_session
    db_clear_messages(sid_)
    st.session_state.history_store[sid_] = InMemoryChatMessageHistory()
    st.success("Chat memory cleared.")
    st.rerun()

# ======================
# Chat
# ======================
if not st.session_state.current_session:
    create_chat()

sid = st.session_state.current_session
hist = get_session_history(sid)

# render history ONCE
for m in hist.messages:
    role = "user" if m.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

can_run = bool(OPENAI_API_KEY) and bool(PINECONE_API_KEY) and bool(files)
question = st.chat_input("Type your question...", disabled=not can_run)

if question:
    # update language preference per session (sticky override supported)
    lang = update_lang_pref(sid, question)

    if not st.session_state.get("connected", False):
        ok = ensure_connected()
        if not ok:
            st.error("Not connected. Check environment variables / API keys.")
            st.stop()

    ensure_indexed(files, CHUNK_SIZE, CHUNK_OVERLAP)

    db_get_or_create_session(sid, st.session_state.session_index.get(sid, {}).get("name", "Chat"))
    db_save_message(sid, "user", question)

    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"session_id": sid}}

    with st.chat_message("assistant"):
        if st.session_state.chain is None:
            st.error("Chain not ready (indexing/retriever not initialized). Try re-uploading files or refresh.")
            st.stop()

        with st.spinner("Thinking..."):
            res = st.session_state.chain.invoke({"question": question, "lang": lang}, config=config)

        mode = res.get("mode", "unknown")
        rewritten = res.get("rewritten", "")
        doc_count = res.get("doc_count", 0)
        web_sources = res.get("web_sources", []) or []
        min_ev_used = res.get("min_evidence_used", None)

        # ✅ Status strip / badge
        cols = st.columns([0.55, 0.45])
        with cols[0]:
            if mode == "rag":
                st.success(f"🟢 RAG mode — used {doc_count} retrieved chunks")
            elif mode == "general":
                st.warning("🟡 Best-effort — general knowledge (no doc coverage)")
            elif mode == "web":
                st.info(f"🟣 Web search — {len(web_sources)} sources")
            else:
                st.caption(f"⚪ Mode: {mode}")

        with cols[1]:
            if mode == "rag":
                st.caption("Order: Docs → Answer")
            elif mode == "general":
                st.caption("Order: Docs → General")
            elif mode == "web":
                st.caption("Order: Docs → General → Web")

        # ✅ Optional diagnostics (default ON; not user-toggle)
        if st.session_state.get("show_debug", True) and rewritten:
            with st.expander("🔎 Diagnostics"):
                st.write("Standalone / rewritten question:")
                st.code(rewritten)
                st.write(f"doc_count(before-compress): {doc_count}")
                if min_ev_used is not None:
                    st.write(f"min_evidence (GPT-decided): {min_ev_used}")
                st.write(f"auto_web: {st.session_state.get('auto_web', True)} | force_web: {st.session_state.get('force_web', False)}")
                st.write(f"lang: {lang}")

        answer_text = res["answer"]
        st.markdown(answer_text)

        # ✅ Show web sources inline when web was used (so user sees SOURCES)
        if mode == "web" and web_sources:
            st.markdown("### Sources")
            for url in web_sources[:10]:
                st.markdown(f"- {url}")

        db_save_message(sid, "assistant", answer_text)
        auto_title_if_needed(sid, question, answer_text)

    # keep the existing sources expander (local + web)
    with st.expander("🔎 Sources"):
        # local sources
        for i, d in enumerate(res.get("sources", []) or [], 1):
            meta = d.metadata
            src = meta.get("filename") or meta.get("source", "source")
            cid = meta.get("chunk_id", "?")
            st.markdown(f"**[{i}]** `{src}#{cid}`\n\n{d.page_content[:400]}…")

        # web sources
        web_sources = res.get("web_sources", []) or []
        if web_sources:
            st.markdown("### Web sources")
            for url in web_sources[:10]:
                st.markdown(f"- {url}")
