# app.py
# =========================================
# Streamlit RAG app (AstraDB + OpenAI + LangChain)
# - Upload PDF/TXT knowledge base
# - Index to AstraDB (vector store)
# - Ask questions with chat memory
# - Standalone question rewriting + Hybrid Retrieval (MQR + BM25)
# - Multi-session (New chat / Reset memory / Delete chat / auto title)
# - Best-effort answering mode (optional)
# =========================================

import os
import tempfile
from operator import itemgetter
from typing import List
import uuid
from datetime import datetime

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
    from langchain_core.documents import Document   # LC >= 0.2
except Exception:
    from langchain.schema import Document           # LC < 0.2

# ---- Astra vector store ----
from langchain_astradb import AstraDBVectorStore

# ---- Hybrid retrieval extras ----
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

# --- SQLite persistence (sessions + messages) ---
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

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
    role = Column(String)      # "user" / "assistant"
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# ---------- DB helpers ----------
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

# ---------- session_state init ----------
def init_session_state():
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "vstore" not in st.session_state:
        st.session_state.vstore = None
    if "bm25" not in st.session_state:
        st.session_state.bm25 = None
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}      # {session_id: InMemoryChatMessageHistory}
    if "session_index" not in st.session_state:
        st.session_state.session_index = {s.id: {"name": s.name} for s in db_list_sessions()}
    if "current_session" not in st.session_state:
        st.session_state.current_session = next(iter(st.session_state.session_index.keys()), None)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "hybrid_retriever" not in st.session_state:
        st.session_state.hybrid_retriever = None
    if "allow_guess" not in st.session_state:
        st.session_state.allow_guess = False
    if "min_evidence" not in st.session_state:
        st.session_state.min_evidence = 1
    # last values for rebuild detection
    if "_allow_guess_prev" not in st.session_state:
        st.session_state._allow_guess_prev = st.session_state.allow_guess
    if "_min_evidence_prev" not in st.session_state:
        st.session_state._min_evidence_prev = st.session_state.min_evidence

init_session_state()

# ---------- chat/session helpers ----------
def create_chat(name: str = "Untitled") -> str:
    """Create a new chat session and select it."""
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

# ---------- connect once ----------
def ensure_connected(
    openai_key: str,
    astra_token: str,
    astra_endpoint: str,
    astra_keyspace: str,
    astra_collection: str,
) -> bool:
    try:
        if not all([openai_key, astra_token, astra_endpoint, astra_keyspace, astra_collection]):
            st.error("Missing credentials. Fill all fields.")
            return False

        os.environ["OPENAI_API_KEY"] = openai_key

        st.session_state.qa_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
        st.session_state.rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        embedding = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)

        st.session_state.vstore = AstraDBVectorStore(
            collection_name=astra_collection,
            embedding=embedding,
            token=astra_token,
            api_endpoint=astra_endpoint.rstrip("/"),
            namespace=astra_keyspace,
        )

        st.session_state.connected = True
        st.session_state.chain = None  # built after indexing
        return True
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return False

# ---------- docs load/index ----------
def load_uploaded_files_to_docs(uploaded_files, csize, coverlap) -> List[Document]:
    all_docs: List[Document] = []

    def split(docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=csize, chunk_overlap=coverlap)
        return splitter.split_documents(docs)

    for uf in uploaded_files or []:
        fname = uf.name
        if fname.lower().endswith(".txt"):
            text = uf.read().decode("utf-8", errors="ignore")
            base_doc = Document(page_content=text, metadata={"filename": fname, "source": "upload"})
            chunks = split([base_doc])
            for i, d in enumerate(chunks):
                d.metadata.update({"chunk_id": i})
            all_docs.extend(chunks)
        elif fname.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)
            pdf_pages = loader.load()
            chunks = split(pdf_pages)
            for i, d in enumerate(chunks):
                d.metadata.update({"filename": fname, "source": "upload", "chunk_id": i})
            all_docs.extend(chunks)

    return all_docs

def index_docs_to_astra(docs: List[Document]):
    if not docs:
        return 0
    ids = []
    for i, d in enumerate(docs):
        fn = d.metadata.get("filename", "doc")
        ids.append(f"{fn}-{d.metadata.get('chunk_id', i)}")
    st.session_state.vstore.add_documents(docs, ids=ids)
    return len(docs)

def build_hybrid_retriever(docs: List[Document]):
    vstore = st.session_state.vstore
    rewriter_llm = st.session_state.rewriter_llm

    base_vec_retriever = vstore.as_retriever(search_kwargs={"k": 25})
    mqr_prompt = ChatPromptTemplate.from_template(
        "Rewrite the user's question into 4 diverse search queries to retrieve relevant chunks. "
        "If a company/domain looks misspelled, include a corrected variant. "
        "Return ONLY the 4 queries, each on its own line.\n\nUser question: {question}"
    )
    mqr = MultiQueryRetriever.from_llm(
        llm=rewriter_llm,
        retriever=base_vec_retriever,
        prompt=mqr_prompt,
    )

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 6
    st.session_state.bm25 = bm25

    hybrid = EnsembleRetriever(retrievers=[mqr, bm25], weights=[0.7, 0.3])
    st.session_state.hybrid_retriever = hybrid
    return hybrid

def build_chain():
    qa_llm = st.session_state.qa_llm
    hybrid_retriever = st.session_state.hybrid_retriever

    # 1) rewrite into standalone question
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Rewrite the user's message into a fully self-contained question about the uploaded knowledge. "
         "If the user mentions a company/domain that looks misspelled, include a corrected variant too. "
         "Return only the rewritten question."),
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
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs_))

    qa_inputs = (
        RunnablePassthrough
        .assign(history=itemgetter("history"),
                question=itemgetter("question"),
                rewritten=standalone_question)
        .assign(raw_docs=itemgetter("rewritten") | hybrid_retriever | RunnableLambda(lambda d: d or []))
        .assign(context=itemgetter("raw_docs") | RunnableLambda(format_docs),
                sources=itemgetter("raw_docs"))
        .assign(doc_count=RunnableLambda(lambda x: len(x["sources"])))
    )

    # 2) grounded-only vs best-effort
    if st.session_state.allow_guess:
        mode_instructions = (
            f"Prefer the provided excerpts. If fewer than {st.session_state.min_evidence} relevant chunks "
            f"were retrieved, you MAY answer from your general knowledge. "
            "Start with a brief disclaimer like: 'Not found in the uploaded files — best-effort answer:' "
            "State assumptions explicitly. Do NOT imply the info came from the files. Do NOT fabricate citations."
        )
    else:
        mode_instructions = (
            "Answer strictly and ONLY from the provided excerpts. "
            "If they are insufficient, say clearly that you couldn't find this in the uploaded files and do NOT guess."
        )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers in English based only on the provided excerpts from the uploaded files. "
         "{mode_instructions}"),
        MessagesPlaceholder("history"),
        ("human", "Question: {question}\n\nRelevant excerpts:\n{context}")
    ]).partial(mode_instructions=mode_instructions)

    debug_print = RunnableLambda(lambda x: (st.sidebar.write(
        f"🧭 Standalone: {x['rewritten']}  •  Sources: {x['doc_count']}"
    ) or x))

    chain = RunnableWithMessageHistory(
        qa_inputs | debug_print | {
            "answer": qa_prompt | qa_llm | StrOutputParser(),
            "sources": itemgetter("sources"),
        },
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="answer",
    )
    st.session_state.chain = chain
    return chain

# ---------- auto title ----------
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
st.set_page_config(page_title="PDF/TXT RAG Chat (AstraDB)", page_icon="💬", layout="wide")
st.title("💬 RAG Chat over your PDFs/TXTs (AstraDB + OpenAI)")

with st.sidebar:
    # --- Credentials + Upload ---
    st.header("🔐 Credentials")
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    astra_token = st.text_input("ASTRA_DB_APPLICATION_TOKEN", value=os.getenv("ASTRA_DB_APPLICATION_TOKEN", ""), type="password")
    astra_endpoint = st.text_input("ASTRA_DB_API_ENDPOINT", value=os.getenv("ASTRA_DB_API_ENDPOINT", ""))
    astra_keyspace = st.text_input("ASTRA_DB_KEYSPACE", value=os.getenv("ASTRA_DB_KEYSPACE", "langchain_db"))
    astra_collection = st.text_input("ASTRA_DB_COLLECTION_NAME", value=os.getenv("ASTRA_DB_COLLECTION_NAME", "pdf_query"))

    st.markdown("---")
    st.header("⚙️ Indexing options")
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)

    st.markdown("---")
    st.header("📄 Upload Knowledge Base")
    files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    connect_clicked = st.button(
        "✅ Connect / Initialize",
        use_container_width=True,
        disabled=not (openai_key and astra_token and astra_endpoint and astra_keyspace and astra_collection and files),
    )

    st.markdown("---")
    st.header("🧠 Answering mode")
    allow_guess = st.checkbox(
        "Allow best-effort answers when files don't cover it",
        value=st.session_state.get("allow_guess", False)
    )
    min_evidence = st.slider(
        "Min retrieved chunks before we consider it 'covered'",
        0, 8, st.session_state.get("min_evidence", 1)
    )
    # persist + rebuild chain on change
    changed = (
        st.session_state._allow_guess_prev != allow_guess
        or st.session_state._min_evidence_prev != min_evidence
    )
    st.session_state.allow_guess = allow_guess
    st.session_state.min_evidence = min_evidence
    st.session_state._allow_guess_prev = allow_guess
    st.session_state._min_evidence_prev = min_evidence
    if changed and st.session_state.hybrid_retriever:
        build_chain()

# --- Sessions management ---
st.header("💬 Sessions")

# New chat
if st.button("➕ New chat", use_container_width=True):
    create_chat()
    st.rerun()

rows = db_list_sessions()

if rows:
    # active selector
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

# Reset memory (keep chat row)
if st.session_state.current_session and st.button("🧹 Reset chat memory", use_container_width=True):
    sid = st.session_state.current_session
    db_clear_messages(sid)
    st.session_state.history_store[sid] = InMemoryChatMessageHistory()
    st.success("Chat memory cleared.")
    st.rerun()

# ---------- connect ----------
if connect_clicked:
    if ensure_connected(openai_key, astra_token, astra_endpoint, astra_keyspace, astra_collection):
        st.success("Connected. You can now index the uploaded files.")

# ---------- indexing ----------
col_left, col_right = st.columns([1, 2], gap="large")
with col_left:
    st.subheader("📚 Indexing")
    idx_btn = st.button(
        "📥 Index uploaded files to Astra",
        use_container_width=True,
        disabled=not (st.session_state.connected and files),
    )
    if idx_btn:
        with st.spinner("Indexing..."):
            new_docs = load_uploaded_files_to_docs(files, chunk_size, chunk_overlap)
            if not new_docs:
                st.warning("No documents loaded.")
            else:
                st.session_state.docs.extend(new_docs)
                n = index_docs_to_astra(new_docs)
                build_hybrid_retriever(st.session_state.docs)
                build_chain()
                st.success(f"Indexed {n} chunks. Hybrid retriever is ready.")

with col_right:
    if st.session_state.docs:
        st.markdown("#### 🧪 Smoke checks")
        st.write(f"Total chunks in memory (BM25 scope): **{len(st.session_state.docs)}**")
        has_selectwize = any("selectwize" in d.page_content.lower() for d in st.session_state.docs)
        st.write(f"Contains 'selectwize' in any chunk? **{has_selectwize}**")

# ---------- chat ----------
# ensure we have an active chat session
if not st.session_state.current_session:
    create_chat()

is_ready = (
    st.session_state.connected
    and st.session_state.chain is not None
    and len(st.session_state.docs) > 0
)

st.markdown("---")
st.subheader("💬 Chat " + ("· 🔓 best-effort ON" if st.session_state.allow_guess else "· 🔒 grounded only"))

sid = st.session_state.current_session
hist = get_session_history(sid)

for m in hist.messages:
    role = "user" if m.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(m.content)

if not is_ready:
    st.info("To start chatting: 1) Upload file(s) → 2) Connect → 3) Index.")
question = st.chat_input("Type your question...", disabled=not is_ready)

if question:
    if not is_ready or st.session_state.chain is None:
        st.stop()

    db_get_or_create_session(sid, st.session_state.session_index.get(sid, {}).get("name", "Chat"))
    db_save_message(sid, "user", question)
    hist.add_user_message(question)

    with st.chat_message("user"):
        st.markdown(question)

    config = {"configurable": {"session_id": sid}}
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = st.session_state.chain.invoke({"question": question}, config=config)

        # small badge if best-effort with no sources
        if st.session_state.allow_guess and not res["sources"]:
            st.caption("🛈 Not found in the uploaded files — best-effort answer")

        answer_text = res["answer"]
        st.markdown(answer_text)

        db_save_message(sid, "assistant", answer_text)
        hist.add_ai_message(answer_text)
        auto_title_if_needed(sid, question, answer_text)

    with st.expander("🔎 Sources"):
        if not res["sources"] and st.session_state.allow_guess:
            st.info("No sources found — answer is best-effort and may include assumptions.")
        for i, d in enumerate(res["sources"], 1):
            meta = d.metadata
            src = meta.get("filename") or meta.get("source", "source")
            cid = meta.get("chunk_id", "?")
            st.markdown(f"**[{i}]** `{src}#{cid}`\n\n{d.page_content[:400]}…")

    st.session_state.chat_log.append({"role": "assistant", "content": answer_text})
