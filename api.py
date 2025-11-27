import os
import re
import uuid
import json
import zipfile
import pickle
import numpy as np
import hashlib
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph_backend import chatbot
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    import fitz
except Exception:
    fitz = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

DATA_DIR = os.path.join(os.getcwd(), "data")
DOC_DIR = os.path.join(DATA_DIR, "documents")
WEB_DIR = os.path.join(DATA_DIR, "web")
JSON_DIR = os.path.join(DATA_DIR, "json")
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
INDEX_DIR = os.path.join(DATA_DIR, "index")
VEC_PATH = os.path.join(INDEX_DIR, "vectorizer.pkl")
MAT_PATH = os.path.join(INDEX_DIR, "matrix.pkl")
IDS_PATH = os.path.join(INDEX_DIR, "chunk_ids.json")
EMB_PATH = os.path.join(INDEX_DIR, "embeddings.npy")
USE_DENSE = os.environ.get("USE_DENSE", "0") == "1"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", "20000000"))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "200"))

for d in [DATA_DIR, DOC_DIR, WEB_DIR, JSON_DIR, CHUNK_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app = FastAPI()

class UrlIngest(BaseModel):
    url: str
    page_id: Optional[str] = None
    name: Optional[str] = None

class JsonRecord(BaseModel):
    id: str
    title: Optional[str] = None
    text: str

class JsonIngest(BaseModel):
    records: List[JsonRecord]

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4
    mode: Optional[str] = "strict"
    retrieval: Optional[str] = "tfidf"

def dedupe_citations(citations: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for c in citations:
        k = (c.get("type"), c.get("id"))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out

def format_references(citations: List[dict]) -> str:
    meta = load_metadata()
    lines = []
    for c in citations:
        t = (c.get("type") or "").lower()
        name = c.get("name") or ""
        sid = c.get("id") or ""
        url = c.get("url") or None
        if t == "web" and url:
            lines.append(f"* Source: URL: `{url}`")
            continue
        if t == "document":
            ext = (meta.get("sources", {}).get(sid, {}).get("ext") or "").replace(".", "").upper()
            doc_name = name or sid
            if ext:
                lines.append(f"* Source: {doc_name} ({ext} filename)")
            else:
                lines.append(f"* Source: {doc_name}")
            continue
        if t == "json":
            nm = (name or "").strip()
            label = "Product ID" if nm and ("product" in nm.lower()) else "Record ID"
            lines.append(f"* Source: {label}: {sid}")
            continue
        label = name if name else sid
        lines.append(f"* Source: {label}")
    return "References:\n" + ("\n".join(lines) if lines else "* Source: None")

def finalize_answer(answer: str, citations: List[dict], origin: str) -> str:
    refs = format_references(citations)
    a = (answer or "").strip()
    if "References" in a:
        return a
    if citations:
        return (a + "\n\n" + refs).strip()
    if (origin or "").lower() == "general":
        return (a + "\n\n" + "References:\n- General Knowledge").strip()
    return a

def strip_general_disclaimer(text: str) -> str:
    s = (text or "")
    lines = s.splitlines()
    out = []
    for ln in lines:
        if re.search(r"(?i)general knowledge.*provided\s+documents", ln):
            continue
        out.append(ln)
    s = "\n".join(out).strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def choose_route(question: str) -> str:
    p = "Classify the question as LOCAL or GLOBAL. Return LOCAL or GLOBAL only.\n\nQuestion:\n" + question
    r = chatbot.invoke({"messages": [HumanMessage(content=p)]}, config={"configurable": {"thread_id": "route"}})
    c = (r["messages"][-1].content or "").strip().upper()
    if "GLOBAL" in c and "LOCAL" not in c:
        return "GLOBAL"
    if "LOCAL" in c and "GLOBAL" not in c:
        return "LOCAL"
    return "LOCAL"

def choose_intent(question: str, topics: List[str]) -> str:
    t = "\n".join([f"- {x}" for x in topics])
    p = (
        "Is the user asking about one of the documents/topics in this list? "
        "Answer LOCAL or GLOBAL only.\n\n" +
        "Topics:\n" + t + "\n\nQuestion:\n" + question
    )
    r = chatbot.invoke({"messages": [HumanMessage(content=p)]}, config={"configurable": {"thread_id": "intent"}})
    c = (r["messages"][-1].content or "").strip().upper()
    if "GLOBAL" in c and "LOCAL" not in c:
        return "GLOBAL"
    if "LOCAL" in c and "GLOBAL" not in c:
        return "LOCAL"
    return "LOCAL"

def load_metadata():
    if not os.path.exists(META_PATH):
        return {"sources": {}, "chunks": {}}
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata(meta):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

def ensure_hash_index(meta):
    changed = False
    meta.setdefault("hash_to_sid", {})
    for sid, sm in list(meta.get("sources", {}).items()):
        if sm.get("type") != "document":
            continue
        if sm.get("hash"):
            if sm["hash"] not in meta["hash_to_sid"]:
                meta["hash_to_sid"][sm["hash"]] = sid
                changed = True
            continue
        ext = sm.get("ext") or ""
        p = os.path.join(DOC_DIR, f"{sid}{ext}")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "rb") as f:
                data = f.read()
            h = hashlib.sha256(data).hexdigest()
            sm["hash"] = h
            meta["hash_to_sid"][h] = sid
            changed = True
        except Exception:
            pass
    return changed

def normalize_text(t: str) -> str:
    s = t.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = "\n".join(ln.strip() for ln in s.split("\n"))
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_main_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            try:
                tag.decompose()
            except Exception:
                pass
        return normalize_text(soup.get_text("\n"))
    except Exception:
        return ""

def extract_pdf_text_primary(path: str, max_pages: int) -> str:
    out = []
    with open(path, "rb") as f:
        reader = PdfReader(f)
        i = 0
        for p in reader.pages:
            if i >= max_pages:
                break
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            out.append(t)
            i += 1
    return normalize_text("\n".join(out))

def extract_pdf_text_fallback(path: str) -> str:
    if fitz is not None:
        try:
            doc = fitz.open(path)
            parts = []
            for i in range(min(doc.page_count, MAX_PDF_PAGES)):
                pg = doc.load_page(i)
                parts.append(pg.get_text("text"))
            return normalize_text("\n".join(parts))
        except Exception:
            pass
    if pdfminer_extract_text is not None:
        try:
            return normalize_text(pdfminer_extract_text(path))
        except Exception:
            pass
    return ""

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    overlap = max(0, min(overlap, chunk_size // 2))
    paras = re.split(r"\n\s*\n", text.strip())
    merged = []
    i = 0
    while i < len(paras):
        p = paras[i].strip()
        if i + 1 < len(paras) and (len(p) <= 120 and (p.isupper() or re.match(r"^(#+\s+|\d+[\.\)]\s+|.+:\s*$)", p) is not None)):
            merged.append((p + "\n" + paras[i + 1]).strip())
            i += 2
        else:
            merged.append(p)
            i += 1
    def split_sents(x: str) -> List[str]:
        return [s.strip() for s in re.split(r"(?<=[\.\!\?])\s+", x) if s.strip()]
    chunks = []
    cur = ""
    for para in merged:
        sents = split_sents(para)
        if not sents:
            sents = [para]
        for s in sents:
            nxt = (cur + (" " if cur else "") + s).strip()
            if len(nxt) <= chunk_size:
                cur = nxt
            else:
                if cur:
                    chunks.append(cur)
                    tail = cur[-overlap:] if overlap > 0 else ""
                    cur = (tail + s).strip()
                else:
                    part = s[:chunk_size]
                    chunks.append(part)
                    tail = part[-overlap:] if overlap > 0 else ""
                    cur = (tail + s[chunk_size:]).strip()
    if cur:
        chunks.append(cur)
    return chunks

def index_build():
    meta = load_metadata()
    texts = []
    ids = []
    for cid, cm in meta.get("chunks", {}).items():
        p = os.path.join(CHUNK_DIR, f"{cid}.txt")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
                if txt.strip():
                    texts.append(txt)
                    ids.append(cid)
    if not texts:
        return None, None, None
    vec = TfidfVectorizer(max_features=50000)
    mat = vec.fit_transform(texts)
    return vec, mat, ids

_vectorizer = None
_matrix = None
_chunk_ids = None
_embedder = None
_embeddings = None

def rebuild_index():
    global _vectorizer, _matrix, _chunk_ids, _embedder, _embeddings
    _vectorizer, _matrix, _chunk_ids = index_build()
    if _vectorizer is not None and _matrix is not None and _chunk_ids is not None:
        try:
            with open(VEC_PATH, "wb") as f:
                pickle.dump(_vectorizer, f)
            with open(MAT_PATH, "wb") as f:
                pickle.dump(_matrix, f)
            with open(IDS_PATH, "w", encoding="utf-8") as f:
                json.dump(_chunk_ids, f, ensure_ascii=False)
        except Exception:
            pass
    if SentenceTransformer is not None:
        meta = load_metadata()
        texts = []
        ids = []
        for cid, cm in meta.get("chunks", {}).items():
            p = os.path.join(CHUNK_DIR, f"{cid}.txt")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    txt = f.read()
                    if txt.strip():
                        texts.append(txt)
                        ids.append(cid)
        if texts:
            try:
                _embedder = SentenceTransformer(EMBED_MODEL)
                embs = _embedder.encode(texts, normalize_embeddings=True)
                _embeddings = np.asarray(embs, dtype=np.float32)
                _chunk_ids = ids
                np.save(EMB_PATH, _embeddings)
            except Exception:
                _embeddings = None

def rebuild_tfidf_index():
    global _vectorizer, _matrix, _chunk_ids
    _vectorizer, _matrix, _chunk_ids = index_build()
    if _vectorizer is not None and _matrix is not None and _chunk_ids is not None:
        try:
            with open(VEC_PATH, "wb") as f:
                pickle.dump(_vectorizer, f)
            with open(MAT_PATH, "wb") as f:
                pickle.dump(_matrix, f)
            with open(IDS_PATH, "w", encoding="utf-8") as f:
                json.dump(_chunk_ids, f, ensure_ascii=False)
        except Exception:
            pass

def append_dense_embeddings(new_ids: List[str]):
    global _embeddings, _embedder, _chunk_ids
    if SentenceTransformer is None:
        return
    try:
        if _embedder is None:
            _embedder = SentenceTransformer(EMBED_MODEL)
        texts = []
        ids = []
        for cid in new_ids:
            p = os.path.join(CHUNK_DIR, f"{cid}.txt")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    t = f.read()
                    if t.strip():
                        texts.append(t)
                        ids.append(cid)
        if not texts:
            return
        embs = _embedder.encode(texts, normalize_embeddings=True)
        arr = np.asarray(embs, dtype=np.float32)
        if _embeddings is None:
            _embeddings = arr
            _chunk_ids = ids
        else:
            _embeddings = np.vstack([_embeddings, arr])
            _chunk_ids.extend(ids)
        np.save(EMB_PATH, _embeddings)
        with open(IDS_PATH, "w", encoding="utf-8") as f:
            json.dump(_chunk_ids, f, ensure_ascii=False)
    except Exception:
        pass

@app.on_event("startup")
def startup_event():
    global _vectorizer, _matrix, _chunk_ids, _embeddings, _embedder
    try:
        if os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH) and os.path.exists(IDS_PATH):
            with open(VEC_PATH, "rb") as f:
                _vectorizer = pickle.load(f)
            with open(MAT_PATH, "rb") as f:
                _matrix = pickle.load(f)
            with open(IDS_PATH, "r", encoding="utf-8") as f:
                _chunk_ids = json.load(f)
    except Exception:
        _vectorizer = None
        _matrix = None
        _chunk_ids = None
    try:
        if os.path.exists(EMB_PATH) and SentenceTransformer is not None:
            _embeddings = np.load(EMB_PATH)
            _embedder = SentenceTransformer(EMBED_MODEL)
    except Exception:
        _embeddings = None
        _embedder = None
    if _vectorizer is None or _matrix is None or not _chunk_ids:
        rebuild_tfidf_index()
    if _embeddings is None and SentenceTransformer is not None:
        rebuild_index()

@app.get("/health")
def health():
    meta = load_metadata()
    sources = len(meta.get("sources", {}))
    chunks = len(meta.get("chunks", {}))
    tfidf_ready = _vectorizer is not None and _matrix is not None and _chunk_ids is not None
    dense_ready = _embeddings is not None and _chunk_ids is not None
    return {"status": "ok", "sources": sources, "chunks": chunks, "index_ready": tfidf_ready or dense_ready, "tfidf_ready": tfidf_ready, "dense_ready": dense_ready}

@app.post("/ingest/doc")
async def ingest_doc(background: BackgroundTasks, file: UploadFile = File(...), doc_id: Optional[str] = Form(None), name: Optional[str] = Form(None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".pdf", ".txt", ".md", ".docx"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    meta = load_metadata()
    meta.setdefault("hash_to_sid", {})
    if ensure_hash_index(meta):
        save_metadata(meta)
    sid = doc_id or str(uuid.uuid4())
    raw_path = os.path.join(DOC_DIR, f"{sid}{ext}")
    content = await file.read()
    if len(content or b"") > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        h = hashlib.sha256(content or b"").hexdigest()
    except Exception:
        h = None
    if h and h in meta.get("hash_to_sid", {}):
        existing_sid = meta["hash_to_sid"][h]
        existing = meta.get("sources", {}).get(existing_sid, {})
        chs = existing.get("chunk_ids", [])
        return {"doc_id": existing_sid, "chunks": len(chs), "duplicate": True}
    with open(raw_path, "wb") as f:
        f.write(content)
    text = ""
    if ext == ".pdf":
        text = extract_pdf_text_primary(raw_path, MAX_PDF_PAGES)
        if len(text) < 500:
            text = extract_pdf_text_fallback(raw_path)
    elif ext == ".docx":
        with open(raw_path, "rb") as f:
            content = f.read()
        try:
            with zipfile.ZipFile(raw_path) as z:
                xml = z.read("word/document.xml")
        except Exception:
            xml = b""
        soup = BeautifulSoup(xml, "xml")
        parts = [t.get_text() for t in soup.find_all("w:t")] if soup else []
        text = normalize_text(" ".join(parts))
    else:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            text = content.decode("latin-1", errors="ignore")
    text = normalize_text(text)
    if not text:
        raise HTTPException(status_code=400, detail="Empty content")
    chunks = chunk_text(text)
    saved_chunks = []
    for ch in chunks:
        cid = str(uuid.uuid4())
        with open(os.path.join(CHUNK_DIR, f"{cid}.txt"), "w", encoding="utf-8") as f:
            f.write(ch)
        meta.setdefault("chunks", {})[cid] = {"source_id": sid, "source_type": "document", "name": name or file.filename, "url": None}
        saved_chunks.append(cid)
    src = {"type": "document", "name": name or file.filename, "ext": ext, "chunk_ids": saved_chunks, "uploaded_at": datetime.now(timezone.utc).isoformat()}
    if h:
        src["hash"] = h
        meta["hash_to_sid"][h] = sid
    meta.setdefault("sources", {})[sid] = src
    save_metadata(meta)
    background.add_task(rebuild_tfidf_index)
    background.add_task(append_dense_embeddings, saved_chunks)
    return {"doc_id": sid, "chunks": len(saved_chunks)}

@app.post("/ingest/url")
def ingest_url(background: BackgroundTasks, body: UrlIngest):
    try:
        r = requests.get(body.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Bad response")
        try:
            text = extract_main_text(r.text)
        except Exception:
            try:
                soup = BeautifulSoup(r.text, "html.parser")
                text = normalize_text(soup.get_text("\n"))
            except Exception:
                text = ""
        if not text:
            raise HTTPException(status_code=400, detail="Empty page")
        title = None
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.string or "").strip() if soup.title and soup.title.string else None
        except Exception:
            title = None
        host = urlparse(body.url).netloc
        inferred_name = body.name or title or host
        meta = load_metadata()
        sid = body.page_id or str(uuid.uuid4())
        with open(os.path.join(WEB_DIR, f"{sid}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        chunks = chunk_text(text)
        saved_chunks = []
        for ch in chunks:
            cid = str(uuid.uuid4())
            with open(os.path.join(CHUNK_DIR, f"{cid}.txt"), "w", encoding="utf-8") as f:
                f.write(ch)
            meta.setdefault("chunks", {})[cid] = {"source_id": sid, "source_type": "web", "name": inferred_name, "url": body.url}
            saved_chunks.append(cid)
        meta.setdefault("sources", {})[sid] = {"type": "web", "name": inferred_name, "url": body.url, "chunk_ids": saved_chunks, "uploaded_at": datetime.now(timezone.utc).isoformat()}
        save_metadata(meta)
        background.add_task(rebuild_tfidf_index)
        background.add_task(append_dense_embeddings, saved_chunks)
        return {"page_id": sid, "chunks": len(saved_chunks)}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Parse failed")

@app.post("/ingest/json")
def ingest_json(background: BackgroundTasks, body: JsonIngest):
    meta = load_metadata()
    stored = []
    all_new = []
    for rec in body.records:
        text = normalize_text(rec.text)
        if not text:
            continue
        sid = rec.id
        with open(os.path.join(JSON_DIR, f"{sid}.txt"), "w", encoding="utf-8") as f:
            f.write(text)
        chunks = chunk_text(text)
        saved_chunks = []
        for ch in chunks:
            cid = str(uuid.uuid4())
            with open(os.path.join(CHUNK_DIR, f"{cid}.txt"), "w", encoding="utf-8") as f:
                f.write(ch)
            meta.setdefault("chunks", {})[cid] = {"source_id": sid, "source_type": "json", "name": rec.title, "url": None}
            saved_chunks.append(cid)
        meta.setdefault("sources", {})[sid] = {"type": "json", "name": rec.title, "chunk_ids": saved_chunks, "uploaded_at": datetime.now(timezone.utc).isoformat()}
        stored.append(sid)
        all_new.extend(saved_chunks)
    save_metadata(meta)
    background.add_task(rebuild_tfidf_index)
    background.add_task(append_dense_embeddings, all_new)
    return {"stored": stored, "chunks": sum(len(meta["sources"][sid]["chunk_ids"]) for sid in stored)}

@app.post("/query")
def query(body: QueryRequest):
    want_dense = ((body.retrieval or "tfidf").lower() == "dense")
    use_dense = want_dense and (_embeddings is not None) and (_embedder is not None) and (_chunk_ids is not None)
    if use_dense:
        qv = _embedder.encode([body.question], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)
        sims = (qv @ _embeddings.T)[0]
    else:
        if _vectorizer is None or _matrix is None or not _chunk_ids:
            raise HTTPException(status_code=400, detail="Index not ready")
        qv = _vectorizer.transform([body.question])
        sims = cosine_similarity(qv, _matrix)[0]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    k = max(1, min(body.top_k or 4, len(order)))
    meta = load_metadata()
    topics = []
    for sid, sm in meta.get("sources", {}).items():
        n = sm.get("name") or sid
        topics.append(n)
    has_sources = len(meta.get("sources", {})) > 0
    selected = []
    used_sources = set()
    for i in order:
        cid = _chunk_ids[i]
        cm = meta["chunks"].get(cid)
        sid = cm.get("source_id")
        if sid in used_sources:
            continue
        used_sources.add(sid)
        selected.append(i)
        if len(selected) >= k:
            break
    if len(selected) < k:
        for i in order:
            if i in selected:
                continue
            selected.append(i)
            if len(selected) >= k:
                break
    contexts = []
    citations = []
    for i in selected:
        cid = _chunk_ids[i]
        cm = meta["chunks"].get(cid)
        p = os.path.join(CHUNK_DIR, f"{cid}.txt")
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
        name = cm.get("name")
        url = cm.get("url")
        sid = cm.get("source_id")
        st = cm.get("source_type")
        header = f"Source: {name or ''} | Type: {st} | ID: {sid} | URL: {url or ''}"
        contexts.append(header + "\n" + txt)
        if sims[i] >= 0.12:
            citations.append({"type": st, "id": sid, "name": name, "url": url, "chunk_id": cid, "snippet": txt[:300]})
    citations = dedupe_citations(citations)
    origin = "local"
    thr = 0.12
    max_sim = max(sims) if len(sims) > 0 else 0.0
    if (body.mode or "strict").lower() != "strict":
        route = choose_intent(body.question, topics) if topics else choose_route(body.question)
        if route == "GLOBAL":
            contexts = []
            citations = []
            origin = "web"
        else:
            if len(contexts) == 0 or max_sim < thr:
                contexts = []
                citations = []
                origin = "web"
    if origin == "web":
        wc, wci = web_search(body.question, body.top_k or 4)
        if wc and wci:
            contexts = wc
            citations = wci
        else:
            slug = quote((body.question or "").strip().replace(" ", "_"))
            fallback_url = f"https://en.wikipedia.org/wiki/{slug}"
            citations = [{"type": "web", "id": f"wiki:{slug}", "name": "Wikipedia", "url": fallback_url, "chunk_id": None, "snippet": None}]
            origin = "general"
    if (body.mode or "strict").lower() != "strict" and origin == "general":
        local_citation_present = any(
            (c.get("type") in {"document", "json", "web"}) and not str(c.get("id", "")).startswith(("wiki:", "serper:", "tavily:"))
            for c in citations
        )
        if local_citation_present:
            prompt = (
                "Answer the question with a concise statement first. "
                "If this answer is general knowledge and not found in provided sources, add a short disclaimer after the answer: 'This is general knowledge and was not found in the provided documents.' "
                "Include a References section only if sources are provided.\n\n"
                + "Question:\n" + body.question + "\n\n"
                + "Sources:\n" + "\n\n".join(contexts)
            )
        else:
            prompt = (
                "Answer the question with a concise statement.\n\n"
                + "Question:\n" + body.question
            )
    else:
        prompt = (
            "Answer the question using the provided sources. Write a clear, well-formatted response. "
            "Include a References section listing each source with its name and URL or ID. "
            "If the answer is not in the sources, say you don't know.\n\n"
            + "Question:\n" + body.question + "\n\n"
            + "Sources:\n" + "\n\n".join(contexts)
        )
    resp = chatbot.invoke({"messages": [HumanMessage(content=prompt)]}, config={"configurable": {"thread_id": "thread-1"}})
    answer = resp["messages"][-1].content
    if origin == "general" and (not citations or all(str(c.get("id", "")).startswith(("wiki:", "serper:", "tavily:")) for c in citations)):
        answer = strip_general_disclaimer(answer)
    final_answer = finalize_answer(answer, citations, origin)
    return {"answer": final_answer, "citations": citations, "origin": origin}
def tavily_search(query: str, k: int):
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None, None
    try:
        r = requests.post("https://api.tavily.com/search", json={"api_key": key, "query": query, "search_depth": "basic", "include_raw_content": False, "max_results": max(1, k)}, timeout=15)
        j = r.json()
        results = j.get("results", [])
        contexts = []
        citations = []
        for res in results:
            url = res.get("url")
            title = res.get("title")
            content = normalize_text((res.get("content") or ""))
            if not content:
                continue
            chs = chunk_text(content)
            text = chs[0] if chs else content
            header = f"Source: {title or ''} | Type: web | ID: tavily:{url} | URL: {url}"
            contexts.append(header + "\n" + text)
            citations.append({"type": "web", "id": f"tavily:{url}", "name": title, "url": url, "chunk_id": None, "snippet": text[:300]})
        return (contexts if contexts else None), (dedupe_citations(citations) if citations else None)
    except Exception:
        return None, None

def serper_search(query: str, k: int):
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        return None, None
    try:
        r = requests.post("https://google.serper.dev/search", headers={"X-API-KEY": key}, json={"q": query, "gl": "us", "hl": "en"}, timeout=15)
        j = r.json()
        organic = j.get("organic", [])[:max(1, k)]
        contexts = []
        citations = []
        for item in organic:
            url = item.get("link")
            title = item.get("title")
            try:
                rr = requests.get(url, timeout=15)
                soup = BeautifulSoup(rr.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                content = normalize_text(soup.get_text("\n"))
            except Exception:
                content = ""
            if not content:
                continue
            chs = chunk_text(content)
            text = chs[0] if chs else content
            header = f"Source: {title or ''} | Type: web | ID: serper:{url} | URL: {url}"
            contexts.append(header + "\n" + text)
            citations.append({"type": "web", "id": f"serper:{url}", "name": title, "url": url, "chunk_id": None, "snippet": text[:300]})
        return (contexts if contexts else None), (dedupe_citations(citations) if citations else None)
    except Exception:
        return None, None

def wikipedia_search(query: str, k: int):
    try:
        sr = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1, "srlimit": 1}, timeout=15)
        sj = sr.json()
        items = sj.get("query", {}).get("search", [])
        if not items:
            return None, None
        pageid = items[0].get("pageid")
        title = items[0].get("title")
        er = requests.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "prop": "extracts", "explaintext": 1, "format": "json", "pageids": pageid}, timeout=15)
        ej = er.json()
        pages = ej.get("query", {}).get("pages", {})
        content = ""
        for pid, pv in pages.items():
            content = pv.get("extract") or ""
        content = normalize_text(content)
        if not content:
            return None, None
        chs = chunk_text(content)
        url = f"https://en.wikipedia.org/?curid={pageid}"
        contexts = []
        citations = []
        for ch in chs[:max(1, k)]:
            header = f"Source: {title or ''} | Type: web | ID: wiki:{pageid} | URL: {url}"
            contexts.append(header + "\n" + ch)
            citations.append({"type": "web", "id": f"wiki:{pageid}", "name": title, "url": url, "chunk_id": None, "snippet": ch[:300]})
        return contexts, dedupe_citations(citations)
    except Exception:
        return None, None

def web_search(query: str, k: int):
    c, ci = wikipedia_search(query, k)
    return (c or []), (ci or [])
