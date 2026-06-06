"""
RAG Chatbot Backend — FastAPI
Endpoints:
  POST /upload       → ingest document text/file
  POST /query        → retrieve + generate answer
  GET  /documents    → list indexed docs
  DELETE /documents/{name} → remove a doc
  GET  /health       → health check
"""

import os, math, re, json, time
from collections import defaultdict
from typing import Optional

import anthropic
import uvicorn
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Anthropic client (safe init) ─────────────────────────────────────────────
# Try to initialize the Anthropic client, but don't crash the app if the
# library or environment is missing/incompatible. In that case we keep
# `client` as `None` and return a helpful mock response from `/query`.
_client = None
_api_key = os.environ.get("ANTHROPIC_API_KEY", "") or None
if _api_key:
    try:
        _client = anthropic.Anthropic(api_key=_api_key)
    except Exception as e:
        # Warn but keep server running. Some anthropic/httpx versions may
        # be incompatible and raise on import/initialization.
        print("Warning: failed to initialize Anthropic client:", e)
        _client = None

client = _client

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the single-file frontend if present (allows opening http://localhost:8000/)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    # Serve static files under /static to avoid shadowing API routes
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    # Serve index.html at root so visiting / loads the UI
    @app.get("/")
    def serve_index():
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        raise HTTPException(404, "index.html not found")
else:
    print("Warning: frontend directory not found at", FRONTEND_DIR)

# ── In-memory Knowledge Base ─────────────────────────────────────────────────
class KnowledgeBase:
    def __init__(self):
        self.docs: dict[str, str] = {}          # name → full text
        self.chunks: list[dict] = []            # [{doc, idx, text}]
        self.inverted: dict[str, list[int]] = defaultdict(list)  # term → chunk indices

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _chunk(self, text: str, size=400, overlap=80) -> list[str]:
        words = text.split()
        result, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i:i + size])
            if chunk.strip():
                result.append(chunk)
            if i + size >= len(words):
                break
            i += size - overlap
        return result

    def add_document(self, name: str, text: str):
        if name in self.docs:
            self.remove_document(name)
        self.docs[name] = text
        start = len(self.chunks)
        raw_chunks = self._chunk(text)
        for idx, chunk_text in enumerate(raw_chunks):
            chunk = {"doc": name, "idx": idx, "text": chunk_text}
            self.chunks.append(chunk)
            for token in set(self._tokenize(chunk_text)):
                self.inverted[token].append(start + idx)

    def remove_document(self, name: str):
        if name not in self.docs:
            return
        del self.docs[name]
        # rebuild chunks and inverted index without this doc
        self.chunks = [c for c in self.chunks if c["doc"] != name]
        self.inverted = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            for token in set(self._tokenize(chunk["text"])):
                self.inverted[token].append(i)

    def retrieve(self, query: str, top_k=4) -> list[dict]:
        """BM25-style TF-IDF retrieval with position bonus."""
        if not self.chunks:
            return []
        N = len(self.chunks)
        q_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in q_tokens:
            if token not in self.inverted:
                continue
            candidate_ids = self.inverted[token]
            df = len(set(candidate_ids))
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for cid in candidate_ids:
                chunk_text = self.chunks[cid]["text"]
                words = chunk_text.lower().split()
                tf = words.count(token) / len(words) if words else 0
                k1, b, avgdl = 1.5, 0.75, 400
                dl = len(words)
                bm25 = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                scores[cid] += bm25

            # bonus for exact phrase
            if query.lower() in " ".join(self._tokenize(self.chunks[cid]["text"])):
                scores[cid] += 1.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in ranked[:top_k]:
            c = self.chunks[cid].copy()
            c["score"] = round(score, 4)
            results.append(c)
        return results

    def list_documents(self) -> list[dict]:
        counts = defaultdict(int)
        for c in self.chunks:
            counts[c["doc"]] += 1
        return [{"name": n, "chunks": counts[n], "chars": len(t)} for n, t in self.docs.items()]

    def total_chunks(self) -> int:
        return len(self.chunks)


kb = KnowledgeBase()

# ── Pydantic models ──────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    top_k: int = 4
    conversation_history: list[dict] = []   # [{role, content}]

class TextIngestRequest(BaseModel):
    name: str
    text: str

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "chunks": kb.total_chunks(), "docs": len(kb.docs)}


@app.get("/documents")
def list_docs():
    return {"documents": kb.list_documents(), "total_chunks": kb.total_chunks()}


@app.post("/ingest/text")
def ingest_text(req: TextIngestRequest):
    if not req.name.strip() or not req.text.strip():
        raise HTTPException(400, "name and text are required")
    kb.add_document(req.name.strip(), req.text.strip())
    return {"status": "ok", "name": req.name, "chunks": kb.total_chunks()}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    # Accept common text files robustly. If the file isn't a known text
    # extension, still attempt to decode as utf-8 (with replacement) so
    # many uploads still work in development.
    fname = file.filename or f"upload-{int(time.time())}"
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Failed to read uploaded file: {e}")

    # Prefer UTF-8, but gracefully degrade
    try:
        text = content.decode("utf-8")
    except Exception:
        text = content.decode("utf-8", errors="replace")

    # Basic reject for clearly binary files (very low text ratio)
    if len(text) > 0:
        printable = sum(1 for c in text if ord(c) >= 32 or c in "\n\r\t")
        ratio = printable / len(text)
        if ratio < 0.2:
            # treat as likely non-text
            raise HTTPException(400, "Uploaded file does not appear to be text")

    kb.add_document(fname, text)
    print(f"Ingested file: {fname} ({len(text)} chars, {kb.total_chunks()} chunks total)")
    return {"status": "ok", "name": fname, "chunks": kb.total_chunks()}


@app.delete("/documents/{name}")
def delete_doc(name: str):
    if name not in kb.docs:
        raise HTTPException(404, "Document not found")
    kb.remove_document(name)
    return {"status": "deleted", "name": name, "remaining_chunks": kb.total_chunks()}


@app.post("/query")
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "query is required")

    retrieved = kb.retrieve(req.query.strip(), top_k=req.top_k)

    if not retrieved:
        context_block = "No relevant context found in the knowledge base."
    else:
        context_block = "\n\n".join(
            f"[Source {i+1} | doc: {c['doc']} | score: {c['score']}]\n{c['text']}"
            for i, c in enumerate(retrieved)
        )

    system_prompt = f"""You are a helpful, precise AI assistant. Answer the user's question using ONLY the provided context below. 
If the context doesn't contain enough information to answer confidently, say so.
Be concise, accurate, and cite which source(s) you are drawing from.

CONTEXT:
{context_block}"""

    messages = req.conversation_history + [{"role": "user", "content": req.query}]

    t0 = time.time()

    # If the Anthropic client couldn't be initialized (missing API key or
    # incompatible library versions), synthesize a local, deterministic
    # answer from the retrieved chunks so the app remains useful offline.
    if client is None:
        elapsed = round(time.time() - t0, 2)

        def first_sentence(s: str) -> str:
            s = s.strip().replace("\n", " ")
            m = re.search(r"([^.?!]+[.?!])", s)
            return m.group(1).strip() if m else (s[:200].strip() + ("..." if len(s) > 200 else ""))

        if not retrieved:
            answer = "I couldn't find relevant context in the uploaded documents. Try adding more documents or rephrasing your question."
            sources = []
        else:
            # Build a concise extractive answer using the top retrieved chunks
            pieces = []
            for i, c in enumerate(retrieved):
                pieces.append(f"[{i+1}] {first_sentence(c['text'])} (source: {c['doc']})")
            answer = "Based on the retrieved documents:\n\n" + "\n".join(pieces)
            sources = [{"doc": c["doc"], "score": c["score"], "preview": c["text"][:200]} for c in retrieved]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved),
            "latency_s": elapsed,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        system=system_prompt,
        messages=messages,
    )
    elapsed = round(time.time() - t0, 2)

    answer = response.content[0].text
    sources = [{"doc": c["doc"], "score": c["score"], "preview": c["text"][:200]} for c in retrieved]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": len(retrieved),
        "latency_s": elapsed,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
