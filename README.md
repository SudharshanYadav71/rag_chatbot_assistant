# Nexus — RAG Chatbot 🔍

A **full-stack Retrieval-Augmented Generation chatbot** built with FastAPI + Anthropic Claude.  
Upload documents, ask questions, and get grounded answers with source attribution.


## 📁 Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py           ← FastAPI app (all RAG logic)
│   └── requirements.txt  ← Python dependencies
└── frontend/
    └── index.html        ← Single-file frontend (open in browser)
```


## ⚡ Quick Start

### 1. Get your Anthropic API key
Sign up at https://console.anthropic.com and create a key.

### 2. Set up the backend


## 📌 GitHub Repository

Target repository: [SudharshanYadav71/rag_chatbot_assistant](https://github.com/SudharshanYadav71/rag_chatbot_assistant)

That GitHub repository is currently empty, so this workspace contains the project source. If you want to publish this code there, initialize git in this folder and push to that remote:

```bash
git init
git remote add origin https://github.com/SudharshanYadav71/rag_chatbot_assistant.git
git add .
git commit -m "Initial project import"
git branch -M main
git push -u origin main
```

---
# Mac
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

The frontend connects to `http://localhost:8000` by default. You can change the API URL in the bottom-left input.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + chunk count |
| GET | `/documents` | List indexed documents |
| POST | `/ingest/text` | Add document from raw text |
| POST | `/ingest/file` | Upload .txt or .md file |
| DELETE | `/documents/{name}` | Remove a document |
| POST | `/query` | RAG query → answer + sources |

### POST /ingest/text
```json
{ "name": "my-doc.txt", "text": "Full document text here..." }
```

### POST /query
```json
{
  "query": "What is machine learning?",
  "top_k": 4,
  "conversation_history": []
}
```

### Response from /query
```json
{
  "answer": "Machine learning is...",
  "sources": [
    { "doc": "ml-intro.txt", "score": 2.34, "preview": "..." }
  ],
  "retrieved_chunks": 4,
  "latency_s": 1.2,
  "input_tokens": 820,
  "output_tokens": 180
}
```

---

## 🧠 How the RAG Pipeline Works

```
User Question
     │
     ▼
 BM25 Retrieval ──── Knowledge Base (chunked docs)
     │
     ▼
 Top-K Chunks (ranked by score)
     │
     ▼
 Context Injection → System Prompt
     │
     ▼
 Claude 3.5 Haiku → Grounded Answer
     │
     ▼
 Answer + Source Attribution
```

1. **Chunking** — Documents split into 400-word overlapping chunks (80-word overlap)
2. **Indexing** — BM25 inverted index built in-memory per token
3. **Retrieval** — Top-K chunks ranked by BM25 score for the query
4. **Generation** — Claude generates answer grounded only in retrieved context
5. **Attribution** — Source document + relevance score shown per answer

---

## 🎨 Frontend Features

- **Dark theme** with purple accent — refined editorial design
- **Drag-and-drop** file upload (.txt, .md)
- **Paste text** directly with a custom document name
- **Top-K slider** — control how many chunks are retrieved
- **Source chips** — click any source to preview the retrieved text
- **Conversation history** — last 3 turns sent for multi-turn context
- **Connection indicator** — live ping to your backend
- **Token & latency stats** per response

---

## 🚀 Deployment Options

### Option A — Hugging Face Spaces (free)
Add a `Dockerfile` that installs requirements and runs uvicorn on port 7860.  
Serve the HTML via `StaticFiles` in FastAPI.

### Option B — Vercel + Railway (free tier)
- **Railway** → deploy the `backend/` folder as a Python service
- **Vercel** → deploy the `frontend/` folder as a static site  
- Update the API URL in the frontend to your Railway URL

### Option C — Local / LAN
Run backend on any machine, open `index.html` on any browser on the same network,  
and set the API URL to `http://<machine-ip>:8000`.

---

<<<<<<< HEAD
=======
## 📌 GitHub Repository

Target repository: [SudharshanYadav71/rag_chatbot_assistant](https://github.com/SudharshanYadav71/rag_chatbot_assistant)

That GitHub repository is currently empty, so this workspace contains the project source. If you want to publish this code there, initialize git in this folder and push to that remote:

```bash
git init
git remote add origin https://github.com/SudharshanYadav71/rag_chatbot_assistant.git
git add .
git commit -m "Initial project import"
git branch -M main
git push -u origin main
```

---

>>>>>>> 607b36e5c2a960f1d6c78aed0e59b6b8853331cd
## 🛠 Extending the Project

| Feature | How |
|---------|-----|
| PDF support | Add `pypdf` and extract text in `/ingest/file` |
| FAISS embeddings | Add `sentence-transformers` + `faiss-cpu`; replace BM25 with vector search |
| Persistent storage | Replace in-memory dicts with SQLite or PostgreSQL |
| Auth | Add API key middleware to FastAPI |
| Streaming | Use `StreamingResponse` + SSE for token-by-token output |
| Reranking | Add a cross-encoder reranker after BM25 retrieval |

---

## 📦 Dependencies

**Backend**
- `fastapi` — web framework
- `uvicorn` — ASGI server
- `anthropic` — Claude API client
- `python-multipart` — file upload support
- `pydantic` — request validation

**Frontend**
- Zero dependencies — vanilla HTML/CSS/JS
- Google Fonts (DM Serif Display, DM Sans, JetBrains Mono)

---

Made for internship project demo. MIT License.
<<<<<<< HEAD
=======
# rag_chatbot_assistant
>>>>>>> 607b36e5c2a960f1d6c78aed0e59b6b8853331cd
