# 🚗 Automotive Graph RAG System — BMW Edition

> **Graph-enhanced Retrieval-Augmented Generation** for the automotive industry
> Powered by **Neo4j + ChromaDB + LangGraph + Ollama** — 100% free, open-source, runs fully locally.

---

## 🏗 Architecture

```
                    ┌──────────────────────────────────────────────────┐
                    │           Gradio Frontend  (Port 7860)            │
                    │   Chat (Streaming) │ Ingest │ Graph │ Stats       │
                    └─────────────────────┬────────────────────────────┘
                                          │ HTTP / SSE
                    ┌─────────────────────▼────────────────────────────┐
                    │           FastAPI Backend  (Port 8000)            │
                    │  /query  /stream/query  /ingest  /graph  /health  │
                    └─────────────────────┬────────────────────────────┘
                                          │
                    ┌─────────────────────▼────────────────────────────┐
                    │           LangGraph Agent  (10 nodes)             │
                    │                                                    │
                    │  1. classify_query   → intent + complexity        │
                    │  2. expand_query     → 2-3 alternative phrasings  │
                    │  3. select_pipeline  → GRAPH / VECTOR / HYBRID    │
                    │  4. retrieve         → Neo4j + ChromaDB           │
                    │  5. rerank           → cross-encoder scoring      │
                    │  6. check_relevance  → relevance gate (NEW)       │
                    │       ├─ relevant   → 7. optimize_prompt          │
                    │       └─ irrelevant → 10. refuse_answer           │
                    │  7. optimize_prompt  → dynamic system prompt      │
                    │  8. generate         → Ollama LLM answer          │
                    │  9. validate         → numeric grounding check    │
                    │  10. refuse_answer   → clean refusal message      │
                    └──────────┬──────────────────────┬────────────────┘
                               │                      │
               ┌───────────────▼──────┐  ┌────────────▼───────────────┐
               │   Neo4j  (Port 7687)  │  │  ChromaDB  (local disk)    │
               │  B-Tree  ·  Fulltext  │  │  HNSW vector index         │
               │  Vector  ·  Graph     │  │  384-dim cosine similarity  │
               │  Vehicle · Engine     │  │  sentence-transformers      │
               │  Feature · Spec       │  └────────────────────────────┘
               │  Recall  · DTC        │
               └──────────────────────┘
```

---

## 🛠 Tech Stack — 100% Free & Open-Source

| Component | Technology | Purpose |
|---|---|---|
| **LLM** | Ollama — `llama3` / `mistral` / `gemma2` | Answer generation, query classification |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Semantic chunk encoding (384-dim) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Relevance scoring of retrieved passages |
| **Graph DB** | Neo4j 5 Community (Docker) | Knowledge graph — nodes, edges, traversal |
| **Vector DB** | ChromaDB (persistent local) | Semantic similarity search (HNSW) |
| **Agent Framework** | LangGraph | 10-node stateful agent pipeline |
| **Backend** | FastAPI + Uvicorn | REST API + SSE streaming |
| **Frontend** | Gradio 4 | Streaming chat UI + graph visualization |
| **Entity Extraction** | spaCy + regex (automotive rules) | BMW NER for graph population |
| **Evaluation** | RAGAS (offline script) | Faithfulness, relevancy, precision, recall |

---

## ⚡ Quick Start

### Prerequisites

- Python **3.10 or 3.11**
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Ollama](https://ollama.com/download)

---

### Step 1 — Navigate to Project

```bash
cd C:/graph_rag_automotive
```

---

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

### Step 4 — Configure Environment

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Key settings in `.env` (defaults work out of the box):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=automotive_rag
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

### Step 5 — Start Neo4j

```bash
# Start Neo4j in background
docker-compose up -d

# Check it is running
docker ps

# Neo4j Browser → http://localhost:7474
# Login: neo4j / automotive_rag
```

---

### Step 6 — Pull Ollama LLM (One Time)

```bash
# Terminal 1 — keep this open
ollama serve

# Terminal 2 — download models (~4-5 GB total)
ollama pull llama3           # main LLM (4.7 GB)
ollama pull nomic-embed-text # embeddings
```

> **Alternatives:** `mistral` (4.1 GB), `phi3` (2.3 GB, lightweight)

---

### Step 7 — Copy BMW PDF

```bash
# Windows
copy "C:\Users\91956\Downloads\2023-bmw-7-30-62.pdf" "data\pdfs\"

# Mac / Linux
cp ~/Downloads/2023-bmw-7-30-62.pdf data/pdfs/
```

---

### Step 8 — Ingest the PDF (Run Once)

```bash
python ingest_bmw.py

# Custom path
python ingest_bmw.py --pdf "data/pdfs/2023-bmw-7-30-62.pdf"
```

Expected output:
```
✅ Ingestion complete!
   File   : 2023-bmw-7-30-62.pdf
   Pages  : 13
   Chunks : 72
   Doc ID : abc123def456

🚀 Now run: python start.py
```

---

### Step 9 — Start the Application

```bash
python start.py
```

```
════════════════════════════════════════════════════
  🚗 Automotive Graph RAG System — Starting Up
════════════════════════════════════════════════════
  FastAPI  → http://localhost:8000/docs
  Gradio   → http://localhost:7860
  Neo4j    → http://localhost:7474

  Press Ctrl+C to stop all services.
════════════════════════════════════════════════════
```

---

### Step 10 — Open in Browser

| Service | URL |
|---|---|
| **Chat UI** | http://localhost:7860 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **Neo4j Browser** | http://localhost:7474 |

---

## 📁 Project Structure

```
graph_rag_automotive/
│
├── backend/
│   ├── main.py                      # FastAPI entry point
│   ├── config.py                    # Settings from .env
│   │
│   ├── agents/
│   │   ├── graph_rag_agent.py       # LangGraph 10-node pipeline
│   │   └── streaming.py             # SSE token streaming generator
│   │
│   ├── graph/
│   │   ├── automotive_schema.py     # Neo4j node/relationship definitions
│   │   ├── neo4j_client.py          # Driver + query helpers
│   │   ├── cypher_generator.py      # LLM → Cypher translation
│   │   └── graph_retriever.py       # Graph traversal retrieval
│   │
│   ├── indexing/
│   │   ├── pdf_processor.py         # PDF → text chunks
│   │   ├── entity_extractor.py      # BMW NER (spaCy + regex)
│   │   ├── graph_builder.py         # Neo4j graph population
│   │   └── vector_indexer.py        # ChromaDB + embeddings
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py      # Graph + vector fusion
│   │   └── reranker.py              # Cross-encoder reranking
│   │
│   ├── pipeline/
│   │   └── ingestion_pipeline.py    # End-to-end PDF ingestion
│   │
│   ├── evaluation/
│   │   └── ragas_evaluator.py       # RAGAS metrics library (offline)
│   │
│   ├── api/
│   │   ├── models.py                # Pydantic request/response schemas
│   │   └── routes/
│   │       ├── query.py             # POST /query/
│   │       ├── stream.py            # POST /stream/query  (SSE)
│   │       ├── ingest.py            # POST /ingest/upload
│   │       └── graph.py             # GET /graph/stats, /visualize
│   │
│   └── utils/
│       └── logger.py                # Loguru structured logging
│
├── frontend/
│   ├── app.py                       # Gradio UI (streaming chat)
│   └── api_client.py                # httpx calls to FastAPI
│
├── data/
│   └── pdfs/                        # Drop BMW PDFs here
│
├── start.py                         # One-command app launcher
├── ingest_bmw.py                    # One-command PDF ingestion
├── evaluate.py                      # Standalone RAGAS test script
├── docker-compose.yml               # Neo4j Community Edition
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔍 LangGraph Agent — 10 Nodes

The agent runs as a **stateful directed graph**. Each node is a pure Python function that mutates the shared `AgentState`.

| Node | Name | What It Does |
|---|---|---|
| 1 | `classify_query` | LLM reads query → assigns intent + complexity (temp=0.0) |
| 2 | `expand_query` | Generates 2 alternative phrasings → better recall |
| 3 | `select_pipeline` | Routes to GRAPH / VECTOR / HYBRID based on keywords + intent |
| 4 | `retrieve` | Runs all expanded queries through selected pipeline |
| 5 | `rerank` | Cross-encoder scores all candidates → keeps top-K |
| 6 | `check_relevance` | **Relevance gate** — blocks generation if docs not relevant |
| 7 | `optimize_prompt` | Builds tone-aware grounded system prompt |
| 8 | `generate` | Ollama LLM produces the answer (temp=0.1) |
| 9 | `validate` | Numeric grounding check — flags ungrounded values |
| 10 | `refuse_answer` | Clean refusal when relevance gate blocks generation |

### Relevance Gate (Node 6)

```
Reranker top score:

  0.0 ──────── 0.3 ──────── 0.6 ──────── 1.0
        REFUSE     LLM CHECK     GENERATE
     (immediate)  (YES/NO call) (immediate)
```

### Pipeline Routing Logic

```
Query keyword/intent          →  Pipeline selected
────────────────────────────────────────────────────
dtc / recall / fault / nhtsa  →  GRAPH_ONLY
dimension / hp / torque / spec →  HYBRID
exploratory / comparative      →  VECTOR_ONLY
general / factual              →  HYBRID  (default)
```

---

## 🕸 Graph Schema

```
(Vehicle) ──HAS_ENGINE──────► (Engine)
(Vehicle) ──HAS_FEATURE─────► (Feature) ──BELONGS_TO_SYSTEM──► (System)
(Vehicle) ──HAS_SPECIFICATION► (Specification)
(Vehicle) ──SUBJECT_TO_RECALL► (Recall)
(System)  ──HAS_COMPONENT───► (Component) ──HAS_DTC──────────► (DiagnosticCode)
(Chunk)   ──SOURCED_FROM────► (Document)
(Chunk)   ──MENTIONS_VEHICLE► (Vehicle)
(Chunk)   ──MENTIONS_FEATURE► (Feature)
(Chunk)   ──NEXT_CHUNK──────► (Chunk)
```

### Neo4j Index Types

| Index | Type | Used For |
|---|---|---|
| `id` constraints | B-Tree | MERGE / MATCH by id — O(log n) |
| `chunk_text_idx` | Lucene fulltext | Keyword search with TF-IDF ranking |
| `chunk_embedding_idx` | HNSW vector | Graph-aware semantic search |

---

## 🌊 Streaming API

The `/stream/query` endpoint returns **Server-Sent Events (SSE)**. Tokens stream word-by-word as the LLM generates.

```
POST /stream/query
Content-Type: application/json

{"query": "What is the horsepower of the 760i?"}
```

SSE events emitted in order:

```
data: {"type": "stage",    "content": "🔍 Classifying query…"}
data: {"type": "stage",    "content": "📋 Intent: factual"}
data: {"type": "stage",    "content": "📚 Retrieving from knowledge base…"}
data: {"type": "stage",    "content": "✅ Top relevance score: 0.941"}
data: {"type": "stage",    "content": "✍️  Generating answer…"}
data: {"type": "token",    "content": "The "}
data: {"type": "token",    "content": "2023 "}
data: {"type": "token",    "content": "BMW "}
...
data: {"type": "metadata", "content": {"pipeline": "hybrid", "confidence": 0.94}}
data: {"type": "done",     "content": ""}
```

If documents are not relevant:
```
data: {"type": "refused",  "content": "❌ Retrieved documents are not relevant enough…"}
data: {"type": "done",     "content": ""}
```

---

## 📡 API Reference

### `POST /query/`
Standard (non-streaming) query.

```json
Request:
{
  "query": "What is the horsepower of the 760i xDrive?",
  "chat_history": [],
  "pipeline": "auto"
}

Response:
{
  "answer": "The 2023 BMW 760i xDrive produces 536 horsepower…",
  "sources": ["Neo4j:Engine", "ChromaDB:p4"],
  "pipeline": "hybrid",
  "pipeline_reason": "Specification query → hybrid coverage",
  "confidence": 0.941,
  "query_intent": "factual",
  "num_results": 5,
  "was_refused": false,
  "refusal_reason": ""
}
```

### `POST /stream/query`
Streaming SSE query — returns tokens as they generate.

### `POST /ingest/upload`
Multipart PDF upload → full ingestion pipeline.

### `GET /graph/stats`
Node counts, relationship counts, ChromaDB stats.

### `GET /graph/visualize?limit=150`
Graph nodes + edges for Plotly visualization.

### `GET /graph/vehicles`
All vehicle nodes with engines and top features.

### `GET /health`
Neo4j, ChromaDB and Ollama connectivity status.

---

## 🧪 Offline Evaluation (RAGAS)

Evaluation runs as a **standalone script** — completely separate from the app. Run it once after ingestion to measure pipeline quality.

```bash
# Run full BMW benchmark (6 built-in questions)
python evaluate.py

# Save results to JSON
python evaluate.py --save results.json

# Evaluate one custom question interactively
python evaluate.py --mode single

# Evaluate from a custom JSON file
python evaluate.py --mode batch --questions-file my_questions.json
```

### Metrics

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Fraction of answer claims supported by retrieved context |
| **Answer Relevancy** | How well the answer addresses the actual question |
| **Context Precision** | Are retrieved chunks truly relevant to the question? |
| **Context Recall** | Does the context contain enough info to answer correctly? |

All scoring uses **local Ollama** as judge — no OpenAI API, no cloud calls.

---

## ⚙️ Configuration

Edit `.env` to tune the system:

```env
# Switch LLM
OLLAMA_MODEL=mistral           # faster alternative to llama3
OLLAMA_MODEL=phi3              # lightweight, less memory

# Larger embeddings (better quality, slower)
HF_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2

# Faster reranker
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2

# Retrieval tuning
CHUNK_SIZE=512                 # tokens per chunk
CHUNK_OVERLAP=64               # overlap between chunks
RETRIEVAL_TOP_K=10             # candidates before reranking
RERANK_TOP_K=5                 # passed to LLM after reranking

# Relevance gate thresholds
GRAPH_SCORE_THRESHOLD=0.6
VECTOR_SCORE_THRESHOLD=0.5
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `Neo4j not reachable` | `docker-compose down && docker-compose up -d` — wait 30s |
| `Ollama model not found` | `ollama pull llama3 && ollama list` |
| `Port already in use` | Change `FASTAPI_PORT` / `GRADIO_PORT` in `.env` |
| `Embeddings slow first run` | Normal — model downloads once (~90 MB), cached after |
| `Import errors` | Activate venv: `venv\Scripts\activate` |
| `Empty graph after ingest` | Check Neo4j is running before `python ingest_bmw.py` |

---

## 📊 Performance Reference

| Setting | Default | Notes |
|---|---|---|
| Chunk size | 512 tokens | Balance between precision and context |
| Retrieval top-K | 10 | Candidates sent to reranker |
| Rerank top-K | 5 | Context chunks sent to LLM |
| Embedding dim | 384 | MiniLM — fast and accurate |
| Neo4j vector index | cosine | Best for semantic similarity |
| LLM temperature | 0.1 (generate) / 0.0 (classify) | Low = grounded, deterministic |

---

## 🏆 Key Design Decisions

**Why Graph + Vector together?**
Vector search finds semantically similar text. Graph traversal finds structurally connected facts. A question like "Does the 760i have any recalls?" needs graph traversal (`Vehicle → Recall`), not similarity search.

**Why the Relevance Gate?**
Preventing generation from irrelevant context eliminates hallucination at the source rather than detecting it after the fact. If the top reranker score is below 0.3, the system refuses cleanly instead of generating a plausible but fabricated answer.

**Why 100% local models?**
Enterprise automotive clients have strict data privacy requirements. No data leaves the machine — Ollama, sentence-transformers, cross-encoder, Neo4j and ChromaDB all run locally.

**Why dynamic pipeline switching?**
Different questions need fundamentally different retrieval strategies. A DTC lookup needs a graph node, not semantic similarity. Routing per query instead of using one strategy for everything produces significantly better answers.
