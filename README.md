# 🚗 Automotive Graph RAG System

> **Graph-enhanced Retrieval-Augmented Generation** for the automotive industry  
> Powered by **Neo4j + ChromaDB + LangGraph + Ollama** — 100% free & open-source, runs locally.

---

## 🏗 Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │              Gradio Frontend (Port 7860)         │
                        │  Chat │ Ingest │ Graph Viz │ Stats │ Examples    │
                        └────────────────────┬────────────────────────────┘
                                             │ HTTP
                        ┌────────────────────▼────────────────────────────┐
                        │              FastAPI Backend (Port 8000)         │
                        │     /query   /ingest   /graph   /health          │
                        └────────────────────┬────────────────────────────┘
                                             │
                        ┌────────────────────▼────────────────────────────┐
                        │              LangGraph Agent Pipeline            │
                        │                                                  │
                        │  1. classify_query   → intent + complexity       │
                        │  2. expand_query     → 2-3 alt phrasings         │
                        │  3. select_pipeline  → auto-route retrieval      │
                        │  4. retrieve         → Neo4j + ChromaDB          │
                        │  5. rerank           → cross-encoder scoring     │
                        │  6. optimize_prompt  → dynamic system prompt     │
                        │  7. generate         → Ollama LLM answer         │
                        │  8. validate         → hallucination guard       │
                        └──────────┬──────────────────────┬───────────────┘
                                   │                      │
                    ┌──────────────▼──────┐  ┌────────────▼────────────────┐
                    │   Neo4j (Graph DB)   │  │   ChromaDB (Vector Store)   │
                    │  Vehicle, Engine,    │  │   all-MiniLM-L6-v2          │
                    │  Feature, Spec,      │  │   cosine similarity         │
                    │  Recall, DTC, Chunk  │  │   384-dim embeddings        │
                    └─────────────────────┘  └─────────────────────────────┘
```

---

## 🛠 Tech Stack (100% Free & Open-Source)

| Component | Technology |
|-----------|-----------|
| **LLM** | [Ollama](https://ollama.com) — `llama3`, `mistral`, `gemma2` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| **Graph DB** | Neo4j 5 (Docker) |
| **Vector DB** | ChromaDB (persistent local) |
| **Agent Framework** | LangGraph |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Gradio 4 |
| **Entity Extraction** | spaCy + regex (automotive domain rules) |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- [Ollama](https://ollama.com/download) installed

### 1. Clone & Configure

```bash
git clone <repo>
cd graph_rag_automotive
cp .env.example .env
```

### 2. Start Neo4j

```bash
docker-compose up -d
# Neo4j Browser → http://localhost:7474
# Login: neo4j / automotive_rag
```

### 3. Install Ollama Models (Free)

```bash
ollama pull llama3          # 4.7 GB — best quality
# OR
ollama pull mistral         # 4.1 GB — faster
# OR  
ollama pull phi3            # 2.3 GB — lightweight
```

### 4. Install Python Dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Ingest the BMW PDF

```bash

# Run ingestion
python -c "
from backend.pipeline.ingestion_pipeline import IngestionPipeline
result = IngestionPipeline().run('data/pdfs/2023-bmw-7-30-62.pdf')
print(result)
"
```

### 6. Start the Backend

```bash
# From project root
python -m backend.main
# OR
uvicorn backend.main:app --reload --port 8000
```

API Docs → http://localhost:8000/docs

### 7. Start the Gradio Frontend

```bash
# In a new terminal
python frontend/app.py
```

Frontend → http://localhost:7860

---

## 📁 Project Structure

```
graph_rag_automotive/
│
├── backend/
│   ├── main.py                    # FastAPI app entry
│   ├── config.py                  # Settings (env-based)
│   │
│   ├── agents/
│   │   └── graph_rag_agent.py     # LangGraph 8-node pipeline
│   │
│   ├── graph/
│   │   ├── automotive_schema.py   # Neo4j node/rel definitions
│   │   ├── neo4j_client.py        # Driver + query helpers
│   │   ├── cypher_generator.py    # LLM → Cypher translation
│   │   └── graph_retriever.py     # Graph traversal retrieval
│   │
│   ├── indexing/
│   │   ├── pdf_processor.py       # PDF → chunks
│   │   ├── entity_extractor.py    # BMW NER (regex + spaCy)
│   │   ├── graph_builder.py       # Neo4j graph population
│   │   └── vector_indexer.py      # ChromaDB + embeddings
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # Graph + vector fusion
│   │   └── reranker.py            # Cross-encoder reranking
│   │
│   ├── pipeline/
│   │   └── ingestion_pipeline.py  # End-to-end PDF ingestion
│   │
│   ├── api/
│   │   ├── models.py              # Pydantic schemas
│   │   └── routes/
│   │       ├── query.py           # POST /query/
│   │       ├── ingest.py          # POST /ingest/upload
│   │       └── graph.py           # GET /graph/stats, /visualize
│   │
│   └── utils/
│       └── logger.py              # Loguru structured logging
│
├── frontend/
│   ├── app.py                     # Gradio 5-tab UI
│   └── api_client.py              # httpx calls to FastAPI
│
├── data/
│   └── pdfs/                      # Drop BMW PDFs here
│
├── docker-compose.yml             # Neo4j Community
├── requirements.txt
└── .env.example
```

---

## 🔍 LangGraph Agent Deep Dive

The agent runs **8 sequential nodes**, each a Python function mutating the shared `AgentState`:

| Node | Function | Description |
|------|----------|-------------|
| 1 | `classify_query` | LLM classifies intent (factual/diagnostic/exploratory/comparative) and complexity |
| 2 | `expand_query` | Generates 2 alternative phrasings for better recall |
| 3 | `select_pipeline` | Routes to VECTOR / GRAPH / HYBRID based on query type |
| 4 | `retrieve` | Executes retrieval across all expanded queries |
| 5 | `rerank` | Cross-encoder scores all candidates, keeps top-K |
| 6 | `optimize_prompt` | Builds tone-aware system prompt with grounded context |
| 7 | `generate` | Ollama LLM produces the answer |
| 8 | `validate` | Hallucination guard — flags ungrounded numeric claims |

### Pipeline Selection Logic

```
Query type           →  Pipeline selected
─────────────────────────────────────────
DTC / Recall         →  GRAPH_ONLY   (precise structured lookup)
Dimensions / Specs   →  HYBRID       (graph + semantic)
Exploratory / Why    →  VECTOR_ONLY  (semantic search)
General              →  HYBRID       (best coverage)
```

---

## 🕸 Graph Schema

```
(Vehicle)──HAS_ENGINE──────►(Engine)
(Vehicle)──HAS_FEATURE─────►(Feature)──BELONGS_TO_SYSTEM──►(System)
(Vehicle)──HAS_SPECIFICATION►(Specification)
(Vehicle)──SUBJECT_TO_RECALL►(Recall)
(System)───HAS_COMPONENT───►(Component)──HAS_DTC──────────►(DiagnosticCode)
(Chunk)────SOURCED_FROM────►(Document)
(Chunk)────MENTIONS_VEHICLE►(Vehicle)
(Chunk)────NEXT_CHUNK──────►(Chunk)
```

---

## 🧪 API Reference

### POST /query/
```json
{
  "query": "What is the horsepower of the 760i xDrive?",
  "chat_history": [],
  "pipeline": "auto"
}
```
Response:
```json
{
  "answer": "The 2023 BMW 760i xDrive ...",
  "sources": ["Neo4j:Engine", "ChromaDB:p4"],
  "pipeline": "hybrid",
  "pipeline_reason": "Specification query → hybrid coverage",
  "confidence": 0.823,
  "query_intent": "factual",
  "num_results": 5
}
```

### POST /ingest/upload
Multipart PDF upload — triggers full ingestion pipeline.

### GET /graph/stats
Returns node counts, relationship counts, ChromaDB stats.

### GET /graph/visualize?limit=150
Returns graph nodes + edges for Plotly visualization.

### GET /health
Returns Neo4j, ChromaDB, and Ollama connectivity status.

---

## 🔧 Configuration

Edit `.env` to change models:

```env
OLLAMA_MODEL=mistral          # Change LLM
HF_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2  # Larger embeddings
RERANKER_MODEL=cross-encoder/ms-marco-TinyBERT-L-2-v2   # Faster reranker
CHUNK_SIZE=1024               # Larger chunks
RERANK_TOP_K=8                # More context for LLM
```

---

## 📈 Performance Notes

| Setting | Value | Impact |
|---------|-------|--------|
| Chunk size | 512 | Balance precision vs context |
| Retrieval top-k | 10 | Candidates before reranking |
| Rerank top-k | 5 | Passed to LLM prompt |
| Embed dim | 384 | MiniLM — fast & accurate |
| Neo4j vector index | cosine | Best for semantic similarity |
