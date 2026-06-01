"""
app.py — Gradio Frontend for the Automotive Graph RAG System
─────────────────────────────────────────────────────────────
Tabs:
  🚗 Chat          — Streaming conversational Q&A with pipeline selector
  📄 Ingest        — PDF upload + batch scan trigger
  🕸  Graph View   — Interactive Neo4j graph visualization (Plotly)
  📊 Stats         — System health & index statistics
  💡 Examples      — Quick-start sample queries for the BMW 7 Series

Evaluation is a separate offline test script — run: python evaluate.py
"""
from __future__ import annotations

import os
import sys
import json
import traceback
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from frontend.api_client import (
    query as api_query,
    upload_pdf,
    get_graph_stats,
    get_graph_data,
    health_check,
    scan_data_dir,
)

# ─── Theme & Styling ──────────────────────────────────────────────────────────
BMW_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    button_primary_background_fill="#1c69d4",
    button_primary_background_fill_hover="#0d4fa3",
    button_primary_text_color="white",
)

PIPELINE_CHOICES = ["auto", "hybrid", "graph_only", "vector_only"]

EXAMPLE_QUERIES = [
    "What is the horsepower of the 2023 BMW 760i xDrive?",
    "What are the dimensions and wheelbase of the BMW 7 Series?",
    "List all safety and ADAS features available on the BMW 7 Series.",
    "How does the BMW iDrive 8.5 infotainment system work?",
    "What engine options are available for the 2023 BMW 7 Series?",
    "Does the BMW 7 Series have a plug-in hybrid variant?",
    "What is the 0-60 mph time for the BMW M760e?",
    "What suspension systems are available on the 7 Series?",
    "Compare the 740i and 760i engine specifications.",
    "What are the standard comfort features of the BMW 7 Series?",
]


# ── Streaming Chat Logic ──────────────────────────────────────────────────────
FASTAPI_BASE = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")


def chat_stream(
    message: str,
    history: list[list[str]],
    pipeline: str,
):
    """
    Streaming chat handler — connects to /stream/query SSE endpoint.
    Yields (history, metadata) tuples as tokens arrive so Gradio
    renders them word-by-word in real time.
    """
    if not message.strip():
        yield history, ""
        return

    # Convert Gradio history to API format
    api_history = []
    for h in history:
        if h[0]:
            api_history.append({"role": "user",    "content": h[0]})
        if h[1]:
            api_history.append({"role": "assistant","content": h[1]})

    # Append user message with empty assistant slot
    history = history + [[message, ""]]
    meta = "*Connecting to pipeline…*"
    yield history, meta

    accumulated  = ""
    stage_lines  = []
    final_meta   = ""

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{FASTAPI_BASE}/stream/query",
                json={"query": message, "chat_history": api_history},
                headers={"Accept": "text/event-stream"},
            ) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if not raw:
                        continue

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype   = event.get("type", "")
                    content = event.get("content", "")

                    if etype == "stage":
                        stage_lines.append(content)
                        meta = "\n".join(stage_lines[-4:])   # show last 4 stages
                        yield history, meta

                    elif etype == "token":
                        accumulated += content
                        history[-1][1] = accumulated
                        yield history, meta

                    elif etype == "metadata":
                        pipe_used   = content.get("pipeline", "")
                        pipe_reason = content.get("pipeline_reason", "")
                        confidence  = content.get("confidence", 0.0)
                        intent      = content.get("query_intent", "")
                        n_results   = content.get("num_results", 0)
                        sources     = content.get("sources", [])

                        src_text = ""
                        if sources:
                            src_text = "\n\n📎 **Sources:** " + " | ".join(
                                f"`{s}`" for s in sources[:5]
                            )
                        history[-1][1] = accumulated + src_text

                        final_meta = (
                            f"**Pipeline:** `{pipe_used}` — {pipe_reason}\n"
                            f"**Intent:** `{intent}` | "
                            f"**Confidence:** `{confidence:.2%}` | "
                            f"**Results used:** `{n_results}`"
                        )
                        yield history, final_meta

                    elif etype == "refused":
                        history[-1][1] = content
                        final_meta = "⚠️ Query refused by relevance gate"
                        yield history, final_meta

                    elif etype == "error":
                        history[-1][1] = f"❌ **Error:** {content}"
                        yield history, f"Error: {content}"

                    elif etype == "done":
                        yield history, final_meta or meta
                        return

    except httpx.ConnectError:
        history[-1][1] = "❌ **Cannot connect to FastAPI backend.**\n\nMake sure it is running on port 8000."
        yield history, "❌ Connection failed"
    except Exception as exc:
        history[-1][1] = f"❌ **Unexpected error:** {exc}"
        yield history, f"Error: {exc}"


def clear_chat() -> tuple:
    return [], []


# ── Evaluation Logic ──────────────────────────────────────────────────────────


# ── Upload Logic ──────────────────────────────────────────────────────────────
def handle_upload(files) -> str:
    if not files:
        return "⚠️ No file selected."
    results = []
    for file in files:
        try:
            result = upload_pdf(file.name)
            results.append(
                f"✅ **{result['file']}** — "
                f"{result['chunks']} chunks from {result['pages']} pages"
            )
        except Exception as exc:
            results.append(f"❌ Failed: {exc}")
    return "\n\n".join(results)


def handle_scan() -> str:
    try:
        result = scan_data_dir()
        ingested = result.get("ingested", [])
        lines = [f"Scanned data directory — {len(ingested)} PDFs processed:"]
        for r in ingested:
            if r.get("status") == "success":
                lines.append(f"  ✅ {r['file']} — {r['chunks']} chunks")
            else:
                lines.append(f"  ❌ {r.get('message', 'unknown error')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Scan failed: {exc}"


# ── Graph Visualization ───────────────────────────────────────────────────────
NODE_COLORS = {
    "Vehicle":        "#1c69d4",
    "Engine":         "#e74c3c",
    "Feature":        "#2ecc71",
    "System":         "#f39c12",
    "Component":      "#9b59b6",
    "Specification":  "#1abc9c",
    "Recall":         "#e67e22",
    "DiagnosticCode": "#c0392b",
    "Document":       "#7f8c8d",
    "Chunk":          "#bdc3c7",
}


def build_graph_figure(limit: int = 120) -> go.Figure:
    """Build a Plotly force-directed graph."""
    try:
        data = get_graph_data(limit=limit)
    except Exception as exc:
        fig = go.Figure()
        fig.add_annotation(text=f"Could not load graph: {exc}",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color="red"))
        return fig

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        fig = go.Figure()
        fig.add_annotation(text="No graph data found. Please ingest a PDF first.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14))
        return fig

    # Assign positions using a simple circular layout
    import math
    id_to_pos: dict[str, tuple[float, float]] = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        # Group by type in rough rings
        radius = 1.0
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        id_to_pos[node["id"]] = (x, y)

    # Edge traces
    edge_x, edge_y, edge_labels = [], [], []
    for edge in edges:
        src = id_to_pos.get(edge["source"])
        tgt = id_to_pos.get(edge["target"])
        if src and tgt:
            edge_x += [src[0], tgt[0], None]
            edge_y += [src[1], tgt[1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="#aaaaaa"),
        hoverinfo="none",
    )

    # Node traces by type
    type_groups: dict[str, list] = {}
    for node in nodes:
        t = node.get("type", "Unknown")
        type_groups.setdefault(t, []).append(node)

    node_traces = []
    for ntype, group_nodes in type_groups.items():
        xs = [id_to_pos[n["id"]][0] for n in group_nodes if n["id"] in id_to_pos]
        ys = [id_to_pos[n["id"]][1] for n in group_nodes if n["id"] in id_to_pos]
        labels = [n.get("label", n["id"])[:30] for n in group_nodes if n["id"] in id_to_pos]
        color = NODE_COLORS.get(ntype, "#95a5a6")
        node_traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                name=ntype,
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                marker=dict(size=14, color=color, line=dict(width=1, color="white")),
                hovertemplate=f"<b>{ntype}</b><br>%{{text}}<extra></extra>",
            )
        )

    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            title=dict(
                text="🕸 BMW Automotive Knowledge Graph",
                font=dict(size=18, color="#1c69d4"),
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="white",
            height=650,
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#ddd",
                borderwidth=1,
            ),
        ),
    )
    return fig


def refresh_graph(limit: int) -> go.Figure:
    return build_graph_figure(int(limit))


# ── Stats Panel ───────────────────────────────────────────────────────────────
def get_stats_text() -> str:
    try:
        stats = get_graph_stats()
        health = health_check()

        lines = [
            "## 📊 System Statistics\n",
            "### Neo4j Graph Nodes",
        ]
        for label, count in stats.get("nodes", {}).items():
            bar = "█" * min(count, 30)
            lines.append(f"  `{label:<20}` {count:>5}  {bar}")

        lines.append("\n### Relationships")
        for rel, count in stats.get("relationships", {}).items():
            lines.append(f"  `{rel:<25}` {count:>5}")

        lines.append(f"\n### ChromaDB Vector Store")
        lines.append(f"  Total chunks indexed: **{stats.get('total_chunks', 0)}**")
        lines.append(f"  Collection: `{stats.get('collection', 'N/A')}`")

        lines.append("\n### System Health")
        lines.append(f"  Neo4j:    {'✅' if health.get('neo4j') else '❌'}")
        lines.append(f"  ChromaDB: {'✅' if health.get('chromadb') else '❌'}")
        lines.append(f"  Ollama:   {'✅' if health.get('ollama') else '❌'}")
        lines.append(f"  LLM Model: `{health.get('model', 'N/A')}`")

        return "\n".join(lines)
    except Exception as exc:
        return f"❌ Could not fetch stats: {exc}\n\nMake sure the FastAPI backend is running."


# ── Build Gradio UI ───────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        theme=BMW_THEME,
        title="🚗 Automotive Graph RAG — BMW 7 Series",
        css="""
            .gr-chatbot { border-radius: 12px; }
            .meta-box { font-size: 0.85em; color: #555; background: #f0f4f8;
                        padding: 8px 12px; border-radius: 8px; }
            footer { display: none !important; }
            .tab-nav button { font-weight: 600; }
            #title-row { text-align: center; padding: 8px 0 4px 0; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="title-row">
          <h1 style="color:#1c69d4; margin:0;">
            🚗 Automotive Graph RAG System
          </h1>
          <p style="color:#555; margin:4px 0 0 0;">
            BMW 7 Series 2023 · Powered by Neo4j + ChromaDB + LangGraph + Ollama
          </p>
        </div>
        """)

        with gr.Tabs():
            # ══════════════════════════════════════════════════════════════════
            #  TAB 1 — Chat
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("🚗 Chat"):
                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="BMW 7 Series Assistant",
                            height=500,
                            bubble_full_width=False,
                            avatar_images=(
                                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BMW.svg/60px-BMW.svg.png",
                                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BMW.svg/60px-BMW.svg.png",
                            ),
                        )
                        meta_box = gr.Markdown(
                            value="*Pipeline info will appear here after first query.*",
                            elem_classes=["meta-box"],
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask anything about the BMW 7 Series 2023…",
                                label="",
                                scale=5,
                                container=False,
                            )
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)
                            clear_btn = gr.Button("Clear 🗑", scale=1)

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")
                        pipeline_sel = gr.Radio(
                            choices=PIPELINE_CHOICES,
                            value="auto",
                            label="Retrieval Pipeline",
                            info="'auto' lets the agent decide",
                        )
                        gr.Markdown("---")
                        gr.Markdown("### 💡 Example Queries")
                        for ex in EXAMPLE_QUERIES[:6]:
                            gr.Button(ex[:55] + ("…" if len(ex) > 55 else ""),
                                      size="sm", variant="secondary").click(
                                fn=lambda q=ex: q,
                                outputs=msg_input,
                            )

                # ── Event handlers (streaming) ────────────────────────────────
                send_btn.click(
                    fn=chat_stream,
                    inputs=[msg_input, chatbot, pipeline_sel],
                    outputs=[chatbot, meta_box],
                ).then(
                    fn=lambda: "",
                    outputs=[msg_input],
                )
                msg_input.submit(
                    fn=chat_stream,
                    inputs=[msg_input, chatbot, pipeline_sel],
                    outputs=[chatbot, meta_box],
                ).then(
                    fn=lambda: "",
                    outputs=[msg_input],
                )
                clear_btn.click(
                    fn=clear_chat,
                    outputs=[chatbot, chatbot],
                )

            # ══════════════════════════════════════════════════════════════════
            #  TAB 2 — Ingest
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("📄 Ingest Documents"):
                gr.Markdown("""
                ## Upload BMW Documentation
                Upload PDF files (spec sheets, owner manuals, service guides).
                The system will extract text, generate embeddings, and build the knowledge graph.
                """)
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload PDF(s)",
                            file_types=[".pdf"],
                            file_count="multiple",
                        )
                        upload_btn = gr.Button("📤 Ingest PDF(s)", variant="primary")
                        upload_result = gr.Markdown(label="Ingestion Result")
                        upload_btn.click(
                            fn=handle_upload,
                            inputs=[file_upload],
                            outputs=[upload_result],
                        )
                    with gr.Column():
                        gr.Markdown("### Batch Ingest from Data Directory")
                        gr.Markdown(
                            "Scans `./data/pdfs/` folder and ingests all PDFs found."
                        )
                        scan_btn = gr.Button("🔍 Scan & Ingest Data Folder", variant="secondary")
                        scan_result = gr.Markdown()
                        scan_btn.click(
                            fn=handle_scan,
                            outputs=[scan_result],
                        )

                gr.Markdown("""
                ---
                ### Ingestion Pipeline Steps
                1. **PDF Parsing** — pdfplumber extracts clean text per page
                2. **Chunking** — RecursiveCharacterTextSplitter (512 tokens, 64 overlap)
                3. **Embedding** — `sentence-transformers/all-MiniLM-L6-v2` (local, free)
                4. **ChromaDB** — vectors stored for semantic search
                5. **Entity Extraction** — BMW-specific regex + spaCy NER
                6. **Graph Building** — Nodes & edges written to Neo4j
                """)

            # ══════════════════════════════════════════════════════════════════
            #  TAB 3 — Graph Visualization
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("🕸 Knowledge Graph"):
                with gr.Row():
                    limit_slider = gr.Slider(
                        minimum=20, maximum=300, value=120, step=10,
                        label="Max nodes to display",
                    )
                    refresh_btn = gr.Button("🔄 Refresh Graph", variant="primary")

                graph_plot = gr.Plot(
                    label="BMW Knowledge Graph",
                    value=build_graph_figure(120),
                )

                gr.Markdown("""
                **Node colours:**
                🔵 Vehicle  🔴 Engine  🟢 Feature  🟡 System  🟣 Component
                🟦 Specification  🟠 Recall  🔴 DiagnosticCode  ⚫ Document
                """)

                refresh_btn.click(
                    fn=refresh_graph,
                    inputs=[limit_slider],
                    outputs=[graph_plot],
                )
                limit_slider.change(
                    fn=refresh_graph,
                    inputs=[limit_slider],
                    outputs=[graph_plot],
                )

            # ══════════════════════════════════════════════════════════════════
            #  TAB 4 — Stats
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("📊 System Stats"):
                refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                stats_md = gr.Markdown(value=get_stats_text())
                refresh_stats_btn.click(fn=get_stats_text, outputs=[stats_md])

                gr.Markdown("""
                ---
                ### Architecture Overview

                ```
                User Query
                    │
                    ▼
                ┌─────────────────────────────────────────┐
                │            LangGraph Agent               │
                │                                         │
                │  classify → expand → select_pipeline    │
                │       ↓                                 │
                │  ┌──────────┐    ┌─────────────┐        │
                │  │ Neo4j    │    │  ChromaDB   │        │
                │  │ Graph    │    │  Vectors    │        │
                │  │Retriever │    │  Retriever  │        │
                │  └──────────┘    └─────────────┘        │
                │       └─────────────┘                   │
                │             ▼                           │
                │    Cross-Encoder Reranker               │
                │             ▼                           │
                │    Prompt Optimizer                     │
                │             ▼                           │
                │    Ollama LLM (llama3)                  │
                │             ▼                           │
                │    Hallucination Guard                  │
                └─────────────────────────────────────────┘
                    │
                    ▼
                  Answer + Sources + Pipeline Metadata
                ```
                """)

            # ══════════════════════════════════════════════════════════════════
            #  TAB 5 — Examples
            # ══════════════════════════════════════════════════════════════════
            with gr.TabItem("💡 Examples"):
                gr.Markdown("## Sample Queries for BMW 7 Series 2023\nClick any query to copy it to the chat.")
                for cat, queries in {
                    "🏎 Performance": [
                        "What is the 0-60 mph time for the 2023 BMW 760i xDrive?",
                        "How much horsepower does the M760e produce?",
                        "What is the top speed of the BMW 740i?",
                    ],
                    "📐 Specifications": [
                        "What are the exterior dimensions of the BMW 7 Series?",
                        "What is the wheelbase of the 2023 BMW 7 Series?",
                        "How much cargo space does the BMW 7 Series have?",
                    ],
                    "⚙️ Technology": [
                        "What ADAS features are available on the 7 Series?",
                        "How does BMW iDrive 8.5 work?",
                        "What is the BMW Digital Key Plus feature?",
                    ],
                    "🔋 Powertrain": [
                        "What engine options are in the 2023 BMW 7 Series lineup?",
                        "Explain the BMW M760e hybrid system.",
                        "What is the electric range of the BMW i7?",
                    ],
                    "🛡 Safety": [
                        "What safety features come standard on the BMW 7 Series?",
                        "How does the BMW Emergency Call feature work?",
                    ],
                }.items():
                    gr.Markdown(f"### {cat}")
                    for q in queries:
                        gr.Markdown(f"- *{q}*")

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False,
        favicon_path=None,
        show_api=False,
    )
