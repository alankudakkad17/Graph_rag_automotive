"""
start.py — One-command launcher for the full Graph RAG stack.
Starts FastAPI backend + Gradio frontend in parallel subprocesses.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent

def run():
    print("=" * 60)
    print("  🚗 Automotive Graph RAG System — Starting Up")
    print("=" * 60)

    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    # Start FastAPI
    print("\n▶  Starting FastAPI backend on http://localhost:8000 …")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT,
        env=env,
    )

    time.sleep(3)   # give backend time to start

    # Start Gradio
    print("▶  Starting Gradio frontend on http://localhost:7860 …\n")
    frontend = subprocess.Popen(
        [sys.executable, "frontend/app.py"],
        cwd=ROOT,
        env=env,
    )

    print("=" * 60)
    print("  FastAPI  → http://localhost:8000/docs")
    print("  Gradio   → http://localhost:7860")
    print("  Neo4j    → http://localhost:7474")
    print("\n  Press Ctrl+C to stop all services.")
    print("=" * 60)

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n⏹  Shutting down …")
        backend.terminate()
        frontend.terminate()
        print("✓  Done.")

if __name__ == "__main__":
    run()
