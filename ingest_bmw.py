"""
ingest_bmw.py — Quick script to ingest the BMW 7 Series PDF.
Run this ONCE after docker-compose up, before starting the app.

Usage:
    python ingest_bmw.py
    python ingest_bmw.py --pdf "path/to/your.pdf"
"""
import sys
import argparse
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description="Ingest BMW PDF into Graph RAG")
    parser.add_argument(
        "--pdf",
        default=r"C:\Users\91956\Downloads\2023-bmw-7-30-62.pdf",
        help="Path to the BMW PDF file",
    )
    args = parser.parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        # Try data/pdfs folder
        alt = Path("data/pdfs") / pdf_path.name
        if alt.exists():
            pdf_path = alt
        else:
            print(f"❌ PDF not found: {pdf_path}")
            print("   Copy your PDF to ./data/pdfs/ or pass --pdf <path>")
            sys.exit(1)

    print(f"📄 Ingesting: {pdf_path}")
    print("   This may take 1-3 minutes depending on PDF size …\n")

    from backend.pipeline.ingestion_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    result = pipeline.run(pdf_path)

    if result["status"] == "success":
        print(f"\n✅ Ingestion complete!")
        print(f"   File    : {result['file']}")
        print(f"   Pages   : {result['pages']}")
        print(f"   Chunks  : {result['chunks']}")
        print(f"   Doc ID  : {result['doc_id']}")
        print(f"\n🚀 Now run:  python start.py")
    else:
        print(f"\n❌ Ingestion failed: {result.get('message')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
