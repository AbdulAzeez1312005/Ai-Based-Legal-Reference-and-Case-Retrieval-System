"""Build generic provision-aware chunks and safely upsert them to Pinecone."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from legal_rag import records_for_folder

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".cache" / "huggingface"))


def connect_index(dimension: int):
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in backend/.env.")
    name = os.getenv("PINECONE_INDEX", "legal-reference-m4")
    client = Pinecone(api_key=api_key)
    if name not in client.list_indexes().names():
        print(f"Creating Pinecone vector index {name!r} ({dimension} dimensions).")
        client.create_index(
            name=name, dimension=dimension, metric="cosine",
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
            ),
        )
        while not client.describe_index(name).status["ready"]:
            time.sleep(1)
    description = client.describe_index(name)
    actual_dimension = getattr(description, "dimension", dimension)
    if actual_dimension != dimension:
        raise RuntimeError(
            f"Index {name!r} has dimension {actual_dimension}; "
            f"the selected embedding model produces {dimension}. "
            "Use an index with the matching dimension."
        )
    return client.Index(name), name


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest supported legal files from backend/data2 into Pinecone."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse documents and report chunks without contacting Pinecone.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation; Pinecone records are only upserted.")
    args = parser.parse_args()

    records, report = records_for_folder(BASE_DIR / "data2")
    print(f"Sources found: {len(report['files'])}; records prepared: {report['records']}")
    for row in report["files"]:
        print(f"  {row['source']}: {row['records']} chunks, "
              f"{row['provisions']} detected provisions ({row['document_type']})")
    for row in report["skipped"]:
        print(f"  SKIPPED {row['source']}: {row['reason']}")
    if not records:
        print("No extractable legal text was found; nothing was uploaded.")
        return 1
    if args.dry_run:
        print("Dry run complete. Pinecone was not changed.")
        return 0
    if not args.yes:
        answer = input(
            f"Upsert {len(records)} deterministic records into "
            f"{os.getenv('PINECONE_INDEX', 'legal-reference-m4')!r}? (y/n): "
        ).strip().lower()
        if answer != "y":
            print("Cancelled. Pinecone was not changed.")
            return 0

    from sentence_transformers import SentenceTransformer

    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    dimension = int(embedder.get_embedding_dimension())
    index, index_name = connect_index(dimension)
    namespace = os.getenv("PINECONE_NAMESPACE", "")
    batch_size = max(1, int(os.getenv("UPSERT_BATCH_SIZE", "64")))
    total = 0
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        vectors = embedder.encode(
            [row["text"] for row in batch],
            batch_size=min(32, len(batch)),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        payload = [
            {"id": row["id"], "values": vector, "metadata": row["metadata"]}
            for row, vector in zip(batch, vectors)
        ]
        index.upsert(vectors=payload, namespace=namespace)
        total += len(payload)
        print(f"Upserted {total}/{len(records)}")
    stats = index.describe_index_stats()
    namespace_stats = getattr(stats, "namespaces", {}).get(namespace, None)
    print(f"Finished. Upserted {total} vectors to {index_name}/"
          f"{namespace or '(default namespace)'}.")
    print(f"Namespace vector count reported by Pinecone: "
          f"{getattr(namespace_stats, 'vector_count', 'unknown')}")
    print("No Pinecone vectors were deleted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
