from __future__ import annotations
from typing import Dict, Any
from tasks.vector_pinecone import upsert_vectors_from_ndjson

def task_pinecone_upsert(
    bucket_name: str,
    prefix_chunks: str,   # "rag/chunks/2025/"
    aws_conn_id: str,
    index_name: str,      # "boletines-2025"
    namespace: str,       # "2025"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 128,
    **kwargs
) -> Dict[str, Any]:
    out = upsert_vectors_from_ndjson(
        bucket=bucket_name,
        prefix_chunks=prefix_chunks,
        aws_conn_id=aws_conn_id,
        index_name=index_name,
        namespace=namespace,
        model_name=model_name,
        batch_size=batch_size,
    )
    print(f"✅ Upsert Pinecone: {out}")
    return out
