# plugins/tasks/pinecone_upsert_op_task.py
from __future__ import annotations
from typing import Dict, Any, Optional

from tasks.vector_pinecone_op import upsert_vectors_from_ndjson_op

def task_pinecone_upsert_op(
    bucket_name: str,
    prefix_chunks: str,             # ej: "rag/chunks_op/2025/"
    aws_conn_id: Optional[str],
    index_name: str,                # ej: "boletines-2025"
    namespace: Optional[str],       # ej: "2025"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 128,
    min_chars: int = 20,            # filtra textos muy cortos
    normalize: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Upsert específico para el pipeline por-OP (no pisa el task original).
    Lee NDJSON desde `prefix_chunks` y sube a Pinecone.
    """
    out = upsert_vectors_from_ndjson_op(
        bucket=bucket_name,
        prefix_chunks=prefix_chunks,
        aws_conn_id=aws_conn_id,
        index_name=index_name,
        namespace=namespace,
        model_name=model_name,
        batch_size=batch_size,
        min_chars=min_chars,
        normalize=normalize,
    )
    print(f"✅ Upsert Pinecone (OP): {out}")
    return out
