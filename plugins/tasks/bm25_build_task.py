# plugins/tasks/bm25_build_task.py
from __future__ import annotations

import os
import json
import pickle
from typing import Dict, Any, List, DefaultDict
from collections import defaultdict
from datetime import datetime

from tasks.s3_utilities import list_keys, read_text, upload_bytes, upload_json
from tasks.documents import Document
from tasks.bm25_index import BM25Index  # <- de tu notebook copiado a plugins/tasks

def task_build_bm25_from_ndjson(
    bucket_name: str,
    prefix_chunks: str,   # p.ej. "rag/chunks/2025/"
    prefix_models: str,   # p.ej. "rag/models/2025/"
    aws_conn_id: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Lee NDJSON de chunks en s3://bucket/prefix_chunks,
    arma docs + chunks_map y construye BM25Index.
    Guarda bm25.pkl en s3://bucket/prefix_models.
    """
    ndjson_keys: List[str] = list_keys(
        bucket=bucket_name, prefix=prefix_chunks, aws_conn_id=aws_conn_id, suffix=".ndjson"
    )
    if not ndjson_keys:
        print(f"ℹ️ No hay NDJSON en s3://{bucket_name}/{prefix_chunks}")
        return {"emb": 0, "bm25": 0, "models_prefix": prefix_models}

    # Reconstruir {doc_id: [chunk_texts]} y metadatos mínimos para Document
    chunks_map: DefaultDict[str, List[str]] = defaultdict(list)
    doc_meta: Dict[str, Dict[str, Any]] = {}  # doc_id -> {source, page?}

    total_lines = 0
    for key in sorted(ndjson_keys):
        body = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id") or ""
            text   = obj.get("text") or ""
            source = obj.get("source") or ""
            page   = obj.get("page", None)
            if not doc_id or not text:
                continue
            chunks_map[doc_id].append(text)
            if doc_id not in doc_meta:
                doc_meta[doc_id] = {"source": source, "page": page}
            total_lines += 1

    if not chunks_map:
        print("ℹ️ No hay chunks válidos para indexar.")
        return {"emb": 0, "bm25": 0, "models_prefix": prefix_models}

    # Armar lista de Document (texto base opcional: concatenación de chunks)
    docs: List[Document] = []
    for doc_id, chunk_list in chunks_map.items():
        meta = doc_meta.get(doc_id, {})
        # Texto base: concatenación (no usado por BM25 si trabaja sobre chunks_map, pero es coherente)
        base_text = "\n\n".join(chunk_list)
        docs.append(
            Document(
                id=doc_id,
                text=base_text,
                source=os.path.basename(meta.get("source", "")),
                page=meta.get("page"),
            )
        )

    # Construir índice BM25
    print(f"🔧 Construyendo BM25Index sobre {len(docs)} docs y {sum(len(v) for v in chunks_map.values())} chunks...")
    index = BM25Index(docs, chunks_map)

    # Guardar como pickle
    blob = pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL)
    out_key = f"{prefix_models.rstrip('/')}/bm25.pkl"
    upload_bytes(bucket=bucket_name, key=out_key, data=blob, aws_conn_id=aws_conn_id)
    print(f"✅ BM25 guardado en s3://{bucket_name}/{out_key}")

    # Manifest/summary
    summary = {
        "bucket": bucket_name,
        "prefix_chunks": prefix_chunks,
        "prefix_models": prefix_models,
        "bm25_key": out_key,
        "docs": len(docs),
        "chunks": sum(len(v) for v in chunks_map.values()),
        "lines_read": total_lines,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    manifest_key = f"{prefix_models.rstrip('/')}/_bm25_manifest.json"
    upload_json(bucket=bucket_name, key=manifest_key, obj=summary, aws_conn_id=aws_conn_id)
    print(f"📄 Manifest: s3://{bucket_name}/{manifest_key}")

    return {"bm25_key": out_key, "manifest_key": manifest_key, "docs": summary["docs"], "chunks": summary["chunks"]}
