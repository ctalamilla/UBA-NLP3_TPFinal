# plugins/tasks/classify_chunks_agent_task.py
from __future__ import annotations
import json
import os
import re
from typing import Dict, Any, List, Optional

from tasks.s3_utilities import list_keys, read_text, upload_text
from tasks.agent_classifier import ClassifierAgent

def _derive_ids(rec: dict) -> (str, int):
    """
    Intenta derivar doc_id base e índice del chunk.
    - Si hay 'doc_id' y 'chunk_id' los usa.
    - Si no, intenta parsear con patrones comunes (_cNN o ::cNN).
    """
    doc_id = rec.get("doc_id") or ""
    chunk_id = rec.get("chunk_id") or ""

    # idx desde chunk_id
    idx = 0
    m = re.search(r"[cC](\d+)$", os.path.basename(chunk_id))
    if not m:
        m = re.search(r"::[cC](\d+)$", chunk_id)
    if m:
        idx = int(m.group(1))

    # si no hay doc_id, usa base del chunk_id
    if not doc_id:
        doc_id = chunk_id.split("::", 1)[0] or os.path.splitext(os.path.basename(chunk_id))[0]

    return doc_id, idx

def task_classify_chunks_agent(
    bucket_name: str,
    src_prefix: str = "rag/chunks/2025/",
    dst_prefix: str = "rag/chunks_labeled/2025/",
    aws_conn_id: str = "minio_s3",
    **kwargs
) -> Dict[str, Any]:
    """
    Lee NDJSON con chunks, clasifica cada chunk (heurística + LLM fallback),
    y escribe NDJSON con campo 'classification' agregado.
    """
    keys = list_keys(bucket=bucket_name, prefix=src_prefix, aws_conn_id=aws_conn_id, suffix=".ndjson")
    if not keys:
        print(f"ℹ️ No hay NDJSON en s3://{bucket_name}/{src_prefix}")
        return {"processed": 0, "dst_prefix": dst_prefix, "items": []}

    agent = ClassifierAgent()  # usa OPENAI_API_KEY si está presente
    results: List[Dict[str, Any]] = []

    for key in sorted(keys):
        blob = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
        out_lines: List[str] = []
        count = 0

        for line in blob.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "") or ""

            base_doc_id, idx = _derive_ids(rec)
            cls = agent.classify_chunk(text, doc_id=base_doc_id, idx=idx)

            # guardamos tal cual pide tu notebook, en bloque 'classification'
            rec["classification"] = cls
            out_lines.append(json.dumps(rec, ensure_ascii=False))
            count += 1

        base = os.path.basename(key)
        out_key = f"{dst_prefix.rstrip('/')}/{base}"
        upload_text(bucket=bucket_name, key=out_key, text="\n".join(out_lines), aws_conn_id=aws_conn_id)
        print(f"✅ Clasificados {count} chunks → s3://{bucket_name}/{out_key}")
        results.append({"src": key, "dst": out_key, "chunks": count})

    return {"processed": len(results), "dst_prefix": dst_prefix, "items": results}
