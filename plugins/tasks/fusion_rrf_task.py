# plugins/tasks/fusion_rrf_task.py
from __future__ import annotations

import re, json, pickle
from datetime import datetime
from typing import Dict, Any, List

from tasks.s3_utilities import download_to_tmp, read_text, upload_json
from tasks.bm25_index import BM25Index
from tasks.vector_pinecone import query_index
from tasks.fusion import rrf_combine  # ← TU función de la notebook

def _safe_get_doc_id(index: BM25Index, idx: int) -> str:
    if hasattr(index, "get_doc_id"):
        return index.get_doc_id(idx)
    if hasattr(index, "doc_ids"):
        return index.doc_ids[idx]
    if hasattr(index, "_doc_ids"):
        return index._doc_ids[idx]
    return ""

def _snippet_for_chunk(bucket: str, prefix_chunks: str, chunk_id: str, aws_conn_id: str) -> Dict[str, Any]:
    base = chunk_id.split("::", 1)[0]
    key = f"{prefix_chunks.rstrip('/')}/{base}.ndjson"
    try:
      body = read_text(bucket=bucket, key=key, aws_conn_id=aws_conn_id)
      for line in body.splitlines():
          line = line.strip()
          if not line:
              continue
          obj = json.loads(line)
          if obj.get("id") == chunk_id:
              return {
                  "text": obj.get("text", ""),
                  "page": obj.get("page"),
                  "pdf_key": obj.get("source"),
                  "doc_id": obj.get("doc_id"),
              }
    except Exception:
      pass
    return {"text": "", "page": None, "pdf_key": None, "doc_id": None}

def task_fusion_query(
    bucket_name: str,
    aws_conn_id: str,
    # BM25
    bm25_model_key: str,             # ej: "rag/models/2025/bm25.pkl"
    top_k_bm25: int = 50,
    # Pinecone
    pc_index_name: str = "boletines-2025",
    pc_namespace: str = "2025",
    top_k_vec: int = 50,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # RRF
    rrf_k: int = 60,
    top_k_fused: int = 10,
    # para snippets
    prefix_chunks: str = "rag/chunks/2025/",
    # consulta
    query: str = "",
    # salida
    out_prefix: str = "rag/fusion/2025/",
    **kwargs
) -> Dict[str, Any]:
    if not query or not str(query).strip():
        print("⚠️ Query vacía.")
        return {"query": query, "results": []}

    # 1) BM25: cargar y consultar
    local_pkl = download_to_tmp(bucket=bucket_name, key=bm25_model_key, aws_conn_id=aws_conn_id)
    with open(local_pkl, "rb") as f:
        bm25_index: BM25Index = pickle.load(f)
    bm25_res = bm25_index.search(query, top_k=top_k_bm25)  # [(idx, score), ...]
    bm25_ids = [_safe_get_doc_id(bm25_index, idx) for idx, _ in bm25_res if _safe_get_doc_id(bm25_index, idx)]

    # 2) Pinecone: consultar
    vec = query_index(
        index_name=pc_index_name,
        query=query,
        top_k=top_k_vec,
        model_name=model_name,
        namespace=pc_namespace,
    )
    vec_ids = [m["id"] for m in vec.get("matches", []) if m.get("id")]

    # 3) Fusión RRF EXACTA a tu notebook
    fused_ids = rrf_combine(bm25_ids, vec_ids, k=float(rrf_k))[:top_k_fused]

    # 4) Armar resultados con snippet
    results: List[Dict[str, Any]] = []
    for rank, cid in enumerate(fused_ids, start=1):
        meta = _snippet_for_chunk(bucket=bucket_name, prefix_chunks=prefix_chunks, chunk_id=cid, aws_conn_id=aws_conn_id)
        snippet = " ".join((meta.get("text") or "").split())[:280]
        results.append({
            "rank": rank,
            "chunk_id": cid,
            "page": meta.get("page"),
            "pdf_key": meta.get("pdf_key"),
            "doc_id": meta.get("doc_id"),
            "snippet": snippet,
        })

    # 5) Subir JSON
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_q = re.sub(r"[^A-Za-z0-9_-]+", "_", query.strip())[:60] or "query"
    out_key = f"{out_prefix.rstrip('/')}/fusion_{safe_q}_{ts}.json"
    payload = {
        "query": query,
        "params": {
            "top_k_bm25": top_k_bm25, "top_k_vec": top_k_vec,
            "rrf_k": rrf_k, "top_k_fused": top_k_fused,
            "pc_index": pc_index_name, "pc_namespace": pc_namespace,
            "bm25_model_key": bm25_model_key, "prefix_chunks": prefix_chunks,
            "model_name": model_name,
        },
        "results": results,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    upload_json(bucket=bucket_name, key=out_key, obj=payload, aws_conn_id=aws_conn_id)
    print(f"✅ Fusión RRF subida a s3://{bucket_name}/{out_key}  (top={len(results)})")

    # Log amigable
    print(f"🔀 RRF para {query!r}")
    for r in results:
        print(f"#{r['rank']:02d}  {r['chunk_id']}  page={r.get('page')}  src={r.get('pdf_key') or '-'}")
        if r.get("snippet"):
            print(f"      {r['snippet']}")

    return {"out_key": out_key, "count": len(results)}
