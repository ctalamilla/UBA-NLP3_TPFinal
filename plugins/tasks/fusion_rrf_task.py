# plugins/tasks/fusion_rrf_task.py
from __future__ import annotations

import re, json, pickle
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from tasks.s3_utilities import download_to_tmp, read_text, upload_json
from tasks.bm25_index import BM25Index
from tasks.vector_pinecone import query_index
from tasks.fusion import rrf_combine  # tu función RRF


def _safe_get_doc_id(index: BM25Index, idx: int) -> str:
    if hasattr(index, "get_doc_id"):
        return index.get_doc_id(idx)
    if hasattr(index, "doc_ids"):
        return index.doc_ids[idx]
    if hasattr(index, "_doc_ids"):
        return index._doc_ids[idx]
    return ""


def _best_chunk_for_page(
    body: str,
    base_name: str,        # '22043_2025-10-01'
    page_num: int,         # 1
    query: str
) -> Optional[Dict[str, Any]]:
    """
    Dado el NDJSON del documento (body) y una page, elige el mejor chunk por
    intersección de tokens con la query. Retorna el objeto del chunk.
    """
    q_tokens = set((query or "").lower().split())
    best: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)

    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue

        if obj.get("page") != page_num:
            continue
        txt = (obj.get("text") or "").strip().lower()
        if not txt:
            continue

        toks = set(txt.split())
        overlap = float(len(q_tokens & toks))
        if overlap >= best[0]:
            best = (overlap, obj)

    return best[1]


def _snippet_for_any(
    bucket: str,
    prefix_chunks: str,
    any_id: str,              # puede ser chunk_id o page_id
    aws_conn_id: str,
    query: str = ""
) -> Dict[str, Any]:
    """
    Si 'any_id' tiene '::' buscamos el chunk exacto.
    Si es page_id (ej: '22043_2025-10-01_p1'), elegimos el mejor chunk de esa página.
    Devuelve: text, page, pdf_key, doc_id, boletin, fecha, op, classification, page_id, chunk_id
    """
    is_chunk = "::" in (any_id or "")
    if is_chunk:
        base = any_id.split("::", 1)[0]                 # '22043_2025-10-01'
        page_seg = any_id.split("::", 2)[1]             # 'p1'
        page_num = int(page_seg.lstrip("p") or "1")
    else:
        # page_id → '22043_2025-10-01_p1'
        base, page_tail = (any_id or "").rsplit("_p", 1)
        page_num = int((page_tail or "1"))

    key = f"{prefix_chunks.rstrip('/')}/{base}.ndjson"

    try:
        body = read_text(bucket=bucket, key=key, aws_conn_id=aws_conn_id)
    except Exception:
        return {"text": "", "page": None, "pdf_key": None, "doc_id": None,
                "boletin": None, "fecha": None, "op": None,
                "classification": None, "page_id": any_id, "chunk_id": None}

    if is_chunk:
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") == any_id:
                return {
                    "text": obj.get("text", ""),
                    "page": obj.get("page"),
                    "pdf_key": obj.get("source"),
                    "doc_id": obj.get("doc_id"),
                    "boletin": obj.get("boletin"),
                    "fecha": obj.get("fecha"),
                    "op": obj.get("op"),
                    "classification": obj.get("classification"),
                    "page_id": obj.get("doc_id"),
                    "chunk_id": obj.get("id"),
                }
        # si no apareció exacto, caemos a best-by-page
    # best chunk para la página
    best = _best_chunk_for_page(body=body, base_name=base, page_num=page_num, query=query)
    if best:
        return {
            "text": best.get("text", ""),
            "page": best.get("page"),
            "pdf_key": best.get("source"),
            "doc_id": best.get("doc_id"),
            "boletin": best.get("boletin"),
            "fecha": best.get("fecha"),
            "op": best.get("op"),
            "classification": best.get("classification"),
            "page_id": best.get("doc_id"),
            "chunk_id": best.get("id"),
        }

    return {"text": "", "page": page_num, "pdf_key": None, "doc_id": f"{base}_p{page_num}",
            "boletin": None, "fecha": None, "op": None, "classification": None,
            "page_id": f"{base}_p{page_num}", "chunk_id": None}


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
    prefix_chunks: str = "rag/chunks_op/2025/",
    # consulta
    query: str = "",
    # salida
    out_prefix: str = "rag/fusion/2025/",
    **kwargs
) -> Dict[str, Any]:
    if not query or not str(query).strip():
        print("⚠️ Query vacía.")
        return {"query": query, "results": []}

    # 1) BM25
    local_pkl = download_to_tmp(bucket=bucket_name, key=bm25_model_key, aws_conn_id=aws_conn_id)
    with open(local_pkl, "rb") as f:
        bm25_index: BM25Index = pickle.load(f)
    bm25_res = bm25_index.search(query, top_k=top_k_bm25)  # [(doc_global_idx, score), ...]
    bm25_page_ids = [_safe_get_doc_id(bm25_index, idx) for idx, _ in bm25_res if _safe_get_doc_id(bm25_index, idx)]

    # 2) Pinecone
    vec = query_index(
        index_name=pc_index_name,
        query=query,
        top_k=top_k_vec,
        model_name=model_name,
        namespace=pc_namespace,
    )
    vec_matches = vec.get("matches", []) if isinstance(vec, dict) else []
    # Convertimos chunk_id → page_id para fusionar en el mismo espacio que BM25
    vec_page_ids: List[str] = []
    seen = set()
    for m in vec_matches:
        cid = m.get("id") or ""
        if "::" in cid:
            base, pseg, *_ = cid.split("::")
            pid = f"{base}_{pseg}"  # base_pN
        else:
            pid = cid if "_p" in cid else f"{cid}_p1"
        if pid not in seen:
            seen.add(pid)
            vec_page_ids.append(pid)

    # 3) Fusión RRF
    fused_page_ids = rrf_combine(bm25_page_ids, vec_page_ids, k=float(rrf_k))[:top_k_fused]

    # 4) Armar resultados con snippet/meta
    results: List[Dict[str, Any]] = []
    for rank, pid in enumerate(fused_page_ids, start=1):
        meta = _snippet_for_any(
            bucket=bucket_name,
            prefix_chunks=prefix_chunks,
            any_id=pid,
            aws_conn_id=aws_conn_id,
            query=query,
        )
        snippet = " ".join((meta.get("text") or "").split())[:280]
        results.append({
            "rank": rank,
            "page_id": pid,
            "chunk_id": meta.get("chunk_id"),
            "snippet": snippet,
            "pdf_key": meta.get("pdf_key"),
            "doc_id": meta.get("doc_id"),
            "boletin": meta.get("boletin"),
            "fecha": meta.get("fecha"),
            "op": meta.get("op"),
            "classification": meta.get("classification"),
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
        print(f"#{r['rank']:02d}  {r['page_id']:>20}  ->  {r.get('chunk_id') or '-':>20}  src={r.get('pdf_key') or '-'}")
        if r.get("snippet"):
            print(f"      {r['snippet']}")

    return {"out_key": out_key, "count": len(results)}
