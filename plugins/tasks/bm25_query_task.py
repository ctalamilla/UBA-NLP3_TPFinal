# plugins/tasks/bm25_query_task.py
from __future__ import annotations

import os
import re
import pickle
from typing import Dict, Any, List, Optional

from tasks.s3_utilities import download_to_tmp
# Import para que pickle resuelva la clase si fue pickeada como plugins.tasks.bm25_index.BM25Index
from tasks.bm25_index import BM25Index  # noqa: F401

_PAGE_RE = re.compile(r"^(?P<base>.+)_p(?P<page>\d+)$")

def _parse_doc_id(doc_id: Optional[str]):
    m = _PAGE_RE.match(doc_id or "")
    if not m:
        return doc_id, None
    return m.group("base"), int(m.group("page"))

def _safe_get_doc_id(index, idx: int) -> Optional[str]:
    if hasattr(index, "get_doc_id"):
        return index.get_doc_id(idx)
    if hasattr(index, "doc_ids"):
        return index.doc_ids[idx]
    if hasattr(index, "_doc_ids"):
        return index._doc_ids[idx]
    return None

def _safe_get_chunk(index, idx: int) -> str:
    if hasattr(index, "get_chunk"):
        return index.get_chunk(idx)
    if hasattr(index, "chunks"):
        return index.chunks[idx]
    if hasattr(index, "_chunks"):
        return index._chunks[idx]
    # último recurso: si rank_bm25 estuviera expuesto (no suele servir para texto)
    if hasattr(index, "_bm25") and hasattr(index._bm25, "corpus"):
        try:
            # corpus es tokenizado, no texto crudo; lo unimos como fallback
            return " ".join(index._bm25.corpus[idx])
        except Exception:
            pass
    return ""

def task_query_bm25(
    bucket_name: str,
    model_key: str,                 # ej: "rag/models/2025/bm25.pkl"
    aws_conn_id: str,
    query: str,
    top_k: int = 5,
    prefix_pdfs: Optional[str] = None,  # ej: "boletines/2025/"
    **kwargs
) -> Dict[str, Any]:
    if not query or not str(query).strip():
        print("⚠️ Query vacía. No se ejecuta búsqueda.")
        return {"query": query, "results": []}

    # 1) Descargar modelo
    local_pkl = download_to_tmp(bucket=bucket_name, key=model_key, aws_conn_id=aws_conn_id)
    if not os.path.exists(local_pkl):
        raise FileNotFoundError(f"No se encontró el modelo local tras descargar: {local_pkl}")

    # 2) Cargar índice
    with open(local_pkl, "rb") as f:
        index = pickle.load(f)

    if not hasattr(index, "search"):
        raise TypeError("El objeto cargado no parece un índice BM25 válido (falta método 'search').")

    # 3) Buscar
    results = index.search(query, top_k=top_k)  # -> List[Tuple[chunk_idx, score]]

    # 4) Formatear salida
    print(f"🔎 Query: {query!r} | Top-{top_k} resultados")
    out_items: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(results, start=1):
        doc_id = _safe_get_doc_id(index, idx)
        chunk  = _safe_get_chunk(index, idx)
        base, page = _parse_doc_id(doc_id or "")
        pdf_key = f"{prefix_pdfs.rstrip('/')}/{base}.pdf" if (prefix_pdfs and base) else None
        snippet = " ".join((chunk or "").split())[:240]

        print(f"#{rank:02d}  score={float(score):.4f}  doc_id={doc_id or '-'}  page={page}  pdf={pdf_key or '-'}")
        if snippet:
            print(f"      {snippet}")

        out_items.append({
            "rank": rank,
            "score": float(score),
            "doc_id": doc_id,
            "page": page,
            "pdf_key": pdf_key,
            "snippet": snippet,
        })

    return {
        "query": query,
        "top_k": top_k,
        "results": out_items,
        "model_key": model_key,
    }
