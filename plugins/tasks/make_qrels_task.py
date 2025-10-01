# plugins/tasks/make_qrels_task.py
from __future__ import annotations

import os
import csv
import io
import pickle
from typing import Dict, Any, List, Optional, Set

from tasks.s3_utilities import download_to_tmp, upload_bytes, list_keys, read_text
from tasks.bm25_index import BM25Index  # noqa

def _safe_get_doc_id(index, idx: int) -> str:
    if hasattr(index, "get_doc_id"):
        return index.get_doc_id(idx)
    if hasattr(index, "doc_ids"):
        return index.doc_ids[idx]
    if hasattr(index, "_doc_ids"):
        return index._doc_ids[idx]
    return ""

def task_make_qrels_from_bm25(
    bucket_name: str,
    model_key: str,            # ej: "rag/models/2025/bm25.pkl"
    aws_conn_id: str,
    qrels_key: str,            # ej: "rag/qrels/2025/qrels.csv" (OUTPUT)
    query: str,                # ej: "contratación pública vial"
    top_k_pos: int = 10,       # positivos (label=1)
    add_negatives: bool = False,
    negatives_from_chunks_prefix: Optional[str] = None,  # ej: "rag/chunks/2025/"
    negatives_count: int = 20,
    **kwargs
) -> Dict[str, Any]:
    """
    Crea qrels.csv a partir de los top-K resultados de BM25 para 'query'.
    Formato: query,doc_id,label (1 = relevante). Opcionalmente agrega negativos (label=0).
    """
    if not query or not str(query).strip():
        raise ValueError("query vacía para generar qrels")

    # 1) Descargar modelo y cargar índice
    local_pkl = download_to_tmp(bucket=bucket_name, key=model_key, aws_conn_id=aws_conn_id)
    with open(local_pkl, "rb") as f:
        index = pickle.load(f)
    if not hasattr(index, "search"):
        raise TypeError("El objeto cargado no parece un BM25Index válido.")

    # 2) Buscar top-K positivos
    results = index.search(query, top_k=top_k_pos)
    pos_ids: List[str] = [_safe_get_doc_id(index, idx) for idx, _ in results]
    pos_ids = [d for d in pos_ids if d]  # limpiar vacíos

    rows: List[List[str]] = []
    for did in pos_ids:
        rows.append([query, did, "1"])

    # 3) (Opcional) agregar negativos (label=0)
    if add_negatives:
        neg_pool: Set[str] = set()
        # fuentes: todos los doc_ids del índice
        if hasattr(index, "doc_ids"):
            neg_pool.update(index.doc_ids)
        elif hasattr(index, "_doc_ids"):
            neg_pool.update(index._doc_ids)

        # también podemos extraer de NDJSON si nos pasan el prefijo
        if negatives_from_chunks_prefix:
            ndjson_keys = list_keys(bucket=bucket_name, prefix=negatives_from_chunks_prefix,
                                    aws_conn_id=aws_conn_id, suffix=".ndjson")
            # leer rápido y sumar doc_id
            for key in ndjson_keys[:50]:  # límite para no leer demasiado
                body = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
                for line in body.splitlines():
                    try:
                        import json
                        obj = json.loads(line)
                        did = obj.get("doc_id") or ""
                        if did:
                            neg_pool.add(did)
                    except Exception:
                        continue

        neg_pool = list(neg_pool - set(pos_ids))
        neg_pool = neg_pool[:max(0, negatives_count)]
        for did in neg_pool:
            rows.append([query, did, "0"])

    # 4) Escribir CSV en memoria y subir
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["query", "doc_id", "label"])
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")

    upload_bytes(bucket=bucket_name, key=qrels_key, data=data, aws_conn_id=aws_conn_id)
    print(f"✅ qrels generado en s3://{bucket_name}/{qrels_key} (pos={len(pos_ids)}, neg={len(rows)-len(pos_ids)})")

    return {"qrels_key": qrels_key, "positives": len(pos_ids), "rows": len(rows)}
