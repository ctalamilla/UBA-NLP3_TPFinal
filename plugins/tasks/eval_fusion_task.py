# plugins/tasks/eval_fusion_task.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from tasks.s3_utilities import (
    list_keys, read_text, download_to_tmp, upload_json, exists
)
from tasks.qrels_utils import load_qrels        # qrels: {query: {doc_id: rel_int}}
from tasks import metrics as METR               # average_precision_at_k, ndcg_at_k, recall_at_k, mrr


# ---------------------------
# Helpers de normalización
# ---------------------------
def _safe_qname(q: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (q or "").strip())[:60] or "query"

def _stem(x: str) -> str:
    if not x:
        return ""
    base = os.path.basename(x)
    return os.path.splitext(base)[0]

def _strip_page_suffix(x: str) -> str:
    # quita sufijo _pN si existe
    return re.sub(r"_p\d+$", "", x or "")

def _norm_doc_id(x: str) -> str:
    """
    Normaliza IDs a formato base:
    - quita ruta y extensión
    - quita sufijo de página (_pN)
    Ej: 'boletines/2025/22043_2025-10-01.pdf' → '22043_2025-10-01'
        '22043_2025-10-01_p1'                → '22043_2025-10-01'
    """
    return _strip_page_suffix(_stem(x or ""))

def _pred_doc_id_from_result(r: Dict[str, Any]) -> str:
    """
    Extrae el mejor doc_id "base" desde un resultado de fusión.
    Soporta varias variantes:
      - pdf_key (preferida, viene con ruta .pdf)
      - doc_id (tipo 22043_2025-10-01_p1)
      - chunk_id (tomamos la parte base antes de '::' y normalizamos)
      - page_id (idem doc_id)
    """
    pdf_key = r.get("pdf_key")
    if pdf_key:
        return _norm_doc_id(pdf_key)

    doc_id = r.get("doc_id")
    if doc_id:
        return _norm_doc_id(doc_id)

    chunk_id = r.get("chunk_id")
    if chunk_id:
        base = (chunk_id or "").split("::", 1)[0]
        return _norm_doc_id(base)

    page_id = r.get("page_id")
    if page_id:
        return _norm_doc_id(page_id)

    return ""


def _pick_latest_fusion_key(keys: List[str], query: Optional[str]) -> Optional[str]:
    """
    Selecciona el último (por nombre) JSON de fusión.
    Si viene query, intenta matchear el 'safe' de la query dentro del nombre.
    """
    if not keys:
        return None
    keys = sorted(k for k in keys if k.lower().endswith(".json"))
    if not keys:
        return None
    if query:
        sq = _safe_qname(query)
        cand = [k for k in keys if f"fusion_{sq}_" in os.path.basename(k)]
        if cand:
            return cand[-1]
    return keys[-1]


# ---------------------------
# Task principal
# ---------------------------
def task_eval_fusion(
    bucket_name: str,
    aws_conn_id: str,
    fusion_prefix: str = "rag/fusion/2025/",
    qrels_key: str = "rag/qrels/2025/qrels.csv",
    metrics_prefix: str = "rag/metrics/2025/",
    query: str = "",
    ks: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, Any]:
    ks = ks or [1, 3, 5, 10]

    # 0) Chequeo qrels
    if not exists(bucket=bucket_name, key=qrels_key, aws_conn_id=aws_conn_id):
        msg = f"⏭️ qrels no encontrado: s3://{bucket_name}/{qrels_key} — skip eval"
        print(msg)
        return {"skipped": True, "reason": "no_qrels"}

    # 1) Elegir archivo de fusión a evaluar
    fusion_keys = list_keys(bucket=bucket_name, prefix=fusion_prefix, aws_conn_id=aws_conn_id, suffix=".json")
    sel_key = _pick_latest_fusion_key(fusion_keys, query=query or None)
    if not sel_key:
        print(f"⏭️ No se encontraron archivos de fusión en s3://{bucket_name}/{fusion_prefix}")
        return {"skipped": True, "reason": "no_fusion_files"}

    print(f"📝 Evaluando fusión: s3://{bucket_name}/{sel_key}")

    # 2) Cargar ranking fusionado
    fusion_blob = read_text(bucket=bucket_name, key=sel_key, aws_conn_id=aws_conn_id)
    fusion_obj = json.loads(fusion_blob)
    fusion_query = (fusion_obj.get("query") or "").strip()
    results = fusion_obj.get("results", [])

    # Predicciones normalizadas (doc-level)
    pred_ids: List[str] = []
    seen: set = set()
    for r in results:
        did = _pred_doc_id_from_result(r)
        if did and did not in seen:
            seen.add(did)
            pred_ids.append(did)

    # 3) Cargar qrels
    local_qrels = download_to_tmp(bucket=bucket_name, key=qrels_key, aws_conn_id=aws_conn_id)
    qrels = load_qrels(local_qrels)  # dict: {query_text: {doc_id: rel, ...}, ...}

    # Elegir la clave de qrels (uso la query del JSON o la provista)
    q_key = fusion_query or (query or "")
    if q_key not in qrels:
        sq = _safe_qname(q_key)
        candidates = [k for k in qrels.keys() if _safe_qname(k) == sq]
        if candidates:
            q_key = candidates[0]
    if q_key not in qrels:
        print(f"⏭️ La query '{fusion_query or query}' no existe en qrels — skip eval.")
        return {"skipped": True, "reason": "query_not_in_qrels", "fusion_file": sel_key}

    rels_map: Dict[str, int] = qrels[q_key]  # {doc_id: rel_int (>0 = relevante)}

    # 4) Normalizar TAMBIÉN los doc_ids del qrels
    rel_ids = { _norm_doc_id(d) for d, rel in rels_map.items() if rel and int(rel) > 0 }

    # Debug opcional
    if os.getenv("EVAL_DEBUG", "0") == "1":
        inter = set(pred_ids) & rel_ids
        print(f"pred_ids (norm) top10: {pred_ids[:10]}")
        print(f"rel_ids  (norm) top10: {sorted(list(rel_ids))[:10]}")
        print(f"intersection size: {len(inter)}  -> {sorted(list(inter))[:10]}")

    # 5) Calcular métricas
    metrics_out: Dict[str, Any] = {
        "query": q_key,
        "fusion_file": sel_key,
        "k": ks,
        "total_rel": len(rel_ids),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    ap_at     = {k: METR.average_precision_at_k(pred_ids, rel_ids, k) for k in ks}
    ndcg_at   = {k: METR.ndcg_at_k(pred_ids, rel_ids, k) for k in ks}
    recall_at = {k: METR.recall_at_k(pred_ids, rel_ids, k) for k in ks}
    mrr_value = METR.mrr(pred_ids, rel_ids)

    metrics_out["AP@k"]     = ap_at
    metrics_out["nDCG@k"]   = ndcg_at
    metrics_out["Recall@k"] = recall_at
    metrics_out["MRR"]      = mrr_value

    # 6) Subir métricas a MinIO
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sq = _safe_qname(q_key)
    out_key = f"{metrics_prefix.rstrip('/')}/fusion_eval_{sq}_{ts}.json"
    upload_json(bucket=bucket_name, key=out_key, obj=metrics_out, aws_conn_id=aws_conn_id)

    # Resumen en logs
    print("✅ Métricas fusión:")
    for k in ks:
        print(f"  k={k:>2}  AP={ap_at[k]:.4f}  nDCG={ndcg_at[k]:.4f}  Recall={recall_at[k]:.4f}")
    print(f"  MRR={mrr_value:.4f}")
    print(f"⬆️ Subidas a s3://{bucket_name}/{out_key}")

    return {"out_key": out_key, "k": ks, "metrics": metrics_out}
