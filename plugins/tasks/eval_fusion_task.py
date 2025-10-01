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
from tasks.qrels_utils import load_qrels             # ← helper para leer qrels
from tasks import metrics as METR                    # ← tus métricas (pred_ids/rel_ids)


def _safe_qname(q: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", (q or "").strip())[:60] or "query"


def _base_doc_id(chunk_or_doc_id: str) -> str:
    """
    Los resultados de fusión vienen por chunk (id con '::').
    Para evaluar a nivel documento, tomamos la parte base antes de '::'.
    Si ya es doc_id simple, se devuelve tal cual.
    """
    if not chunk_or_doc_id:
        return ""
    return chunk_or_doc_id.split("::", 1)[0]


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
    ranked_chunk_ids = [r.get("chunk_id") for r in results if r.get("chunk_id")]
    ranked_doc_ids = [_base_doc_id(cid) for cid in ranked_chunk_ids]  # pred_ids a nivel documento

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

    # 4) Calcular métricas con TU API (pred_ids + rel_ids + k)
    pred_ids = ranked_doc_ids
    rel_ids = {d for d, r in rels_map.items() if r > 0}

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

    # 5) Subir métricas a MinIO
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sq = _safe_qname(q_key)
    out_key = f"{metrics_prefix.rstrip('/')}/fusion_eval_{sq}_{ts}.json"
    upload_json(bucket=bucket_name, key=out_key, obj=metrics_out, aws_conn_id=aws_conn_id)

    print("✅ Métricas fusión:")
    for k in ks:
        print(f"  k={k:>2}  AP={ap_at[k]:.4f}  nDCG={ndcg_at[k]:.4f}  Recall={recall_at[k]:.4f}")
    print(f"  MRR={mrr_value:.4f}")
    print(f"⬆️ Subidas a s3://{bucket_name}/{out_key}")

    return {"out_key": out_key, "k": ks, "metrics": metrics_out}