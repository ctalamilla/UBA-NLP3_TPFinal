# plugins/tasks/eval_bm25_task.py
from __future__ import annotations

import os
import csv
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List, Set

from tasks.s3_utilities import download_to_tmp, exists, upload_json, upload_bytes
from tasks.bm25_index import BM25Index  # para resolver pickle
from tasks import io_utils as IO       # ← usa tus funciones del notebook
from tasks import metrics as M         # ← usa tus métricas del notebook
from airflow.exceptions import AirflowSkipException

def _safe_get_doc_id(index, idx: int) -> str:
    if hasattr(index, "get_doc_id"):
        return index.get_doc_id(idx)
    if hasattr(index, "doc_ids"):
        return index.doc_ids[idx]
    if hasattr(index, "_doc_ids"):
        return index._doc_ids[idx]
    return ""


def task_eval_bm25(
    bucket_name: str,
    model_key: str,            # ej: "rag/models/2025/bm25.pkl"
    qrels_key: str,            # ej: "rag/qrels/2025/qrels.csv" (query,doc_id,label>0)
    aws_conn_id: str,
    prefix_eval: str,          # ej: "rag/eval/2025/"
    k_list: List[int] = [5, 10],
    top_k_search: int = 50,    # tope para buscar por query
    **kwargs
) -> Dict[str, Any]:
    """
    Evalúa el índice BM25 contra qrels y guarda resultados (JSON y CSV) en MinIO.
    Requiere:
      - plugins/tasks/io_utils.py (load_qrels_csv)
      - plugins/tasks/metrics.py  (precision/recall/mrr/ndcg)
    """
    # 1) Descargar y cargar el modelo
    local_pkl = download_to_tmp(bucket=bucket_name, key=model_key, aws_conn_id=aws_conn_id)
    with open(local_pkl, "rb") as f:
        index = pickle.load(f)
    if not hasattr(index, "search"):
        raise TypeError("El objeto cargado no parece un BM25Index válido.")
    # 2) Verificar qrels
    if not exists(bucket_name, qrels_key, aws_conn_id):
        msg = f"⏭️ qrels no encontrado: s3://{bucket_name}/{qrels_key} — skip eval"
        print(msg)
        raise AirflowSkipException(msg)
    # 2) Descargar qrels y cargar con TU io_utils
    local_qrels = download_to_tmp(bucket=bucket_name, key=qrels_key, aws_conn_id=aws_conn_id)
    qrels: Dict[str, Set[str]] = IO.load_qrels_csv(local_qrels)  # query -> set(doc_id_relevante)

    queries = list(qrels.keys())
    if not queries:
        print("ℹ️ No hay queries en qrels.")
        return {"queries": 0}

    # 3) Iterar queries, buscar y calcular métricas
    per_query = []
    agg = {f"P@{k}": 0.0 for k in k_list}
    agg.update({f"R@{k}": 0.0 for k in k_list})
    agg["MRR@{}".format(max(k_list))] = 0.0
    agg["nDCG@{}".format(max(k_list))] = 0.0

    for q in queries:
        rel = qrels[q]
        results = index.search(q, top_k=top_k_search)  # [(idx, score), ...]
        ranked_doc_ids = [_safe_get_doc_id(index, idx) for idx, _ in results]

        row: Dict[str, Any] = {"query": q, "relevant": len(rel), "returned": len(ranked_doc_ids)}
        # Métricas por k
        for k in k_list:
            try:
                p_at_k = M.precision_at_k(rel, ranked_doc_ids, k)
                r_at_k = M.recall_at_k(rel, ranked_doc_ids, k)
            except Exception:
                # Si los nombres difieren, podés adaptar acá; mantenemos el fallo suave.
                p_at_k, r_at_k = 0.0, 0.0
            row[f"P@{k}"] = float(p_at_k)
            row[f"R@{k}"] = float(r_at_k)

        # MRR y nDCG al máximo K
        K = max(k_list)
        try:
            mrr_k = M.mrr_at_k(rel, ranked_doc_ids, K)
        except Exception:
            mrr_k = 0.0
        try:
            ndcg_k = M.ndcg_at_k(rel, ranked_doc_ids, K)
        except Exception:
            ndcg_k = 0.0

        row[f"MRR@{K}"] = float(mrr_k)
        row[f"nDCG@{K}"] = float(ndcg_k)

        # Acumular
        for k in k_list:
            agg[f"P@{k}"] += row[f"P@{k}"]
            agg[f"R@{k}"] += row[f"R@{k}"]
        agg[f"MRR@{K}"] += row[f"MRR@{K}"]
        agg[f"nDCG@{K}"] += row[f"nDCG@{K}"]

        per_query.append(row)

    N = len(per_query)
    macro = {k: (v / N if N else 0.0) for k, v in agg.items()}

    # 4) Subir reportes a MinIO
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report = {
        "bucket": bucket_name,
        "model_key": model_key,
        "qrels_key": qrels_key,
        "queries": N,
        "k_list": k_list,
        "top_k_search": top_k_search,
        "macro": macro,
        "per_query": per_query,
        "generated_at": ts,
    }
    out_json_key = f"{prefix_eval.rstrip('/')}/bm25_eval_{ts}.json"
    upload_json(bucket=bucket_name, key=out_json_key, obj=report, aws_conn_id=aws_conn_id)
    print(f"📄 Eval JSON: s3://{bucket_name}/{out_json_key}")

    # CSV resumido
    headers = ["query", "relevant", "returned"] + [f"P@{k}" for k in k_list] + [f"R@{k}" for k in k_list] + [f"MRR@{max(k_list)}", f"nDCG@{max(k_list)}"]
    lines = [",".join(headers)]
    for r in per_query:
        line = [str(r.get(h, "")) for h in headers]
        lines.append(",".join(line))
    out_csv_key = f"{prefix_eval.rstrip('/')}/bm25_eval_{ts}.csv"
    upload_bytes(bucket=bucket_name, key=out_csv_key, data=("\n".join(lines)).encode("utf-8"), aws_conn_id=aws_conn_id)
    print(f"📑 Eval CSV: s3://{bucket_name}/{out_csv_key}")

    return {"queries": N, "macro": macro, "json_key": out_json_key, "csv_key": out_csv_key}
