from __future__ import annotations
import os, pickle
from typing import Dict, Any, List
from tasks.s3_utilities import download_to_tmp, upload_bytes
from tasks.bm25_index import BM25Index  # noqa

def task_dump_doc_ids(
    bucket_name: str,
    model_key: str,
    out_key: str,           # p.ej. "rag/qrels/2025/_docids.txt"
    aws_conn_id: str,
    **kwargs
) -> Dict[str, Any]:
    local_pkl = download_to_tmp(bucket=bucket_name, key=model_key, aws_conn_id=aws_conn_id)
    with open(local_pkl, "rb") as f:
        index = pickle.load(f)

    # obtener doc_ids únicos
    if hasattr(index, "doc_ids"):
        doc_ids = list(dict.fromkeys(index.doc_ids))  # únicos y en orden
    elif hasattr(index, "_doc_ids"):
        doc_ids = list(dict.fromkeys(index._doc_ids))
    else:
        # fallback vacío
        doc_ids = []

    blob = ("\n".join(doc_ids)).encode("utf-8")
    upload_bytes(bucket=bucket_name, key=out_key, data=blob, aws_conn_id=aws_conn_id)
    print(f"📄 DocIDs exportados: s3://{bucket_name}/{out_key} ({len(doc_ids)} únicos)")
    return {"count": len(doc_ids), "out_key": out_key}
