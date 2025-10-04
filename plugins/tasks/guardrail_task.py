from __future__ import annotations
import json
import hashlib
from typing import Dict, Any, List

from tasks.s3_utilities import list_keys, read_text, upload_text
import tasks.agente_verificador as NG  # ← usa tu verificador tal cual

def _hash_text(text: str) -> str:
    """Hash estable para dedupe."""
    norm = " ".join((text or "").split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

def task_guardrail_chunks(
    bucket_name: str,
    in_prefix: str = "rag/chunks_labeled/2025/",
    out_prefix: str = "rag/chunks_curated/2025/",
    aws_conn_id: str = "minio_s3",
) -> Dict[str, Any]:
    """
    Lee NDJSON etiquetados (classifier) → filtra con verificar_chunk_llm → escribe NDJSON “curated”.
    Mantiene los campos originales del registro.
    """
    ndjson_keys = list_keys(bucket=bucket_name, prefix=in_prefix, aws_conn_id=aws_conn_id, suffix=".ndjson")
    if not ndjson_keys:
        print(f"ℹ️ No hay NDJSON en s3://{bucket_name}/{in_prefix}")
        return {"processed_files": 0, "written_files": [], "in_records": 0, "out_records": 0}

    total_in = total_out = 0
    written_files: List[str] = []

    for key in sorted(ndjson_keys):
        raw = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
        out_lines: List[str] = []
        seen = set()
        kept = 0

        for line in raw.splitlines():
            line = (line or "").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            text = (rec.get("text") or "").strip()
            total_in += 1

            # Llama a tu verificador de la notebook
            if not NG.verificar_chunk_llm(text):
                continue

            h = _hash_text(text)
            if h in seen:
                continue
            seen.add(h)

            out_lines.append(json.dumps(rec, ensure_ascii=False))
            kept += 1

        if out_lines:
            out_key = key.replace(in_prefix.rstrip("/"), out_prefix.rstrip("/"), 1)
            upload_text(bucket=bucket_name, key=out_key, text="\n".join(out_lines) + "\n", aws_conn_id=aws_conn_id)
            print(f"✅ Curado: {kept}/{total_in} → s3://{bucket_name}/{out_key}")
            written_files.append(out_key)
            total_out += kept

    return {
        "processed_files": len(ndjson_keys),
        "written_files": written_files,
        "in_records": total_in,
        "out_records": total_out,
    }
