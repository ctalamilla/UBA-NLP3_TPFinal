# tasks/extract_texts_by_op_task.py
from __future__ import annotations
import os, re, json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import boto3

# Reutilizamos tus utilidades existentes
from tasks.utils_op_split import split_pdf_por_op, parse_base_id
# Estas helpers deben existir en tu repo (las usás en otras tareas)
from tasks.s3_utilities import list_pdfs, download_to_tmp, upload_text

# -----------------------
# S3 helpers locales
# -----------------------
def _build_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio_admin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio_admin"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

def _put_json(s3, bucket: str, key: str, data: Dict[str, Any]):
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")

# -----------------------
# Derivar prefijos
# -----------------------
def _auto_meta_prefix(prefix_txt: str) -> str:
    """
    Si no se pasa prefix_meta, lo derivamos:
    - .../text_op/...  -> .../text_op_meta/...
    - otro caso        -> sufijo '_meta/'.
    """
    p = (prefix_txt or "").rstrip("/")
    if not p:
        return "rag/text_op_meta/2025/"
    if "/text_op/" in p:
        return p.replace("/text_op/", "/text_op_meta/") + "/"
    return p + "_meta/"

# -----------------------
# Tarea principal
# -----------------------
def task_extract_texts_by_op(
    bucket_name: str,
    prefix_pdfs: str,
    prefix_txt: str,
    aws_conn_id: str = "minio_s3",    # compat
    ignore_first_pages: int = 2,
    ignore_last_pages: int = 1,
    prefix_meta: Optional[str] = None,  # <- NUEVO: dónde guardar metadatos por-doc
    **kwargs
) -> Dict[str, Any]:
    """
    Por cada PDF:
      - Separa por OP (bloques).
      - Sube 1 .txt por bloque con OP a prefix_txt.
      - Escribe 1 .meta.json por cada .txt a prefix_meta.
      - Escribe un manifest JSONL con todos los items procesados.

    Devuelve un resumen con samples.
    """
    s3 = _build_s3()
    prefix_meta = prefix_meta or _auto_meta_prefix(prefix_txt)

    print(f"[extract_texts_by_op] bucket={bucket_name}")
    print(f"[extract_texts_by_op] prefix_pdfs={prefix_pdfs}")
    print(f"[extract_texts_by_op] prefix_txt={prefix_txt}")
    print(f"[extract_texts_by_op] prefix_meta={prefix_meta} (auto-derivado si None)")

    pdf_keys: List[str] = list_pdfs(bucket=bucket_name, prefix=prefix_pdfs, aws_conn_id=aws_conn_id)
    if not pdf_keys:
        print(f"ℹ️ No hay PDFs en s3://{bucket_name}/{prefix_pdfs}")
        return {
            "processed_pdfs": 0,
            "generated_docs": 0,
            "bucket": bucket_name,
            "prefix_txt": prefix_txt,
            "prefix_meta": prefix_meta,
            "items_sample": [],
        }

    total_docs = 0
    items: List[Dict[str, Any]] = []
    manifest_lines: List[str] = []
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for key in sorted(pdf_keys):
        # 1) Descargar PDF temporal
        local_pdf = Path(download_to_tmp(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id))

        # 2) base (boletín y fecha)
        base = os.path.splitext(os.path.basename(key))[0]   # ej: 22036_2025-09-22
        boletin, fecha = parse_base_id(base)

        # 3) split por OP
        bloques = split_pdf_por_op(local_pdf, ignore_first_pages=ignore_first_pages, ignore_last_pages=ignore_last_pages)

        # 4) por cada bloque con OP: subir TXT + escribir META
        doc_count = 0
        for b in bloques:
            op = b.get("op")
            if not op:
                # Si querés subir todo igual, descomentá:
                # op = f"IDX{b['doc_index']:03d}"
                # (por defecto, sólo bloques con OP)
                continue

            op_safe = re.sub(r"[^A-Za-z0-9]+", "", op)
            txt_key = f"{prefix_txt.rstrip('/')}/{boletin}_OP{op_safe}_{fecha}.txt"

            # subir texto
            upload_text(bucket=bucket_name, key=txt_key, text=b["texto"], aws_conn_id=aws_conn_id)

            # armar metadato por-doc
            meta_rec = {
                "status": "extracted",
                "created_at": now_iso,
                "pdf_key": key,
                "txt_key": txt_key,
                "boletin": boletin,
                "fecha": fecha,
                "op": op,
                "doc_index": int(b.get("doc_index") or 0),
                "chars": int(len(b.get("texto") or "")),
            }

            # meta sidecar por TXT
            meta_key = f"{prefix_meta.rstrip('/')}/{os.path.basename(txt_key).replace('.txt', '.meta.json')}"
            _put_json(s3, bucket_name, meta_key, meta_rec)

            # mantener lista
            items.append(meta_rec)
            manifest_lines.append(json.dumps(meta_rec, ensure_ascii=False))
            total_docs += 1
            doc_count += 1

        print(f"✅ {doc_count} TXT(s)+META(s) desde: s3://{bucket_name}/{key}")

    # 5) escribir MANIFEST JSONL (agregado de esta corrida)
    manifest_key = f"{prefix_meta.rstrip('/')}/_manifest.jsonl"
    s3.put_object(
        Bucket=bucket_name,
        Key=manifest_key,
        Body=("\n".join(manifest_lines) + ("\n" if manifest_lines else "")).encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    print(f"[extract_texts_by_op] manifest escrito: s3://{bucket_name}/{manifest_key} ({len(manifest_lines)} líneas)")

    summary = {
        "processed_pdfs": len(pdf_keys),
        "generated_docs": total_docs,
        "bucket": bucket_name,
        "prefix_pdfs": prefix_pdfs,
        "prefix_txt": prefix_txt,
        "prefix_meta": prefix_meta,
        "manifest_key": manifest_key,
        "generated_at": now_iso,
        "items_sample": items[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
