# plugins/tasks/chunk_from_txt_task_op.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Preferir utilidades existentes; si no están, usar fallback con boto3
# -----------------------------------------------------------------------------
HAVE_UTILS = False
try:
    # Variante 1: s3_utilities (tu guía)
    from tasks.s3_utilities import list_keys as _list_keys
    from tasks.s3_utilities import read_text as _read_text
    from tasks.s3_utilities import upload_bytes as _upload_bytes
    from tasks.s3_utilities import upload_json as _upload_json
    HAVE_UTILS = True
except Exception:
    try:
        # Variante 2: s3_utils (ya usada en otras tareas)
        from tasks.s3_utils import list_keys as _list_keys
        from tasks.s3_utils import read_text as _read_text
        from tasks.s3_utils import upload_text as _upload_text
        HAVE_UTILS = True
        _upload_bytes = lambda bucket, key, data, aws_conn_id=None: _upload_text(
            bucket=bucket, key=key, text=data.decode("utf-8"), aws_conn_id=aws_conn_id
        )
        def _upload_json(bucket, key, obj, aws_conn_id=None):
            _upload_text(bucket=bucket, key=key, text=json.dumps(obj, ensure_ascii=False, indent=2), aws_conn_id=aws_conn_id)
    except Exception:
        HAVE_UTILS = False

if not HAVE_UTILS:
    import boto3
    from botocore.config import Config

    def _boto_client():
        endpoint = os.getenv("S3_ENDPOINT_URL") or os.getenv("MINIO_ENDPOINT_URL") or "http://minio:9000"
        ak = os.getenv("AWS_ACCESS_KEY_ID", "minio_admin")
        sk = os.getenv("AWS_SECRET_ACCESS_KEY", "minio_admin")
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )

    def _list_keys(bucket: str, prefix: str, aws_conn_id: Optional[str] = None, suffix: Optional[str] = None) -> List[str]:
        s3 = _boto_client()
        keys: List[str] = []
        token = None
        while True:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=token) if token else \
                   s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            for it in resp.get("Contents", []):
                k = it["Key"]
                if not suffix or k.lower().endswith(suffix.lower()):
                    keys.append(k)
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return keys

    def _read_text(bucket: str, key: str, aws_conn_id: Optional[str] = None) -> str:
        s3 = _boto_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8", errors="replace")

    def _upload_bytes(bucket: str, key: str, data: bytes, aws_conn_id: Optional[str] = None) -> None:
        s3 = _boto_client()
        s3.put_object(Bucket=bucket, Key=key, Body=data)

    def _upload_json(bucket: str, key: str, obj: Any, aws_conn_id: Optional[str] = None) -> None:
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        _upload_bytes(bucket, key, data, aws_conn_id)

# -----------------------------------------------------------------------------
# Chunking: reutilizamos tu función de tokens
# -----------------------------------------------------------------------------
from tasks.documents import chunk_text  # 👈 ya la usas en tu loader

# -----------------------------------------------------------------------------
# Helpers de parsing y sidecars
# -----------------------------------------------------------------------------
_RX_BASE = re.compile(
    r"^(?P<boletin>\d{5})_OP(?P<op>[A-Za-z0-9]+)_(?P<fecha>\d{4}-\d{2}-\d{2})$"
)

def _parse_txt_base(base: str) -> Optional[Tuple[str, str, str]]:
    """
    '22032_OPSA100051525_2025-09-16' -> ('22032','SA100051525','2025-09-16')
    """
    m = _RX_BASE.match(base)
    if not m:
        return None
    return m.group("boletin"), m.group("op"), m.group("fecha")

def _sidecar_key_for(base: str, meta_prefix: Optional[str]) -> Optional[str]:
    if not meta_prefix:
        return None
    return f"{meta_prefix.rstrip('/')}/{base}.meta.json"

def _s3_exists(bucket: str, key: str) -> bool:
    try:
        # usamos las utils si existen
        if HAVE_UTILS:
            # no todas exponen head; intentamos leer metadatos de forma ligera
            # fallback: usar list_keys con filtro exacto
            prefix = os.path.dirname(key) + "/"
            name = os.path.basename(key)
            for k in _list_keys(bucket, prefix, suffix=".meta.json"):
                if k.endswith(name):
                    return True
            return False
        # fallback boto3
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL") or os.getenv("MINIO_ENDPOINT_URL") or "http://minio:9000",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio_admin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio_admin"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            config=Config(signature_version="s3v4"),
        )
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def _read_json(bucket: str, key: str, aws_conn_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        raw = _read_text(bucket=bucket, key=key, aws_conn_id=aws_conn_id)
        return json.loads(raw)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# TASK principal
# -----------------------------------------------------------------------------
def task_chunk_txt_op(
    bucket_name: str,
    prefix_txt: str,         # p.ej. "rag/text_op/2025/"
    prefix_pdfs: str,        # p.ej. "boletines/2025/"
    prefix_chunks: str,      # p.ej. "rag/chunks_op/2025/"
    aws_conn_id: Optional[str] = None,
    # Metadatos/sidecars (opcional pero recomendado)
    meta_prefix: Optional[str] = None,  # p.ej. "rag/text_op_meta/2025/"
    # Chunking
    max_tokens_chunk: int = 300,
    overlap: int = 80,
    # Control
    fail_on_empty: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Agrupa todos los TXT por (boletin, fecha) y genera un único NDJSON por PDF:
      s3://{bucket}/{prefix_chunks}/{boletin}_{fecha}.ndjson

    - Cada línea es un chunk con:
        chunk_id = "{boletin}_{fecha}::p1::{idx}"
        source   = "{prefix_pdfs}/{boletin}_{fecha}.pdf"
        page     = 1
        doc_id   = "{boletin}_{fecha}_p1"
        + metadatos heredados: op, txt_key, y si existe sidecar -> classification, etc.
    """
    # 1) listar TXT de OP
    txt_keys: List[str] = _list_keys(bucket=bucket_name, prefix=prefix_txt, aws_conn_id=aws_conn_id, suffix=".txt")
    if not txt_keys:
        msg = f"ℹ️ No hay TXT en s3://{bucket_name}/{prefix_txt}"
        print(msg)
        if fail_on_empty:
            raise FileNotFoundError(msg)
        return {"processed_groups": 0, "chunks": 0, "items": []}

    # 2) agrupar por (boletin, fecha)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key in sorted(txt_keys):
        base = os.path.splitext(os.path.basename(key))[0]
        parsed = _parse_txt_base(base)
        if not parsed:
            print(f"⚠️ Nombre inesperado (no parsea OP): {base}")
            # skip silencioso para no romper
            continue
        boletin, op_code, fecha = parsed
        pdf_base = f"{boletin}_{fecha}"
        groups.setdefault((boletin, fecha), []).append(
            {"txt_key": key, "base": base, "boletin": boletin, "fecha": fecha, "op": op_code}
        )

    if not groups:
        msg = f"⚠️ No se pudo agrupar ningún TXT por (boletin, fecha) desde {prefix_txt}"
        print(msg)
        if fail_on_empty:
            raise RuntimeError(msg)
        return {"processed_groups": 0, "chunks": 0, "items": []}

    total_chunks = 0
    outputs: List[Dict[str, Any]] = []

    # 3) procesar cada grupo => un NDJSON por PDF
    for (boletin, fecha), items in groups.items():
        pdf_base = f"{boletin}_{fecha}"
        ndjson_key = f"{prefix_chunks.rstrip('/')}/{pdf_base}.ndjson"
        source_pdf = f"{prefix_pdfs.rstrip('/')}/{pdf_base}.pdf"

        # ordenar por op para estabilidad
        items.sort(key=lambda x: (x["op"], x["txt_key"]))

        lines: List[str] = []
        global_idx = 0

        for it in items:
            txt_key = it["txt_key"]
            base    = it["base"]
            op_code = it["op"]

            # texto
            try:
                text = _read_text(bucket=bucket_name, key=txt_key, aws_conn_id=aws_conn_id)
            except Exception as e:
                print(f"⚠️ No se pudo leer TXT '{txt_key}': {e}")
                continue

            # sidecar (si existe)
            sidecar_key = _sidecar_key_for(base, meta_prefix)
            sidecar = None
            if sidecar_key and _s3_exists(bucket_name, sidecar_key):
                sidecar = _read_json(bucket_name, sidecar_key, aws_conn_id)

            # chunkear
            chunks = chunk_text(text or "", max_tokens_chunk, overlap)
            if not chunks:
                # fallback: 1 chunk mínimo si hay texto
                raw = " ".join((text or "").split())
                if raw:
                    words = raw.split()
                    if len(words) > max_tokens_chunk:
                        words = words[:max_tokens_chunk]
                    chunks = [" ".join(words)]
                else:
                    continue  # realmente vacío

            # armar líneas NDJSON
            for ch in chunks:
                rec = {
                    "chunk_id": f"{pdf_base}::p1::{global_idx}",
                    "source": source_pdf,
                    "page": 1,
                    "chunk_index": global_idx,
                    "text": ch,
                    "doc_id": f"{pdf_base}_p1",
                    # metadatos heredados:
                    "boletin": boletin,
                    "fecha": fecha,
                    "op": op_code,
                    "txt_key": txt_key,
                }
                if isinstance(sidecar, dict):
                    # Guardamos todo el sidecar bajo 'meta', y exponemos 'classification' directo.
                    rec["meta"] = sidecar
                    if "classification" in sidecar:
                        rec["classification"] = sidecar["classification"]

                lines.append(json.dumps(rec, ensure_ascii=False))
                global_idx += 1

        if not lines:
            print(f"ℹ️ Grupo vacío (sin chunks útiles): {pdf_base}")
            continue

        # subir NDJSON del grupo
        _upload_bytes(
            bucket=bucket_name,
            key=ndjson_key,
            data=("\n".join(lines)).encode("utf-8"),
            aws_conn_id=aws_conn_id,
        )
        print(f"✅ NDJSON subido: s3://{bucket_name}/{ndjson_key}  (chunks={len(lines)})")

        outputs.append({
            "pdf_base": pdf_base,
            "ndjson_key": ndjson_key,
            "chunks": len(lines),
            "source_pdf": source_pdf,
            "group_items": len(items),
        })
        total_chunks += len(lines)

    # 4) manifest
    manifest = {
        "bucket": bucket_name,
        "prefix_txt": prefix_txt,
        "prefix_pdfs": prefix_pdfs,
        "prefix_chunks": prefix_chunks,
        "meta_prefix": meta_prefix,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "groups": len(outputs),
        "total_chunks": total_chunks,
        "items": outputs[:50],  # muestra
    }
    manifest_key = f"{prefix_chunks.rstrip('/')}/_chunks_manifest_op.json"
    _upload_json(bucket=bucket_name, key=manifest_key, obj=manifest, aws_conn_id=aws_conn_id)
    print(f"📄 Manifest OP: s3://{bucket_name}/{manifest_key}")

    return {
        "processed_groups": len(outputs),
        "chunks": total_chunks,
        "manifest_key": manifest_key,
        "sample": outputs[:5],
    }
