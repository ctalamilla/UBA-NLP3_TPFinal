# plugins/tasks/s3_utilities.py
from __future__ import annotations

import os
import io
import json
import tempfile
from typing import Iterable, List, Optional

from airflow.providers.amazon.aws.hooks.s3 import S3Hook


# -------------------------
# Hook / helpers
# -------------------------
def get_hook(aws_conn_id: str = "minio_s3") -> S3Hook:
    """Devuelve un S3Hook configurado (MinIO o AWS), según la conexión de Airflow."""
    return S3Hook(aws_conn_id=aws_conn_id)


def _ensure_list(x) -> List[str]:
    if not x:
        return []
    return list(x)


# -------------------------
# Listing
# -------------------------
def list_keys(
    bucket: str,
    prefix: str = "",
    aws_conn_id: str = "minio_s3",
    suffix: Optional[str] = None,
) -> List[str]:
    """
    Lista claves bajo un prefix. Si `suffix` se especifica, filtra por sufijo (p.ej. '.pdf').
    """
    hook = get_hook(aws_conn_id)
    keys = _ensure_list(hook.list_keys(bucket_name=bucket, prefix=prefix))
    if suffix:
        keys = [k for k in keys if k.lower().endswith(suffix.lower())]
    return keys


def list_pdfs(bucket: str, prefix: str, aws_conn_id: str = "minio_s3") -> List[str]:
    """Atajo para listar sólo PDFs."""
    return list_keys(bucket=bucket, prefix=prefix, aws_conn_id=aws_conn_id, suffix=".pdf")


# -------------------------
# Downloads
# -------------------------
def download_to_dir(
    bucket: str,
    key: str,
    local_dir: str,
    aws_conn_id: str = "minio_s3",
    preserve_file_name: bool = True,   # se mantiene por compatibilidad, no lo usamos
) -> str:
    """
    Descarga usando boto3 al path FINAL exacto:
      <local_dir>/<basename(key)>
    Retorna ese path.
    """
    import os
    os.makedirs(local_dir, exist_ok=True)
    dest = os.path.join(local_dir, os.path.basename(key))

    hook = get_hook(aws_conn_id)
    client = hook.get_conn()

    # descarga directa al destino exacto
    client.download_file(bucket, key, dest)

    # sanity check
    if not os.path.exists(dest):
        # ayuda de depuración
        try:
            print("⚠️ Archivos en el tmp tras descargar:", os.listdir(local_dir))
        except Exception:
            pass
        raise FileNotFoundError(f"No se encontró el archivo descargado en: {dest}")

    return dest


def download_to_tmp(bucket: str, key: str, aws_conn_id: str = "minio_s3") -> str:
    """Crea un tempdir y descarga ahí el objeto. Retorna el path local del archivo."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    return download_to_dir(bucket=bucket, key=key, local_dir=tmpdir, aws_conn_id=aws_conn_id)

# -------------------------
# Uploads (archivo / bytes / texto / json)
# -------------------------
def upload_file(
    bucket: str,
    key: str,
    filename: str,
    aws_conn_id: str = "minio_s3",
    replace: bool = True,
) -> None:
    hook = get_hook(aws_conn_id)
    hook.load_file(filename=filename, key=key, bucket_name=bucket, replace=replace)


def upload_bytes(
    bucket: str,
    key: str,
    data: bytes,
    aws_conn_id: str = "minio_s3",
    replace: bool = True,
) -> None:
    hook = get_hook(aws_conn_id)
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        hook.load_file(filename=tmp, key=key, bucket_name=bucket, replace=replace)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def upload_text(
    bucket: str,
    key: str,
    text: str,
    aws_conn_id: str = "minio_s3",
    replace: bool = True,
    encoding: str = "utf-8",
) -> None:
    upload_bytes(bucket=bucket, key=key, data=text.encode(encoding), aws_conn_id=aws_conn_id, replace=replace)


def upload_json(
    bucket: str,
    key: str,
    obj,
    aws_conn_id: str = "minio_s3",
    replace: bool = True,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    blob = json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent).encode("utf-8")
    upload_bytes(bucket=bucket, key=key, data=blob, aws_conn_id=aws_conn_id, replace=replace)


# -------------------------
# Read / existence / delete / copy
# -------------------------
def read_text(bucket: str, key: str, aws_conn_id: str = "minio_s3", encoding: str = "utf-8") -> str:
    """
    Lee un objeto de S3/MinIO y lo decodifica como texto.
    Compatible con providers donde read_key no soporta 'encoding'.
    """
    hook = get_hook(aws_conn_id)
    try:
        # Algunas versiones devuelven ya str
        data = hook.read_key(key=key, bucket_name=bucket)  # <- sin 'encoding'
        if isinstance(data, bytes):
            return data.decode(encoding, errors="replace")
        return data
    except TypeError:
        # Fallback robusto: usar get_key y decodificar
        obj = hook.get_key(key=key, bucket_name=bucket)
        body = obj.get()["Body"].read()
        return body.decode(encoding, errors="replace")
    
def exists(bucket: str, key: str, aws_conn_id: str = "minio_s3") -> bool:
    hook = get_hook(aws_conn_id)
    try:
        obj = hook.get_key(key=key, bucket_name=bucket)
        return obj is not None
    except Exception:
        return False


def delete_key(bucket: str, key: str, aws_conn_id: str = "minio_s3") -> None:
    hook = get_hook(aws_conn_id)
    client = hook.get_conn()
    client.delete_object(Bucket=bucket, Key=key)


def copy_object(
    bucket_src: str,
    key_src: str,
    bucket_dst: str,
    key_dst: str,
    aws_conn_id: str = "minio_s3",
) -> None:
    hook = get_hook(aws_conn_id)
    client = hook.get_conn()
    client.copy_object(
        CopySource={"Bucket": bucket_src, "Key": key_src},
        Bucket=bucket_dst,
        Key=key_dst,
    )


# -------------------------
# Buckets / signed URL
# -------------------------
def ensure_bucket(bucket: str, aws_conn_id: str = "minio_s3") -> None:
    """
    Crea el bucket si no existe. Funciona en MinIO y AWS (sin policy).
    """
    hook = get_hook(aws_conn_id)
    client = hook.get_conn()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        # MinIO: create_bucket suele no requerir LocationConstraint.
        try:
            client.create_bucket(Bucket=bucket)
        except Exception:
            # fallback AWS con LocationConstraint (si lo necesitás, ajustá la región)
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": "us-east-1"},
            )


def presigned_url(bucket: str, key: str, aws_conn_id: str = "minio_s3", expires_in: int = 3600) -> str:
    """Genera una URL firmada temporal (si el endpoint lo soporta)."""
    hook = get_hook(aws_conn_id)
    client = hook.get_conn()
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )