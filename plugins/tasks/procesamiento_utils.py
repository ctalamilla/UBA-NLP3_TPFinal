import subprocess
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def listar_pdfs_minio(**kwargs):
    """
    Se conecta a MinIO vía S3Hook y lista todos los PDFs bajo el prefix.
    """
    params = kwargs.get("params", {}) or {}
    bucket_name = params.get("bucket_name", "respaldo2")
    prefix = params.get("prefix", "boletines/2025/")
    aws_conn_id = params.get("aws_conn_id", "minio_s3")

    hook = S3Hook(aws_conn_id=aws_conn_id)

    # listar las keys en el bucket
    keys = hook.list_keys(bucket_name=bucket_name, prefix=prefix) or []
    pdf_keys = [k for k in keys if k.lower().endswith(".pdf")]

    if not pdf_keys:
        print(f"ℹ️ No se encontraron PDFs en s3://{bucket_name}/{prefix}")
        return []

    print(f"🗂️ PDFs en s3://{bucket_name}/{prefix}:")
    for key in pdf_keys:
        print(f"  - {key}")

    return pdf_keys
