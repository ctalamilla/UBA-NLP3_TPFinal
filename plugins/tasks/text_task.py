# plugins/tasks/text_task.py
from __future__ import annotations
import os
from pathlib import Path  # <<-- agregar
from datetime import datetime
from typing import Dict, Any, List

from tasks.s3_utilities import list_pdfs, download_to_tmp, upload_text
from tasks.loader_pdfs import pdf_to_documents
from tasks.documents import Document

def task_extract_texts(
    bucket_name: str,
    prefix_pdfs: str,
    prefix_txt: str,
    aws_conn_id: str,
    **kwargs
) -> Dict[str, Any]:
    pdf_keys: List[str] = list_pdfs(bucket=bucket_name, prefix=prefix_pdfs, aws_conn_id=aws_conn_id)
    if not pdf_keys:
        print(f"ℹ️ No hay PDFs en s3://{bucket_name}/{prefix_pdfs}")
        return {"processed": 0, "prefix_txt": prefix_txt, "items": []}

    items = []
    for key in sorted(pdf_keys):
        # 1) bajar PDF temporal (devuelve str) y convertir a Path
        local_pdf_str = download_to_tmp(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
        local_pdf = Path(local_pdf_str)  # <<-- FIX: convertir a Path

        # 2) extraer documentos (una lista de Document por página)
        docs: List[Document] = pdf_to_documents(local_pdf)

        # 3) concatenar el texto de las páginas
        full_text = "\n\n".join([d.text for d in docs if (d.text or "").strip()])

        # 4) subir .txt a MinIO
        base = os.path.splitext(os.path.basename(key))[0]
        out_key = f"{prefix_txt.rstrip('/')}/{base}.txt"
        upload_text(bucket=bucket_name, key=out_key, text=full_text, aws_conn_id=aws_conn_id)

        items.append({"pdf_key": key, "txt_key": out_key, "pages": len(docs), "chars": len(full_text)})
        print(f"✅ TXT subido: s3://{bucket_name}/{out_key}  (páginas={len(docs)}, chars={len(full_text)})")

    summary = {
        "processed": len(items),
        "bucket": bucket_name,
        "prefix_pdfs": prefix_pdfs,
        "prefix_txt": prefix_txt,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "items": items,
    }
    print(f"✅ Total TXT generados: {len(items)} en s3://{bucket_name}/{prefix_txt}")
    return summary


