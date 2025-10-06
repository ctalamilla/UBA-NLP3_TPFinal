# plugins/tasks/chunk_from_txt_task.py
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Dict, Any, List

from tasks.s3_utilities import list_keys, read_text, upload_bytes, upload_json


# ✅ usamos TUS funciones/clases del notebook
from tasks.documents import Document
from tasks.loader_pdfs import documents_to_chunks


def task_chunk_txt(
    bucket_name: str,
    prefix_txt: str,        # p.ej. "rag/text/2025/"
    prefix_pdfs: str,       # p.ej. "boletines/2025/" (para referenciar 'source' en los metadatos)
    prefix_chunks: str,     # p.ej. "rag/chunks/2025/"
    aws_conn_id: str,
    max_tokens_chunk: int = 300,
    overlap: int = 80,
    **kwargs
) -> Dict[str, Any]:
    """
    1) Lista TXT en s3://bucket/prefix_txt
    2) Por cada TXT:
       - construye un Document (id=basename_p1, page=1, source=basename.pdf)
       - genera chunks con documents_to_chunks([doc], ...)
       - sube <basename>.ndjson a prefix_chunks
    3) Sube un manifest con conteos totales.
    """
    txt_keys: List[str] = list_keys(bucket=bucket_name, prefix=prefix_txt, aws_conn_id=aws_conn_id, suffix=".txt")
    if not txt_keys:
        print(f"ℹ️ No hay TXT en s3://{bucket_name}/{prefix_txt}")
        return {"processed": 0, "pages": 0, "chunks": 0, "items": []}

    items = []
    total_pages = 0
    total_chunks = 0

    for key in sorted(txt_keys):
        base = os.path.splitext(os.path.basename(key))[0]  # sin .txt
        text = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)

        # Un Document por TXT (page=1 simbólica)
        doc = Document(
            id=f"{base}_p1",
            text=text or "",
            source=f"{base}.pdf",   # referenciamos el PDF original por nombre
            page=1,
        )

        # Usamos EXACTAMENTE tu función de chunking a nivel documentos
        # documents_to_chunks espera List[Document]
        chunks_map = documents_to_chunks([doc], max_tokens_chunk=max_tokens_chunk, overlap=overlap)
        chunks = chunks_map.get(doc.id, [])

        # Armamos NDJSON con metadatos consistentes
        lines = []
        for i, ch in enumerate(chunks):
            obj = {
                "id": f"{base}::p1::{i}",
                "source": f"{prefix_pdfs.rstrip('/')}/{base}.pdf",
                "page": 1,
                "chunk_index": i,
                "text": ch,
                "doc_id": doc.id,
            }
            lines.append(json.dumps(obj, ensure_ascii=False))
        ndjson_blob = ("\n".join(lines)).encode("utf-8")

        out_key = f"{prefix_chunks.rstrip('/')}/{base}.ndjson"
        upload_bytes(bucket=bucket_name, key=out_key, data=ndjson_blob, aws_conn_id=aws_conn_id)

        items.append({"pdf": f"{base}.pdf", "txt_key": key, "ndjson_key": out_key, "chunks": len(chunks)})
        total_pages += 1
        total_chunks += len(chunks)
        print(f"✅ NDJSON subido: s3://{bucket_name}/{out_key}  (chunks={len(chunks)})")

    manifest = {
        "bucket": bucket_name,
        "prefix_txt": prefix_txt,
        "prefix_pdfs": prefix_pdfs,
        "prefix_chunks": prefix_chunks,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_txt": len(items),
        "total_pages": total_pages,     # 1 por TXT en este esquema
        "total_chunks": total_chunks,
        "items": items,
    }
    manifest_key = f"{prefix_chunks.rstrip('/')}/_chunks_manifest.json"
    upload_json(bucket=bucket_name, key=manifest_key, obj=manifest, aws_conn_id=aws_conn_id)
    print(f"📄 Manifest: s3://{bucket_name}/{manifest_key}")

    return {"processed": len(items), "pages": total_pages, "chunks": total_chunks, "manifest_key": manifest_key}

