# from __future__ import annotations
# import json
# import hashlib
# from typing import Dict, Any, List

# from tasks.s3_utilities import list_keys, read_text, upload_text
# import tasks.agente_verificador as NG  # ← usa tu verificador tal cual

# def _hash_text(text: str) -> str:
#     """Hash estable para dedupe."""
#     norm = " ".join((text or "").split()).lower()
#     return hashlib.sha1(norm.encode("utf-8")).hexdigest()

# def task_guardrail_chunks(
#     bucket_name: str,
#     in_prefix: str = "rag/chunks_op/2025/",
#     out_prefix: str = "rag/chunks_op_curated/2025/",
#     aws_conn_id: str = "minio_s3",
# ) -> Dict[str, Any]:
#     """
#     Lee NDJSON etiquetados (classifier) → filtra con verificar_chunk_llm → escribe NDJSON “curated”.
#     Mantiene los campos originales del registro.
#     """
#     ndjson_keys = list_keys(bucket=bucket_name, prefix=in_prefix, aws_conn_id=aws_conn_id, suffix=".ndjson")
#     if not ndjson_keys:
#         print(f"ℹ️ No hay NDJSON en s3://{bucket_name}/{in_prefix}")
#         return {"processed_files": 0, "written_files": [], "in_records": 0, "out_records": 0}

#     total_in = total_out = 0
#     written_files: List[str] = []

#     for key in sorted(ndjson_keys):
#         raw = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
#         out_lines: List[str] = []
#         seen = set()
#         kept = 0

#         for line in raw.splitlines():
#             line = (line or "").strip()
#             if not line:
#                 continue
#             try:
#                 rec = json.loads(line)
#             except Exception:
#                 continue

#             text = (rec.get("text") or "").strip()
#             total_in += 1

#             # Llama a tu verificador de la notebook
#             if not NG.verificar_chunk_llm(text):
#                 continue

#             h = _hash_text(text)
#             if h in seen:
#                 continue
#             seen.add(h)

#             out_lines.append(json.dumps(rec, ensure_ascii=False))
#             kept += 1

#         if out_lines:
#             out_key = key.replace(in_prefix.rstrip("/"), out_prefix.rstrip("/"), 1)
#             upload_text(bucket=bucket_name, key=out_key, text="\n".join(out_lines) + "\n", aws_conn_id=aws_conn_id)
#             print(f"✅ Curado: {kept}/{total_in} → s3://{bucket_name}/{out_key}")
#             written_files.append(out_key)
#             total_out += kept

#     return {
#         "processed_files": len(ndjson_keys),
#         "written_files": written_files,
#         "in_records": total_in,
#         "out_records": total_out,
#     }
from __future__ import annotations
import json
import hashlib
from typing import Dict, Any, List
from datetime import datetime

from tasks.s3_utilities import list_keys, read_text, upload_text, upload_json
import tasks.agente_verificador as NG  # ← usa tu verificador tal cual


def _hash_text(text: str) -> str:
    """Hash estable para dedupe."""
    norm = " ".join((text or "").split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def task_guardrail_chunks(
    bucket_name: str,
    in_prefix: str = "rag/chunks_op/2025/",
    out_prefix: str = "rag/chunks_op_curated/2025/",
    aws_conn_id: str = "minio_s3",
) -> Dict[str, Any]:
    """
    Lee NDJSON etiquetados (classifier) → filtra con verificar_chunk_llm → escribe NDJSON "curated".
    Genera manifiesto completo de chunks rechazados.
    """
    print("="*70)
    print("🛡️  GUARDRAIL CHUNKS - Iniciando proceso de curación")
    print("="*70)
    print(f"📂 Input:  s3://{bucket_name}/{in_prefix}")
    print(f"📂 Output: s3://{bucket_name}/{out_prefix}")
    print()
    
    # Listar archivos a procesar
    ndjson_keys = list_keys(
        bucket=bucket_name, 
        prefix=in_prefix, 
        aws_conn_id=aws_conn_id, 
        suffix=".ndjson"
    )
    
    if not ndjson_keys:
        print(f"ℹ️  No hay archivos NDJSON en s3://{bucket_name}/{in_prefix}")
        return {
            "processed_files": 0,
            "written_files": [],
            "in_records": 0,
            "out_records": 0,
            "rejected_records": 0,
            "manifest_key": None
        }
    
    print(f"📋 Archivos encontrados: {len(ndjson_keys)}")
    print()
    
    # Contadores globales
    total_in = 0
    total_out = 0
    total_rejected = 0
    written_files: List[str] = []
    rejected_chunks: List[Dict[str, Any]] = []
    
    # Procesar cada archivo
    for idx, key in enumerate(sorted(ndjson_keys), 1):
        filename = key.split("/")[-1]
        print(f"[{idx}/{len(ndjson_keys)}] 📄 Procesando: {filename}")
        
        # Leer archivo
        try:
            raw = read_text(bucket=bucket_name, key=key, aws_conn_id=aws_conn_id)
        except Exception as e:
            print(f"  ❌ Error leyendo archivo: {e}")
            continue
        
        # Contadores por archivo
        file_in = 0
        file_out = 0
        file_rejected_guardrail = 0
        file_rejected_duplicate = 0
        
        out_lines: List[str] = []
        seen = set()
        
        # Procesar cada línea (chunk)
        for line_num, line in enumerate(raw.splitlines(), 1):
            line = (line or "").strip()
            if not line:
                continue
            
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"  ⚠️  Línea {line_num}: Error parseando JSON - {e}")
                continue
            
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            
            file_in += 1
            total_in += 1
            
            # Verificación con LLM
            if not NG.verificar_chunk_llm(text):
                file_rejected_guardrail += 1
                total_rejected += 1
                
                # Registrar en manifest
                rejected_chunks.append({
                    "file": filename,
                    "chunk_id": rec.get("id") or rec.get("chunk_id") or f"{filename}::{line_num}",
                    "reason": "failed_guardrail",
                    "text_preview": text[:150] + "..." if len(text) > 150 else text,
                    "text_length": len(text),
                    "boletin": rec.get("boletin"),
                    "fecha": rec.get("fecha"),
                    "op": rec.get("op"),
                    "page": rec.get("page"),
                    "line_number": line_num,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                continue
            
            # Deduplicación
            h = _hash_text(text)
            if h in seen:
                file_rejected_duplicate += 1
                total_rejected += 1
                
                # Registrar en manifest
                rejected_chunks.append({
                    "file": filename,
                    "chunk_id": rec.get("id") or rec.get("chunk_id") or f"{filename}::{line_num}",
                    "reason": "duplicate",
                    "text_preview": text[:150] + "..." if len(text) > 150 else text,
                    "text_length": len(text),
                    "boletin": rec.get("boletin"),
                    "fecha": rec.get("fecha"),
                    "op": rec.get("op"),
                    "page": rec.get("page"),
                    "line_number": line_num,
                    "duplicate_hash": h,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                continue
            
            seen.add(h)
            out_lines.append(json.dumps(rec, ensure_ascii=False))
            file_out += 1
            total_out += 1
        
        # Guardar archivo curado (si hay chunks aceptados)
        if out_lines:
            out_key = key.replace(in_prefix.rstrip("/"), out_prefix.rstrip("/"), 1)
            
            try:
                upload_text(
                    bucket=bucket_name,
                    key=out_key,
                    text="\n".join(out_lines) + "\n",
                    aws_conn_id=aws_conn_id
                )
                written_files.append(out_key)
                
                # Calcular % de aceptación
                acceptance_rate = (file_out / file_in * 100) if file_in > 0 else 0
                
                print(f"  ✅ Curado: {file_out}/{file_in} chunks ({acceptance_rate:.1f}% aceptados)")
                if file_rejected_guardrail > 0:
                    print(f"     ❌ Guardrail: {file_rejected_guardrail}")
                if file_rejected_duplicate > 0:
                    print(f"     🔄 Duplicados: {file_rejected_duplicate}")
                print(f"     📤 Guardado en: {out_key.split('/')[-1]}")
                
            except Exception as e:
                print(f"  ❌ Error guardando archivo curado: {e}")
        else:
            print(f"  ⚠️  Sin chunks válidos - archivo no generado")
        
        print()
    
    # Guardar manifest de rechazados
    manifest_key = None
    if rejected_chunks:
        manifest_key = f"{out_prefix.rstrip('/')}/REJECTED_MANIFEST.json"
        
        manifest = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "input_prefix": in_prefix,
                "output_prefix": out_prefix,
                "total_files_processed": len(ndjson_keys),
                "total_chunks_in": total_in,
                "total_chunks_out": total_out,
                "total_chunks_rejected": total_rejected,
            },
            "summary": {
                "by_reason": {
                    "failed_guardrail": sum(1 for r in rejected_chunks if r["reason"] == "failed_guardrail"),
                    "duplicate": sum(1 for r in rejected_chunks if r["reason"] == "duplicate"),
                },
                "rejection_rate": (total_rejected / total_in * 100) if total_in > 0 else 0,
                "acceptance_rate": (total_out / total_in * 100) if total_in > 0 else 0,
            },
            "rejected_chunks": rejected_chunks
        }
        
        try:
            upload_json(
                bucket=bucket_name,
                key=manifest_key,
                obj=manifest,
                aws_conn_id=aws_conn_id
            )
            print(f"📋 Manifest de rechazados guardado: s3://{bucket_name}/{manifest_key}")
            print()
        except Exception as e:
            print(f"❌ Error guardando manifest: {e}")
            print()
    
    # Resumen final
    print("="*70)
    print("📊 RESUMEN FINAL")
    print("="*70)
    print(f"Archivos procesados:      {len(ndjson_keys)}")
    print(f"Archivos curados creados: {len(written_files)}")
    print()
    print(f"Chunks totales entrada:   {total_in:,}")
    print(f"Chunks totales salida:    {total_out:,}")
    print(f"Chunks rechazados:        {total_rejected:,}")
    print()
    
    if total_in > 0:
        acceptance_rate = (total_out / total_in * 100)
        rejection_rate = (total_rejected / total_in * 100)
        print(f"Tasa de aceptación:       {acceptance_rate:.2f}%")
        print(f"Tasa de rechazo:          {rejection_rate:.2f}%")
        print()
    
    if rejected_chunks:
        failed_guardrail = sum(1 for r in rejected_chunks if r["reason"] == "failed_guardrail")
        duplicates = sum(1 for r in rejected_chunks if r["reason"] == "duplicate")
        
        print("Rechazos por razón:")
        print(f"  ❌ Guardrail fallido:   {failed_guardrail:,} ({failed_guardrail/total_rejected*100:.1f}%)")
        print(f"  🔄 Duplicados:          {duplicates:,} ({duplicates/total_rejected*100:.1f}%)")
        print()
        
        # Mostrar ejemplos de rechazos
        print("Ejemplos de rechazos:")
        print()
        
        # Guardrail failures
        guardrail_examples = [r for r in rejected_chunks if r["reason"] == "failed_guardrail"][:3]
        if guardrail_examples:
            print("  ❌ Guardrail (primeros 3):")
            for i, ex in enumerate(guardrail_examples, 1):
                print(f"     {i}. Archivo: {ex['file']}")
                print(f"        Chunk ID: {ex['chunk_id']}")
                print(f"        Texto: {ex['text_preview'][:100]}...")
                print()
        
        # Duplicates
        duplicate_examples = [r for r in rejected_chunks if r["reason"] == "duplicate"][:3]
        if duplicate_examples:
            print("  🔄 Duplicados (primeros 3):")
            for i, ex in enumerate(duplicate_examples, 1):
                print(f"     {i}. Archivo: {ex['file']}")
                print(f"        Chunk ID: {ex['chunk_id']}")
                print(f"        Texto: {ex['text_preview'][:100]}...")
                print()
    
    print("="*70)
    print("✅ Proceso completado")
    print("="*70)
    print()
    
    return {
        "processed_files": len(ndjson_keys),
        "written_files": written_files,
        "in_records": total_in,
        "out_records": total_out,
        "rejected_records": total_rejected,
        "manifest_key": manifest_key,
        "summary": {
            "acceptance_rate": (total_out / total_in * 100) if total_in > 0 else 0,
            "rejection_rate": (total_rejected / total_in * 100) if total_in > 0 else 0,
            "by_reason": {
                "failed_guardrail": sum(1 for r in rejected_chunks if r["reason"] == "failed_guardrail"),
                "duplicate": sum(1 for r in rejected_chunks if r["reason"] == "duplicate"),
            }
        }
    }