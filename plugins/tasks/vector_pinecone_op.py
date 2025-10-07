# plugins/tasks/vector_pinecone_op.py
from __future__ import annotations
import os, json
from typing import List, Dict, Any, Optional, Iterable, Tuple

# ---- Pinecone (nuevo SDK con fallback legacy) ----
try:
    from pinecone import Pinecone, ServerlessSpec
    _PC_STYLE = "new"
except Exception:
    import pinecone as _pc_legacy
    Pinecone = None
    ServerlessSpec = None
    _PC_STYLE = "legacy"

from sentence_transformers import SentenceTransformer
import numpy as np

# Reutilizamos tus utilidades S3 existentes
from tasks.s3_utilities import list_keys, read_text

# ======================================================
# Embeddings cache
# ======================================================
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

def _get_encoder(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]

def _encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    ).astype("float32")
    return embs

# ======================================================
# Pinecone helpers (compat new/legacy)
# ======================================================
def _pc_client():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY no está definido en el entorno.")
    if _PC_STYLE == "new":
        return Pinecone(api_key=api_key)
    # legacy
    env = os.environ.get("PINECONE_ENVIRONMENT") or os.environ.get("PINECONE_ENV") or "us-east1-gcp"
    _pc_legacy.init(api_key=api_key, environment=env)
    return _pc_legacy

def _ensure_index(index_name: str, dim: int, metric: str = "cosine",
                  cloud: str = "aws", region: Optional[str] = None) -> None:
    pc = _pc_client()
    if _PC_STYLE == "new":
        try:
            spec = ServerlessSpec(cloud=cloud, region=region or os.environ.get("PINECONE_REGION", "us-east-1"))
            pc.create_index(name=index_name, dimension=dim, metric=metric, spec=spec)
        except Exception as e:
            msg = str(e).lower()
            if "already exists" not in msg and "resource_conflict" not in msg:
                raise
    else:
        if index_name not in pc.list_indexes():
            pc.create_index(name=index_name, dimension=dim, metric=metric)

def _get_index(index_name: str):
    pc = _pc_client()
    if _PC_STYLE == "new":
        return pc.Index(index_name)
    return pc.Index(index_name)

# ======================================================
# NDJSON reader (robusto a nombres de campo)
# ======================================================
def _iter_ndjson_chunks_op(
    bucket: str,
    prefix_chunks: str,
    aws_conn_id: Optional[str],
    min_chars: int = 20
) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """
    Itera sobre todos los .ndjson encontrados en prefix_chunks.
    Yields: (chunk_id, text, metadata)
    - Admite claves: id / chunk_id para el ID
    - Admite claves: text / chunk / content para el texto
    - Pasa metadatos heredados (doc_id, page, source, boletin, fecha, op, classification.categoria)
    """
    ndjson_keys = list_keys(bucket=bucket, prefix=prefix_chunks, aws_conn_id=aws_conn_id, suffix=".ndjson")
    print(f"[PC/OP] ndjson_files={len(ndjson_keys)} prefix={prefix_chunks}")
    for key in sorted(ndjson_keys):
        body = read_text(bucket=bucket, key=key, aws_conn_id=aws_conn_id)
        if not body:
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            cid  = rec.get("id") or rec.get("chunk_id")
            txt  = rec.get("text") or rec.get("chunk") or rec.get("content")
            if not cid or not txt or len((txt or "").strip()) < min_chars:
                continue

            meta = {
                "doc_id":  rec.get("doc_id"),
                "page":    rec.get("page"),
                "source":  rec.get("source"),
                "boletin": rec.get("boletin"),
                "fecha":   rec.get("fecha"),
                "op":      rec.get("op"),
            }
            # categoría si viene anidada bajo classification
            cat = None
            if isinstance(rec.get("classification"), dict):
                cat = rec["classification"].get("categoria")
            if cat:
                meta["categoria"] = cat

            yield cid, txt, meta

# ======================================================
# Upsert OP
# ======================================================
def upsert_vectors_from_ndjson_op(
    bucket: str,
    prefix_chunks: str,
    aws_conn_id: Optional[str],
    index_name: str,
    namespace: Optional[str],
    model_name: str,
    batch_size: int = 128,
    min_chars: int = 20,
    normalize: bool = True,
) -> Dict[str, Any]:
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    parsed = 0
    for cid, txt, meta in _iter_ndjson_chunks_op(bucket, prefix_chunks, aws_conn_id, min_chars=min_chars):
        ids.append(cid); texts.append(txt); metas.append(meta)
        parsed += 1

    print(f"[PC/OP] parsed_kept={parsed} (tras min_chars>={min_chars})")
    if not texts:
        print("⚠️ No hay vectores para indexar (OP).")
        return {"indexed": 0, "dim": 0, "namespace": namespace or ""}

    enc = _get_encoder(model_name)
    embs = _encode_texts(enc, texts, batch_size=batch_size, normalize=normalize)
    dim = int(embs.shape[1])
    print(f"[PC/OP] embedding model={model_name} dim={dim} total_vectors={len(ids)}")

    _ensure_index(index_name=index_name, dim=dim, metric="cosine")
    index = _get_index(index_name)

    # upsert por lotes pequeños para robustez
    B = 100
    total = len(ids)
    for i in range(0, total, B):
        sub_ids  = ids[i:i+B]
        sub_embs = embs[i:i+B]
        sub_meta = metas[i:i+B]
        vectors = [
            {"id": sub_ids[j], "values": sub_embs[j].tolist(), "metadata": sub_meta[j] or {}}
            for j in range(len(sub_ids))
        ]
        index.upsert(vectors=vectors, namespace=namespace)

    print(f"✅ Upsert Pinecone (OP): indexed={total}, dim={dim}, ns={namespace}")
    return {"indexed": total, "dim": dim, "namespace": namespace or ""}

# (opcional) query helper compatible
def query_index_op(
    index_name: str,
    query: str,
    top_k: int,
    model_name: str,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    enc = _get_encoder(model_name)
    q = _encode_texts(enc, [query], batch_size=1, normalize=True)[0]
    index = _get_index(index_name)
    res = index.query(vector=q.tolist(), top_k=top_k, include_metadata=True, namespace=namespace)
    matches = []
    # normaliza respuesta
    if isinstance(res, dict):
        raw = res.get("matches", [])
        for m in raw:
            matches.append({"id": m.get("id"), "score": float(m.get("score", 0.0)), "metadata": m.get("metadata", {})})
    else:
        for m in getattr(res, "matches", []):
            matches.append({"id": m.id, "score": float(m.score), "metadata": dict(getattr(m, "metadata", {}) or {})})
    return {"query": query, "top_k": top_k, "matches": matches, "namespace": namespace or ""}
