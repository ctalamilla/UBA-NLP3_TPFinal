# plugins/tasks/vector_pinecone.py
from __future__ import annotations

import os, io, json, math
from typing import List, Dict, Any, Optional, Iterable, Tuple
import numpy as np

# ✅ compatibilidad cliente Pinecone (nuevo y legacy)
try:
    from pinecone import Pinecone, ServerlessSpec
    _PC_STYLE = "new"
except Exception:
    import pinecone as _pc_legacy
    Pinecone = None
    ServerlessSpec = None
    _PC_STYLE = "legacy"

from sentence_transformers import SentenceTransformer
from tasks.s3_utilities import list_keys, read_text

# -------- helpers de embeddings --------
_model_cache: Dict[str, SentenceTransformer] = {}

def get_encoder(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    ).astype("float32")
    return embs

# -------- compat pinecone --------
def _pc_client() -> Any:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY no está definido en el entorno.")

    if _PC_STYLE == "new":
        return Pinecone(api_key=api_key)
    else:
        # cliente legacy
        env = os.environ.get("PINECONE_ENVIRONMENT") or os.environ.get("PINECONE_ENV") or "us-east1-gcp"
        _pc_legacy.init(api_key=api_key, environment=env)
        return _pc_legacy

def ensure_index(index_name: str, dim: int, metric: str = "cosine", cloud: str = "aws", region: Optional[str] = None) -> None:
    pc = _pc_client()
    if _PC_STYLE == "new":
        # intentar crear; si existe, ignorar
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
    else:
        return pc.Index(index_name)

# -------- leer NDJSON de MinIO --------
def iter_ndjson_chunks(bucket: str, prefix_chunks: str, aws_conn_id: str) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """
    Yields: (chunk_id, text, metadata)
    metadata incluye: doc_id, page, source
    """
    keys = list_keys(bucket=bucket, prefix=prefix_chunks, aws_conn_id=aws_conn_id, suffix=".ndjson")
    for key in sorted(keys):
        body = read_text(bucket=bucket, key=key, aws_conn_id=aws_conn_id)
        for line in body.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("id")
            txt = obj.get("text") or ""
            if not cid or not txt.strip():
                continue
            meta = {
                "doc_id": obj.get("doc_id"),
                "page": obj.get("page"),
                "source": obj.get("source"),
            }
            yield cid, txt, meta

# -------- upsert desde NDJSON --------
def upsert_vectors_from_ndjson(
    bucket: str,
    prefix_chunks: str,
    aws_conn_id: str,
    index_name: str,
    namespace: Optional[str],
    model_name: str,
    batch_size: int = 128,
    normalize: bool = True,
) -> Dict[str, Any]:
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for cid, txt, meta in iter_ndjson_chunks(bucket, prefix_chunks, aws_conn_id):
        ids.append(cid); texts.append(txt); metas.append(meta)

    if not texts:
        return {"indexed": 0, "dim": 0}

    enc = get_encoder(model_name)
    embs = encode_texts(enc, texts, batch_size=batch_size, normalize=normalize)
    dim = int(embs.shape[1])

    ensure_index(index_name=index_name, dim=dim, metric="cosine")

    index = _get_index(index_name)

    # upsert en lotes
    B = 100  # tamaño lote seguro
    total = len(ids)
    for i in range(0, total, B):
        sl = slice(i, i + B)
        vectors = [
            {"id": ids[j], "values": embs[j].tolist(), "metadata": metas[j] or {}}
            for j in range(sl.start, min(sl.stop, total))
        ]
        if _PC_STYLE == "new":
            index.upsert(vectors=vectors, namespace=namespace)
        else:
            index.upsert(vectors=vectors, namespace=namespace)
    return {"indexed": len(ids), "dim": dim, "namespace": namespace or ""}

# -------- query --------
def query_index(
    index_name: str,
    query: str,
    top_k: int,
    model_name: str,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    enc = get_encoder(model_name)
    q = encode_texts(enc, [query], batch_size=1, normalize=True)[0]
    index = _get_index(index_name)
    res = index.query(vector=q.tolist(), top_k=top_k, include_metadata=True, namespace=namespace)
    # normalizar salida a dict sencillo
    matches = []
    # ambos clientes devuelven estructura similar
    for m in (res.get("matches") if isinstance(res, dict) else res.matches):
        matches.append({
            "id": m["id"] if isinstance(m, dict) else m.id,
            "score": float(m.get("score", getattr(m, "score", 0.0))),
            "metadata": m.get("metadata", getattr(m, "metadata", {})),
        })
    return {"query": query, "top_k": top_k, "matches": matches, "namespace": namespace or ""}
