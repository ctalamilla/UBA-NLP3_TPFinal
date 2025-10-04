# fastapi_app/vector_pinecone_api.py
from __future__ import annotations

import os, time
from typing import Optional, List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# -----------------------
# Cliente Pinecone
# -----------------------
def _pc() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY no está configurada.")
    return Pinecone(api_key=api_key)

def ensure_index(
    index_name: str,
    dim: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: Optional[str] = None,
) -> None:
    pc = _pc()
    region = region or os.getenv("PINECONE_REGION", "us-east-1")
    existing = [it["name"] for it in pc.list_indexes().get("indexes", [])]
    if index_name in existing:
        return
    pc.create_index(
        name=index_name,
        dimension=dim,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    for _ in range(30):
        info = pc.describe_index(index_name)
        if info and info.get("status", {}).get("ready"):
            return
        time.sleep(2)
    raise RuntimeError(f"Index '{index_name}' no quedó listo a tiempo.")

# -----------------------
# Embeddings cache
# -----------------------
_EMB_CACHE: Dict[str, SentenceTransformer] = {}

def _embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMB_CACHE:
        _EMB_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMB_CACHE[model_name]

def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    emb = _embedder(model_name)
    return emb.encode(texts, normalize_embeddings=True).tolist()

# -----------------------
# Query
# -----------------------
def query_index(
    index_name: str,
    query_text: str,
    top_k: int = 5,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    pc = _pc()
    index = pc.Index(index_name)
    qvec = embed_texts([query_text], model_name=model_name)[0]
    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace or None,
    )
    matches = res.get("matches", []) if isinstance(res, dict) else []
    out: List[Dict[str, Any]] = []
    for m in matches:
        out.append(
            {
                "id": m.get("id"),
                "score": m.get("score"),
                "metadata": m.get("metadata") or {},
            }
        )
    return out
