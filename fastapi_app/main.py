# fastapi_app/main.py
import os
from typing import Optional

from fastapi import FastAPI, APIRouter, Query
from pydantic import BaseModel

from .pipeline import RAGPipeline, build_openai
from .s3_boto import build_s3
from .vector_pinecone_api import query_index  # raw endpoint opcional

# --- Config vía env ---
S3_BUCKET      = os.getenv("S3_BUCKET", "respaldo2")
BM25_MODEL_KEY = os.getenv("BM25_MODEL_KEY", "rag/models/2025/bm25.pkl")
CHUNKS_PREFIX  = os.getenv("CHUNKS_PREFIX", "rag/chunks_labeled/2025/")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "boletines-2025")
PINECONE_NS    = os.getenv("PINECONE_NAMESPACE", "2025")
EMB_MODEL      = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

app = FastAPI(title="RAG API", version="0.5.0")
router = APIRouter()

pipeline: Optional[RAGPipeline] = None

@app.on_event("startup")
def _startup():
    global pipeline
    pipeline = RAGPipeline(
        s3_bucket=S3_BUCKET,
        bm25_key=BM25_MODEL_KEY,
        chunks_prefix=CHUNKS_PREFIX,
        pinecone_index=PINECONE_INDEX,
        pinecone_ns=PINECONE_NS,
        emb_model=EMB_MODEL,
        s3_client=build_s3(),
        openai_client=build_openai(),
    )

@router.get("/health")
def health():
    return {"ok": True}

# --------- RAW vector (opcional) ----------
class VectorQueryIn(BaseModel):
    query: str
    top_k: int = 5
    index_name: str = PINECONE_INDEX
    namespace: Optional[str] = PINECONE_NS
    model_name: str = EMB_MODEL

@router.post("/vector/query")
def vector_query(payload: VectorQueryIn):
    hits = query_index(
        index_name=payload.index_name,
        query_text=payload.query,
        top_k=payload.top_k,
        model_name=payload.model_name,
        namespace=payload.namespace,
    )
    return {"query": payload.query, "top_k": payload.top_k, "results": hits}

# --------- RAG completo ----------
class RAGQueryIn(BaseModel):
    query: str
    k_bm25: int = 50
    k_vec: int = 50
    k_final: int = 6
    per_page: int = 3
    do_rerank: bool = False
    debug: bool = True

@router.post("/query")
def rag_query(payload: RAGQueryIn):
    if pipeline is None:
        return {"error": "pipeline not initialized"}
    out = pipeline.run(
        query=payload.query,
        k_bm25=payload.k_bm25,
        k_vec=payload.k_vec,
        k_final=payload.k_final,
        per_page=payload.per_page,
        do_rerank=payload.do_rerank,
        debug=payload.debug,
    )
    return out

# --------- Conveniencia GET /ask ----------
@router.get("/ask")
def ask(
    q: str = Query(..., description="Consulta del usuario"),
    k_bm25: int = 50,
    k_vec: int = 50,
    k_final: int = 6,
    per_page: int = 3,
    do_rerank: bool = False,
    debug: bool = True,
):
    if pipeline is None:
        return {"error": "pipeline not initialized"}
    out = pipeline.run(
        query=q,
        k_bm25=k_bm25,
        k_vec=k_vec,
        k_final=k_final,
        per_page=per_page,
        do_rerank=do_rerank,
        debug=debug,
    )
    return out

# Montamos rutas sin prefijo y con /api
app.include_router(router)
app.include_router(router, prefix="/api")
