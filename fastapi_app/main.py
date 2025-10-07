# # fastapi_app/main.py
# import os
# from typing import Optional

# from fastapi import FastAPI, APIRouter, Query
# from pydantic import BaseModel

# from .pipeline import RAGPipeline, build_openai
# from .s3_boto import build_s3
# from .vector_pinecone_api import query_index  # raw endpoint opcional

# # --- Config vía env ---
# S3_BUCKET      = os.getenv("S3_BUCKET", "respaldo2")
# BM25_MODEL_KEY = os.getenv("BM25_MODEL_KEY", "rag/models/2025/bm25.pkl")
# CHUNKS_PREFIX  = os.getenv("CHUNKS_PREFIX", "rag/chunks_op/2025/")#"rag/chunks_labeled/2025/"
# PINECONE_INDEX = os.getenv("PINECONE_INDEX", "boletines-2025")
# PINECONE_NS    = os.getenv("PINECONE_NAMESPACE", "2025")
# EMB_MODEL      = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# app = FastAPI(title="RAG API", version="0.5.0")
# router = APIRouter()

# pipeline: Optional[RAGPipeline] = None

# @app.on_event("startup")
# def _startup():
#     global pipeline
#     pipeline = RAGPipeline(
#         s3_bucket=S3_BUCKET,
#         bm25_key=BM25_MODEL_KEY,
#         chunks_prefix=CHUNKS_PREFIX,
#         pinecone_index=PINECONE_INDEX,
#         pinecone_ns=PINECONE_NS,
#         emb_model=EMB_MODEL,
#         s3_client=build_s3(),
#         openai_client=build_openai(),
#     )

# @router.get("/health")
# def health():
#     return {"ok": True}

# # --------- RAW vector (opcional) ----------
# class VectorQueryIn(BaseModel):
#     query: str
#     top_k: int = 5
#     index_name: str = PINECONE_INDEX
#     namespace: Optional[str] = PINECONE_NS
#     model_name: str = EMB_MODEL

# @router.post("/vector/query")
# def vector_query(payload: VectorQueryIn):
#     hits = query_index(
#         index_name=payload.index_name,
#         query_text=payload.query,
#         top_k=payload.top_k,
#         model_name=payload.model_name,
#         namespace=payload.namespace,
#     )
#     return {"query": payload.query, "top_k": payload.top_k, "results": hits}

# # --------- RAG completo ----------
# class RAGQueryIn(BaseModel):
#     query: str
#     k_bm25: int = 50
#     k_vec: int = 50
#     k_final: int = 6
#     per_page: int = 3
#     do_rerank: bool = False
#     debug: bool = True

# @router.post("/query")
# def rag_query(payload: RAGQueryIn):
#     if pipeline is None:
#         return {"error": "pipeline not initialized"}
#     out = pipeline.run(
#         query=payload.query,
#         k_bm25=payload.k_bm25,
#         k_vec=payload.k_vec,
#         k_final=payload.k_final,
#         per_page=payload.per_page,
#         do_rerank=payload.do_rerank,
#         debug=payload.debug,
#     )
#     return out

# # --------- Conveniencia GET /ask ----------
# @router.get("/ask")
# def ask(
#     q: str = Query(..., description="Consulta del usuario"),
#     k_bm25: int = 50,
#     k_vec: int = 50,
#     k_final: int = 6,
#     per_page: int = 3,
#     do_rerank: bool = False,
#     debug: bool = True,
# ):
#     if pipeline is None:
#         return {"error": "pipeline not initialized"}
#     out = pipeline.run(
#         query=q,
#         k_bm25=k_bm25,
#         k_vec=k_vec,
#         k_final=k_final,
#         per_page=per_page,
#         do_rerank=do_rerank,
#         debug=debug,
#     )
#     return out

# # Montamos rutas sin prefijo y con /api
# app.include_router(router)
# app.include_router(router, prefix="/api")
# fastapi_app/main.py
# fastapi_app/main.py
import os
import logging
from typing import Optional

from fastapi import FastAPI, APIRouter, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline, build_openai
from .s3_boto import build_s3
from .vector_pinecone_api import query_index

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config vía env ---
S3_BUCKET      = os.getenv("S3_BUCKET", "respaldo2")
BM25_MODEL_KEY = os.getenv("BM25_MODEL_KEY", "rag/models/2025/bm25.pkl")
CHUNKS_PREFIX  = os.getenv("CHUNKS_PREFIX", "rag/chunks_op/2025/")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "boletines-2025")
PINECONE_NS    = os.getenv("PINECONE_NAMESPACE", "2025")
EMB_MODEL      = os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

app = FastAPI(
    title="RAG API - Boletín Oficial",
    version="0.6.0",
    description="API RAG para consultas sobre el Boletín Oficial"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()
pipeline: Optional[RAGPipeline] = None

@app.on_event("startup")
async def startup():
    global pipeline
    try:
        logger.info("🚀 Inicializando RAG Pipeline...")
        logger.info(f"   - S3 Bucket: {S3_BUCKET}")
        logger.info(f"   - BM25 Key: {BM25_MODEL_KEY}")
        logger.info(f"   - Chunks Prefix: {CHUNKS_PREFIX}")
        logger.info(f"   - Pinecone Index: {PINECONE_INDEX}")
        
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
        logger.info("✅ Pipeline inicializado correctamente")
    except Exception as e:
        logger.error(f"❌ Error inicializando pipeline: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    logger.info("👋 Cerrando API...")

@router.get("/")
def root():
    return {
        "service": "RAG API - Boletín Oficial",
        "version": "0.6.0",
        "status": "ok" if pipeline else "initializing",
        "endpoints": ["/health", "/ask", "/query", "/vector/query"]
    }

@router.get("/health")
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return {
        "status": "healthy",
        "pipeline": "ready",
        "config": {
            "s3_bucket": S3_BUCKET,
            "chunks_prefix": CHUNKS_PREFIX,
            "pinecone_index": PINECONE_INDEX,
        }
    }

# --------- RAW vector (opcional) ----------
class VectorQueryIn(BaseModel):
    query: str = Field(..., description="Texto de consulta")
    top_k: int = Field(5, ge=1, le=100, description="Número de resultados")
    index_name: str = Field(PINECONE_INDEX, description="Nombre del índice")
    namespace: Optional[str] = Field(PINECONE_NS, description="Namespace de Pinecone")
    model_name: str = Field(EMB_MODEL, description="Modelo de embeddings")

@router.post("/vector/query")
def vector_query(payload: VectorQueryIn):
    """Consulta directa al índice vectorial (sin BM25 ni rerank)"""
    try:
        hits = query_index(
            index_name=payload.index_name,
            query_text=payload.query,
            top_k=payload.top_k,
            model_name=payload.model_name,
            namespace=payload.namespace,
        )
        return {
            "query": payload.query,
            "top_k": payload.top_k,
            "results": hits,
            "count": len(hits)
        }
    except Exception as e:
        logger.error(f"Error en vector_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------- RAG completo ----------
class RAGQueryIn(BaseModel):
    query: str = Field(..., description="Consulta del usuario")
    k_bm25: int = Field(50, ge=1, le=200, description="Top-K para BM25")
    k_vec: int = Field(50, ge=1, le=200, description="Top-K para búsqueda vectorial")
    k_final: int = Field(6, ge=1, le=20, description="Número final de chunks")
    per_page: int = Field(3, ge=1, le=10, description="Chunks por página")
    do_rerank: bool = Field(False, description="Aplicar rerank con CrossEncoder")
    debug: bool = Field(False, description="Incluir información de debug")

@router.post("/query")
def rag_query(payload: RAGQueryIn):
    """
    Consulta RAG completa con:
    - Búsqueda híbrida (BM25 + Vector)
    - Fusión RRF
    - Rerank opcional
    - Generación de respuesta con LLM
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"📝 Query: {payload.query[:100]}...")
        out = pipeline.run(
            query=payload.query,
            k_bm25=payload.k_bm25,
            k_vec=payload.k_vec,
            k_final=payload.k_final,
            per_page=payload.per_page,
            do_rerank=payload.do_rerank,
            debug=payload.debug,
        )
        logger.info(f"✅ Respuesta generada con {len(out.get('results', []))} chunks")
        return out
    except Exception as e:
        logger.error(f"❌ Error en rag_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------- Endpoint GET conveniente ----------
@router.get("/ask")
def ask(
    q: str = Query(..., description="Consulta del usuario", min_length=3),
    k_bm25: int = Query(50, ge=1, le=200, description="Top-K para BM25"),
    k_vec: int = Query(50, ge=1, le=200, description="Top-K vectorial"),
    k_final: int = Query(6, ge=1, le=20, description="Chunks finales"),
    per_page: int = Query(3, ge=1, le=10, description="Chunks por página"),
    do_rerank: bool = Query(False, description="Activar rerank"),
    debug: bool = Query(False, description="Info de debug"),
):
    """
    Endpoint GET simplificado para consultas rápidas.
    Ejemplo: /ask?q=quiebras en Salta&k_final=5
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"🔍 GET /ask: {q[:100]}...")
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
    except Exception as e:
        logger.error(f"❌ Error en /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --------- Stats (opcional) ----------
@router.get("/stats")
def stats():
    """Estadísticas del índice BM25"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        bm25 = pipeline.bm25
        return {
            "total_docs": len(bm25.doc_ids),
            "sample_doc_ids": bm25.doc_ids[:10] if hasattr(bm25, 'doc_ids') else [],
            "bm25_key": pipeline.bm25_key,
            "chunks_prefix": pipeline.chunks_prefix,
        }
    except Exception as e:
        logger.error(f"Error obteniendo stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Montamos rutas
app.include_router(router)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)