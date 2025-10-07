# # fastapi_app/vector_pinecone_api.py
# from __future__ import annotations

# import os, time
# from typing import Optional, List, Dict, Any

# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer

# # -----------------------
# # Cliente Pinecone
# # -----------------------
# def _pc() -> Pinecone:
#     api_key = os.getenv("PINECONE_API_KEY")
#     if not api_key:
#         raise RuntimeError("PINECONE_API_KEY no está configurada.")
#     return Pinecone(api_key=api_key)

# def ensure_index(
#     index_name: str,
#     dim: int,
#     metric: str = "cosine",
#     cloud: str = "aws",
#     region: Optional[str] = None,
# ) -> None:
#     pc = _pc()
#     region = region or os.getenv("PINECONE_REGION", "us-east-1")
#     existing = [it["name"] for it in pc.list_indexes().get("indexes", [])]
#     if index_name in existing:
#         return
#     pc.create_index(
#         name=index_name,
#         dimension=dim,
#         metric=metric,
#         spec=ServerlessSpec(cloud=cloud, region=region),
#     )
#     for _ in range(30):
#         info = pc.describe_index(index_name)
#         if info and info.get("status", {}).get("ready"):
#             return
#         time.sleep(2)
#     raise RuntimeError(f"Index '{index_name}' no quedó listo a tiempo.")

# # -----------------------
# # Embeddings cache
# # -----------------------
# _EMB_CACHE: Dict[str, SentenceTransformer] = {}

# def _embedder(model_name: str) -> SentenceTransformer:
#     if model_name not in _EMB_CACHE:
#         _EMB_CACHE[model_name] = SentenceTransformer(model_name)
#     return _EMB_CACHE[model_name]

# def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
#     emb = _embedder(model_name)
#     return emb.encode(texts, normalize_embeddings=True).tolist()

# # -----------------------
# # Query
# # -----------------------
# def query_index(
#     index_name: str,
#     query_text: str,
#     top_k: int = 5,
#     model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#     namespace: Optional[str] = None,
# ) -> List[Dict[str, Any]]:
#     pc = _pc()
#     index = pc.Index(index_name)
#     qvec = embed_texts([query_text], model_name=model_name)[0]
#     res = index.query(
#         vector=qvec,
#         top_k=top_k,
#         include_metadata=True,
#         namespace=namespace or None,
#     )
#     matches = res.get("matches", []) if isinstance(res, dict) else []
#     out: List[Dict[str, Any]] = []
#     for m in matches:
#         out.append(
#             {
#                 "id": m.get("id"),
#                 "score": m.get("score"),
#                 "metadata": m.get("metadata") or {},
#             }
#         )
#     return out
# fastapi_app/vector_pinecone_api.py
# fastapi_app/vector_pinecone_api.py
# fastapi_app/vector_pinecone_api.py
from __future__ import annotations

import os
import time
import logging
from typing import Optional, List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# -----------------------
# Cliente Pinecone
# -----------------------
def _pc() -> Pinecone:
    """Crea cliente Pinecone con API key del entorno"""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY no está configurada en variables de entorno")
    return Pinecone(api_key=api_key)

def ensure_index(
    index_name: str,
    dim: int,
    metric: str = "cosine",
    cloud: str = "aws",
    region: Optional[str] = None,
) -> None:
    """
    Asegura que el índice existe en Pinecone.
    Si no existe, lo crea y espera a que esté listo.
    """
    pc = _pc()
    region = region or os.getenv("PINECONE_REGION", "us-east-1")
    
    try:
        existing_indexes = pc.list_indexes()
        existing_names = [it["name"] for it in existing_indexes.get("indexes", [])]
        
        if index_name in existing_names:
            logger.info(f"✅ Índice '{index_name}' ya existe")
            return
        
        logger.info(f"🔨 Creando índice '{index_name}' (dim={dim}, metric={metric})")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        
        for i in range(30):
            info = pc.describe_index(index_name)
            if info and info.get("status", {}).get("ready"):
                logger.info(f"✅ Índice '{index_name}' listo después de {i*2}s")
                return
            logger.debug(f"⏳ Esperando índice... ({i*2}s)")
            time.sleep(2)
        
        raise RuntimeError(f"❌ Índice '{index_name}' no quedó listo después de 60s")
        
    except Exception as e:
        logger.error(f"❌ Error con índice Pinecone: {e}")
        raise

# -----------------------
# Embeddings con cache
# -----------------------
_EMB_CACHE: Dict[str, SentenceTransformer] = {}

def _embedder(model_name: str) -> SentenceTransformer:
    """Obtiene modelo de embeddings (usa cache)"""
    if model_name not in _EMB_CACHE:
        logger.info(f"📥 Cargando modelo de embeddings: {model_name}")
        try:
            _EMB_CACHE[model_name] = SentenceTransformer(model_name)
            logger.info(f"✅ Modelo cargado: {model_name}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo {model_name}: {e}")
            raise
    return _EMB_CACHE[model_name]

def embed_texts(texts: List[str], model_name: str) -> List[List[float]]:
    """Genera embeddings para una lista de textos"""
    if not texts:
        return []
    
    try:
        emb = _embedder(model_name)
        vectors = emb.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
    except Exception as e:
        logger.error(f"❌ Error generando embeddings: {e}")
        raise

# -----------------------
# Query a Pinecone
# -----------------------
def query_index(
    index_name: str,
    query_text: str,
    top_k: int = 5,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    namespace: Optional[str] = None,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Consulta el índice vectorial de Pinecone
    
    Args:
        index_name: Nombre del índice en Pinecone
        query_text: Texto de consulta
        top_k: Número de resultados a retornar
        model_name: Modelo para generar embedding de la query
        namespace: Namespace de Pinecone (opcional)
        filter_dict: Filtros metadata (opcional)
    
    Returns:
        Lista de matches con id, score y metadata
    """
    if not query_text or not query_text.strip():
        logger.warning("⚠️ Query vacía recibida")
        return []
    
    try:
        pc = _pc()
        index = pc.Index(index_name)
        
        # Generar embedding
        logger.debug(f"🔍 Query a '{index_name}': '{query_text[:100]}...'")
        qvec = embed_texts([query_text], model_name=model_name)[0]
        
        # Consultar índice
        query_params = {
            "vector": qvec,
            "top_k": top_k,
            "include_metadata": True,
        }
        
        if namespace:
            query_params["namespace"] = namespace
        
        if filter_dict:
            query_params["filter"] = filter_dict
        
        res = index.query(**query_params)
        
        # Procesar matches - manejar objeto Pinecone
        matches = []
        if hasattr(res, 'matches'):
            matches = res.matches or []
        elif isinstance(res, dict):
            matches = res.get("matches", [])
        
        out: List[Dict[str, Any]] = []
        for m in matches:
            # Manejar tanto objetos como dicts
            match_id = m.id if hasattr(m, 'id') else m.get("id")
            match_score = m.score if hasattr(m, 'score') else m.get("score")
            match_meta = m.metadata if hasattr(m, 'metadata') else m.get("metadata", {})
            
            out.append({
                "id": match_id,
                "score": match_score,
                "metadata": match_meta or {},
            })
        
        if out:
            logger.info(f"✅ Encontrados {len(out)} matches (top score: {out[0]['score']:.4f})")
        else:
            logger.warning(f"⚠️ No se encontraron matches para la query")
        
        return out
        
    except Exception as e:
        logger.error(f"❌ Error consultando Pinecone: {e}", exc_info=True)
        raise

# -----------------------
# Batch upsert
# -----------------------
def batch_upsert(
    index_name: str,
    vectors: List[tuple],
    namespace: Optional[str] = None,
    batch_size: int = 100,
) -> int:
    """
    Inserta vectores en batch al índice
    
    Args:
        index_name: Nombre del índice
        vectors: Lista de tuplas (id, vector, metadata)
        namespace: Namespace opcional
        batch_size: Tamaño de lote
    
    Returns:
        Número total de vectores insertados
    """
    if not vectors:
        logger.warning("⚠️ No hay vectores para insertar")
        return 0
    
    try:
        pc = _pc()
        index = pc.Index(index_name)
        
        total = len(vectors)
        logger.info(f"📤 Insertando {total} vectores en batches de {batch_size}")
        
        inserted = 0
        for i in range(0, total, batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)
            inserted += len(batch)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"   Progreso: {inserted}/{total}")
        
        logger.info(f"✅ {inserted} vectores insertados exitosamente")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ Error en batch upsert: {e}")
        raise

# -----------------------
# Delete by filter
# -----------------------
def delete_by_filter(
    index_name: str,
    filter_dict: Dict[str, Any],
    namespace: Optional[str] = None,
) -> None:
    """
    Elimina vectores que coincidan con el filtro
    
    Args:
        index_name: Nombre del índice
        filter_dict: Diccionario de filtros metadata
        namespace: Namespace opcional
    """
    try:
        pc = _pc()
        index = pc.Index(index_name)
        
        logger.info(f"🗑️ Eliminando vectores con filtro: {filter_dict}")
        index.delete(filter=filter_dict, namespace=namespace)
        logger.info("✅ Vectores eliminados exitosamente")
        
    except Exception as e:
        logger.error(f"❌ Error eliminando vectores: {e}")
        raise