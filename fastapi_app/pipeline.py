# fastapi_app/pipeline.py
from __future__ import annotations
import os
import json
import pickle
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from tasks.bm25_index import BM25Index
from .s3_boto import build_s3
from .vector_pinecone_api import ensure_index, query_index
from .performance import PerformanceTracker
logger = logging.getLogger(__name__)

# -----------------------
# OpenAI
# -----------------------
def build_openai() -> Optional[OpenAI]:
    api = os.getenv("OPENAI_API_KEY")
    if api:
        logger.info("✅ OpenAI client configurado")
        return OpenAI(api_key=api)
    logger.warning("⚠️ OPENAI_API_KEY no configurada - modo sin LLM")
    return None

# -----------------------
# NDJSON helpers (S3)
# -----------------------
def read_ndjson_lines(s3, bucket: str, key: str) -> List[Dict[str, Any]]:
    """Lee y parsea un archivo NDJSON desde S3"""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        raw = obj["Body"].read().decode("utf-8", errors="replace")
        out = []
        for i, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Error parseando línea {i} en {key}: {e}")
                continue
        return out
    except Exception as e:
        logger.error(f"Error leyendo NDJSON {key}: {e}")
        raise

def page_id_to_ndjson_key(chunks_prefix: str, page_id: str) -> str:
    """
    Convierte page_id a la ruta del NDJSON
    Ej: '22036_2025-09-22_p1' -> 'rag/chunks_op/2025/22036_2025-09-22.ndjson'
    """
    base = page_id.rsplit("_p", 1)[0]
    return f"{chunks_prefix.rstrip('/')}/{base}.ndjson"

# -----------------------
# Rerank (opcional)
# -----------------------
def optional_rerank(
    query: str,
    candidates: List[Tuple[str, str, Dict[str, Any]]]
) -> List[Tuple[str, str, Dict[str, Any], float]]:
    """Reordena candidatos usando CrossEncoder si RERANK_MODEL está configurado"""
    model_name = os.getenv("RERANK_MODEL")
    if not model_name:
        logger.debug("Rerank deshabilitado (RERANK_MODEL no configurado)")
        return [(cid, txt, meta, 0.0) for (cid, txt, meta) in candidates]
    
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"🔄 Aplicando rerank con {model_name}")
        ce = CrossEncoder(model_name)
        pairs = [(query, txt) for _, txt, _ in candidates]
        scores = ce.predict(pairs)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        result = [
            (candidates[i][0], candidates[i][1], candidates[i][2], float(scores[i]))
            for i in order
        ]
        logger.info(f"✅ Rerank completado - Top score: {result[0][3]:.4f}")
        return result
    except Exception as e:
        logger.error(f"❌ Error en rerank: {e}")
        return [(cid, txt, meta, 0.0) for (cid, txt, meta) in candidates]

# -----------------------
# Resumen y respuesta con LLM
# -----------------------
def rag_summary_llm(
    client: Optional[OpenAI],
    query: str,
    chunks: List[str],
    max_chars: int = 500,
    tracker: Optional[PerformanceTracker] = None  # ← NUEVO parámetro
) -> Tuple[str, Optional[Any]]:  # ← NUEVO tipo de retorno
    """Genera resumen del contexto usando LLM"""
    if not chunks:
        return "", None  # ← CAMBIADO
    
    if not client:
        logger.debug("Sin OpenAI client - retornando contexto truncado")
        return "\n\n".join(chunks)[:max_chars], None  # ← CAMBIADO
    
    joined = "\n\n".join(f"- {c}" for c in chunks)[:4000]
    prompt = (
        f"Resumí de forma concisa y factual el siguiente contexto para responder la consulta.\n"
        f"Consulta: {query}\n\nContexto:\n{joined}\n\n"
        f"Devolvé SOLO el resumen (máx {max_chars} caracteres), sin viñetas ni comentarios."
    )
    
    try:
        model = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
        logger.debug(f"Generando resumen con {model}")
        
        start_time = time.time()  # ← NUEVO
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        elapsed = time.time() - start_time  # ← NUEVO
        
        summary = (out.choices[0].message.content or "")[:max_chars].strip()
        logger.info(f"✅ Resumen generado ({len(summary)} chars, {elapsed:.2f}s)")  # ← MODIFICADO
        
        # Track metrics  # ← NUEVO BLOQUE
        if tracker:
            tracker.timings["llm_summary_time"] = elapsed
            tracker.add_llm_metrics("summary", out.usage, model)
        
        return summary, out.usage  # ← CAMBIADO
    except Exception as e:
        logger.error(f"❌ Error generando resumen: {e}")
        return "\n\n".join(chunks)[:max_chars], None  # ← CAMBIADO

def answer_llm(
    client: Optional[OpenAI],
    query: str,
    context_chunks: List[str],
    summary: str,
    metadata_list: Optional[List[Dict[str, Any]]] = None,
    tracker: Optional[PerformanceTracker] = None  # ← NUEVO parámetro
) -> Tuple[str, Optional[Any]]:  # ← NUEVO tipo de retorno
    """Genera respuesta final usando LLM con contexto enriquecido y metadatos estructurados"""
    if not context_chunks:
        return "No hay contexto disponible.", None  # ← CAMBIADO
    
    if not client:
        logger.debug("Sin OpenAI client - retornando respuesta por defecto")
        return "No está especificado en las fuentes.", None  # ← CAMBIADO
    
    ctx = "\n\n".join(context_chunks)[:6000]
    
    # Construir sección de metadatos estructurados
    metadata_section = ""
    if metadata_list:
        metadata_section = "\n\nMETADATOS ESTRUCTURADOS DE LAS FUENTES:\n"
        for i, meta in enumerate(metadata_list, 1):
            meta_info = []
            if meta.get("boletin"):
                meta_info.append(f"Boletín: {meta['boletin']}")
            if meta.get("fecha"):
                meta_info.append(f"Fecha: {meta['fecha']}")
            if meta.get("op"):
                meta_info.append(f"OP: {meta['op']}")
            
            if meta_info:
                metadata_section += f"Fuente {i}: {' | '.join(meta_info)}\n"
    
    prompt = (
        "Usá SOLO la información del CONTEXTO y los METADATOS para responder la CONSULTA.\n"
        "Los metadatos te dan información precisa sobre boletines, fechas y números de OP.\n"
        "Incluí referencias específicas en tu respuesta (ej: 'Según el Boletín X del fecha Y...').\n"
        "Si la respuesta no está en el contexto, decí 'No está especificado en las fuentes.'\n\n"
        f"CONSULTA: {query}\n"
        f"{metadata_section}\n"
        f"RESUMEN CONTEXTO:\n{summary}\n\n"
        f"CONTEXTO COMPLEMENTARIO:\n{ctx}\n"
    )
    
    try:
        model = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini")
        logger.debug(f"Generando respuesta con {model}")
        
        start_time = time.time()  # ← NUEVO
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        elapsed = time.time() - start_time  # ← NUEVO
        
        answer = (out.choices[0].message.content or "").strip()
        logger.info(f"✅ Respuesta generada ({len(answer)} chars, {elapsed:.2f}s)")  # ← MODIFICADO
        
        # Track metrics  # ← NUEVO BLOQUE
        if tracker:
            tracker.timings["llm_answer_time"] = elapsed
            tracker.add_llm_metrics("answer", out.usage, model)
        
        return answer, out.usage  # ← CAMBIADO
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "Error al generar la respuesta.", None  # ← CAMBIADO

def verificar_respuesta_llm(
    client: Optional[OpenAI],
    query: str,
    respuesta: str,
    context_chunks: List[str],
    tracker: Optional[PerformanceTracker] = None  # ← NUEVO parámetro
) -> Tuple[str, Optional[Any]]:  # ← NUEVO tipo de retorno
    """
    Verifica si la respuesta generada está respaldada por los documentos
    Retorna una evaluación con ✅ / ⚠️ / ❌
    """
    if not client:
        return "⚠️ (verificador LLM no disponible)", None  # ← CAMBIADO
    
    if not context_chunks or not respuesta:
        return "⚠️ (sin contexto o respuesta para verificar)", None  # ← CAMBIADO
    
    evidencias = "\n\n".join(context_chunks)[:4000]
    prompt = f"""Tu tarea es verificar si la respuesta está coherente y respaldada por los documentos recuperados.

Evaluá según estos criterios:
- ✅ TOTALMENTE RESPALDADA: Toda la información de la respuesta está explícitamente en los documentos
- ⚠️ PARCIALMENTE RESPALDADA: Parte de la respuesta está respaldada, pero hay información que no aparece en los documentos
- ❌ NO RESPALDADA: La respuesta contiene afirmaciones que NO están en los documentos

CONSULTA: {query}

RESPUESTA GENERADA: {respuesta}

DOCUMENTOS RECUPERADOS:
{evidencias}

Proporciona tu evaluación en este formato:
[ICONO] Evaluación breve
- Punto específico 1
- Punto específico 2 (si aplica)
"""
    
    try:
        model = os.getenv("OPENAI_OUT_GUARD_MODEL", "gpt-4o-mini")
        logger.debug(f"Verificando respuesta con {model}")
        
        start_time = time.time()  # ← NUEVO
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        elapsed = time.time() - start_time  # ← NUEVO
        
        verification = (out.choices[0].message.content or "⚠️").strip()
        logger.info(f"✅ Verificación completada ({elapsed:.2f}s)")  # ← MODIFICADO
        
        # Track metrics  # ← NUEVO BLOQUE
        if tracker:
            tracker.timings["llm_verification_time"] = elapsed
            tracker.add_llm_metrics("verification", out.usage, model)
        
        return verification, out.usage  # ← CAMBIADO
    except Exception as e:
        logger.error(f"❌ Error en verificación: {e}")
        return "⚠️ (error en verificador)", None  # ← CAMBIADO

# -----------------------
# RRF combine
# -----------------------
def rrf_combine(
    list_a: List[str],
    list_b: List[str],
    k: float = 60.0,
    top_k: Optional[int] = None
) -> List[str]:
    """Fusiona dos listas usando Reciprocal Rank Fusion"""
    scores: Dict[str, float] = {}
    
    for rank, x in enumerate(list_a, start=1):
        scores[x] = scores.get(x, 0.0) + 1.0 / (k + rank)
    
    for rank, x in enumerate(list_b, start=1):
        scores[x] = scores.get(x, 0.0) + 1.0 / (k + rank)
    
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    items = [item_id for item_id, _ in ordered]
    
    result = items[:top_k] if top_k else items
    logger.debug(f"RRF fusion: {len(list_a)} + {len(list_b)} -> {len(result)} items")
    return result

# -----------------------
# RAG Pipeline
# -----------------------
class RAGPipeline:
    """Pipeline RAG híbrido con BM25 + Vector + RRF"""
    
    def __init__(
        self,
        s3_bucket: str,
        bm25_key: str,
        chunks_prefix: str,
        pinecone_index: str,
        pinecone_ns: Optional[str],
        emb_model: str,
        s3_client=None,
        openai_client: Optional[OpenAI] = None
    ):
        logger.info("🔧 Inicializando RAGPipeline...")
        
        self.bucket = s3_bucket
        self.bm25_key = bm25_key
        self.chunks_prefix = chunks_prefix
        self.pinecone_index = pinecone_index
        self.pinecone_ns = pinecone_ns
        self.emb_model = emb_model

        self.s3 = s3_client or build_s3()
        self.oa = openai_client or build_openai()

        # Cargar BM25
        logger.info(f"📦 Cargando BM25 desde s3://{self.bucket}/{self.bm25_key}")
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.bm25_key)
            self.bm25: BM25Index = pickle.loads(obj["Body"].read())
            
            if not hasattr(self.bm25, "search") or not hasattr(self.bm25, "doc_ids"):
                raise RuntimeError("BM25Index no tiene los métodos esperados")
            
            logger.info(f"✅ BM25 cargado - {len(self.bm25.doc_ids)} documentos")
        except Exception as e:
            logger.error(f"❌ Error cargando BM25: {e}")
            raise

        # Verificar Pinecone (dim configurable por ENV)
        logger.info(f"🔍 Verificando índice Pinecone: {self.pinecone_index}")
        try:
            emb_dim = int(os.getenv("EMB_DIM", "384"))
            ensure_index(index_name=self.pinecone_index, dim=emb_dim, metric="cosine")
            logger.info(f"✅ Índice Pinecone listo (dim={emb_dim})")
        except Exception as e:
            logger.error(f"❌ Error con Pinecone: {e}")
            raise

    def bm25_best_pages(self, query: str, top_k: int) -> List[str]:
        """Recupera las mejores páginas usando BM25"""
        hits = self.bm25.search(query, top_k=top_k)
        pages: List[str] = []
        seen = set()
        
        for gi, score in hits:
            pid = str(self.bm25.doc_ids[gi])
            if pid not in seen:
                seen.add(pid)
                pages.append(pid)
        
        logger.debug(f"BM25: {len(pages)} páginas únicas de {len(hits)} hits")
        return pages

    def pinecone_best_pages(self, query: str, top_k: int) -> List[str]:
        """Recupera las mejores páginas usando búsqueda vectorial"""
        try:
            logger.debug(f"Consultando Pinecone: query='{query[:50]}...', top_k={top_k}")
            matches = query_index(
                index_name=self.pinecone_index,
                query_text=query,
                top_k=top_k,
                model_name=self.emb_model,
                namespace=self.pinecone_ns
            )
            
            if not matches:
                logger.warning(f"⚠️ Pinecone no retornó matches")
                return []
            
            logger.info(f"📊 Pinecone retornó {len(matches)} matches")
            
            pages, seen = [], set()
            for m in matches:
                cid = m.get("id") or ""
                
                if "::" in cid:
                    base, pseg, *_ = cid.split("::")
                    p = pseg if pseg.startswith("p") else "p1"
                    pid = f"{base}_{p}"
                else:
                    pid = cid if "_p" in cid else f"{cid}_p1"
                
                if pid not in seen:
                    seen.add(pid)
                    pages.append(pid)
            
            logger.info(f"✅ Pinecone: {len(pages)} páginas únicas extraídas")
            return pages
            
        except Exception as e:
            logger.error(f"❌ Error en pinecone_best_pages: {e}", exc_info=True)
            return []

    def build_candidates_from_pages(
        self,
        query: str,
        page_ids: List[str],
        per_page: int = 3
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Construye lista de candidatos desde las páginas con scoring mejorado y propagación completa de metadata"""
        out: List[Tuple[str, str, Dict[str, Any]]] = []
        q_tokens = set((query or "").lower().split())
        
        # Detectar términos clave (nombres propios, números)
        query_words = (query or "").split()
        key_terms = set()
        for w in query_words:
            cleaned = w.strip('.,;:¿?¡!').lower()
            # Es clave si: empieza con mayúscula, tiene dígitos, o es un apellido común
            if (w and w[0].isupper()) or any(c.isdigit() for c in w) or len(cleaned) > 6:
                key_terms.add(cleaned)
        
        logger.debug(f"Términos clave detectados: {key_terms}")
        
        for pid in page_ids:
            k = page_id_to_ndjson_key(self.chunks_prefix, pid)
            
            try:
                recs = read_ndjson_lines(self.s3, self.bucket, k)
            except self.s3.exceptions.NoSuchKey:
                logger.warning(f"⚠️ NDJSON no encontrado: {k}")
                continue
            except Exception as e:
                logger.error(f"❌ Error leyendo {k}: {e}")
                continue

            scored: List[Tuple[float, Dict[str, Any]]] = []
            for r in recs:
                text = (r.get("text") or "").strip()
                if not text:
                    continue
                toks = set(text.lower().split())
                
                # Scoring con pesos:
                # - Términos clave: 3.0 puntos
                # - Términos normales: 1.0 punto
                score = 0.0
                for qt in q_tokens:
                    if qt in toks:
                        score += 3.0 if qt in key_terms else 1.0
                
                scored.append((score, r))

            scored.sort(key=lambda x: x[0], reverse=True)
            taken = 0
            
            for overlap_score, r in scored:
                if taken >= per_page:
                    break
                
                txt = (r.get("text") or "").strip()
                if not txt:
                    continue

                cid = r.get("id") or r.get("chunk_id")
                if not cid:
                    doc_id = r.get("doc_id") or pid.rsplit("_p", 1)[0]
                    local = r.get("chunk_index")
                    cpage = r.get("page") or 1
                    if local is None:
                        local = taken
                    cid = f"{doc_id}::p{cpage}::{local}"

                # Propagar TODO el metadata original, excepto 'text' e 'id'
                meta = {kk: vv for kk, vv in r.items() if kk not in ("text", "id")}
                
                # Asegurar campos clásicos
                meta.setdefault("source", r.get("source"))
                meta.setdefault("page", r.get("page"))
                meta.setdefault("doc_id", r.get("doc_id") or cid.split("::", 1)[0])
                meta.setdefault("boletin", r.get("boletin"))
                meta.setdefault("fecha", r.get("fecha"))
                meta.setdefault("op", r.get("op"))
                
                # Añadir trazabilidad
                meta["ndjson_key"] = k
                
                out.append((cid, txt, meta))
                taken += 1
        
        # Deduplicar por chunk_id
        seen = set()
        dedup: List[Tuple[str, str, Dict[str, Any]]] = []
        for cid, txt, meta in out:
            if cid in seen:
                continue
            seen.add(cid)
            dedup.append((cid, txt, meta))
        
        logger.info(f"📄 Construidos {len(dedup)} candidatos (dedup) desde {len(page_ids)} páginas")
        return dedup

    def run(
        self,
        query: str,
        k_bm25: int = 50,
        k_vec: int = 50,
        k_final: int = 10,
        per_page: int = 5,
        rrf_k: float = 60.0,
        do_rerank: bool = False,
        debug: bool = False
    ) -> Dict[str, Any]:
        """Ejecuta el pipeline RAG completo"""
        
        # Overridables por ENV (para tunear sin redeploy)
        k_bm25 = int(os.getenv("RAG_K_BM25", k_bm25))
        k_vec = int(os.getenv("RAG_K_VEC", k_vec))
        k_final = int(os.getenv("RAG_K_FINAL", k_final))
        per_page = int(os.getenv("RAG_PER_PAGE", per_page))
        rrf_k = float(os.getenv("RAG_RRF_K", rrf_k))
        
        logger.info(f"🚀 Ejecutando RAG para query: '{query[:100]}...'")
        logger.info(f"Params => k_bm25={k_bm25} k_vec={k_vec} k_final={k_final} per_page={per_page} rrf_k={rrf_k} rerank={do_rerank}")
        tracker = PerformanceTracker()
        tracker.start()
        # 1. Búsqueda híbrida
        logger.info(f"🔍 BM25 top-{k_bm25}")
        with tracker.track("bm25_time"):
            bm25_pages = self.bm25_best_pages(query, top_k=k_bm25)
        
        logger.info(f"🔍 Vector top-{k_vec}")
        with tracker.track("vector_time"):
            pc_pages = self.pinecone_best_pages(query, top_k=k_vec)
        
        # 2. Fusión RRF
        with tracker.track("fusion_time"):
            if pc_pages:
                logger.info("🔀 Aplicando RRF fusion")
                fused_pages = rrf_combine(
                    bm25_pages,
                    pc_pages,
                    k=rrf_k,
                    top_k=max(k_final * 3, 20)
                )
            else:
                logger.warning("⚠️ Pinecone vacío, usando solo BM25")
                fused_pages = bm25_pages[:max(k_final * 3, 20)]

         # 3. Construir candidatos
        with tracker.track("candidates_time"):
            candidates = self.build_candidates_from_pages(query, fused_pages, per_page=per_page)
        
        if not candidates:
            logger.warning("⚠️ No se encontraron candidatos")
            performance = tracker.get_summary()
            return {
                "query": query,
                "answer": "No hay contexto disponible.",
                "summary": "",
                "verification": "⚠️ (sin candidatos para verificar)",
                "results": [],
                "performance": performance,
                "debug": {
                    "bm25_pages": bm25_pages[:10],
                    "pinecone_pages": pc_pages[:10],
                    "fused_pages": fused_pages[:10],
                    "candidates": 0
                } if debug else None
            }

        # 4. Rerank
        with tracker.track("rerank_time"):
            if do_rerank:
                ranked = optional_rerank(query, candidates)
            else:
                ranked = [(cid, txt, meta, 0.0) for (cid, txt, meta) in candidates]
        
        final = ranked[:k_final]
        
        # 5. Enriquecer chunks con metadatos para el LLM
        ctx_enriched = []
        metadata_list = []
        for cid, txt, meta, score in final:
            # Guardar metadatos estructurados
            metadata_list.append(meta)
            
            # Construir encabezado con metadatos para el contexto textual
            metadata_header = []
            if meta.get("boletin"):
                metadata_header.append(f"Boletín N° {meta['boletin']}")
            if meta.get("fecha"):
                metadata_header.append(f"Fecha: {meta['fecha']}")
            if meta.get("op"):
                metadata_header.append(f"OP: {meta['op']}")
            
            # Si hay classification con label
            if meta.get("classification") and isinstance(meta["classification"], dict):
                label = meta["classification"].get("label")
                if label and label != "Clasificación incierta":
                    metadata_header.append(f"Categoría: {label}")
            
            # Armar chunk enriquecido
            if metadata_header:
                header = "[" + " | ".join(metadata_header) + "]"
                enriched = f"{header}\n{txt}"
            else:
                enriched = txt
            
            ctx_enriched.append(enriched)
        
        logger.info(f"✅ {len(final)} chunks finales seleccionados")

        # 6. Generación LLM con metadatos estructurados
        logger.info("🤖 Generando respuesta con LLM")
        summary, summary_usage = rag_summary_llm(
            self.oa, query, ctx_enriched, max_chars=500, tracker=tracker
        )
        answer, answer_usage = answer_llm(
            self.oa, query, ctx_enriched, summary, 
            metadata_list=metadata_list, tracker=tracker
        )
        
        # 7. Verificación de respuesta
        logger.info("🔍 Verificando respuesta contra documentos")
        verification, verification_usage = verificar_respuesta_llm(
            self.oa, query, answer, ctx_enriched, tracker=tracker
        )
        
        # 8. Obtener métricas de performance
        performance = tracker.get_summary()
        
        # Log performance
        logger.info(f"⏱️  Total time: {performance['total_time']:.3f}s")
        logger.info(f"🎫 Total tokens: {performance['tokens']['total']:,}")
        logger.info(f"💰 Total cost: ${performance['cost']['total_usd']:.6f}")
        
        
        # 9. Construir respuesta con metadata completo
        payload = {
            "query": query,
            "summary": summary,
            "answer": answer if answer.strip() else "No está especificado en las fuentes.",
            "verification": verification,
            "results": [
                {
                    "chunk_id": cid,
                    "score": float(score),
                    "text": txt,
                    # Campos individuales para compatibilidad
                    "source": (meta or {}).get("source"),
                    "page": (meta or {}).get("page"),
                    "doc_id": (meta or {}).get("doc_id"),
                    "boletin": (meta or {}).get("boletin"),
                    "fecha": (meta or {}).get("fecha"),
                    "op": (meta or {}).get("op"),
                    # Campo metadata completo con TODO
                    "metadata": meta or {},
                }
                for (cid, txt, meta, score) in final
            ],
            "performance": performance,  # ← NUEVO: métricas de performance
        }
        
        if debug:
            payload["debug"] = {
                "bm25_pages": bm25_pages[:10],
                "pinecone_pages": pc_pages[:10],
                "fused_pages": fused_pages[:10],
                "candidates_count": len(candidates),
                "rerank_applied": do_rerank,
            }
        
        logger.info(f"✅ Pipeline completado")
        return payload