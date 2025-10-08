#fastapi_app/pipeline.py
from __future__ import annotations
import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from tasks.bm25_index import BM25Index
from .s3_boto import build_s3
from .vector_pinecone_api import ensure_index, query_index

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
    max_chars: int = 500
) -> str:
    """Genera resumen del contexto usando LLM"""
    if not chunks:
        return ""
    
    if not client:
        logger.debug("Sin OpenAI client - retornando contexto truncado")
        return "\n\n".join(chunks)[:max_chars]
    
    joined = "\n\n".join(f"- {c}" for c in chunks)[:4000]
    prompt = (
        f"Resumí de forma concisa y factual el siguiente contexto para responder la consulta.\n"
        f"Consulta: {query}\n\nContexto:\n{joined}\n\n"
        f"Devolvé SOLO el resumen (máx {max_chars} caracteres), sin viñetas ni comentarios."
    )
    
    try:
        model = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
        logger.debug(f"Generando resumen con {model}")
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        summary = (out.choices[0].message.content or "")[:max_chars].strip()
        logger.info(f"✅ Resumen generado ({len(summary)} chars)")
        return summary
    except Exception as e:
        logger.error(f"❌ Error generando resumen: {e}")
        return "\n\n".join(chunks)[:max_chars]

def answer_llm(
    client: Optional[OpenAI],
    query: str,
    context_chunks: List[str],
    summary: str
) -> str:
    """Genera respuesta final usando LLM y el contexto"""
    if not context_chunks:
        return "No hay contexto disponible."
    
    if not client:
        logger.debug("Sin OpenAI client - retornando respuesta por defecto")
        return "No está especificado en las fuentes."
    
    ctx = "\n\n".join(context_chunks)[:6000]
    prompt = (
        "Usá SOLO la información del CONTEXTO para responder la CONSULTA de forma breve y clara.\n"
        "Si la respuesta no está en el contexto, decí 'No está especificado en las fuentes.'\n\n"
        f"CONSULTA: {query}\n\nRESUMEN CONTEXTO:\n{summary}\n\nCONTEXTO COMPLEMENTARIO:\n{ctx}\n"
    )
    
    try:
        model = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini")
        logger.debug(f"Generando respuesta con {model}")
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = (out.choices[0].message.content or "").strip()
        logger.info(f"✅ Respuesta generada ({len(answer)} chars)")
        return answer
    except Exception as e:
        logger.error(f"❌ Error generando respuesta: {e}")
        return "Error al generar la respuesta."

def verificar_respuesta_llm(
    client: Optional[OpenAI],
    query: str,
    respuesta: str,
    context_chunks: List[str]
) -> str:
    """
    Verifica si la respuesta generada está respaldada por los documentos
    Retorna una evaluación con ✅ / ⚠️ / ❌
    """
    if not client:
        return "⚠️ (verificador LLM no disponible)"
    
    if not context_chunks or not respuesta:
        return "⚠️ (sin contexto o respuesta para verificar)"
    
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
        out = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        verification = (out.choices[0].message.content or "⚠️").strip()
        logger.info(f"✅ Verificación completada")
        return verification
    except Exception as e:
        logger.error(f"❌ Error en verificación: {e}")
        return "⚠️ (error en verificador)"

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

        # Verificar Pinecone
        logger.info(f"🔍 Verificando índice Pinecone: {self.pinecone_index}")
        # Verificar Pinecone (dim configurable por ENV)
        try:
            emb_dim = int(os.getenv("EMB_DIM", "384"))  # <- tu encoder actual es de 384
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
            #"""Construye lista de candidatos desde las páginas y propaga TODO el metadata original."""
            out: List[Tuple[str, str, Dict[str, Any]]] = []
            q_tokens = set((query or "").lower().split())

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

                # Prioriza por intersección de tokens
                scored: List[Tuple[float, Dict[str, Any]]] = []
                for r in recs:
                    text = (r.get("text") or "").strip()
                    if not text:
                        continue
                    toks = set(text.lower().split())
                    score = float(len(q_tokens.intersection(toks)))
                    scored.append((score, r))

                scored.sort(key=lambda x: x[0], reverse=True)
                taken = 0

                for _, r in scored:
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

                    # ⬇️ Propaga TODO el metadata original, excepto 'text' e 'id'
                    meta = {kk: vv for kk, vv in r.items() if kk not in ("text", "id")}
                    # compat con claves clásicas
                    meta.setdefault("source", r.get("source"))
                    meta.setdefault("page", r.get("page"))
                    meta.setdefault("doc_id", r.get("doc_id") or cid.split("::", 1)[0])
                    # muy útil para trazabilidad
                    meta["ndjson_key"] = k

                    out.append((cid, txt, meta))
                    taken += 1

            # Deduplicar por chunk_id (por si vino repetido de BM25/Vector)
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
        k_final: int = 6,
        per_page: int = 3,
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

        # 1) Búsqueda híbrida
        bm25_pages = self.bm25_best_pages(query, top_k=k_bm25)
        pc_pages = self.pinecone_best_pages(query, top_k=k_vec)

        # 2) Fusión RRF (si Pinecone trae vacío, usar BM25 directly)
        if pc_pages:
            fused_pages = rrf_combine(bm25_pages, pc_pages, k=rrf_k, top_k=max(k_final * 3, 20))
        else:
            fused_pages = bm25_pages[:max(k_final * 3, 20)]
        logger.info(f"🔀 Páginas fusionadas: {len(fused_pages)}")

        # 3) Construir candidatos (propaga metadata)
        candidates = self.build_candidates_from_pages(query, fused_pages, per_page=per_page)
        if not candidates:
            logger.warning("⚠️ No se encontraron candidatos")
            return {
                "query": query,
                "answer": "No hay contexto disponible.",
                "summary": "",
                "verification": "⚠️ (sin candidatos para verificar)",
                "results": [],
                "debug": {
                    "bm25_pages": bm25_pages[:10],
                    "pinecone_pages": pc_pages[:10],
                    "fused_pages": fused_pages[:10],
                    "candidates": 0
                } if debug else None
            }

        # 4) Rerank opcional
        if do_rerank:
            ranked = optional_rerank(query, candidates)
        else:
            ranked = [(cid, txt, meta, 0.0) for (cid, txt, meta) in candidates]

        final = ranked[:k_final]
        ctx_texts = [t for _, t, _, _ in final]
        logger.info(f"✅ {len(final)} chunks finales seleccionados")

        # 5) LLM
        summary = rag_summary_llm(self.oa, query, ctx_texts, max_chars=500)
        answer = answer_llm(self.oa, query, ctx_texts, summary)
        verification = verificar_respuesta_llm(self.oa, query, answer, ctx_texts)

        # 6) Payload (ahora incluye 'metadata' completo en cada resultado)
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
                    "source": (meta or {}).get("source"),
                    "page": (meta or {}).get("page"),
                    "doc_id": (meta or {}).get("doc_id"),
                    "boletin": (meta or {}).get("boletin"),
                    "fecha": (meta or {}).get("fecha"),
                    "op": (meta or {}).get("op"),
                    "metadata": meta or {},   # ⬅️ aquí viaja TODO (classification, txt_key, ndjson_key, etc.)
                }
                for (cid, txt, meta, score) in final
            ],
        }

        if debug:
            payload["debug"] = {
                "bm25_pages": bm25_pages[:10],
                "pinecone_pages": pc_pages[:10],
                "fused_pages": fused_pages[:10],
                "candidates_count": len(candidates),
                "rerank_applied": do_rerank,
            }

        logger.info("✅ Pipeline completado")
        return payload