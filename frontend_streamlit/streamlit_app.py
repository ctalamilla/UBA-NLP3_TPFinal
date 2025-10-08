import os
import json
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


# ------------------------------
# Utilidades HTTP
# ------------------------------
def api_base_default() -> str:
    return os.getenv("RAG_API_URL", "http://localhost:8000")

def req(
    base: str,
    method: str,
    path: str,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    url = base.rstrip("/") + path
    try:
        if method.upper() == "GET":
            r = requests.get(url, timeout=timeout)
        else:
            headers = {"Content-Type": "application/json"}
            r = requests.request(method.upper(), url, headers=headers, json=json_body, timeout=timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"_raw": r.text}
    except requests.HTTPError as e:
        detail = e.response.text if e.response is not None else str(e)
        return {"error": f"HTTP {e.response.status_code if e.response else '?'}", "detail": detail}
    except Exception as e:
        return {"error": "request_failed", "detail": str(e)}


# ------------------------------
# Layout
# ------------------------------
st.set_page_config(page_title="RAG Boletín Oficial", layout="wide", page_icon="🔎")

# Header
st.markdown("# 🔎 RAG Boletín Oficial de Salta")
st.markdown("Sistema de consulta híbrido con **BM25** + **Vector Search** + **LLM**")

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    base_url = st.text_input("🌐 API URL", value=api_base_default(), key="cfg_base_url")
    
    # Health check
    if st.button("🏥 Health Check", use_container_width=True):
        health = req(base_url, "GET", "/health")
        if health.get("status") == "healthy":
            st.success("✅ API funcionando")
            with st.expander("Detalles"):
                st.json(health)
        else:
            st.error("❌ API no responde correctamente")
            st.json(health)
    
    st.divider()
    st.caption("**Tabs disponibles:**")
    st.caption("🧠 RAG - Búsqueda híbrida completa")
    st.caption("🧭 Vector - Solo búsqueda semántica")
    st.caption("📚 BM25 - Solo búsqueda léxica")

# Tabs principales
tabs = st.tabs(["🧠 RAG Completo", "🧭 Búsqueda Vectorial", "📚 Búsqueda Léxica (BM25)"])


# ------------------------------
# Tab 1: RAG Completo
# ------------------------------
with tabs[0]:
    st.markdown("## Consulta RAG Híbrida")
    st.markdown("Combina **búsqueda léxica (BM25)** + **búsqueda semántica (Vector)** + **LLM** para generar respuestas")
    
    q = st.text_area(
        "💬 Tu consulta",
        value="¿Qué adjudicaciones simples hay?",
        height=100,
        key="rag_query",
        placeholder="Ejemplo: Patricia Sanso quiebra, en qué boletín se publicó?"
    )

    # Parámetros en columnas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        k_bm25 = st.number_input("📚 k_bm25", min_value=1, max_value=200, value=50, step=10, 
                                 help="Documentos a recuperar con BM25")
    with col2:
        k_vec = st.number_input("🧭 k_vec", min_value=0, max_value=200, value=50, step=10,
                                help="Documentos a recuperar con Vector Search")
    with col3:
        k_final = st.number_input("🎯 k_final", min_value=1, max_value=20, value=6, step=1,
                                  help="Chunks finales para el LLM")
    with col4:
        debug_req = st.checkbox("🛠️ Debug", value=True, help="Mostrar información de debug")

    if st.button("🚀 Ejecutar Consulta RAG", type="primary", use_container_width=True):
        payload = {
            "query": q,
            "k_bm25": int(k_bm25),
            "k_vec": int(k_vec),
            "k_final": int(k_final),
            "debug": debug_req
        }

        with st.spinner("🔄 Consultando API..."):
            data = req(base_url, "POST", "/query", json_body=payload)

        if "error" in data:
            st.error(f"❌ Error: {data['error']}")
            if data.get("detail"):
                st.code(data["detail"], language="text")
        else:
            # Respuesta principal
            st.success("✅ Respuesta generada")
            
            # Mostrar query
            st.markdown(f"**📝 Consulta:** {data.get('query', '')}")
            
            # Respuesta del LLM
            st.markdown("### 🤖 Respuesta")
            st.info(data.get("answer", "—"))

            # Summary y Verificación en columnas
            col_sum, col_ver = st.columns(2)
            
            with col_sum:
                if data.get("summary"):
                    with st.expander("📋 Resumen del Contexto", expanded=False):
                        st.write(data["summary"])
            
            with col_ver:
                if data.get("verification"):
                    with st.expander("🔍 Verificación", expanded=False):
                        verification_text = data["verification"]
                        # Colorear según el resultado
                        if "✅" in verification_text:
                            st.success(verification_text)
                        elif "⚠️" in verification_text:
                            st.warning(verification_text)
                        elif "❌" in verification_text:
                            st.error(verification_text)
                        else:
                            st.write(verification_text)

            # Fuentes/Resultados
            results: List[Dict[str, Any]] = data.get("results", [])
            st.markdown(f"### 📎 Fuentes Consultadas ({len(results)})")
            
            if not results:
                st.info("No hay resultados.")
            else:
                for i, r in enumerate(results, start=1):
                    with st.expander(
                        f"**#{i}** | Boletín {r.get('boletin', '?')} | {r.get('fecha', '?')} | OP: {r.get('op', '?')}",
                        expanded=(i <= 2)  # Primeros 2 expandidos
                    ):
                        # Chunk info
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**🆔 Chunk ID:** `{r.get('chunk_id', '?')}`")
                        with col_b:
                            score = r.get("score")
                            if score is not None:
                                st.metric("Score", f"{score:.4f}" if isinstance(score, (int, float)) else score)
                        
                        # Texto del chunk
                        st.markdown("**📄 Texto:**")
                        st.write(r.get("text", ""))
                        
                        # Metadata completa
                        metadata = r.get("metadata", {})
                        if metadata:
                            with st.expander("🏷️ Metadata Completa"):
                                # Mostrar clasificación si existe
                                classification = metadata.get("classification", {})
                                if classification:
                                    st.markdown("**📊 Clasificación:**")
                                    cat = classification.get("categoria", "N/A")
                                    st.markdown(f"- **Categoría:** `{cat}`")
                                    
                                    extracted = classification.get("extracted", {})
                                    if extracted:
                                        st.markdown("**📋 Datos Extraídos:**")
                                        if extracted.get("numero"):
                                            st.markdown(f"- **Número:** {extracted['numero']}")
                                        if extracted.get("organismo"):
                                            st.markdown(f"- **Organismo:** {extracted['organismo']}")
                                        if extracted.get("personas"):
                                            st.markdown(f"- **Personas:** {', '.join(extracted['personas'])}")
                                        if extracted.get("resumen"):
                                            st.markdown(f"- **Resumen:** {extracted['resumen']}")
                                
                                # Metadata técnica
                                st.markdown("**🔧 Metadata Técnica:**")
                                st.json(metadata)

            # Debug info
            if debug_req and data.get("debug"):
                with st.expander("🛠️ Información de Debug", expanded=False):
                    debug_info = data["debug"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BM25 Pages", len(debug_info.get("bm25_pages", [])))
                    with col2:
                        st.metric("Pinecone Pages", len(debug_info.get("pinecone_pages", [])))
                    with col3:
                        st.metric("Candidatos", debug_info.get("candidates_count", 0))
                    
                    st.json(debug_info)

            # Descarga
            st.download_button(
                "⬇️ Descargar Respuesta Completa (JSON)",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="rag_response.json",
                mime="application/json",
                use_container_width=True
            )


# ------------------------------
# Tab 2: Búsqueda Vectorial
# ------------------------------
with tabs[1]:
    st.markdown("## 🧭 Búsqueda Vectorial (Pinecone)")
    st.markdown("Búsqueda semántica pura usando embeddings")
    
    qv = st.text_area(
        "💬 Consulta semántica",
        value="licitación pública",
        height=80,
        key="vec_query"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        v_topk = st.number_input("🎯 Top K", min_value=1, max_value=100, value=10, step=1)
        v_ns = st.text_input("📁 Namespace", value=os.getenv("PINECONE_NAMESPACE", "2025"))
    with col2:
        v_index = st.text_input("🗂️ Index", value=os.getenv("PINECONE_INDEX", "boletines-2025"))
        v_model = st.text_input(
            "🤖 Modelo",
            value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    if st.button("🔎 Buscar", type="primary", use_container_width=True):
        body = {
            "query": qv,
            "top_k": int(v_topk),
            "index_name": v_index,
            "namespace": v_ns,
            "model_name": v_model,
        }
        
        with st.spinner("🔄 Buscando..."):
            data = req(base_url, "POST", "/vector/query", json_body=body)
        
        if "error" in data:
            st.error(f"❌ Error: {data['error']}")
            if data.get("detail"):
                st.code(data["detail"])
        else:
            st.success(f"✅ {data.get('count', 0)} resultados encontrados")
            
            res = data.get("results", [])
            
            for i, m in enumerate(res, start=1):
                mid = m.get("id", "?")
                score = m.get("score", 0)
                meta = m.get("metadata", {})
                
                with st.expander(f"**#{i}** | ID: `{mid}` | Score: {score:.4f}", expanded=(i == 1)):
                    if meta:
                        # Mostrar campos clave
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if meta.get("boletin"):
                                st.metric("Boletín", meta["boletin"])
                        with col2:
                            if meta.get("fecha"):
                                st.metric("Fecha", meta["fecha"])
                        with col3:
                            if meta.get("op"):
                                st.metric("OP", meta["op"])
                        
                        st.markdown("**📋 Metadata completa:**")
                        st.json(meta)
                    else:
                        st.info("Sin metadata")

            st.download_button(
                "⬇️ Descargar JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="vector_search.json",
                mime="application/json",
                use_container_width=True
            )


# ------------------------------
# Tab 3: BM25
# ------------------------------
with tabs[2]:
    st.markdown("## 📚 Búsqueda Léxica (BM25)")
    st.markdown("Búsqueda por coincidencia de términos (sin embeddings)")
    
    qb = st.text_area(
        "💬 Términos de búsqueda",
        value="edictos",
        height=80,
        key="bm25_query"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        b_kbm25 = st.number_input("📚 k_bm25", min_value=1, max_value=200, value=50, step=10)
    with col2:
        b_kfinal = st.number_input("🎯 Chunks finales", min_value=1, max_value=20, value=6, step=1)

    st.caption("ℹ️ Esta consulta usa k_vec=0 para simular búsqueda BM25 pura")

    if st.button("🔎 Buscar", type="primary", use_container_width=True, key="btn_bm25"):
        body = {
            "query": qb,
            "k_bm25": int(b_kbm25),
            "k_vec": 0,
            "k_final": int(b_kfinal),
            "debug": True
        }
        
        with st.spinner("🔄 Buscando..."):
            data = req(base_url, "POST", "/query", json_body=body)

        if "error" in data:
            st.error(f"❌ Error: {data['error']}")
            if data.get("detail"):
                st.code(data["detail"])
        else:
            st.success("✅ Búsqueda completada")
            
            # Respuesta
            st.markdown("### 🤖 Respuesta Generada")
            st.info(data.get("answer", "—"))

            # Resultados
            results: List[Dict[str, Any]] = data.get("results", [])
            st.markdown(f"### 📄 Chunks Recuperados ({len(results)})")
            
            if not results:
                st.info("No hay resultados.")
            else:
                for i, r in enumerate(results, start=1):
                    with st.expander(
                        f"**#{i}** | Boletín {r.get('boletin', '?')} | {r.get('op', '?')}",
                        expanded=(i <= 2)
                    ):
                        st.markdown(f"**🆔 Chunk:** `{r.get('chunk_id', '?')}`")
                        st.write(r.get("text", ""))
                        
                        # Metadata
                        metadata = r.get("metadata", {})
                        if metadata:
                            with st.expander("🏷️ Metadata"):
                                st.json(metadata)

            # Debug
            if data.get("debug"):
                with st.expander("🛠️ Debug"):
                    st.json(data["debug"])

            st.download_button(
                "⬇️ Descargar JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="bm25_search.json",
                mime="application/json",
                use_container_width=True
            )


# Footer
st.divider()
st.caption("🔎 RAG Boletín Oficial de Salta | v0.6.0")