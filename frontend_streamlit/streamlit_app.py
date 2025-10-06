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
st.set_page_config(page_title="RAG Demo - Streamlit", layout="wide")
st.title("🔎 RAG Demo — Cliente Streamlit")

with st.sidebar:
    st.header("⚙️ Configuración")
    base_url = st.text_input("API base URL", value=api_base_default(), key="cfg_base_url",
                             help="Ej: http://localhost:8000")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Ping /health", key="btn_health"):
            health = req(base_url, "GET", "/health")
            if "ok" in health and health["ok"]:
                st.success("API OK ✅")
            else:
                st.warning(f"Salud no OK: {health}")
    with colB:
        st.caption("Usa las pestañas para RAG / Pinecone / BM25")

tabs = st.tabs(["🧠 RAG", "🧭 Pinecone (vector)", "📚 BM25 (solo texto)"])


# ------------------------------
# Tab RAG
# ------------------------------
with tabs[0]:
    st.subheader("Consulta RAG")
    q = st.text_area("Consulta", value="¿Hay edictos y en qué fechas?",
                     height=110, key="rag_query", placeholder="Escribe tu pregunta…")

    col1, col2, col3 = st.columns(3)
    with col1:
        k_bm25 = st.number_input("k_bm25", min_value=1, max_value=200, value=50, step=1, key="rag_k_bm25")
    with col2:
        k_vec = st.number_input("k_vec", min_value=0, max_value=200, value=50, step=1, key="rag_k_vec")
    with col3:
        k_final = st.number_input("k_final", min_value=1, max_value=50, value=6, step=1, key="rag_k_final")

    debug_req = st.checkbox("Solicitar debug (si la API lo soporta)", value=False, key="rag_debug",
                            help="Algunas versiones de la API ignoran este flag (no es error).")

    if st.button("🔎 Ejecutar RAG", type="primary", key="btn_rag"):
        payload = {"query": q, "k_bm25": int(k_bm25), "k_vec": int(k_vec), "k_final": int(k_final)}
        if debug_req:
            payload["debug"] = True

        with st.spinner("Consultando API /query…"):
            data = req(base_url, "POST", "/query", json_body=payload)

        if "error" in data:
            st.error(f"Error: {data['error']}")
            if data.get("detail"):
                st.code(data["detail"])
        else:
            st.success("Respuesta generada")
            st.markdown(f"**Consulta:** {data.get('query','')}")
            ans_safe = data.get("answer_safe", None)
            if isinstance(ans_safe, bool):
                st.caption(f"Salida verificada: {'✅ SEGURA' if ans_safe else '⚠️ REVISAR'}")

            st.markdown("### 🧩 Respuesta")
            st.write(data.get("answer", "—"))

            if data.get("summary"):
                with st.expander("Resumen del contexto"):
                    st.write(data["summary"])

            if data.get("verification"):
                with st.expander("Verificación del verificador (agente)"):
                    st.write(data["verification"])

            results: List[Dict[str, Any]] = data.get("results", [])
            st.markdown(f"### 📎 Fuentes ({len(results)})")
            if not results:
                st.info("No hay resultados/citas.")
            else:
                for i, r in enumerate(results, start=1):
                    cid = r.get("chunk_id")
                    text = r.get("text", "")
                    src = r.get("source") or ""
                    page = r.get("page")
                    score = r.get("score")
                    with st.container(border=True):
                        st.write(f"**#{i:02d}**  `chunk_id`: `{cid}`")
                        meta_bits = []
                        if src:
                            meta_bits.append(f"**source:** `{src}`")
                        if page is not None:
                            meta_bits.append(f"**page:** {page}")
                        if score is not None:
                            meta_bits.append(f"**score:** {score:.4f}" if isinstance(score, (int, float)) else f"**score:** {score}")
                        if meta_bits:
                            st.caption(" | ".join(meta_bits))
                        st.write(text)

            if data.get("debug"):
                with st.expander("🛠️ Debug"):
                    st.json(data["debug"])

            st.download_button(
                "⬇️ Descargar JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="rag_response.json",
                mime="application/json",
                key="dl_rag_json"
            )


# ------------------------------
# Tab Pinecone (vector)
# ------------------------------
with tabs[1]:
    st.subheader("Consulta vectorial (Pinecone)")
    qv = st.text_area("Consulta", value="licitación pública", height=100, key="vec_query")
    col1, col2 = st.columns(2)
    with col1:
        v_topk = st.number_input("top_k", min_value=1, max_value=200, value=10, step=1, key="vec_topk")
    with col2:
        v_ns = st.text_input("Namespace", value=os.getenv("PINECONE_NAMESPACE", "2025"), key="vec_ns")

    colm1, colm2 = st.columns(2)
    with colm1:
        v_index = st.text_input("Index name", value=os.getenv("PINECONE_INDEX", "boletines-2025"), key="vec_index")
    with colm2:
        v_model = st.text_input(
            "Modelo de embeddings",
            value=os.getenv("EMB_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            key="vec_model",
        )

    if st.button("🔎 Ejecutar consulta vectorial", key="btn_vec"):
        body = {
            "query": qv,
            "top_k": int(v_topk),
            "index_name": v_index,
            "namespace": v_ns,
            "model_name": v_model,
        }
        with st.spinner("Consultando API /vector/query…"):
            data = req(base_url, "POST", "/vector/query", json_body=body)
        if "error" in data:
            st.error(f"Error: {data['error']}")
            if data.get("detail"):
                st.code(data["detail"])
        else:
            st.success("Resultados de Pinecone")
            st.caption(f"Query: {data.get('query','')}  |  top_k: {data.get('top_k')}")
            res = data.get("results", [])
            st.markdown(f"### Matches ({len(res)})")
            if isinstance(res, dict) and "matches" in res:
                res = res.get("matches", [])
            for i, m in enumerate(res, start=1):
                mid = m.get("id")
                score = m.get("score")
                meta = m.get("metadata") or {}
                with st.container(border=True):
                    st.write(f"**#{i:02d}** `id`: `{mid}`  |  `score`: {score:.4f}" if isinstance(score, (int, float)) else f"**#{i:02d}** `id`: `{mid}`")
                    if meta:
                        with st.expander("metadata"):
                            st.json(meta)

            st.download_button(
                "⬇️ Descargar JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="pinecone_query.json",
                mime="application/json",
                key="dl_vec_json"
            )


# ------------------------------
# Tab BM25 (solo texto)
# ------------------------------
with tabs[2]:
    st.subheader("Consulta BM25 (desactiva vector)")
    qb = st.text_area("Consulta", value="edictos", height=100, key="bm25_query")
    col1, col2 = st.columns(2)
    with col1:
        b_kbm25 = st.number_input("k_bm25", min_value=1, max_value=200, value=50, step=1, key="bm25_kbm25")
    with col2:
        b_kfinal = st.number_input("k_final (citas a mostrar)", min_value=1, max_value=50, value=6, step=1, key="bm25_kfinal")

    st.caption("Tip: esto llama a `/query` con `k_vec=0` para aproximar 'BM25-only'. La API igualmente puede invocar el LLM para redactar la respuesta.")

    if st.button("🔎 Ejecutar BM25-only", key="btn_bm25"):
        body = {
            "query": qb,
            "k_bm25": int(b_kbm25),
            "k_vec": 0,
            "k_final": int(b_kfinal)
        }
        with st.spinner("Consultando API /query (BM25-only)…"):
            data = req(base_url, "POST", "/query", json_body=body)

        if "error" in data:
            st.error(f"Error: {data['error']}")
            if data.get("detail"):
                st.code(data["detail"])
        else:
            st.success("Resultados BM25")
            st.markdown("### 🧩 Respuesta (generada por la API)")
            st.write(data.get("answer", "—"))

            results: List[Dict[str, Any]] = data.get("results", [])
            st.markdown(f"### 📎 Chunks recuperados ({len(results)})")
            if not results:
                st.info("No hay resultados/citas.")
            else:
                for i, r in enumerate(results, start=1):
                    cid = r.get("chunk_id")
                    text = r.get("text", "")
                    with st.container(border=True):
                        st.write(f"**#{i:02d}**  `chunk_id`: `{cid}`")
                        st.write(text)

            st.download_button(
                "⬇️ Descargar JSON",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="bm25_only_response.json",
                mime="application/json",
                key="dl_bm25_json"
            )
