from __future__ import annotations
from typing import Dict, Any
from tasks.vector_pinecone import query_index

def task_pinecone_query(
    index_name: str,
    query: str,
    top_k: int,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    namespace: str = "2025",
    **kwargs
) -> Dict[str, Any]:
    res = query_index(
        index_name=index_name,
        query=query,
        top_k=top_k,
        model_name=model_name,
        namespace=namespace,
    )
    # log amigable
    print(f"🔎 Pinecone TOP-{top_k} para {query!r}")
    for i, m in enumerate(res["matches"], 1):
        meta = m.get("metadata") or {}
        snippet = ""
        # si guardaste 'text' en metadata, podés mostrarlo aquí (yo guardé solo doc_id, page, source)
        print(f"#{i:02d}  score={m['score']:.4f}  id={m['id']}  doc_id={meta.get('doc_id')}  page={meta.get('page')}  src={meta.get('source')}")
    return res
