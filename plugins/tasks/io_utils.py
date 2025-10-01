##################      io_utils.py    ###################

import json, pandas as pd
from pathlib import Path
from typing import Dict, List, Set
#  importá Document desde tu módulo
from tasks.documents import Document

# io_utils.py

def load_docs_jsonl(path: Path) -> List[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)

            # ✅ Priorizar el chunk completo; si no existe, usar resumen
            text = (o.get("text") or o.get("resumen") or "").strip()
            if not text:
                continue

            page = o.get("page", None)
            try:
                page = int(page) if page is not None and str(page).strip().lower() != "none" else None
            except Exception:
                page = None

            doc = Document(
                id=str(o.get("doc_id", o.get("id", ""))),
                text=text,
                source=str(o.get("source", o.get("tipo", ""))),
                page=page
            )
            docs.append(doc)
    return docs



def load_qrels_csv(path: Path) -> Dict[str, Set[str]]:
    df = pd.read_csv(path)
    df = df[df["label"] > 0]
    out: Dict[str, Set[str]] = {}
    for q, sub in df.groupby("query"):
        out[str(q)] = set(map(str, sub["doc_id"].tolist()))
    return out