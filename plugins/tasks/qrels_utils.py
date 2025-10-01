# plugins/tasks/qrels_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union
import csv, json, io


def load_qrels(path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    Devuelve: {"<query>": {"<doc_id>": rel_int, ...}, ...}

    Formatos soportados:
    - CSV con headers: query,doc_id,rel (o variantes: qid/q, doc/docno, relevance/label)
    - JSON dict     : {"query": {"doc_id": rel, ...}, ...}
    - JSONL por fila: {"query": "...", "doc_id": "...", "rel": 1}
    - TREC qrels    : "<qid> 0 <docno> <rel>"
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")

    # 1) JSON dict
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            out: Dict[str, Dict[str, int]] = {}
            for q, d in obj.items():
                out[str(q)] = {str(k): int(v) for k, v in (d or {}).items()}
            return out
    except Exception:
        pass

    lines = text.splitlines()

    # 2) JSONL
    out: Dict[str, Dict[str, int]] = {}
    jsonl_ok = False
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("{") and s.endswith("}"):
            try:
                j = json.loads(s)
                q = str(j.get("query") or j.get("qid") or j.get("q") or "").strip()
                d = str(j.get("doc_id") or j.get("doc") or j.get("docno") or "").strip()
                r = int(j.get("rel") or j.get("relevance") or j.get("label") or 0)
                if q and d:
                    out.setdefault(q, {})[d] = r
                    jsonl_ok = True
            except Exception:
                jsonl_ok = False
                out = {}
                break
    if jsonl_ok:
        return out

    # 3) CSV (headers flexibles)
    try:
        f = io.StringIO(text)
        reader = csv.DictReader(f)
        if reader.fieldnames:
            q_keys = {"query", "qid", "q"}
            d_keys = {"doc_id", "doc", "docno"}
            r_keys = {"rel", "relevance", "label"}
            out = {}
            for row in reader:
                q = next((row[k] for k in reader.fieldnames if k in q_keys and row.get(k)), "").strip()
                d = next((row[k] for k in reader.fieldnames if k in d_keys and row.get(k)), "").strip()
                r_raw = next((row[k] for k in reader.fieldnames if k in r_keys and row.get(k) not in (None, "")), "0")
                if q and d:
                    try:
                        r = int(float(r_raw))
                    except Exception:
                        r = 0
                    out.setdefault(q, {})[d] = r
            if out:
                return out
    except Exception:
        pass

    # 4) TREC qrels: "<qid> 0 <docno> <rel>"
    out = {}
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) == 4 and parts[1] in {"0", "Q0"}:
            qid, _, docno, rel = parts
            try:
                r = int(float(rel))
            except Exception:
                r = 0
            out.setdefault(qid, {})[docno] = r
    if out:
        return out

    raise ValueError(f"No pude interpretar qrels en {p}")
