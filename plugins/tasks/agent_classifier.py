# plugins/tasks/agent_classifier.py
from __future__ import annotations
import os
import re
import json
from typing import Optional, Dict

# ✅ OpenAI (SDK v1.x)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # por si el paquete no está

# --- Heurística de categorías (igual a tu notebook)
CATEGORIES = {
    "DECRETO":     r"DECRETO\s+N[°º]\s*\d+",
    "RESOLUCION":  r"RESOLUCI[ÓO]N\s+N[°º]\s*\d+",
    "LICITACION":  r"LICITACI[ÓO]N",
    "ADJUDICACION":r"ADJUDICACI[ÓO]N",
    "REMATE":      r"REMATE",
    "SUCESORIO":   r"SUCESORIO",
    "QUIEBRA":     r"QUIEBRA",
    "SOCIEDAD":    r"SOCIEDAD",
    "AVISO":       r"AVISO",
    "ASAMBLEA":    r"ASAMBLEA",
    "LEY":         r"LEY\s+N[°º]\s*\d+",
}

def heuristic_classify(text: str) -> str:
    for cat, pattern in CATEGORIES.items():
        if re.search(pattern, text or "", re.IGNORECASE):
            return cat
    return "OTROS"

def _make_client_from_env() -> Optional[OpenAI]:
    """
    Crea el cliente OpenAI si hay OPENAI_API_KEY y el SDK está disponible.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def llm_classify(text: str, client: OpenAI, model: str = None) -> dict:
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = f"""
Devuelve SOLO un JSON válido con los campos:
- doc_id
- tipo
- numero
- fecha
- organismo
- personas
- resumen

Texto:
{text[:1000]}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()

    # limpia bloques ```json ... ```
    if raw.startswith("```"):
        raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except Exception:
        # fallback robusto para no romper pipeline
        return {
            "doc_id": None,
            "tipo": "OTROS",
            "numero": None,
            "fecha": None,
            "organismo": None,
            "personas": [],
            "resumen": (text or "")[:200],
        }

class ClassifierAgent:
    """
    Igual a tu notebook:
    - Primero heurística (barata y rápida).
    - Si devuelve OTROS y hay API Key, usa LLM como fallback.
    """
    def __init__(self, client: Optional[OpenAI] = None, model: Optional[str] = None):
        self.client = client if client is not None else _make_client_from_env()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def classify_chunk(self, chunk: str, doc_id: str, idx: int) -> Dict:
        cat = heuristic_classify(chunk)
        if cat != "OTROS":
            return {
                "doc_id": f"{doc_id}_c{idx}",
                "tipo": cat,
                "numero": None,
                "fecha": None,
                "organismo": None,
                "personas": [],
                "resumen": (chunk or "")[:200],
            }

        if self.client:
            try:
                rec = llm_classify(chunk, client=self.client, model=self.model)
                rec["doc_id"] = f"{doc_id}_c{idx}"
                return rec
            except Exception:
                pass

        # fallback final
        return {
            "doc_id": f"{doc_id}_c{idx}",
            "tipo": "OTROS",
            "numero": None,
            "fecha": None,
            "organismo": None,
            "personas": [],
            "resumen": (chunk or "")[:200],
        }
