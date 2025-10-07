# plugins/tasks/utils_op_split.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import unicodedata

import fitz  # PyMuPDF

# ============================
# Patrones y limpieza
# ============================

# Marca fin de bloque por OP (ajusta si tu boletín usa otra variante)
PATRON_OP_BOUNDARY = re.compile(r"OP\s*N[°º]:\s*[A-Z]*\d{6,}", re.IGNORECASE)

# Extrae el código de OP más “al final” del bloque
PATRON_OP_CODE_LAST = re.compile(r"OP\s*N[°º]?:?\s*([A-Z]*\d{6,})\b", re.IGNORECASE)

RE_PAGINA = re.compile(r"Pág\.\s*N[°º]?\s*\d+", re.IGNORECASE)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_HARD_HYPH = re.compile(r"(\w)-\n(\w)")

def _normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00A0", " ")  # NBSP
    # ligaduras frecuentes
    t = t.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return t

def eliminar_pies_pagina(texto: str) -> str:
    """
    Heurística: si una línea comienza con "Pág. N° X", salto esa línea + 3 siguientes
    """
    lineas = texto.splitlines()
    nuevas, skip = [], 0
    for ln in lineas:
        if skip > 0:
            skip -= 1
            continue
        if RE_PAGINA.match(ln.strip()):
            skip = 3
            continue
        nuevas.append(ln)
    return "\n".join(nuevas)

def limpieza_basica(texto: str) -> str:
    if not texto:
        return ""
    # deshiphen + compactar espacios
    texto = RE_HARD_HYPH.sub(r"\1\2", texto)
    texto = RE_MULTI_SPACE.sub(" ", texto)
    return texto.strip()

def extraer_op_final(texto: str) -> Optional[str]:
    last = None
    for m in PATRON_OP_CODE_LAST.finditer(texto or ""):
        last = m.group(1)
    return last

def parse_base_id(base: str) -> Tuple[str, str]:
    """
    '22036_2025-09-22' -> ('22036', '2025-09-22')
    Si no matchea, (base, '0000-00-00')
    """
    m = re.match(r"(?P<bol>\d{5})(?:[_-](?P<date>\d{4}-\d{2}-\d{2}))?$", base)
    if m:
        bol = m.group("bol")
        date = m.group("date") or "0000-00-00"
        return bol, date
    return base, "0000-00-00"

# ============================
# Split principal
# ============================

def split_pdf_por_op(
    pdf_path: Path,
    ignore_first_pages: int = 2,
    ignore_last_pages: int = 1
) -> List[Dict]:
    """
    Lee el PDF con PyMuPDF, concatena el texto de páginas [ignore_first_pages : len - ignore_last_pages],
    segmenta por el patrón de OP, limpia pies de página y espacios, y devuelve:
      [{"texto": str, "op": str|None, "doc_index": int}, ...]
    """
    if not pdf_path.exists():
        raise FileNotFoundError(str(pdf_path))

    doc = fitz.open(str(pdf_path))
    try:
        p0 = max(0, int(ignore_first_pages))
        p1 = len(doc) - max(0, int(ignore_last_pages))
        p1 = max(p0, p1)

        texto_sumario = ""
        for i in range(p0, p1):
            texto_sumario += doc[i].get_text()

        texto_sumario = _normalize_text(texto_sumario)

        matches = list(PATRON_OP_BOUNDARY.finditer(texto_sumario))
        bloques: List[str] = []
        if not matches:
            bloques = [texto_sumario.strip()]
        else:
            start_idx = 0
            for m in matches:
                end_idx = m.end()
                bloque = texto_sumario[start_idx:end_idx].strip()
                if bloque:
                    bloques.append(bloque)
                start_idx = end_idx
            if start_idx < len(texto_sumario):
                tail = texto_sumario[start_idx:].strip()
                if tail:
                    bloques.append(tail)

        out: List[Dict] = []
        for k, b in enumerate(bloques, start=1):
            b2 = eliminar_pies_pagina(b)
            b2 = limpieza_basica(b2)
            op = extraer_op_final(b2)
            out.append({"texto": b2, "op": op, "doc_index": k})
        return out
    finally:
        doc.close()
