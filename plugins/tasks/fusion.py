# fusion.py
from typing import List, Dict, Optional

def rrf_combine(
    *ranked_lists: List[str],
    k: float = 60.0,
    top_k: Optional[int] = None,
    weights: Optional[List[float]] = None,
) -> List[str]:
    """
    Reciprocal Rank Fusion (RRF) con soporte de múltiples listas.
    - ranked_lists: listas ordenadas (BM25, vectorial, etc.)
    - k: constante RRF (más grande => más suave)
    - top_k: recorta el resultado final (opcional)
    - weights: peso por lista (misma longitud que ranked_lists), opcional

    Devuelve una lista fusionada de IDs (sin repetir), ordenada por score RRF.
    """
    scores: Dict[str, float] = {}

    if weights is not None and len(weights) != len(ranked_lists):
        raise ValueError("weights debe tener la misma cantidad de elementos que ranked_lists")

    # Acumular puntajes RRF
    for li, ranked in enumerate(ranked_lists):
        if not ranked:
            continue
        w = 1.0 if not weights else float(weights[li])
        for rank, item in enumerate(ranked, start=1):  # rank empieza en 1
            if not item:
                continue
            scores[item] = scores.get(item, 0.0) + w * (1.0 / (k + rank))

    # Tie-break estable por primera aparición en cualquier lista
    first_pos: Dict[str, int] = {}
    for ranked in ranked_lists:
        if not ranked:
            continue
        for pos, item in enumerate(ranked):
            if item not in first_pos:
                first_pos[item] = pos

    fused = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], first_pos.get(kv[0], 10**9))
    )
    out = [item for item, _ in fused]
    return out[:top_k] if top_k else out
