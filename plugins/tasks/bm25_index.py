##############    bm25_index.py      ##################

from typing import Dict, List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
#from .documents import Document, simple_tokenize

from tasks.documents import Document, simple_tokenize

class BM25Index:
    def __init__(self, docs: List[Document], chunks_per_doc: Dict[str, List[str]]):
        self.doc_ids: List[str] = []
        self.chunks: List[str] = []
        for d in docs:
            for ch in chunks_per_doc[d.id]:
                self.doc_ids.append(d.id)
                self.chunks.append(ch)
        self.tokenized = [simple_tokenize(t) for t in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        idx_sorted = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[int(i)])) for i in idx_sorted]