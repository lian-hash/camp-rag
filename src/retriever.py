"""检索：向量召回 + BM25 召回，两路分数归一化后加权融合。"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .chunker import Chunk
from .embedder import BaseEmbedder
from .text import tokenize


class BM25:
    """经典 BM25，k1 控制词频饱和，b 控制长度归一化强度。"""

    def __init__(self, tokenized: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized)
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0

        self.tf: list[Counter] = [Counter(t) for t in tokenized]
        df = Counter()
        for t in tokenized:
            for tok in set(t):
                df[tok] += 1
        self.df = df

    def idf(self, tok: str) -> float:
        df = self.df.get(tok, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        q = [t for t in query_tokens]
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not q:
            return out
        for i in range(self.n_docs):
            tf_d = self.tf[i]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0))
            s = 0.0
            for tok in q:
                f = tf_d.get(tok, 0)
                if f == 0:
                    continue
                s += self.idf(tok) * (f * (self.k1 + 1)) / (f + norm)
            out[i] = s
        return out


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    """
    alpha 控制两路权重：
      alpha = 1.0  只用向量召回
      alpha = 0.0  只用 BM25
      0 < alpha < 1  加权融合
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: BaseEmbedder,
        alpha: float = 0.5,
        use_bm25: bool = True,
    ):
        self.chunks = chunks
        self.alpha = alpha
        self.use_bm25 = use_bm25

        self.texts = [c.text for c in chunks]
        for c in chunks:
            c.tokens = tokenize(c.text)

        self.embedder = embedder
        self.embedder.fit(self.texts)
        self.doc_vecs = self.embedder.encode(self.texts)

        self.bm25 = BM25([c.tokens for c in chunks]) if use_bm25 else None

    def vector_scores(self, query: str) -> np.ndarray:
        qv = self.embedder.encode([query], is_query=True)[0]
        return self.doc_vecs @ qv

    def bm25_scores(self, query: str) -> np.ndarray:
        if self.bm25 is None:
            return np.zeros(len(self.chunks))
        return self.bm25.scores(tokenize(query))

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        v = self.vector_scores(query)
        if self.use_bm25 and self.bm25 is not None:
            b = self.bm25_scores(query)
            # 两路量纲差很远，先各自归一化再加权
            fused = self.alpha * _minmax(v) + (1 - self.alpha) * _minmax(b)
            raw = v
        else:
            fused = v
            raw = v

        order = np.argsort(-fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in order]

    def top1_raw_score(self, query: str) -> float:
        """返回原始向量相似度的最高分，用于判断「是否该拒答」。"""
        v = self.vector_scores(query)
        return float(v.max())
