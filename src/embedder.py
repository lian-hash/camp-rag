"""Embedding 后端。

两种实现，同一套接口：
- TfidfEmbedder：纯 numpy 手写 TF-IDF，零额外依赖，作为基线。
- BgeEmbedder：sentence-transformers + bge-small-zh-v1.5，真正的语义向量。

接口统一成 fit(texts) / encode(texts) -> (n, d) 的 L2 归一化矩阵，
这样后面的向量检索只管做点积就等于余弦相似度。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .text import tokenize


class BaseEmbedder:
    name = "base"

    def fit(self, texts: list[str]) -> "BaseEmbedder":
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(BaseEmbedder):
    """手写 TF-IDF。没有预训练模型，纯粹靠词频，作为可对照的基线。"""

    name = "tfidf"

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        tokenized = [tokenize(t) for t in texts]
        docs = [set(t) for t in tokenized]

        vocab = {}
        for doc_tokens in tokenized:
            for tok in doc_tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self.vocab = vocab

        n = len(texts)
        df = np.zeros(len(vocab), dtype=np.float64)
        for doc in docs:
            for tok in doc:
                df[vocab[tok]] += 1
        # 平滑 idf，避免除零
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if not toks:
                continue
            counts = Counter(toks)
            total = sum(counts.values())
            for tok, c in counts.items():
                idx = self.vocab.get(tok)
                if idx is not None:
                    mat[i, idx] = (c / total) * self.idf[idx]
        return _l2_normalize(mat)


class BgeEmbedder(BaseEmbedder):
    """bge-small-zh-v1.5 语义向量。

    直接用 transformers 加载，不强依赖 sentence-transformers。
    首次使用会联网下载模型（约 95MB）。

    bge 系列有个约定：查询要加一句指令前缀，文档不加。
    不按这个约定做，检索效果会明显变差——这也是它和「通用句向量」的区别。
    """

    name = "bge"
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            import os

            from transformers import AutoModel, AutoTokenizer

            # 先按「本地已有缓存」加载，这样训练/评测可以完全离线跑，
            # 也避免 huggingface_hub 每次额外请求 API 检查 chat template。
            try:
                self._tok = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=True
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except Exception:
                # 本地没有缓存，第一次跑才联网下载
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                self._tok = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
        return self._tok, self._model

    def fit(self, texts: list[str]) -> "BgeEmbedder":
        return self

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        import torch

        tok, model = self._load()
        if is_query:
            texts = [self.QUERY_INSTRUCTION + t for t in texts]
        inputs = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**inputs)
        # bge 用 [CLS] 位置的向量作为句向量
        emb = out.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype("float32")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_embedder(name: str) -> BaseEmbedder:
    if name == "tfidf":
        return TfidfEmbedder()
    if name == "bge":
        return BgeEmbedder()
    raise ValueError(f"未知 embedder: {name}，可选 tfidf / bge")
