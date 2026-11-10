"""命令行入口。

用法：
    python main.py stats
    python main.py ask "装备在配送路上了还能取消吗"
    python main.py eval --embedder tfidf --alpha 0.5
    python main.py eval --embedder bge --alpha 0.5
    python main.py compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 下 PowerShell 默认编码可能让中文输出乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.chunker import build_chunks                  # noqa: E402
from src.embedder import build_embedder                  # noqa: E402
from src.evaluate import load_questions, print_summary, run_eval  # noqa: E402
from src.retriever import HybridRetriever                # noqa: E402
from src.text import using_jieba                         # noqa: E402


def build_retriever(args) -> HybridRetriever:
    chunks = build_chunks(ROOT / "corpus", mode=args.chunk_mode, max_chars=args.chunk_size)
    embedder = build_embedder(args.embedder)
    return HybridRetriever(chunks, embedder, alpha=args.alpha)


def cmd_stats(args) -> None:
    chunks = build_chunks(ROOT / "corpus", mode=args.chunk_mode, max_chars=args.chunk_size)
    docs = {c.doc for c in chunks}
    print(f"语料文档数 : {len(docs)}")
    print(f"分块数     : {len(chunks)}（max_chars={args.chunk_size}）")
    print(f"平均块长度 : {sum(len(c.text) for c in chunks) / len(chunks):.0f} 字符")
    print(f"分词器     : {'jieba' if using_jieba() else '内置 bigram（未装 jieba）'}")
    print("\n每篇文档块数：")
    for d in sorted(docs):
        n = sum(1 for c in chunks if c.doc == d)
        print(f"  {d:<28} {n}")


def cmd_ask(args) -> None:
    retriever = build_retriever(args)
    hits = retriever.search(args.query, k=args.k)
    print(f"\n查询：{args.query}\n")
    for i, (chunk, score) in enumerate(hits, 1):
        print(f"[{i}] {score:.4f}  {chunk.doc}  ›  {chunk.section}")
        print(f"    {chunk.text[:160].replace(chr(10), ' ')}...\n")
    print("以上只是检索结果，没有经过 LLM 生成。")
    print("答案要自己从这些片段里读——这也是这个项目刻意做的取舍：")
    print("把 R 做扎实，G 留成可替换的接口。")


def cmd_eval(args) -> None:
    retriever = build_retriever(args)
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    summary = run_eval(retriever, questions, k=args.k, verbose=args.verbose)
    print_summary(
        summary,
        title=f"embedder={args.embedder}  alpha={args.alpha}  chunk={args.chunk_size}",
    )


def cmd_compare(args) -> None:
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    configs = [
        ("tfidf 纯 BM25", "tfidf", 0.0),
        ("tfidf 纯向量", "tfidf", 1.0),
        ("tfidf 混合 0.5", "tfidf", 0.5),
    ]
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        configs += [
            ("bge 纯向量", "bge", 1.0),
            ("bge 混合 0.7", "bge", 0.7),
            ("bge 混合 0.5", "bge", 0.5),
        ]
    except ImportError:
        print("（未安装 torch/transformers，跳过 bge 配置）")

    rows = []
    for label, emb, alpha in configs:
        chunks = build_chunks(ROOT / "corpus", mode=args.chunk_mode, max_chars=args.chunk_size)
        retriever = HybridRetriever(chunks, build_embedder(emb), alpha=alpha)
        s = run_eval(retriever, questions, k=args.k)
        rows.append((label, s))

    print(f"\n=== 配置对比（chunk={args.chunk_size}, k={args.k}）===")
    print(f"{'配置':<18}{'doc_recall':>12}{'doc_all':>10}{'kw_hit':>10}{'ctx_len':>10}")
    for label, s in rows:
        print(
            f"{label:<18}{s['doc_recall']:>12.3f}{s['doc_recall_all']:>10.3f}"
            f"{s['kw_hit']:>10.3f}{s['ctx_len_mean']:>10.0f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="camp-rag：城野随行营地知识检索")
    p.add_argument("command", choices=["stats", "ask", "eval", "compare"])
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--embedder", default="tfidf", choices=["tfidf", "bge"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=300, dest="chunk_size")
    p.add_argument("--chunk-mode", default="section", choices=["section", "fixed"], dest="chunk_mode")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    if args.command == "stats":
        cmd_stats(args)
    elif args.command == "ask":
        if not args.query:
            p.error("ask 需要一个查询语句")
        cmd_ask(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
