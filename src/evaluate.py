"""评测：把 40 条问题跑一遍，算召回和答案命中，并给出边界题的可分性。"""

from __future__ import annotations

import json
from pathlib import Path

from .retriever import HybridRetriever


def load_questions(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _doc_recall(returned: list[str], gold: list[str]) -> bool:
    """命中任意一个 gold 文档就算召回。"""
    return any(g in returned for g in gold)


def _doc_recall_all(returned: list[str], gold: list[str]) -> bool:
    """gold 文档全部出现在 top-k 里，多跳题才在乎这个。"""
    return all(g in returned for g in gold)


def _kw_hit(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    return all(kw in text for kw in keywords)


def run_eval(
    retriever: HybridRetriever,
    questions: list[dict],
    k: int = 5,
    verbose: bool = False,
) -> dict:
    per_q = []
    for q in questions:
        hits = retriever.search(q["question"], k=k)
        returned_docs = [c.doc for c, _ in hits]
        joined = "\n".join(c.text for c, _ in hits)
        ctx_len = sum(len(c.text) for c, _ in hits)
        top1_raw = retriever.top1_raw_score(q["question"])

        if q["should_refuse"]:
            per_q.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "should_refuse": True,
                    "top1_raw": top1_raw,
                    "doc_hit": None,
                    "doc_all": None,
                    "kw_hit": None,
                    "ctx_len": ctx_len,
                    "top1_doc": returned_docs[0] if returned_docs else None,
                }
            )
            continue

        rec = {
            "id": q["id"],
            "type": q["type"],
            "should_refuse": False,
            "top1_raw": top1_raw,
            "doc_hit": _doc_recall(returned_docs, q["gold_docs"]),
            "doc_all": _doc_recall_all(returned_docs, q["gold_docs"]),
            "kw_hit": _kw_hit(joined, q["gold_keywords"]),
            "ctx_len": ctx_len,
            "top1_doc": returned_docs[0] if returned_docs else None,
        }
        per_q.append(rec)

        if verbose:
            mark = "OK " if rec["doc_hit"] else "MISS"
            print(
                f"[{mark}] {q['id']:<4} {q['question'][:34]:<34} "
                f"top1={rec['top1_doc']}  gold={q['gold_docs']}"
            )

    # 汇总
    answerable = [r for r in per_q if not r["should_refuse"]]
    boundary = [r for r in per_q if r["should_refuse"]]

    summary = {
        "k": k,
        "n_total": len(per_q),
        "n_answerable": len(answerable),
        "n_boundary": len(boundary),
        "doc_recall": _ratio(answerable, "doc_hit"),
        "doc_recall_all": _ratio(answerable, "doc_all"),
        "kw_hit": _ratio(answerable, "kw_hit"),
        "ctx_len_mean": (
            sum(r["ctx_len"] for r in answerable) / len(answerable) if answerable else 0.0
        ),
        "by_type": {},
        "boundary_top1_mean": (
            sum(r["top1_raw"] for r in boundary) / len(boundary) if boundary else 0.0
        ),
        "answerable_top1_mean": (
            sum(r["top1_raw"] for r in answerable) / len(answerable) if answerable else 0.0
        ),
    }
    for t in sorted({r["type"] for r in answerable}):
        sub = [r for r in answerable if r["type"] == t]
        summary["by_type"][t] = {
            "n": len(sub),
            "doc_recall": _ratio(sub, "doc_hit"),
            "kw_hit": _ratio(sub, "kw_hit"),
        }
    summary["per_question"] = per_q
    return summary


def _ratio(rows: list[dict], key: str) -> float:
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r[key]) / len(valid)


def print_summary(summary: dict, title: str = "") -> None:
    if title:
        print(f"\n=== {title} ===")
    print(f"问题数 {summary['n_total']}（可答 {summary['n_answerable']} / 应拒答 {summary['n_boundary']}）")
    print(f"文档召回率 doc_recall@{summary['k']}      : {summary['doc_recall']:.3f}")
    print(f"多跳全命中率 doc_recall_all@{summary['k']} : {summary['doc_recall_all']:.3f}")
    print(f"答案命中率 kw_hit@{summary['k']}         : {summary['kw_hit']:.3f}")
    print(f"top-{summary['k']} 拼接上下文长度        : {summary['ctx_len_mean']:.0f} 字符")
    print(f"边界题 top1 均分   : {summary['boundary_top1_mean']:.4f}")
    print(f"可答题 top1 均分   : {summary['answerable_top1_mean']:.4f}")
    print("按类型：")
    for t, s in summary["by_type"].items():
        print(f"  {t:<12} n={s['n']:<3} doc_recall={s['doc_recall']:.3f}  kw_hit={s['kw_hit']:.3f}")
