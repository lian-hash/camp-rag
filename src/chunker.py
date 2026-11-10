"""语料读取与分块。

分块策略：先按二级标题（## ）切成 section，再按 max_chars 把 section 内的段落
合并成 chunk。这样切点落在语义边界上，而不是硬按字数截断。
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    doc: str          # 来源文件名，如 "04-取消与退款.md"
    section: str      # 所属小节标题
    text: str
    tokens: list[str] = field(default_factory=list)


def load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    """读目录下所有 .md，返回 [(文件名, 正文)]，按文件名排序。"""
    docs = []
    for p in sorted(Path(corpus_dir).glob("*.md")):
        docs.append((p.name, p.read_text(encoding="utf-8")))
    return docs


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按标题切 section，返回 [(标题, 正文)]。"""
    sections = []
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]

    # 第一个标题之前的内容（通常是 # 一级标题）
    head = text[: matches[0].start()].strip()
    if head:
        sections.append(("概述", head))

    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _pack_paragraphs(body: str, max_chars: int, min_chars: int) -> list[str]:
    """把 section 内的段落合并成不超过 max_chars 的块。"""
    paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
    if not paras:
        return []

    blocks = []
    buf = ""
    for p in paras:
        # 单段就超长，按句子再切
        if len(p) > max_chars:
            if buf:
                blocks.append(buf)
                buf = ""
            sentences = re.split(r"(?<=[。！？；])", p)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) <= max_chars:
                    cur += s
                else:
                    if cur:
                        blocks.append(cur.strip())
                    cur = s
            if cur.strip():
                if len(cur.strip()) < min_chars and blocks:
                    blocks[-1] += cur.strip()
                else:
                    blocks.append(cur.strip())
            continue

        if len(buf) + len(p) + 1 <= max_chars:
            buf = buf + p if not buf else buf + "\n" + p
        else:
            if buf:
                blocks.append(buf)
            buf = p

    if buf:
        if len(buf) < min_chars and blocks:
            blocks[-1] += "\n" + buf
        else:
            blocks.append(buf)
    return blocks


def chunk_corpus(
    corpus_dir: Path, max_chars: int = 300, min_chars: int = 80
) -> list[Chunk]:
    """按标题驱动分块，返回 Chunk 列表。"""
    chunks = []
    for doc_name, text in load_corpus(corpus_dir):
        for section_title, body in _split_sections(text):
            for j, block in enumerate(_pack_paragraphs(body, max_chars, min_chars)):
                cid = f"{doc_name}#{section_title}#{j}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc=doc_name,
                        section=section_title,
                        text=block,
                    )
                )
    return chunks


def chunk_corpus_fixed(corpus_dir: Path, max_chars: int = 300) -> list[Chunk]:
    """朴素做法：不看标题，按固定字数硬切。

    用来和标题驱动分块做对照——硬切容易把「结论」和「条件」切到两个块里。
    """
    chunks = []
    for doc_name, text in load_corpus(corpus_dir):
        # 去掉标题行，避免标题被算进正文
        body = _HEADING.sub("", text)
        body = re.sub(r"\n{2,}", "\n", body).strip()
        for i in range(0, len(body), max_chars):
            seg = body[i : i + max_chars].strip()
            if not seg:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_name}#fixed#{i // max_chars}",
                    doc=doc_name,
                    section="fixed",
                    text=seg,
                )
            )
    return chunks


def build_chunks(
    corpus_dir: Path, mode: str = "section", max_chars: int = 300
) -> list[Chunk]:
    if mode == "fixed":
        return chunk_corpus_fixed(corpus_dir, max_chars=max_chars)
    return chunk_corpus(corpus_dir, max_chars=max_chars)
