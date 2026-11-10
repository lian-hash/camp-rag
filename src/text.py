"""中文分词。

优先用 jieba；环境里没装 jieba 时自动降级成「单字 + 双字 bigram」，
保证 clone 下来不装任何东西也能跑出结果。
"""

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_.]+")

# 极简停用词，只去掉对检索几乎无信息量的高频虚词
_STOP = set("的了是在有和与及对把被为这那个我你他它吗呢吧啊就都也还要会能")

_jieba_state = None


def _jieba():
    """懒加载 jieba，只尝试一次。"""
    global _jieba_state
    if _jieba_state is None:
        try:
            import jieba

            jieba.setLogLevel(20)
            _jieba_state = jieba
        except ImportError:
            _jieba_state = False
    return _jieba_state


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索用的 token 列表。"""
    text = text.strip()
    if not text:
        return []

    jb = _jieba()
    if jb:
        words = [w.strip() for w in jb.lcut(text)]
        return [w for w in words if w and w not in _STOP and len(w) > 0]

    # 降级路径：中文单字 + 双字 bigram，外加英文/数字词
    tokens = []
    chars = _CJK.findall(text)
    for c in chars:
        if c not in _STOP:
            tokens.append(c)
    for i in range(len(chars) - 1):
        pair = chars[i] + chars[i + 1]
        if pair[0] not in _STOP and pair[1] not in _STOP:
            tokens.append(pair)
    tokens.extend(w.lower() for w in _WORD.findall(text))
    return tokens


def using_jieba() -> bool:
    return bool(_jieba())
