"""轻量文本工具（占位实现：后续可替换为分词 / embedding）。"""
from __future__ import annotations

import re

_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

_STOP = frozenset({"最近", "的", "和", "与", "各", "按", "每", "查询", "分渠道", "多少"}) | frozenset({
    "LAST", "DAYS", "DAY", "BY", "AND", "THE", "OF", "IN", "FOR", "PER", "TO", "ON", "A", "AN", "WHAT", "IS", "ARE",
    "YESTERDAY", "TODAY", "WEEK", "MONTH", "THIS", "SHOW", "ME", "GET",
})


def extract_terms(text: str) -> list[str]:
    """英文词（大写归一）+ 中文 bigram，去停用词、保序去重。"""
    out: list[str] = [t.upper() for t in _ASCII_TOKEN.findall(text)]
    for run in _CJK_RUN.findall(text):
        for seg in re.split(r"[的和与及或]", run):      # 虚词切开，避免产生跨词噪声 bigram
            if len(seg) == 1:
                out.append(seg)
            out.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    seen: set[str] = set()
    terms: list[str] = []
    for t in out:
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        terms.append(t)
    return terms


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coeff(a: set[str], b: set[str]) -> float:
    """重叠系数 |a∩b| / min(|a|,|b|)：适合"短 pair 被长 query 包含"的相似度。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dice(a: set[str], b: set[str]) -> float:
    """Dice 系数 2|a∩b|/(|a|+|b|)：query 相似度，对长度差异比 Jaccard 宽容、比重叠系数稳健。"""
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def char_bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text.lower())
    return {t[i : i + 2] for i in range(len(t) - 1)} or ({t} if t else set())


def text_similarity(a: str, b: str) -> float:
    """两段文本的字符 bigram Jaccard，用于判定知识块间冗余。"""
    return jaccard(char_bigrams(a), char_bigrams(b))


def estimate_tokens(text: str) -> int:
    """粗估 token：中文约 1 字 1 token，英文约 4 字符 1 token。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    rest = len(text) - cjk
    return cjk + (rest + 3) // 4
