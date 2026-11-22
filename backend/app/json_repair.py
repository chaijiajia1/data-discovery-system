"""LLM 输出被 max_tokens 截断时的 JSON 兜底（移植自旧版 drr-service-debug 的 `_try_partial_parse`）。

典型截断形态：`{"plans": [{...完整...}, {半截...` → 保留完整部分：`{"plans": [{...完整...}]}`。
策略：自尾部向前找 `]` / `}`，在该位置截断并补全外层缺失的括号后重试解析；最多尝试 50 次，避免病态输入卡死。
只处理以 `{` 开头的对象；截在字符串中间的候选放弃。
"""
from __future__ import annotations

import json
from typing import Any

MAX_TRIES = 50


def close_open_brackets(s: str) -> str | None:
    """对 s 做括号平衡：剩余未闭合的 `{` / `[` 自尾部补 `}` / `]`；忽略字符串字面量内的括号。
    括号不平衡、或结束在字符串中间时返回 None。"""
    stack: list[str] = []
    in_str = escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()
            else:
                return None
    if in_str:
        return None
    return s + "".join("}" if c == "{" else "]" for c in reversed(stack))


def try_partial_parse(text: str) -> dict[str, Any] | None:
    if not text or not text.lstrip().startswith("{"):
        return None
    tried = 0
    for i in range(len(text) - 1, -1, -1):
        if text[i] not in "]}":
            continue
        tried += 1
        if tried > MAX_TRIES:
            break
        cand = close_open_brackets(text[: i + 1])
        if cand is None:
            continue
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
