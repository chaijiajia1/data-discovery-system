"""prompt 模板加载：读取 prompts/{name}.md，递归展开 `{{ include: foo.md }}`。"""
from __future__ import annotations

import re

from .settings import settings

_INCLUDE = re.compile(r"\{\{\s*include\s*:\s*([^\s{}]+?)\s*\}\}")


def load_prompt(name: str, _stack: tuple[str, ...] = ()) -> str:
    fname = name if name.endswith(".md") else f"{name}.md"
    if fname in _stack:
        raise RuntimeError(f"prompt include recursion: {' -> '.join(_stack + (fname,))}")
    text = (settings.PROMPTS_DIR / fname).read_text(encoding="utf-8")
    return _INCLUDE.sub(lambda m: load_prompt(m.group(1), _stack + (fname,)), text)
