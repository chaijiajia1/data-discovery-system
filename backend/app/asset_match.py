"""字段匹配的单一实现：业务词 → 候选资产的字段。

⑤ 的字段命中（rules.evaluate_group）与 ④ 的维度覆盖核对（roles/checks.py）共用这里，不再各写一套子串匹配：
字面相同 > 同义（知识桥接）> 部分 > 推断（名称 / 描述 / 标签）。字段有已知角色（metric / dimension）时，「部分」匹配要求角色一致。
"""
from __future__ import annotations

from typing import Any

from .schemas.knowledge import KnowledgeItem
from .schemas.recall import CandidateAsset
from .schemas.selection import Hit


def _cf(s: Any) -> str:
    return str(s or "").strip().casefold()


def is_chart(c: CandidateAsset) -> bool:
    return c.asset_type == "chart"


def render_field(c: CandidateAsset, f: str) -> str:
    """逐字复制候选字段原文：★ 主键标记 + 中文名|英文名。"""
    alias = c.meta.field_aliases.get(f)
    return ("★" if f in c.meta.primary_keys else "") + (f"{alias}|{f}" if alias else f)


def _syn(term: str, knowledge: list[KnowledgeItem]) -> tuple[set[str], list[str]]:
    """term 经知识桥接得到的同义词（不含自身）与命中的知识 id。"""
    t, syn, ids = _cf(term), set(), []
    for k in knowledge:
        names = {_cf(k.title), *(_cf(x) for x in k.terms)}
        if t in names:
            syn |= names - {t}
            ids.append(k.id)
    return syn, ids


def _field_roles_ok(c: CandidateAsset, f: str, kind: str) -> bool:
    r = c.meta.field_roles.get(f)
    return r is None or r == kind


def match_term(term: str, kind: str, c: CandidateAsset, knowledge: list[KnowledgeItem]) -> tuple[Hit | None, str | None, list[str]]:
    """业务词 → 候选字段。返回 (命中, 物理字段名, 参与桥接的知识 id)。字面相同 > 同义（知识桥接）> 部分 > 推断（名称 / 描述 / 标签）。"""
    t = _cf(term)
    if not t:
        return None, None, []
    if is_chart(c):
        pool = c.meta.metric_ids if kind == "metric" else c.meta.supported_dims
        loc = "metric_ids" if kind == "metric" else "supported_dims"
        for how, ok in (("字面相同", lambda x: _cf(x) == t), ("部分", lambda x: len(t) >= 2 and (t in _cf(x) or _cf(x) in t and len(_cf(x)) >= 2))):
            for i, x in enumerate(pool):
                if ok(x):
                    return Hit(term=term, type=kind, matched=str(x), location=f"{loc}[{i}]", how=how), None, []   # type: ignore[arg-type]
        return None, None, []
    fields = c.meta.fields
    syn, kn = _syn(term, knowledge)
    for how, ok in (
        ("字面相同", lambda f, a: t in {_cf(f), _cf(a)}),
        ("同义", lambda f, a: bool(syn & {_cf(f), _cf(a)})),
        ("部分", lambda f, a: len(t) >= 2 and any(t in x or (len(x) >= 2 and x in t) for x in (_cf(f), _cf(a)) if x)),
    ):
        for i, f in enumerate(fields):
            if (how == "字面相同" or _field_roles_ok(c, f, kind)) and ok(f, c.meta.field_aliases.get(f)):
                return (Hit(term=term, type=kind, matched=render_field(c, f), location=f"fields[{i}]", how=how),  # type: ignore[arg-type]
                        f, kn if how == "同义" else [])
    if not fields:                                                         # 没有字段元信息：按名称 / 描述 / 标签推断
        for loc, text in [("name", c.name), ("description", c.meta.description or ""), *[(f"tags[{i}]", x) for i, x in enumerate(c.meta.tags)]]:
            if len(t) >= 2 and t in _cf(text):
                return Hit(term=term, type=kind, matched=str(text), location=loc, how="推断"), None, []     # type: ignore[arg-type]
    return None, None, []


def chart_dims_unlimited(c: CandidateAsset) -> bool:
    """图表的 supported_dims == [] 表示无限制：视为覆盖全部所需维度。"""
    return is_chart(c) and not c.meta.supported_dims


def covered_dims(c: CandidateAsset, dims: list[str], knowledge: list[KnowledgeItem]) -> list[str]:
    """候选资产真正覆盖的维度（保持 dims 的顺序）。仅凭名称 / 描述 / 标签的「推断」命中不算覆盖证据。"""
    if chart_dims_unlimited(c):
        return list(dims)
    out = []
    for d in dims:
        hit, _, _ = match_term(d, "dimension", c, knowledge)
        if hit is not None and hit.how != "推断":
            out.append(d)
    return out
