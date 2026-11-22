"""intent_validate — 协调阶段对 intent 的确定性校验 / 归一化（移植自 old-version `intent_validate.py`）。

职责：
1. domain / sub_domain 封闭枚举校验；
2. domain 概念冲突检测（query 强信号词 vs 判定的 domain）；
3. LLM 偶尔把 list 字段填成 str 的归一化；非法 sub_domain 的确定性剔除。

validate_intent 返回 warnings 列表，不抛错——写进 Intent.warnings，让下游阶段与前端看到提示。

与 old-version 的差异：词匹配改用 `taxonomy.find_hits`（大小写不敏感、ASCII 词单词边界、
忽略中英文间空格），避免 `GD` 命中 `GDP`、`token 消耗` 漏检；同时支持用 `text` 覆盖检测文本
（追问回答并入后的文本）。
"""
from __future__ import annotations

from typing import Any

from .taxonomy import (
    COMMERCIALIZATION_STRONG_TOKENS, DOMAIN_ENUM, ECOM_LIVE_TOKENS, INNER_LOOP_TOKENS, LIVE_FLOW_TOKENS,
    RESOURCE_COST_STRONG_TOKENS, SOCIAL_LIVE_TOKENS, SUB_DOMAIN_ENUM, find_hits, has,
)

__all__ = [
    "DOMAIN_ENUM", "SUB_DOMAIN_ENUM", "normalize_intent_lists", "drop_invalid_sub_domains",
    "detect_domain_concept_conflict", "validate_intent",
]

# schema 规定为 list 的 intent 字段。LLM 偶尔填成 str（如 sub_domain="交易驾驶舱"），
# 下游 list("交易驾驶舱") 会拆成单字符——入口统一归一化为单元素 list。
_LIST_FIELDS = ("sub_domain", "keywords", "dim_terms", "asset_refs", "kb_intents")


def normalize_intent_lists(intent: dict[str, Any]) -> dict[str, Any]:
    """str → 单元素 list（空串 → []）。就地修改并返回；已是 list / None / 其他类型不动。"""
    if not isinstance(intent, dict):
        return intent
    for key in _LIST_FIELDS:
        val = intent.get(key)
        if isinstance(val, str):
            intent[key] = [val] if val.strip() else []
    return intent


def drop_invalid_sub_domains(intent: dict[str, Any]) -> dict[str, Any]:
    """剔除不在对应 domain 封闭枚举内的 sub_domain，保证下游拿到的永远是合法子集。

    只剔除、不猜测映射（改写词与触发词常互不为子串，模糊映射易误纠）。domain 非法/为空时不处理。就地修改并返回。
    """
    if not isinstance(intent, dict):
        return intent
    valid = SUB_DOMAIN_ENUM.get((intent.get("domain") or "").strip())
    subs = intent.get("sub_domain")
    if valid and isinstance(subs, list):
        intent["sub_domain"] = [sd for sd in subs if sd in valid]
    return intent


def detect_domain_concept_conflict(intent: dict[str, Any], text: str | None = None) -> list[str]:
    """检测 query 强信号词与 intent.domain 是否冲突。只警告，不自动改 domain。"""
    warns: list[str] = []
    query = text if text is not None else (intent.get("query") or "")
    domain = (intent.get("domain") or "").strip()
    if not query or not domain:
        return warns

    hit_comm = find_hits(query, COMMERCIALIZATION_STRONG_TOKENS)
    inner_loop_hit = find_hits(query, INNER_LOOP_TOKENS)
    social_live = has(query, *SOCIAL_LIVE_TOKENS, "连麦")

    if inner_loop_hit and domain == "transaction":
        warns.append(
            f"⚠️ domain 概念冲突：query 含内循环/闭环广告词 {inner_loop_hit}，但 intent.domain=transaction。"
            f"这类指标属 commercialization/电商闭环广告（即使含 DGMV 也不切到 transaction）。"
            f"建议改 domain=commercialization, sub_domain=[闭环电商广告]"
        )

    if has(query, "互动直播") and domain != "community":
        warns.append(
            f"⚠️ domain 概念冲突：query 含「互动直播」，但 intent.domain={domain}。互动直播属 community（社区直播），不是电商直播"
        )
    if has(query, "打赏", "ACU", "主播营收", "REDGALA") and domain == "transaction":
        warns.append("⚠️ domain 概念冲突：query 含社区直播词（打赏/ACU/主播营收/REDGALA），但 intent.domain=transaction。建议改 community")

    has_ecom_live = has(query, *ECOM_LIVE_TOKENS)
    is_live_flow_only = has(query, "直播") and has(query, *LIVE_FLOW_TOKENS) and not social_live
    if has_ecom_live and domain == "community":
        warns.append(
            "⚠️ domain 概念冲突：query 含「电商直播/店播/K播/直播带货」词，但 intent.domain=community。电商直播属 transaction"
            "（电商直播同样关心直播间流量/观播/停留等指标，不要因为出现流量词就归 community）"
        )
    elif is_live_flow_only and domain == "community":
        warns.append(
            "⚠️ domain 概念冲突：query 只含「直播+流量/观播/停留」无社区强信号，默认应判 transaction（电商直播主流）。"
            "若是社区直播请补充「互动直播/打赏/ACU」等强信号词"
        )

    if has(query, "电商") and domain == "commercialization" and not inner_loop_hit:
        warns.append(
            "⚠️ domain 概念冲突：query 含「电商」，但 intent.domain=commercialization。"
            "在小红书语境下「电商」=交易，应判 transaction（除非语境是「电商闭环广告」）"
        )
    if has(query, "商业化", "广告主", "广告消耗", "广告流水") and domain == "transaction":
        warns.append("⚠️ domain 概念冲突：query 含商业化广告词，但 intent.domain=transaction，建议改 commercialization")

    if hit_comm and domain == "transaction" and not inner_loop_hit:
        warns.append(f"⚠️ query 含强 commercialization 词 {hit_comm[:3]}，但 domain=transaction，可能误判")

    hit_rc = find_hits(query, RESOURCE_COST_STRONG_TOKENS)
    if hit_rc and domain != "resource_cost":
        warns.append(
            f"⚠️ domain 概念冲突：query 含资源成本强信号词 {hit_rc[:3]}，但 intent.domain={domain}。"
            f"GPU 供需/token 用量类问题属 resource_cost（Rule I），建议改 domain=resource_cost"
        )
    return warns


def validate_intent(intent: dict[str, Any], text: str | None = None) -> list[str]:
    """轻校验，返回 warnings（不抛错）。

    1. domain 是否在五域封闭枚举内；
    2. sub_domain 每项是否在对应 domain 的封闭枚举内；
    3. domain 非空时 sub_domain 不应为空（prompt 要求未命中触发词时退化为语义理解，确保非空）；
    4. domain 概念冲突检测（基于 query 强信号词）。
    """
    warns: list[str] = []
    domain = (intent.get("domain") or "").strip()
    if domain and domain not in DOMAIN_ENUM:
        warns.append(f"domain `{domain}` 不在合法枚举 {sorted(DOMAIN_ENUM)}；将按空 domain 处理")
        return warns  # domain 错就不再校 sub_domain

    sub_doms = intent.get("sub_domain")
    if sub_doms is not None and not isinstance(sub_doms, list):
        warns.append(f"sub_domain 必须是 list，收到 {type(sub_doms).__name__}")
        return warns

    if domain and sub_doms:
        valid = SUB_DOMAIN_ENUM.get(domain, set())
        for sd in sub_doms:
            if sd not in valid:
                hits = sorted(v for v in valid if sd in v or v in sd)
                hint = f"，建议改为 {hits}" if hits else "（请参考 sub_domain 触发词表）"
                warns.append(f"sub_domain `{sd}` 不在 domain=`{domain}` 的封闭枚举内{hint}")
    elif domain and not sub_doms:
        warns.append(f"sub_domain 为空但 domain=`{domain}` 已识别——按 prompt 要求应退化语义理解产出非空集合，请检查 intent_analysis 是否遗漏")

    warns.extend(detect_domain_concept_conflict(intent, text))
    return warns
