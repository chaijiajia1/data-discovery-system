"""② 协调 · intent 的规则兜底（LLM 不可用 / 输出不可解析时使用，铁律 3）。

domain 判定顺序对齐 old-version `prompts/intent_analysis.md` 的 Rule A–K（每次命中记录 rule_id，铁律 4）：

    Rule I 资源成本 → Rule K 国际化(交易语义归 transaction+海外) → Rule A 闭环/内循环广告
    → Rule B/C 社区直播 → Rule D/E 电商直播 → Rule G/H 商业化强信号 → Rule F 「电商」
    → DGMV 特殊规则 → 交易 / 社区强信号计数 → Rule J 「流水/客户数」泛指

规则只做保守判定：没有信号就返回 domain=None（宁缺毋滥），dim_terms 一律不抽（漏抽只是错失富化，错抽会触发无效维值召回）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...schemas.intent import AssetRef, Intent
from ...schemas.knowledge import KnowledgeKind, PreInjectionResult
from .extractor import heuristic_extract
from .intent_validate import validate_intent
from .taxonomy import (
    ANALYSIS_TRIGGERS, COMMERCIALIZATION_STRONG_TOKENS, COMMUNITY_STRONG_TOKENS, DATA_FETCH_TRIGGERS, DEFAULT_SUB_DOMAIN,
    DEFINITION_TRIGGERS, DISCOVERY_TRIGGERS, ECOM_LIVE_TOKENS, GMV_TOKENS, INNER_LOOP_TOKENS, KB_INTENT_TRIGGERS,
    LIVE_FLOW_TOKENS, OVERSEAS_CONTEXT_TOKENS, OVERSEAS_SIGNAL_TOKENS, OVERSEAS_TRADE_TOKENS, OVERSEAS_UNIQUE_TOKENS,
    RESOURCE_COST_STRONG_TOKENS, SOCIAL_LIVE_TOKENS, SUB_DOMAIN_TRIGGERS, TRANSACTION_STRONG_TOKENS, find_hits, first_pos, has,
)

MAX_SUB_DOMAINS = 4
MAX_KEYWORDS = 12
KNOWLEDGE_DOMAIN_TOP = 5          # 用预注入知识兜底判 domain 时，最多看前几块


@dataclass
class DomainDecision:
    domain: str | None = None
    forced_subs: list[str] = field(default_factory=list)    # 规则强制收录的 sub_domain（如 Rule A → 闭环电商广告）
    rule_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _overseas_range_hit(text: str) -> bool:
    return any(find_hits(text, trig) for trig in SUB_DOMAIN_TRIGGERS["overseas"].values())


def infer_domain(text: str) -> DomainDecision:
    D = DomainDecision

    rc = find_hits(text, RESOURCE_COST_STRONG_TOKENS)
    if rc:
        return D("resource_cost", [], ["domain.rule_i.resource_cost"], [f"Rule I：命中 {rc[:3]}，domain=resource_cost"])

    # Rule K：海外 + 交易语义 → transaction（sub_domain 含 海外），不算国际化
    if has(text, "海外", "跨境", "出海") and has(text, *OVERSEAS_TRADE_TOKENS, "跨境"):
        return D("transaction", ["海外"], ["domain.rule_k.overseas_trade"], ["Rule K：海外信号但语义为交易/电商 → transaction，sub_domain 含 海外"])
    if find_hits(text, OVERSEAS_UNIQUE_TOKENS) or (
        (find_hits(text, OVERSEAS_SIGNAL_TOKENS) or (has(text, "海外") and has(text, *OVERSEAS_CONTEXT_TOKENS))) and _overseas_range_hit(text)
    ):
        return D("overseas", [], ["domain.rule_k.overseas"], ["Rule K：Rednote 信号 + 国际化子域范围命中"])

    inner = find_hits(text, INNER_LOOP_TOKENS)
    if inner:
        return D("commercialization", ["闭环电商广告"], ["domain.rule_a.inner_loop"], [f"Rule A：含 {inner}，DGMV 特殊规则例外，domain=commercialization"])

    if has(text, "互动直播", *SOCIAL_LIVE_TOKENS):
        return D("community", [], ["domain.rule_b_c.social_live"], ["Rule B/C：互动直播/打赏/ACU 等 → community"])
    if has(text, *ECOM_LIVE_TOKENS):
        if has(text, "连麦"):
            return D("community", [], ["domain.rule_d.protected_social_live"], ["Rule D 保护：电商直播词与连麦并存，按社区直播处理"])
        return D("transaction", [], ["domain.rule_d.ecom_live"], ["Rule D：电商直播词 → transaction"])
    if has(text, "直播") and has(text, *LIVE_FLOW_TOKENS):
        return D("transaction", [], ["domain.rule_e.live_flow"], ["Rule E：直播+流量/观播/停留且无社区强信号 → transaction"])

    comm = find_hits(text, COMMERCIALIZATION_STRONG_TOKENS)
    if comm or has(text, "商业化", "广告主", "广告消耗", "广告流水"):
        return D("commercialization", [], ["domain.rule_g_h.commercialization"], [f"Rule G/H：商业化强信号 {comm[:3]}"])

    if has(text, "电商"):
        return D("transaction", [], ["domain.rule_f.ecommerce"], ["Rule F：「电商」= 交易"])
    if has(text, *GMV_TOKENS):
        return D("transaction", [], ["domain.dgmv.transaction"], ["DGMV 特殊规则：含 GMV 系指标 → transaction"])

    trans, comty = find_hits(text, TRANSACTION_STRONG_TOKENS), find_hits(text, COMMUNITY_STRONG_TOKENS)
    if trans or comty:
        if len(trans) != len(comty):
            dom = "transaction" if len(trans) > len(comty) else "community"
        else:                                                  # 平票：先出现者优先
            dom = "transaction" if first_pos(text, trans) <= first_pos(text, comty) else "community"
        hits = trans if dom == "transaction" else comty
        return D(dom, [], [f"domain.signals.{dom}"], [f"强信号词计数 → {dom}（{hits[:3]}）"])

    if has(text, "流水", "客户数"):
        return D("commercialization", ["产品平台"], ["domain.rule_j.flow_or_customers"], ["Rule J：「流水/客户数」泛指且无交易/社区信号 → commercialization，sub_domain 含 产品平台"])
    return D()


def infer_sub_domains(text: str, domain: str, forced: list[str] | None = None) -> tuple[list[str], list[str]]:
    """(sub_domain, rule_ids)。触发词命中的全部收录（上限 MAX_SUB_DOMAINS）；无命中取 domain 默认值，保证非空。"""
    subs: list[str] = []
    ids: list[str] = []
    for s in forced or []:
        subs.append(s)
        ids.append(f"sub_domain.forced.{s}")
    scored: list[tuple[int, int, str]] = []
    for idx, (sub, trig) in enumerate(SUB_DOMAIN_TRIGGERS[domain].items()):
        hits = find_hits(text, (*trig, sub))
        if hits:
            scored.append((-len(hits), idx, sub))
    scored.sort()
    picked = [s for _, _, s in scored]
    if len(picked) > 1 and "common" in picked:
        picked.remove("common")                    # 泛概念兜底，有更具体的就不要
    for s in picked:
        if s not in subs:
            subs.append(s)
            ids.append(f"sub_domain.trigger.{s}")
    if not subs:
        default = "Token消耗" if domain == "resource_cost" and has(text, "token") else DEFAULT_SUB_DOMAIN[domain]
        return [default], [f"sub_domain.default.{domain}"]
    return subs[:MAX_SUB_DOMAINS], ids[:MAX_SUB_DOMAINS]


_TIME = re.compile(
    r"(?:最近|近|过去)\s*\d+\s*(?:天|日|周|个月|月|小时)|last\s*\d+\s*days?"
    r"|\d{4}\s*年(?:\s*\d{1,2}\s*月)?|\d{1,2}\s*月份?"
    r"|昨天|昨日|前天|今天|今日|上周|本周|上个月|上月|本月|去年|今年|最近|近期",
    re.I,
)
_ASSET_REF = re.compile(r"(?<!\w)(dataset|dashboard|report|chart|metric|table)/[^\s?]*\?[^\s]*?\bid=(\d+)", re.I)


def extract_time_hint(text: str) -> str | None:
    m = _TIME.search(text or "")
    return m.group(0) if m else None


def extract_asset_refs(text: str) -> list[AssetRef]:
    seen: set[tuple[str, str]] = set()
    out: list[AssetRef] = []
    for kind, aid in _ASSET_REF.findall(text or ""):
        key = (kind.lower(), aid)
        if key not in seen:
            seen.add(key)
            out.append(AssetRef(kind=key[0], id=aid))
    return out


def infer_intent_type(text: str, has_signal: bool) -> tuple[str, str | None]:
    if has(text, *DISCOVERY_TRIGGERS):
        return "discovery", None
    if has(text, *DEFINITION_TRIGGERS):
        return "definition", None
    if has(text, *ANALYSIS_TRIGGERS):
        if has(text, "预测"):
            return "analysis", "prediction"
        if has(text, "对比"):
            return "analysis", "compare"
        if has(text, "趋势", "走势") and not has(text, "为什么", "原因", "归因", "拆解", "拆下", "拆一下"):
            return "analysis", "trend"
        return "analysis", "attribution"
    if has_signal or has(text, *DATA_FETCH_TRIGGERS):
        return "data_fetch", None
    return "other", None


def infer_kb_intents(text: str) -> list[str]:
    hits = [k for k, trig in KB_INTENT_TRIGGERS.items() if has(text, *trig)]
    return hits or ["write_sql"]          # prompt：无合适时用 write_sql 兜底


_DOMAIN_TOKEN_SETS = (
    RESOURCE_COST_STRONG_TOKENS, COMMERCIALIZATION_STRONG_TOKENS, TRANSACTION_STRONG_TOKENS,
    COMMUNITY_STRONG_TOKENS, OVERSEAS_UNIQUE_TOKENS,
)
_KNOWLEDGE_KEYWORD_KINDS = (KnowledgeKind.METRIC_DEF, KnowledgeKind.DIMENSION, KnowledgeKind.BUSINESS_TERM)


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def _keywords(text: str, domain: str | None, subs: list[str], pre: PreInjectionResult | None, metrics: list[str], dims: list[str]) -> list[str]:
    kw = list(metrics)
    for toks in _DOMAIN_TOKEN_SETS:
        kw += find_hits(text, toks)
    if domain:
        for s in subs:
            kw += find_hits(text, SUB_DOMAIN_TRIGGERS[domain].get(s, ()))
    for it in (pre.injected if pre else []):
        if it.kind in _KNOWLEDGE_KEYWORD_KINDS and has(text, it.title):
            kw.append(it.title)
    kw += dims
    return _dedupe(kw)[:MAX_KEYWORDS]


def _knowledge_text(pre: PreInjectionResult | None) -> str:
    items = [i for i in (pre.injected if pre else []) if i.kind != KnowledgeKind.PAIR][:KNOWLEDGE_DOMAIN_TOP]
    return " ".join(f"{i.title} {i.content}" for i in items)


_TOKEN_SCOPE_WORDS = ("App", "问一问", "业务", "seal", "codewiz", "内部工具", "质效")


def rule_intent(query: str, pre: PreInjectionResult | None, *, session_id: str = "", text: str | None = None, fallback_reason: str = "") -> Intent:
    """规则版意图识别。`text` 为用于识别的文本（query + 追问回答），缺省即 query；Intent.query 始终是原问题。"""
    text = text if text is not None else query
    decision = infer_domain(text)
    warnings: list[str] = []
    rule_ids = list(decision.rule_ids)
    notes = list(decision.notes)

    if decision.domain is None and pre:
        kd = infer_domain(_knowledge_text(pre))
        if kd.domain:
            decision = DomainDecision(kd.domain, [], [], [])
            rule_ids.append("domain.from_knowledge")
            warnings.append(f"query 本身无业务域信号，domain=`{kd.domain}` 由预注入知识推断（{'; '.join(kd.rule_ids)}），请确认")

    domain = decision.domain
    subs: list[str] = []
    if domain:
        subs, sub_ids = infer_sub_domains(text, domain, decision.forced_subs)
        rule_ids += sub_ids
        if "Token消耗" in subs and not has(text, *_TOKEN_SCOPE_WORDS):
            notes.append("Rule I scope 歧义：token 消耗未指明业务端（App/问一问）还是内部工具端（seal/codewiz），建议向用户澄清")

    base = heuristic_extract(text, pre)
    time_hint = extract_time_hint(text)
    has_signal = bool(domain or base["metrics"] or time_hint)
    itype, subtype = infer_intent_type(text, has_signal)
    if itype == "other":
        notes.append("未识别到数据查询信号（无业务域 / 指标 / 时间）")

    draft = {"query": query, "domain": domain, "sub_domain": subs}
    warnings = validate_intent(draft, text) + warnings
    return Intent(
        session_id=session_id, query=query, intent_type=itype, intent_sub_type=subtype,
        asset_refs=extract_asset_refs(text), domain=domain, sub_domain=subs,
        kb_intents=infer_kb_intents(text) if domain == "commercialization" else [],
        keywords=_keywords(text, domain, subs, pre, base["metrics"], base["dimensions"]), dim_terms=[],
        time_hint=time_hint, notes="；".join(notes) or None,
        source="rule", rule_ids=rule_ids, fallback_reason=fallback_reason, warnings=warnings,
    )
