"""⑤ 资产选择 · 规则引擎（对齐 old-version `prompts/asset_selection.md` 的核心选用原则 1-8）。

按口径（指标）分组：每个口径从候选里选出恰好 1 个主资产，其余归 alternates / discarded；所有候选必须且只能落在三档之一。

    1  primary 恰好 1 个                     5  cube 表：非 cube 优先；选了 cube 必须限定切面
    2  chart 严格四条件优先                   6  粒度（主键含所需维度）与 scope（覆盖范围 ≥ query）双过滤
    3  字段覆盖度 > 资产类型 > 数据层级 > 热度  7  用户粒度宽表：先限定切片维度，聚合用求和
    4  OneDash 高权重（大盘 / 趋势类 query）   8  无权限 → 降级到 alternates（可申请），不直接丢弃

规则只读候选里**客观存在**的信息：缺失的元信息（无字段、无主键、无分区）不会被臆测，而是作为盲区（signals）如实标注。
没有字段元信息的候选（如只有 BI 检索结果）不会被当成「无目标指标字段」丢弃，而是按名称 / 描述 / 标签做「推断」命中（覆盖度只记 0.5）并标记风险。
LLM 复核（agent.py）复用这里的判定做硬护栏：LLM 不能选出规则判为不合规的资产。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ...schemas.intent import Intent
from ...schemas.knowledge import KnowledgeItem
from ...schemas.recall import CandidateAsset, Conflict, Evidence, EvidenceKind
from ...schemas.selection import (
    AssetDecision, FieldRef, GroupOutcome, Hit, Miss, RankScore, Reason, SelectionResult,
)
from ...asset_match import _cf, chart_dims_unlimited, is_chart, match_term, render_field
from ...schemas.task import TaskContract

GRAIN_RANK = {"minute": 0, "hour": 1, "day": 2, "week": 3, "month": 4, "quarter": 5, "year": 6}
CUBE_TABLE_RE = re.compile(r"_cube_1d_|_cube_nd_|_cube_|_summary_|_grouping_", re.I)
APP_PREFIX, DETAIL_PREFIX = ("app_",), ("dws_", "dm_", "dwd_", "ods_")
APP_WORDS, DETAIL_WORDS = ("宽表", "汇总", "聚合", "分析"), ("明细", "流水", "事实")
SUMMARY_WORDS = ("汇总", "聚合", "大盘", "分析")
SUMMARY_TABLE_RE = re.compile(r"_sum_|agg_|overview", re.I)
BROAD_WORDS = ("大盘", "核心", "概览", "整体", "趋势", "走势")
GLOBAL_SCOPE_WORDS = ("全站", "大盘", "整体", "全量")
NARROW_SCOPE_WORDS = ("直播", "搜索", "店播", "买手", "点点")          # 场域 / 渠道类窄 scope；按需扩充
USER_PK = ("uid", "user_id", "did", "device_id")
USER_METRIC_RE = re.compile(r"dau|uv|用户数|日活", re.I)
SLICE_FIELD_RE = re.compile(r"^(uv_pv_tag|type|.+_tag|is_.+_user)$", re.I)
DATE_FIELD_RE = re.compile(r"^(dt|dtm|ds|date|day|stat_date|biz_date)$", re.I)
COVERAGE_CLEARLY_LOWER = 0.25
HEAT_BASE = 1.5          # 热度按 1.5 倍一档分桶：同一档内视为「不显著」


@dataclass
class DimRef:
    """维值解析结果：具名维度取值（term）对应的字段。asset_id 为空 = 适用于任何有该字段的资产。"""

    term: str
    field: str
    asset_id: str | None = None


@dataclass
class SelectionInput:
    contract: TaskContract
    candidates: list[CandidateAsset]
    perms: dict[str, tuple[bool, str]] = field(default_factory=dict)    # asset_id → (granted, reason)；缺省 = 有权限
    knowledge: list[KnowledgeItem] = field(default_factory=list)
    dim_refs: list[DimRef] = field(default_factory=list)
    evidences: list[Evidence] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)

    @property
    def intent(self) -> Intent | None:
        return self.contract.intent


# ═══════════════════════════ 候选事实 ═══════════════════════════
def table_base(c: CandidateAsset) -> str:
    return (c.meta.main_table or c.asset_id.split(":")[-1]).split(".")[-1].lower()


def is_cube(c: CandidateAsset) -> bool:
    return bool(c.meta.cube_filters) or bool(CUBE_TABLE_RE.search(c.meta.main_table or ""))


def layer_of(c: CandidateAsset) -> str:
    """app（应用层）| detail（明细层）| unknown。"""
    base = table_base(c)
    if base.startswith(APP_PREFIX):
        return "app"
    if base.startswith(DETAIL_PREFIX):
        return "detail"
    name = f"{c.name} {c.meta.description or ''}"
    if any(w in name for w in DETAIL_WORDS):
        return "detail"
    if any(w in name for w in APP_WORDS):
        return "app"
    return "unknown"


def is_summary(c: CandidateAsset) -> bool:
    return any(w in f"{c.name} {c.meta.description or ''}" for w in SUMMARY_WORDS) or bool(SUMMARY_TABLE_RE.search(c.meta.main_table or ""))


def scope_of(c: CandidateAsset) -> set[str]:
    text = f"{c.name} {c.meta.description or ''}"
    if any(w in text for w in GLOBAL_SCOPE_WORDS):
        return set()
    return {w for w in NARROW_SCOPE_WORDS if w in text}


def query_scope(contract: TaskContract) -> set[str]:
    return {w for w in NARROW_SCOPE_WORDS if w in contract.query}


def is_broad_query(contract: TaskContract) -> bool:
    it = contract.intent
    return any(w in contract.query for w in BROAD_WORDS) or bool(it and it.intent_sub_type == "trend")


def has_fields(c: CandidateAsset) -> bool:
    return bool(c.meta.fields)


def signals_for(c: CandidateAsset) -> list[str]:
    """后端客观计算的盲区信号：LLM 与规则理由里的「盲区」只能引用这里真实存在的项。"""
    out: list[str] = []
    m = c.meta
    if m.extra.get("stub"):
        out.append("Knowledge-channel placeholder only; asset metadata missing")
    if not is_chart(c):
        if not m.fields:
            out.append("Field metadata missing (field coverage cannot be judged)")
        else:
            n = sum(1 for f in m.fields if m.field_aliases.get(f))
            if n / len(m.fields) < 0.6:
                out.append(f"Low Chinese-alias coverage: {n}/{len(m.fields)}")
            if not m.primary_keys:
                out.append("Primary-key info missing (grain cannot be judged)")
        if not m.partition_field:
            out.append("Partition field unknown (scan range cannot be limited)")
    return out


@dataclass
class Facts:
    c: CandidateAsset
    chart: bool
    cube: bool
    layer: str
    summary: bool
    scope: set[str]
    fields_known: bool
    denied: tuple[bool, str] | None = None        # (True, reason) = 无权限
    signals: list[str] = field(default_factory=list)


# ═══════════════════════════ 每个口径一组 ═══════════════════════════
@dataclass
class Eval:
    """一个候选在一个口径下的评估结果。"""

    f: Facts
    hits: list[Hit] = field(default_factory=list)
    misses: list[Miss] = field(default_factory=list)
    metric_field: str | None = None
    dim_fields: dict[str, str] = field(default_factory=dict)
    cov: float = 0.0
    kn_ids: list[str] = field(default_factory=list)
    verdict: str = "eligible"                       # eligible | alternate | discarded
    conclusion: str = ""
    rules: list[str] = field(default_factory=list)
    downweights: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    needs_permission: bool = False
    chart_ok: bool = False
    hard: str | None = None                         # 硬不合规的 rule_id（LLM 护栏用）

    @property
    def c(self) -> CandidateAsset:
        return self.f.c


@dataclass
class GroupResult:
    caliber: str
    primary: AssetDecision | None
    alternates: dict[str, AssetDecision] = field(default_factory=dict)
    discarded: dict[str, AssetDecision] = field(default_factory=dict)
    knowledge_refs: list[str] = field(default_factory=list)
    mode: str = "rule"
    fallback_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    evals: dict[str, Eval] = field(default_factory=dict)


# ═══════════════════════════ ④ 的证据：⑤ 读它，不重算 ═══════════════════════════
MAX_UPSTREAM_NOTES, MAX_CROSS_CHECKS = 6, 2


def _ev(inp: SelectionInput, asset_id: str, **match: Any) -> list[Evidence]:
    return [e for e in inp.evidences if e.asset_id == asset_id and all(getattr(e, k) == v for k, v in match.items())]


def freshness_unknown(inp: SelectionInput, asset_id: str) -> bool:
    """④ 核对过新鲜度、但没取到最新分区（④ 没核对过就不臆测）。"""
    fr = _ev(inp, asset_id, kind=EvidenceKind.FRESHNESS, source="partition_meta")
    return bool(fr) and not any(e.value for e in fr)


def upstream_signals(inp: SelectionInput, asset_id: str) -> list[str]:
    """④ 产出的客观盲区信号：和 signals_for 一样，理由里的「盲区」只能引用这里真实存在的项。"""
    out: list[str] = []
    if freshness_unknown(inp, asset_id):
        out.append("Latest partition unknown (Phase 4 check could not get it; freshness cannot be judged)")
    if any(e.value == [] for e in _ev(inp, asset_id, kind=EvidenceKind.OTHER, source="cross_check")):
        out.append("Shares no fields with other candidates (Phase 4 cross-check)")
    return out


def _grain_conflict_values(inp: SelectionInput, asset_id: str) -> list[str]:
    return [str(v) for c in inp.conflicts if c.asset_id == asset_id and c.topic == EvidenceKind.GRAIN for v in c.values]


def _grain_conflict_acceptable(inp: SelectionInput, c: CandidateAsset, want: str) -> bool:
    """④ 对该资产的粒度给出了互相冲突的结论（合并时只留了一个值），其中有满足要求粒度的：⑤ 不能据合并后的单值直接丢弃。"""
    return any(v in GRAIN_RANK and GRAIN_RANK[v] <= GRAIN_RANK[want] for v in map(str.lower, _grain_conflict_values(inp, c.asset_id)))


def upstream_irrelevant(inp: SelectionInput, asset_id: str, metric: str | None) -> str | None:
    """④ 的 LLM 对着这个口径判定「不相关」的理由（软信号：只降权，不作丢弃依据；硬护栏仍在 _judge）。"""
    if not metric:
        return None
    for e in _ev(inp, asset_id, source="llm_filter", value="irrelevant"):
        if _cf(e.subject) == _cf(metric):
            return e.statement.split(" — ")[-1].strip() or "no reason given"
    return None


def upstream_notes(inp: SelectionInput, asset_id: str, metric: str | None) -> list[dict[str, Any]]:
    """给选择 Agent 的 prompt：④ 已经做过的核对（粒度冲突 / LLM 相关性评估 / 新鲜度 / 口径 / 佐证）。
    不含 FIELD 证据：维度覆盖 ⑤ 用同一个匹配器自己算，不重复给。只带本口径的评估；总量封顶。"""
    out: list[dict[str, Any]] = []
    gv = _grain_conflict_values(inp, asset_id)
    if gv:
        out.append({"kind": "conflict", "source": "merger", "value": gv, "statement": f"Phase 4 gave conflicting grains {gv} for this asset; none was picked automatically"})
    for e in _ev(inp, asset_id, source="llm_filter"):
        if metric and _cf(e.subject) == _cf(metric):
            out.append({"kind": e.kind.value, "source": e.source, "value": e.value, "statement": e.statement})
    for e in _ev(inp, asset_id, kind=EvidenceKind.FRESHNESS, source="partition_meta"):
        out.append({"kind": e.kind.value, "source": e.source, "value": e.value, "statement": e.statement})
    for e in _ev(inp, asset_id, kind=EvidenceKind.CALIBER)[:2]:
        out.append({"kind": e.kind.value, "source": e.source, "value": None, "statement": e.statement[:120]})
    out += [{"kind": e.kind.value, "source": e.source, "value": e.value, "statement": e.statement} for e in _ev(inp, asset_id, source="cross_check")[:MAX_CROSS_CHECKS]]
    return out[:MAX_UPSTREAM_NOTES]


def soft_demote(inp: SelectionInput, evals: dict[str, "Eval"], metric: str | None) -> None:
    """④ 的软信号（粒度冲突 / LLM 判定不相关）：有不受影响的候选时让位到备选；全都受影响时不让主资产落空，只留下降权说明。"""
    elig = [e for e in evals.values() if e.verdict == "eligible"]
    soft: dict[str, list[tuple[str, str, str]]] = {}
    for e in elig:
        notes: list[tuple[str, str, str]] = []
        if "grain_conflict" in e.rule_ids:
            notes.append(("grain_conflict", f"Phase 4 gave conflicting grains {_grain_conflict_values(inp, e.c.asset_id)} for this asset; none was picked automatically", "grain: Phase 4 conflict left unresolved"))
        why = upstream_irrelevant(inp, e.c.asset_id, metric)
        if why:
            notes.append(("upstream_irrelevant", f"Phase 4 judged it irrelevant to '{metric}': {why}", f"Phase 4 relevance check: irrelevant ({why})"))
        if notes:
            soft[e.c.asset_id] = notes
    clean = [e for e in elig if e.c.asset_id not in soft]
    for e in elig:
        notes = soft.get(e.c.asset_id)
        if not notes:
            continue
        e.rule_ids += [rid for rid, _, _ in notes if rid not in e.rule_ids]
        e.downweights += [dw for _, _, dw in notes]
        if clean:
            e.verdict, e.conclusion = "alternate", "；".join(msg for _, msg, _ in notes)


def build_facts(inp: SelectionInput) -> dict[str, Facts]:
    out: dict[str, Facts] = {}
    for c in inp.candidates:
        p = inp.perms.get(c.asset_id)
        out[c.asset_id] = Facts(c=c, chart=is_chart(c), cube=is_cube(c), layer=layer_of(c), summary=is_summary(c), scope=scope_of(c),
                                fields_known=has_fields(c) or is_chart(c), denied=(True, p[1]) if p and not p[0] else None,
                                signals=signals_for(c) + upstream_signals(inp, c.asset_id))
    return out


def _is_date_or_metric(c: CandidateAsset, f: str, metrics: list[str]) -> bool:
    if DATE_FIELD_RE.match(f) or f == c.meta.partition_field or c.meta.field_aliases.get(f) in ("日期", "时间"):
        return True
    if c.meta.field_roles.get(f) == "metric":
        return True
    return any(_cf(m) in {_cf(f), _cf(c.meta.field_aliases.get(f))} for m in metrics)


def dim_term_support(inp: SelectionInput) -> dict[str, dict[str, str | None]]:
    """具名维度取值 → {asset_id: 支撑字段}；没有任何资产支撑 = 空 dict。"""
    metrics = [str(m) for m in (inp.contract.metrics.value or [])]
    out: dict[str, dict[str, str | None]] = {}
    for term in (inp.intent.dim_terms if inp.intent else []):
        refs = [r for r in inp.dim_refs if r.term == term]
        sup: dict[str, str | None] = {}
        for c in inp.candidates:
            if is_chart(c):
                continue
            if refs:
                for r in refs:
                    if r.asset_id not in (None, c.asset_id):
                        continue
                    f = next((f for f in c.meta.fields if _cf(r.field) in {_cf(f), _cf(c.meta.field_aliases.get(f))}), None)
                    if f:
                        sup[c.asset_id] = f
                        break
            elif not c.meta.fields:                       # 没有字段元信息：无法判断它能不能按该取值过滤——未知不是「不支持」（盲区由 fields_unknown 等信号如实标注）
                sup[c.asset_id] = None
            elif any(not _is_date_or_metric(c, f, metrics) for f in c.meta.fields):     # 未解析的取值：至少要有一个可过滤的维度字段
                sup[c.asset_id] = None
        out[term] = sup
    return out


def _heat(c: CandidateAsset) -> int:
    h = c.meta.heat_30d if c.meta.heat_30d is not None else (c.meta.ref_count or 0)
    return int(math.log(max(h, 0) + 1, HEAT_BASE))


def _type_rank(c: CandidateAsset) -> int:
    return 0 if c.asset_type in ("dataset", "chart", "dashboard", "metric") else 1


def _layer_rank(c: CandidateAsset) -> int:
    return {"app": 0, "unknown": 1, "detail": 2}[layer_of(c)]


def rank_dims(e: Eval) -> dict[str, float]:
    return {"coverage": round(e.cov, 4), "type_rank": 1.0 if _type_rank(e.c) == 0 else 0.5, "layer_rank": {0: 1.0, 1: 0.5, 2: 0.0}[_layer_rank(e.c)],
            "heat": round(min(1.0, _heat(e.c) / 12), 4), "retrieval": round(e.c.retrieval_score, 4)}


SCORE_WEIGHTS = {"coverage": 0.5, "type_rank": 0.1, "layer_rank": 0.1, "heat": 0.2, "retrieval": 0.1}   # 综合分各项权重（Σ=1）


def _total(d: dict[str, float]) -> float:
    return round(sum(w * d[k] for k, w in SCORE_WEIGHTS.items()), 4)


def evaluate_group(inp: SelectionInput, facts: dict[str, Facts], metric: str | None, support: dict[str, dict[str, str | None]]) -> dict[str, Eval]:
    """一个口径下，逐候选：命中 / 覆盖度 / 硬过滤（丢弃）/ 门槛（降级到备选）。"""
    ct = inp.contract
    dims = [str(d) for d in (ct.dimensions.value or [])]
    want = ct.grain.value
    evals: dict[str, Eval] = {}
    for aid, f in facts.items():
        c, e = f.c, Eval(f=f)
        terms = ([(metric, "metric")] if metric else []) + [(d, "dimension") for d in dims]
        for term, kind in terms:
            hit, fld, kn = match_term(term, kind, c, inp.knowledge)
            if hit is None:
                e.misses.append(Miss(term=term, type=kind, note="not found in " + ("metric_ids" if kind == "metric" else "supported_dims") if f.chart
                                     else ("field metadata missing; not found in name / description / tags either" if not c.meta.fields else "not found in fields")))  # type: ignore[arg-type]
                continue
            e.hits.append(hit)
            e.kn_ids += [k for k in kn if k not in e.kn_ids]
            if fld:
                if kind == "metric":
                    e.metric_field = fld
                else:
                    e.dim_fields[term] = fld
        if chart_dims_unlimited(c):        # 图表 supported_dims == [] 表示无限制：视为覆盖全部所需维度
            for d in dims:
                if not any(h.term == d for h in e.hits):
                    e.hits.append(Hit(term=d, type="dimension", matched="[]", location="supported_dims", how="推断"))
                    e.misses = [m for m in e.misses if m.term != d]
        # 覆盖度：字段 / 知识桥接命中记 1，仅凭名称 / 描述「推断」命中只记 0.5（不如有真实字段的资产可信）
        weight = {h.term: (0.5 if h.how == "推断" and h.location in ("name", "description") or h.location.startswith("tags") else 1.0) for h in e.hits}
        e.cov = round(sum(weight.values()) / len(terms), 4) if terms else 1.0
        _judge(inp, e, metric, dims, want, support)
        evals[aid] = e
    relax_if_empty(evals, metric)
    return evals


def relax_if_empty(evals: dict[str, Eval], metric: str | None) -> None:
    """没有任何满足全部门槛的候选时，放宽门槛，别让主资产落空（原则 1：候选里有与口径相关的，primary 不允许为空）。
    可放宽的 = 只是没过「软」门槛（chart 四条件 / 某个维值没字段）的备选：不是硬不合规、没有权限问题、且确实命中了口径。
    放宽后照常按选用原则挑最好的，并显式标注（`fallback_candidate` / 风险标记 `fallback_primary`），不装作它满足了全部条件。"""
    if any(e.verdict == "eligible" for e in evals.values()):
        return
    for e in evals.values():
        if e.verdict != "alternate" or e.hard is not None or e.needs_permission:
            continue
        if metric and not any(h.type == "metric" for h in e.hits):
            continue
        e.downweights.append(f"relaxed gate: no candidate met every condition, used as fallback — {e.conclusion}")
        e.verdict, e.conclusion = "eligible", ""
        e.rule_ids.append("fallback_candidate")


def _judge(inp: SelectionInput, e: Eval, metric: str | None, dims: list[str], want: Any, support: dict[str, dict[str, str | None]]) -> None:
    c, f = e.c, e.f
    mhit = any(h.type == "metric" for h in e.hits)

    def disc(rule: str, msg: str, hard: bool = True) -> None:
        e.verdict, e.conclusion = "discarded", msg
        e.rule_ids.append(rule)
        e.hard = rule if hard else None

    def alt(rule: str, msg: str, *, down: str | None = None, hard: str | None = None) -> None:
        if e.verdict == "eligible":
            e.verdict, e.conclusion = "alternate", msg
            e.hard = hard
        e.rule_ids.append(rule)
        if down:
            e.downweights.append(down)

    # ── 丢弃：硬不合规 ──
    have = c.meta.grain
    if not f.chart and want in GRAIN_RANK and have in GRAIN_RANK and GRAIN_RANK[have] > GRAIN_RANK[want]:
        if not _grain_conflict_acceptable(inp, c, want):
            return disc("grain_hard", f"Asset grain {have} is coarser than the requested grain {want}; cannot drill down")
        e.rule_ids.append("grain_conflict")                  # ④ 合并时静默择一了；冲突显式保留，交给 soft_demote 处理而不是直接丢弃
    if not f.chart and f.fields_known and has_fields(c) and metric and not mhit:
        e.rule_ids.append("no_metric_field")
        return disc("no_metric_field", f"No target metric field ({metric})")
    if not f.chart and c.meta.primary_keys:
        for d in dims:                       # 命中该维度的字段必须是主键之一（含知识桥接后的同义命中）
            if e.dim_fields.get(d) not in c.meta.primary_keys:
                return disc("pk_granularity", f"{c.asset_id} has grain {'/'.join(c.meta.primary_keys)}; cannot analyse by {d}")
    qs = query_scope(inp.contract)
    if not f.chart and f.scope and not f.scope <= qs:
        return disc("scope_insufficient", f"{c.asset_id} scope is {'/'.join(sorted(f.scope))}; cannot cover the site-wide / overall scope the query needs")

    # ── 门槛：降级到备选 ──
    if f.chart:
        fails = []
        if c.meta.is_strict is not True:
            fails.append("is_strict is not true")
        if f.denied:
            fails.append(f"perm.granted=false（{f.denied[1]}）")
        if metric and not any(h.type == "metric" for h in e.hits):
            fails.append("metric_ids do not cover the requested metric")
        if any(m.type == "dimension" for m in e.misses):
            fails.append("supported_dims do not cover the requested dimension")
        if fails:
            e.needs_permission = bool(f.denied)
            return alt("chart_conditions", "Chart conditions not met: " + "; ".join(fails), down=f"chart strict conditions: {'; '.join(fails)}")
        e.chart_ok = True
        return
    if f.denied:
        e.needs_permission = True
        e.rules.append("permission demotion")
        return alt("permission", f"No permission ({f.denied[1]}); demoted to alternates, can be requested", hard="permission")
    for term, sup in support.items():
        if c.asset_id not in sup:
            if sup:
                return alt("dim_term_unsupported", f"{c.asset_id} has no field for {term}; cannot filter by it", down=f"dimension-value filter: no field for {term}")
            return alt("dim_term_unsupported", f"No candidate supports filtering by {term}", hard="dim_term_unsupported")
    if not has_fields(c) and not mhit:                   # 无字段元信息且名称 / 描述也没有推断命中：无法判断，待补全
        return alt("fields_unknown", "Asset metadata missing (no fields); field coverage cannot be judged, use after it is completed")


def _key(e: Eval, onedash_on: bool, trend: bool) -> tuple:
    c = e.c
    return (0 if (onedash_on and c.meta.is_onedash) else 1, -round(e.cov, 2), 0 if (not trend or e.f.summary) else 1, _type_rank(c),
            _layer_rank(c), -_heat(c), -round(c.retrieval_score, 4), c.asset_id)


def _loses_to(w: Eval, e: Eval, onedash_on: bool, trend: bool) -> str:
    if onedash_on and w.c.meta.is_onedash and not e.c.meta.is_onedash:
        return "OneDash high weight"
    if round(w.cov, 2) != round(e.cov, 2):
        return f"Field coverage {e.cov:.2f} is lower than the primary {w.cov:.2f}"
    if trend and w.f.summary and not e.f.summary:
        return "Trend analysis prefers summary grain"
    if _type_rank(w.c) != _type_rank(e.c):
        return "Asset type hive_table ranks below dataset"
    if _layer_rank(w.c) != _layer_rank(e.c):
        return "Data layer: application layer ranks above detail layer"
    if _heat(w.c) != _heat(e.c):
        return "Popularity is clearly lower than the primary"
    return "Retrieval score is lower than the primary"


def _pick(evals: dict[str, Eval], contract: TaskContract) -> tuple[Eval | None, bool, bool]:
    """从 eligible 里选 primary：chart 严格四条件 > OneDash（大盘 / 趋势类）> 覆盖度 > 趋势汇总 > 类型 > 层级 > 热度 > 召回分。"""
    elig = [e for e in evals.values() if e.verdict == "eligible"]
    if not elig:
        return None, False, False
    charts = [e for e in elig if e.chart_ok]
    if charts:
        return min(charts, key=lambda e: (-e.c.retrieval_score, e.c.asset_id)), False, False
    # cube：存在非 cube 的可用资产 → cube 让位
    if any(not e.f.cube for e in elig):
        for e in [e for e in elig if e.f.cube]:
            e.verdict, e.conclusion = "alternate", "Cube table yields to a non-cube version (non-cube is preferred when it covers the metric)"
            e.rule_ids.append("cube_yield")
            e.downweights.append("cube: a usable non-cube asset exists")
            if e.c.meta.is_onedash and is_broad_query(contract):
                e.downweights.append("OneDash high weight: cube table suppressed")
        elig = [e for e in elig if e.verdict == "eligible"]
    # scope：query 限定在窄 scope 时，scope 匹配的表优先，过宽的全站宽表降级
    qs = query_scope(contract)
    if qs and any(e.f.scope for e in elig):
        for e in [e for e in elig if not e.f.scope]:
            e.verdict, e.conclusion = "alternate", f"Scope too wide: the query is limited to {'/'.join(sorted(qs))} and a scope-matching candidate exists"
            e.rule_ids.append("scope_wide")
            e.downweights.append("scope: site-wide table yields to a scope-matching candidate")
        elig = [e for e in elig if e.verdict == "eligible"]
    broad, trend = is_broad_query(contract), bool(contract.intent and contract.intent.intent_sub_type == "trend")
    top_cov = max(e.cov for e in elig)
    od = False
    if broad:
        for e in elig:
            if e.c.meta.is_onedash and e.f.cube:
                e.downweights.append("OneDash high weight: cube table suppressed")
        od_ok = [e for e in elig if e.c.meta.is_onedash and not e.f.cube and e.cov >= top_cov - COVERAGE_CLEARLY_LOWER]
        for e in elig:
            if e.c.meta.is_onedash and not e.f.cube and e.cov < top_cov - COVERAGE_CLEARLY_LOWER:
                e.downweights.append("OneDash high weight: field coverage is clearly lower than other candidates")
        od = bool(od_ok)
        if od:
            elig = od_ok + [e for e in elig if e not in od_ok]
    return min(elig, key=lambda e: _key(e, od, trend)), od, trend


def select_group(inp: SelectionInput, facts: dict[str, Facts], metric: str | None, support: dict[str, dict[str, str | None]]) -> GroupResult:
    evals = evaluate_group(inp, facts, metric, support)
    soft_demote(inp, evals, metric)
    win, od, trend = _pick(evals, inp.contract)
    label = metric or inp.contract.query
    g = GroupResult(caliber=label, primary=None, evals=evals, keywords=[label, *[str(d) for d in (inp.contract.dimensions.value or [])]])
    elig = [e for e in evals.values() if e.verdict == "eligible"]
    if win is not None:
        g.primary = _primary_decision(inp, win, evals, od, trend, label)      # 先算赢家的理由（依赖「其余可用候选」），再把它们降为备选
        g.knowledge_refs = list(win.kn_ids)
        for e in elig:
            if e is not win:
                if win.chart_ok:
                    e.conclusion = "All four strict chart conditions met; preferred over dataset / hive_table"
                    e.rule_ids.append("chart_first")
                else:
                    e.conclusion = _loses_to(win, e, od, trend)
                e.verdict = "alternate"
    # 其余：按口径视角分到备选 / 丢弃
    for aid, e in evals.items():
        if g.primary is not None and aid == g.primary.asset_id:
            continue
        if e.verdict == "eligible":
            e.verdict = "alternate"
        d = _decision(e, "alternate" if e.verdict == "alternate" else "discarded", label)
        (g.alternates if e.verdict == "alternate" else g.discarded)[aid] = d
    return g


def _reason(e: Eval, conclusion: str, rules: list[str] | None = None) -> Reason:
    return Reason(conclusion=conclusion, hits=list(e.hits), misses=list(e.misses), rules=rules if rules is not None else list(e.rules),
                  downweights=list(dict.fromkeys(e.downweights)), blind_spots=list(e.f.signals))


def _decision(e: Eval, role: str, label: str) -> AssetDecision:
    return AssetDecision(asset_id=e.c.asset_id, role=role, reason=_reason(e, e.conclusion or "Unrelated to the metric"), covers=[label] if role == "alternate" else [],  # type: ignore[arg-type]
                         needs_permission=e.needs_permission, rule_ids=list(dict.fromkeys(e.rule_ids)),
                         score=_score(e))


def _score(e: Eval) -> RankScore:
    d = rank_dims(e)
    return RankScore(asset_id=e.c.asset_id, dims=d, weights=dict(SCORE_WEIGHTS), total=_total(d))


def _primary_decision(inp: SelectionInput, w: Eval, evals: dict[str, Eval], od: bool, trend: bool, label: str) -> AssetDecision:
    c, others = w.c, [e for e in evals.values() if e is not w and e.verdict == "eligible"]
    close = [e for e in others if round(e.cov, 2) == round(w.cov, 2)]
    rules: list[str] = []
    if w.chart_ok:
        rules.append("chart strict conditions")
    if od:
        rules.append("OneDash high weight")
    if not w.chart_ok:
        if any(round(e.cov, 2) < round(w.cov, 2) for e in others):
            rules.append("highest field coverage")
        if trend and w.f.summary and any(not e.f.summary for e in close):
            rules.append("trend analysis prefers summary grain")
        if _type_rank(c) == 0 and any(_type_rank(e.c) == 1 for e in close):
            rules.append("asset type dataset > hive_table")
        same = [e for e in close if _type_rank(e.c) == _type_rank(c)]
        if _layer_rank(c) == 0 and any(_layer_rank(e.c) > 0 for e in same):
            rules.append("application layer first")
        if any(_heat(c) > _heat(e.c) for e in same if _layer_rank(e.c) == _layer_rank(c)):
            rules.append("popularity leads")
    if w.f.cube:
        rules.append("cube requires slice filter")
    # 降级到明细层要说明
    note = ""
    if layer_of(c) == "detail":
        lacking = [e for e in evals.values() if e is not w and layer_of(e.c) == "app" and e.c.asset_type == c.asset_type
                   and (any(h.type == "metric" for h in e.hits) or e.hard == "pk_granularity")]
        if lacking:
            dim = next((d for d in (inp.contract.dimensions.value or [])), "the requested dimension")
            note = f"Application-layer {lacking[0].c.asset_id} lacks {dim}; using detail-layer {c.asset_id}"
    fields = build_query_fields(inp, w, label)
    if w.f.cube:
        pins = ", ".join(w.c.meta.cube_filters) or "slice"
        note = (note + "; " if note else "") + f"Cube table; the {pins} slice must be pinned"
    user_grain = any(f.field_name and "sum" in f.note for f in fields)
    if user_grain:
        rules.append("user-grain wide table: sum")
    parts = [f"Covers {label} (field coverage {w.cov:.2f})"]
    if note:
        parts.append(note)
    if rules:
        parts.append("Basis: " + ", ".join(rules[:3]))
    if w.kn_ids:
        parts.append("Caliber reference " + ", ".join(w.kn_ids))
    if w.f.denied is None and w.hits and any(h.how in ("同义", "推断") for h in w.hits):
        parts.append("includes knowledge-bridged / inferred hits")
    return AssetDecision(asset_id=c.asset_id, role="primary", reason=_reason(w, "；".join(parts), rules), covers=[label], query_fields=fields,
                         rule_ids=list(dict.fromkeys(w.rule_ids)), score=_score(w))


def build_query_fields(inp: SelectionInput, w: Eval, label: str) -> list[FieldRef]:
    c, out = w.c, []
    seen: set[str] = set()

    def add(ref: FieldRef) -> None:
        if ref.field_name not in seen:
            seen.add(ref.field_name)
            out.append(ref)

    def alias(f: str) -> str:
        return c.meta.field_aliases.get(f, "")

    if w.f.chart:
        for h in w.hits:
            add(FieldRef(field_name=h.matched, field_alias=h.term, role="metric" if h.type == "metric" else "dimension"))
        return out
    if w.metric_field:
        add(FieldRef(field_name=w.metric_field, field_alias=alias(w.metric_field), role="metric"))
    for d, f in w.dim_fields.items():
        add(FieldRef(field_name=f, field_alias=alias(f), role="dimension"))
    for term, sup in dim_term_support(inp).items():       # 具名维度取值：来自维值解析，逐字取其值
        f = sup.get(c.asset_id)
        if f:
            add(FieldRef(field_name=f, field_alias=alias(f), role="dimension", dim_values=[term]))
            if not any(h.term == term for h in w.hits):
                w.hits.append(Hit(term=term, type="dimension", matched=render_field(c, f), location=f"fields[{c.meta.fields.index(f)}]", how="字面相同"))
    pf = c.meta.partition_field
    if pf and pf in c.meta.fields:
        tr = inp.contract.time_range.value
        add(FieldRef(field_name=pf, field_alias=alias(pf), role="dimension", note="Partition field; an explicit time range is required" + (f" ({tr})" if tr else "")))
    for k, v in c.meta.cube_filters.items():          # cube：切面字段必须限定
        add(FieldRef(field_name=k, field_alias=alias(k), role="dimension", dim_values=[v]))
    if w.f.cube and not c.meta.cube_filters:
        w.rule_ids.append("cube_requires_filter")
    if set(USER_PK) & {p.lower() for p in c.meta.primary_keys} and w.metric_field and USER_METRIC_RE.search(w.metric_field + alias(w.metric_field)):
        for f in c.meta.fields:
            if SLICE_FIELD_RE.match(f):
                add(FieldRef(field_name=f, field_alias=alias(f), role="dimension", dim_values=[],
                             note="User-grain wide table: pin the slice values first, aggregate with sum, never distinct count"))
        w.rule_ids.append("user_grain")
    return out


# ═══════════════════════════ 汇总 ═══════════════════════════
def assemble(inp: SelectionInput, groups: list[GroupResult], *, mode: str = "rule", warnings: list[str] | None = None) -> SelectionResult:
    """把各口径的结果合并成三档：同一资产是多个口径的主资产时合并为一条；备选 / 丢弃取其最相关的口径视角。"""
    cmap = {c.asset_id: c for c in inp.candidates}
    prim: dict[str, AssetDecision] = {}
    for g in groups:
        if g.primary is None:
            continue
        p = g.primary
        cur = prim.get(p.asset_id)
        if cur is None:
            prim[p.asset_id] = p.model_copy(deep=True)
            continue
        cur.covers = list(dict.fromkeys([*cur.covers, *p.covers]))
        cur.reason.hits += [h for h in p.reason.hits if (h.term, h.type) not in {(x.term, x.type) for x in cur.reason.hits}]
        cur.reason.rules = list(dict.fromkeys([*cur.reason.rules, *p.reason.rules]))
        cur.reason.conclusion += "；" + p.reason.conclusion
        names = {f.field_name for f in cur.query_fields}
        cur.query_fields += [f for f in p.query_fields if f.field_name not in names]
        cur.rule_ids = list(dict.fromkeys([*cur.rule_ids, *p.rule_ids]))
    alts: dict[str, AssetDecision] = {}
    disc: dict[str, AssetDecision] = {}
    for aid in cmap:
        if aid in prim:
            continue
        a = [g.alternates[aid] for g in groups if aid in g.alternates]
        if a:
            d = a[0].model_copy(deep=True)
            d.covers = list(dict.fromkeys(x for y in a for x in y.covers))
            alts[aid] = d
        else:
            ds = [g.discarded[aid] for g in groups if aid in g.discarded]
            if ds:
                disc[aid] = ds[0]
            else:                                           # 没有口径可评估（理论上不会发生）
                disc[aid] = AssetDecision(asset_id=aid, role="discarded", reason=Reason(conclusion="Unrelated to any metric"))
    order = {aid: i for i, aid in enumerate(cmap)}
    primary = sorted(prim.values(), key=lambda d: next(i for i, g in enumerate(groups) if g.primary and g.primary.asset_id == d.asset_id))
    alternates = sorted(alts.values(), key=lambda d: order[d.asset_id])
    discarded = sorted(disc.values(), key=lambda d: order[d.asset_id])
    refs = list(dict.fromkeys(k for g in groups for k in g.knowledge_refs if g.primary is not None))
    ranking = _ranking(groups, primary, alternates, discarded)
    res = SelectionResult(
        primary=primary, alternates=alternates, discarded=discarded, knowledge_refs=refs, ranking=ranking, warnings=list(warnings or []),
        groups=[GroupOutcome(caliber=g.caliber, primary=g.primary.asset_id if g.primary else None, mode=g.mode, fallback_reason=g.fallback_reason,   # type: ignore[arg-type]
                             keywords=g.keywords, warnings=g.warnings) for g in groups],
        mode=mode,  # type: ignore[arg-type]
    )
    res.risk_flags = _risk_flags(inp, res, groups)
    for d in res.primary:                       # primary 的理由里引用被参考的知识（命中知识桥接的条目）
        pass
    return res


def _ranking(groups: list[GroupResult], primary, alternates, discarded) -> list[RankScore]:
    best: dict[str, RankScore] = {}
    for g in groups:
        for aid, e in g.evals.items():
            s = _score(e)
            if aid not in best or s.total > best[aid].total:
                best[aid] = s
    tier = {d.asset_id: i for i, ds in enumerate((primary, alternates, discarded)) for d in ds}
    out = sorted(best.values(), key=lambda s: (tier.get(s.asset_id, 9), -s.total, s.asset_id))
    for d in (*primary, *alternates, *discarded):
        d.score = best.get(d.asset_id, d.score)
    return out


def _risk_flags(inp: SelectionInput, res: SelectionResult, groups: list[GroupResult]) -> list[str]:
    flags: list[str] = []
    if not res.primary:
        flags.append("no_asset_selected")
    missing = [g.caliber for g in groups if g.primary is None]
    if missing:
        flags.append(f"metrics_not_covered:{sorted(missing)}")
    for term, sup in dim_term_support(inp).items():
        if not sup:
            flags.append(f"dim_term_unsupported:{term}")
        elif not any(r.term == term for r in inp.dim_refs):
            flags.append(f"dim_value_unresolved:{term}")
    prim = {d.asset_id for d in res.primary}
    caliber_assets = {e.asset_id for e in inp.evidences if e.kind == EvidenceKind.CALIBER}
    cmap = {c.asset_id: c for c in inp.candidates}
    for d in res.primary:
        c = cmap[d.asset_id]
        if d.asset_id not in caliber_assets:
            flags.append(f"no_caliber_evidence:{d.asset_id}")
        if not is_chart(c) and not c.meta.partition_field:
            flags.append(f"partition_unknown:{d.asset_id}")
        if freshness_unknown(inp, d.asset_id):
            flags.append(f"freshness_unknown:{d.asset_id}")
        if "fallback_candidate" in d.rule_ids:
            flags.append(f"fallback_primary:{d.asset_id}")
        if not is_chart(c) and not c.meta.fields:
            flags.append(f"fields_unknown:{d.asset_id}")
        if "cube_requires_filter" in d.rule_ids:
            flags.append(f"cube_requires_filter:{d.asset_id}")
        if "user_grain" in d.rule_ids:
            flags.append(f"slice_values_unknown:{d.asset_id}" if any(f.dim_values == [] for f in d.query_fields) else f"slice_field_missing:{d.asset_id}")
    flags += [f"conflict:{c.asset_id}:{c.topic.value}" for c in inp.conflicts if c.asset_id in prim]
    flags += [f"needs_permission:{d.asset_id}" for d in res.alternates if d.needs_permission]
    if inp.intent and any("概念冲突" in w for w in inp.intent.warnings):
        flags.append("intent_conflict")
        res.warnings += [w for w in inp.intent.warnings if "概念冲突" in w and w not in res.warnings]
    return list(dict.fromkeys(flags))


def metric_groups(contract: TaskContract) -> list[str | None]:
    ms = [str(m) for m in (contract.metrics.value or [])]
    return ms or [None]


def rule_select(inp: SelectionInput) -> SelectionResult:
    facts, support = build_facts(inp), dim_term_support(inp)
    groups = [select_group(inp, facts, m, support) for m in metric_groups(inp.contract)]
    if not inp.contract.metrics.value:
        groups[0].caliber = inp.contract.query
    return assemble(inp, groups)
