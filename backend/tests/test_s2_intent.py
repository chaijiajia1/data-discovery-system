"""② 协调 · intent：校验（移植自 old-version intent_validate）/ 规则兜底 / Coordination agent / 阶段与 HTTP。"""
import json

import pytest
from fastapi.testclient import TestClient

from app.deps import Deps
from app.errors import UpstreamError
from app.main import create_app
from app.schemas.intent import Intent
from app.schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin, PreInjectionResult
from app.stages.s2_coordination import CoordinationStage
from app.stages.s2_coordination.agent import CoordinationAgent, build_intent, parse_llm_json
from app.stages.s2_coordination.intent_rules import rule_intent
from app.stages.s2_coordination.intent_validate import (
    DOMAIN_ENUM, SUB_DOMAIN_ENUM, detect_domain_concept_conflict, drop_invalid_sub_domains,
    normalize_intent_lists, validate_intent,
)
from app.stages.s2_coordination.taxonomy import find_hits


class FakeLLM:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.prompts = reply, exc, []

    async def complete(self, *, prompt, system=None, json_mode=False):
        self.prompts.append(prompt)
        if self.exc:
            raise self.exc
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


def _pre(*items):
    return PreInjectionResult(injected=list(items))


def _k(title, content, kind=KnowledgeKind.METRIC_DEF, i=None):
    return KnowledgeItem(id=i or f"k_{title}", kind=kind, title=title, content=content, origin=KnowledgeOrigin.RETRIEVAL)


# ── 词表匹配 ──────────────────────────────────────────────────────────────
def test_find_hits_respects_ascii_boundary_and_case():
    assert find_hits("看看 GDP 走势", ["GD"]) == []
    assert find_hits("自然堂dgmv走势", ["DGMV", "dgmv", "GMV"]) == ["DGMV"]      # 大小写去重、DGMV 内的 GMV 不重复命中
    assert find_hits("本月h200缺口", ["H200"]) == ["H200"]
    assert find_hits("token消耗", ["token消耗"]) == ["token消耗"]


# ── intent_validate（对齐 old-version）────────────────────────────────────
def test_enum_has_five_domains():
    assert DOMAIN_ENUM == {"transaction", "community", "commercialization", "resource_cost", "overseas"}
    assert "闭环电商广告" in SUB_DOMAIN_ENUM["commercialization"] and "GPU资源" in SUB_DOMAIN_ENUM["resource_cost"]


def test_normalize_str_list_fields_to_single_element_list():
    d = normalize_intent_lists({"sub_domain": "交易驾驶舱", "keywords": " ", "dim_terms": ["a"], "asset_refs": None})
    assert d["sub_domain"] == ["交易驾驶舱"] and d["keywords"] == [] and d["dim_terms"] == ["a"] and d["asset_refs"] is None


def test_drop_invalid_sub_domains_keeps_valid_only():
    d = drop_invalid_sub_domains({"domain": "commercialization", "sub_domain": ["种草外溢", "种草广告"]})
    assert d["sub_domain"] == ["种草广告"]
    assert drop_invalid_sub_domains({"domain": "", "sub_domain": ["x"]})["sub_domain"] == ["x"]     # domain 为空不处理


def test_validate_invalid_domain_returns_early():
    w = validate_intent({"domain": "finance", "sub_domain": ["x"]})
    assert len(w) == 1 and "不在合法枚举" in w[0]


def test_validate_sub_domain_outside_enum_suggests_close_match():
    w = validate_intent({"query": "q", "domain": "commercialization", "sub_domain": ["种草"]})
    assert any("种草" in x and "种草广告" in x for x in w)


def test_validate_empty_sub_domain_with_domain():
    w = validate_intent({"query": "q", "domain": "transaction", "sub_domain": []})
    assert any("sub_domain 为空" in x for x in w)


@pytest.mark.parametrize("query,domain,needle", [
    ("千帆推广 DGMV 近7天", "transaction", "内循环"),
    ("互动直播 DAU", "transaction", "互动直播"),
    ("打赏 趋势", "transaction", "社区直播词"),
    ("店播 GMV", "community", "电商直播"),
    ("直播间 观播 停留", "community", "直播+流量"),
    ("电商 GMV", "commercialization", "电商"),
    ("广告消耗 趋势", "transaction", "商业化广告词"),
    ("H200 缺口", "community", "resource_cost"),
])
def test_detect_domain_conflicts(query, domain, needle):
    assert any(needle in w for w in detect_domain_concept_conflict({"query": query, "domain": domain}))


def test_detect_domain_conflict_silent_when_consistent():
    assert detect_domain_concept_conflict({"query": "店播 GMV", "domain": "transaction"}) == []
    assert detect_domain_concept_conflict({"query": "看看 GDP", "domain": "transaction"}) == []


# ── 规则兜底 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("query,domain,subs", [
    ("自然堂 近 7 天 dgmv", "transaction", None),
    ("千帆推广 DGMV 近 7 天趋势", "commercialization", ["闭环电商广告"]),
    ("互动直播 ACU 走势", "community", ["互动直播"]),
    ("店播 直播间 观播 停留", "transaction", None),
    ("本月 H200 缺口多少？哪些项目缺口最大", "resource_cost", ["GPU资源"]),
    ("小红书 App 问一问的日均 token 消耗是多少", "resource_cost", ["Token消耗"]),
    ("rednote 海外 DAU 和新设备次留最近的走势", "overseas", ["用户规模与留存"]),
    ("海外 GMV 上月多少", "transaction", ["海外"]),
    ("近7天流水", "commercialization", ["产品平台"]),
    ("广告主 近7天 消耗", "commercialization", None),
    ("最近30天各渠道的DAU和GMV", "transaction", None),
    ("DAU 近7天", "community", ["社区横向"]),
])
def test_rule_intent_domain(query, domain, subs):
    it = rule_intent(query, None)
    assert it.domain == domain and it.source == "rule"
    assert it.sub_domain, "domain 已识别时 sub_domain 不得为空"
    assert all(s in SUB_DOMAIN_ENUM[domain] for s in it.sub_domain)
    for s in subs or []:
        assert s in it.sub_domain
    assert not any("概念冲突" in w for w in it.warnings), it.warnings    # 规则结论不应自相矛盾


def test_rule_intent_rule_ids_are_traceable():
    it = rule_intent("千帆推广 DGMV", None)
    assert "domain.rule_a.inner_loop" in it.rule_ids
    assert any(r.startswith("sub_domain.") for r in it.rule_ids)


def test_rule_intent_no_signal():
    it = rule_intent("帮我看看数据", None)
    assert it.domain is None and it.sub_domain == [] and it.intent_type == "other"


def test_rule_intent_domain_from_injected_knowledge_when_query_has_no_signal():
    pre = _pre(_k("自然堂", "品牌维度，关联 DGMV、店铺 指标"))
    it = rule_intent("自然堂 近7天", pre)
    assert it.domain == "transaction" and "domain.from_knowledge" in it.rule_ids
    assert any("预注入知识" in w for w in it.warnings)


def test_rule_intent_query_signal_beats_knowledge():
    pre = _pre(_k("GMV", "DGMV 店铺 买手"))
    assert rule_intent("DAU 近7天", pre).domain == "community"


@pytest.mark.parametrize("query,t,sub", [
    ("DGMV 怎么算的？", "definition", None),
    ("为什么上周 K 播流水跌了？拆下渠道", "analysis", "attribution"),
    ("GMV 下周预测", "analysis", "prediction"),
    ("DAU 两周对比", "analysis", "compare"),
    ("有哪些看板能看 KOL 数据", "discovery", None),
    ("自然堂 近 7 天 dgmv", "data_fetch", None),
])
def test_rule_intent_type(query, t, sub):
    it = rule_intent(query, None)
    assert it.intent_type == t and it.intent_sub_type == sub


def test_rule_intent_time_hint_and_asset_refs():
    assert rule_intent("自然堂 近 7 天 dgmv", None).time_hint == "近 7 天"
    assert rule_intent("上月 DAU", None).time_hint == "上月"
    assert rule_intent("DAU", None).time_hint is None
    refs = rule_intent("这个数据集 https://x.com/dataset/list?id=2479 的 DAU", None).asset_refs
    assert [(r.kind, r.id) for r in refs] == [("dataset", "2479")]


def test_rule_intent_keywords_and_dim_terms():
    it = rule_intent("最近30天各渠道的DAU和GMV", _pre(_k("渠道", "获取渠道", KnowledgeKind.DIMENSION)))
    assert {"DAU", "GMV", "渠道"} <= set(it.keywords)
    assert it.dim_terms == []            # 无 LLM 时宁可不抽（漏抽只是错失富化）


def test_rule_intent_kb_intents_only_for_commercialization():
    assert rule_intent("聚光 消耗 近7天", None).kb_intents == ["data_fetch_bi"]
    assert rule_intent("聚光 是什么", None).kb_intents == ["definition"]
    assert rule_intent("商业化 广告 啊", None).kb_intents == ["write_sql"]            # 兜底
    assert rule_intent("店播 GMV", None).kb_intents == []


def test_rule_intent_token_scope_ambiguity_noted():
    assert "scope" in rule_intent("token 消耗 多少", None).notes
    assert not (rule_intent("问一问 token 消耗 多少", None).notes or "").count("scope")


def test_rule_intent_clarification_text_participates():
    it = rule_intent("帮我看看数据", None, text="帮我看看数据 GPU H200")
    assert it.domain == "resource_cost" and it.query == "帮我看看数据"


# ── LLM 输出归一化 / 校验 ──────────────────────────────────────────────────
def _norm(raw, query="q", text=None):
    return build_intent(raw, query=query, session_id="s1", source="llm", text=text or query)


def test_build_intent_normalizes_str_lists_and_drops_invalid_sub_domain():
    it = _norm({"domain": "commercialization", "sub_domain": ["种草外溢", "种草广告"], "keywords": "聚光", "intent_type": "data_fetch"})
    assert it.sub_domain == ["种草广告"] and it.keywords == ["聚光"]
    assert any("种草外溢" in w for w in it.warnings)             # 丢弃前的问题仍保留给下游


def test_build_intent_invalid_domain_becomes_none():
    it = _norm({"domain": "finance", "sub_domain": ["x"]})
    assert it.domain is None and it.sub_domain == [] and any("不在合法枚举" in w for w in it.warnings)


def test_build_intent_bad_intent_type_and_sub_type():
    assert _norm({"intent_type": "weird"}).intent_type == "other"
    assert _norm({"intent_type": "data_fetch", "intent_sub_type": "trend"}).intent_sub_type is None   # 仅 analysis 保留
    assert _norm({"intent_type": "analysis", "intent_sub_type": "trend"}).intent_sub_type == "trend"


def test_build_intent_fills_empty_sub_domain_by_rule_with_trace():
    it = _norm({"domain": "resource_cost", "sub_domain": []}, query="H200 缺口")
    assert it.sub_domain == ["GPU资源"] and any(r.startswith("sub_domain.") for r in it.rule_ids)
    assert it.source == "llm"


def test_build_intent_kb_intents_rules():
    assert _norm({"domain": "transaction", "kb_intents": ["definition"]}).kb_intents == []
    it = _norm({"domain": "commercialization", "sub_domain": ["common"], "kb_intents": ["bogus", "definition"]})
    assert it.kb_intents == ["definition"]
    assert _norm({"domain": "commercialization", "sub_domain": ["common"], "kb_intents": []}).kb_intents == ["write_sql"]


def test_build_intent_dim_terms_subset_of_keywords_and_blocked_for_two_domains():
    it = _norm({"domain": "transaction", "sub_domain": ["商品数据"], "keywords": ["dgmv"], "dim_terms": ["自然堂"]})
    assert it.dim_terms == ["自然堂"] and "自然堂" in it.keywords
    it = _norm({"domain": "resource_cost", "sub_domain": ["GPU资源"], "keywords": ["H200"], "dim_terms": ["H200"]})
    assert it.dim_terms == []


def test_build_intent_warns_when_domain_missing_but_signal_present():
    it = _norm({"domain": None}, query="H200 缺口")
    assert it.domain is None and any("强信号" in w for w in it.warnings)


def test_build_intent_asset_refs_coercion():
    it = _norm({"asset_refs": [{"kind": "dataset", "id": 2479}, "junk", {"kind": "x"}]})
    assert [(r.kind, r.id) for r in it.asset_refs] == [("dataset", "2479")]


def test_build_intent_overrides_passthrough_fields():
    it = _norm({"query": "LLM 改写", "session_id": "evil"}, query="原问题")
    assert it.query == "原问题" and it.session_id == "s1"


def test_parse_llm_json_variants():
    assert parse_llm_json('{"a": 1}') == {"a": 1}
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_llm_json('好的，结果如下：{"a": {"b": 2}} 以上') == {"a": {"b": 2}}
    for bad in ("", "no json", "[1, 2]"):
        with pytest.raises(ValueError):
            parse_llm_json(bad)


# ── Coordination agent：query + 预注入知识 → intent ─────────────────────────
LLM_OK = {
    "intent": {"intent_type": "data_fetch", "domain": "transaction", "sub_domain": ["商品数据"], "keywords": ["自然堂", "dgmv"],
               "dim_terms": ["自然堂"], "time_hint": "近 7 天", "notes": None},
    "slots": {"metrics": ["DGMV"], "caliber": None, "grain": None, "time_range": "last_7_days", "dimensions": [], "filters": []},
}


async def test_agent_injects_query_and_knowledge_into_prompt():
    llm = FakeLLM(LLM_OK)
    pre = _pre(_k("DGMV", "成交金额，含已支付订单"), _k("历史", "历史相似问题「自然堂 dgmv」最终选用资产 dws.x", KnowledgeKind.PAIR))
    out = await CoordinationAgent(llm).run("自然堂 近 7 天 dgmv", pre, session_id="s1")
    p = llm.prompts[0]
    assert "自然堂 近 7 天 dgmv" in p and "[metric_def] DGMV: 成交金额，含已支付订单" in p and "历史相似问题" in p
    assert "{{" not in p                                              # 占位符全部替换
    assert "闭环电商广告" in p and "GPU资源" in p                      # 词表由代码渲染进 prompt，与校验同源
    assert out.intent.source == "llm" and out.intent.domain == "transaction" and out.intent.session_id == "s1"
    assert out.slots["time_range"] == "last_7_days"


async def test_agent_includes_clarification_answers_in_prompt():
    llm = FakeLLM(LLM_OK)
    await CoordinationAgent(llm).run("帮我看看数据", None, clarifications={"metrics": ["ZZ_UNIQ_ANSWER"]})
    assert "- metrics: ZZ_UNIQ_ANSWER" in llm.prompts[0]


async def test_agent_query_cannot_inject_placeholders():
    llm = FakeLLM(LLM_OK)
    await CoordinationAgent(llm).run("看下 {{knowledge}} {{taxonomy}}", _pre(_k("DGMV", "SECRET-KNOWLEDGE")))
    p = llm.prompts[0]
    assert "看下 {{knowledge}} {{taxonomy}}" in p and p.count("SECRET-KNOWLEDGE") == 1


async def test_agent_accepts_fenced_json():
    out = await CoordinationAgent(FakeLLM("```json\n" + json.dumps(LLM_OK, ensure_ascii=False) + "\n```")).run("q", None)
    assert out.intent.source == "llm" and out.intent.domain == "transaction"


@pytest.mark.parametrize("llm", [
    FakeLLM(exc=UpstreamError("llm", "boom")),
    FakeLLM("not json at all"),
    FakeLLM('{"no_intent_key": 1}'),
    FakeLLM('[1,2,3]'),
])
async def test_agent_falls_back_to_rules(llm):
    out = await CoordinationAgent(llm).run("本月 H200 缺口多少", None)
    assert out.intent.source == "rule" and out.intent.domain == "resource_cost" and out.intent.fallback_reason
    assert isinstance(out.slots, dict)


async def test_agent_default_null_llm_runs_rules_only():
    out = await CoordinationAgent(Deps().llm).run("最近30天各渠道的DAU和GMV", None)
    assert out.intent.source == "rule" and "LLM" in out.intent.fallback_reason
    assert out.slots["metrics"] == ["DAU", "GMV"] and out.slots["time_range"] == "last_30_days"


async def test_agent_missing_slots_object_uses_heuristic_slots():
    out = await CoordinationAgent(FakeLLM({"intent": LLM_OK["intent"]})).run("最近30天各渠道的DAU", None)
    assert out.intent.source == "llm" and out.slots["metrics"] == ["DAU"]


# ── 阶段 / 管线 / HTTP ────────────────────────────────────────────────────
async def test_stage_attaches_intent_to_contract(make_ctx):
    c = await CoordinationStage().run(make_ctx("最近30天各渠道的DAU和GMV"))
    assert isinstance(c.intent, Intent) and c.intent.domain == "transaction" and c.intent.query == c.query
    assert c.status == "ready"


async def test_stage_with_llm_uses_llm_intent_and_slots(make_ctx):
    deps = Deps(llm=FakeLLM(LLM_OK))
    c = await CoordinationStage().run(make_ctx("自然堂 近 7 天 dgmv", deps=deps, session_id="sx"))
    assert c.intent.source == "llm" and c.intent.session_id == "sx" and c.intent.time_hint == "近 7 天"
    assert c.metrics.value == ["DGMV"]


async def test_stage_clarification_answer_feeds_intent(make_ctx):
    c = await CoordinationStage().run(make_ctx("帮我看看数据", clarifications={"metrics": ["GPU"]}))
    assert c.status == "ready" and c.intent.domain == "resource_cost"
    assert c.intent.query == "帮我看看数据"


async def test_stage_intent_when_nothing_recognized(make_ctx):
    c = await CoordinationStage().run(make_ctx("帮我看看数据"))
    assert c.status == "ready" and c.intent.domain is None and c.intent.intent_type == "other"


def test_http_stage2_returns_intent_after_stage1():
    client = TestClient(create_app(Deps()))
    r = client.post("/api/stage2", json={"query": "本月 H200 缺口多少", "session_id": "ui"})
    assert r.status_code == 200
    body = r.json()
    assert body["contract"]["intent"]["domain"] == "resource_cost" and body["contract"]["intent"]["session_id"] == "ui"
    assert "injected" in body["preinjection"]


def test_sse_stage2_event_carries_intent():
    client = TestClient(create_app(Deps()))
    ev = client.post("/api/discover/sync", json={"query": "最近7天DAU"}).json()["events"]
    s2 = next(e for e in ev if e["event"] == "stage2")
    assert s2["result"]["intent"]["domain"] in DOMAIN_ENUM



# ── dim_terms 是「具名维度取值」，不是维度名：LLM 把维度名（渠道）也填进去会触发无效的维值解析，并让 ⑤ 把所有候选判成不支持 ───────────────────
async def test_a_dimension_name_is_not_a_dim_term(make_ctx):
    reply = {"intent": {"domain": "community", "sub_domain": [], "intent_type": "data_fetch", "keywords": ["DAU", "渠道"], "dim_terms": ["渠道", "自然堂"],
                        "time_hint": "近 7 天"},
             "slots": {"metrics": ["DAU"], "dimensions": ["渠道"], "time_range": "last_7_days"}}
    c = await CoordinationStage().run(make_ctx("近7天各渠道的自然堂DAU", deps=Deps(llm=FakeLLM(reply))))
    assert c.intent.dim_terms == ["自然堂"] and c.dimensions.value == ["渠道"] and "渠道" in c.intent.keywords
