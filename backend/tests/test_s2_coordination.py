from app.schemas.task import ValueSource
from app.stages.s2_coordination import CoordinationStage
from app.stages.s2_coordination.extractor import heuristic_extract


def test_heuristic_extract():
    d = heuristic_extract("最近30天各渠道的DAU和GMV", None)
    assert d["metrics"] == ["DAU", "GMV"]
    assert d["time_range"] == "last_30_days"
    assert d["dimensions"] == ["渠道"]


async def test_defaults_filled_with_provenance(make_ctx):
    c = await CoordinationStage().run(make_ctx("DAU"))
    assert c.status == "ready"
    assert c.time_range.value == "last_7_days" and c.time_range.source == ValueSource.DEFAULT_RULE
    assert c.time_range.rule_id == "default.time_range.last_7_days"
    assert c.metrics.source == ValueSource.USER


async def test_missing_metric_falls_back_to_keywords_without_asking(make_ctx):
    """提取不到 metric → 不追问；用 intent.keywords 兜底，并标注来源可追溯。"""
    c = await CoordinationStage().run(make_ctx("千帆推广投放效果"))
    assert c.status == "ready"
    assert c.intent.keywords and c.metrics.value == c.intent.keywords
    assert c.metrics.source == ValueSource.INFERRED
    assert c.metrics.rule_id == "fallback.metrics.from_keywords"
    assert "keywords" in c.metrics.note


async def test_keyword_fallback_excludes_dimensions(make_ctx, monkeypatch):
    from app.schemas.intent import Intent
    from app.stages.s2_coordination import agent as agent_mod
    from app.stages.s2_coordination.agent import CoordinationOutput

    async def fake_run(self, query, pre, **kw):
        return CoordinationOutput(intent=Intent(query=query, keywords=["自然堂", "渠道", "活跃"], dim_terms=["自然堂"]),
                                  slots={"dimensions": ["渠道"]})
    monkeypatch.setattr(agent_mod.CoordinationAgent, "run", fake_run)
    c = await CoordinationStage().run(make_ctx("自然堂各渠道活跃"))
    assert c.metrics.value == ["活跃"] and c.metrics.source == ValueSource.INFERRED


async def test_no_metric_and_no_keywords_still_ready_and_never_asks(make_ctx):
    c = await CoordinationStage().run(make_ctx("帮我看看数据"))
    assert c.status == "ready" and not c.metrics.value      # 下游回落到整句 query
    assert not hasattr(c, "open_questions")


async def test_extracted_metric_is_not_overridden_by_keywords(make_ctx):
    c = await CoordinationStage().run(make_ctx("最近30天各渠道的DAU"))
    assert c.metrics.value == ["DAU"] and c.metrics.source == ValueSource.USER


async def test_supplementary_input_still_wins_over_fallback(make_ctx):
    c = await CoordinationStage().run(make_ctx("千帆推广投放效果", clarifications={"metrics": ["DAU"]}))
    assert c.metrics.value == ["DAU"] and c.metrics.source == ValueSource.USER
