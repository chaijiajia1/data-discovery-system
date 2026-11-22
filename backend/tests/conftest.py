from __future__ import annotations

import pytest

from app.deps import Deps
from app.services.fakes import FakeEngine
from app.settings import settings as _settings
from app.stages import RunContext


@pytest.fixture
def deps() -> Deps:
    return Deps()


@pytest.fixture
def make_ctx(deps):
    def _make(query: str, **kw) -> RunContext:
        return RunContext(query=query, deps=kw.pop("deps", deps), **kw)
    return _make


@pytest.fixture
def fake_engine():
    return FakeEngine


@pytest.fixture(autouse=True)
def _handoff_off_by_default(monkeypatch):
    """多数用例在数「某通道恰好被调用几次」，测的是基线协调行为；发现驱动的转交按设计会增加探查，由 test_s4_handoff.py 专门覆盖（那里会打开）。"""
    monkeypatch.setattr(_settings, "COLLAB_HANDOFF_ENABLED", False)


@pytest.fixture(autouse=True)
def _llm_level_off_by_default(monkeypatch):
    """多数用例用单个 FakeLLM 脚本化『规划提案』等其他 LLM 调用（并断言第一条 prompt）；入口定级的 LLM 判定由 test_s3_level_llm.py 专门打开。"""
    monkeypatch.setattr(_settings, "PLANNER_LLM_LEVEL_ENABLED", False)
