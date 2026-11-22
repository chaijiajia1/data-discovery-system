"""规划 Agent：LLM 只负责「额外提案」，其余全部确定性。

提案经与规则候选完全相同的可行性过滤与评分；LLM 未配置 / 调用失败 / 输出不可解析 / 提案非法 → 忽略，回落到规则候选（铁律 3）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from ...errors import UpstreamError
from ...llm_client import LLMClient
from ...prompts import load_prompt
from ...roles import RoleSpec
from ...schemas.workflow import CandidateWorkflow, EdgeKind, PlanContext, WorkflowEdge, WorkflowGraph
from ..s2_coordination.agent import parse_llm_json
from .candidates import make_node, node_id, repair_width

logger = logging.getLogger(__name__)
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def _context_json(pc: PlanContext) -> str:
    return json.dumps({
        "calibers": pc.metrics, "channels": pc.channels, "dimensions": pc.dimensions, "domain": pc.domain, "max_parallel": pc.max_parallel,
        "weights": {"coverage": pc.w_coverage, "parallel": pc.w_parallel, "cost": pc.w_cost},
        "pending_pairs": [u.key for u in pc.pending_units()], "covered_by_injection": [u.key for u in pc.units if u.covered],
        "required_dims": pc.required_dims, "satisfied_dims": pc.satisfied_dims,
    }, ensure_ascii=False, indent=2)


def _roles_text(pc: PlanContext, specs: dict[str, RoleSpec]) -> str:
    return "\n".join(
        f"- `{s.name}`（phase={s.phase}, 通道={sorted(s.channels) or '—'}, max_concurrency={pc.role_limits.get(s.name)}）：{s.description}；"
        f"提供维度={sorted(s.dims)}" for s in specs.values()
    )


def build_prompt(pc: PlanContext, specs: dict[str, RoleSpec]) -> str:
    values = {"context": _context_json(pc), "roles": _roles_text(pc, specs)}
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), load_prompt("planning"))


def _graph_from(raw: dict[str, Any], pc: PlanContext, specs: dict[str, RoleSpec]) -> WorkflowGraph | None:
    nodes, ids = [], set()
    for nd in raw.get("nodes") or []:
        if not isinstance(nd, dict) or nd.get("role") not in specs:
            return None                                              # 非法角色：整张图丢弃
        spec = specs[nd["role"]]
        tg = [str(t) for t in (nd.get("targets") or []) if str(t) in pc.metrics] if spec.phase == "recall" else []
        node = make_node(spec, tg)
        if node.id in ids:
            continue
        ids.add(node.id)
        nodes.append(node)
    if not nodes:
        return None
    edges = []
    for e in raw.get("edges") or []:
        if not isinstance(e, dict) or e.get("src") not in ids or e.get("dst") not in ids or e["src"] == e["dst"]:
            return None
        edges.append(WorkflowEdge(src=e["src"], dst=e["dst"], kind=EdgeKind.THEN if e.get("kind") == "then" else EdgeKind.DEP))
    g = WorkflowGraph(nodes=nodes, edges=edges)
    try:
        g.layers()
    except Exception:  # noqa: BLE001 - 成环等
        return None
    return repair_width(g, pc)


class PlanningAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def propose(self, pc: PlanContext, specs: dict[str, RoleSpec], *, limit: int, taken_names: set[str]) -> list[CandidateWorkflow]:
        if limit <= 0:
            return []
        try:
            data = parse_llm_json(await self.llm.complete(prompt=build_prompt(pc, specs), json_mode=True))
        except (UpstreamError, ValueError) as e:
            logger.info("planning LLM proposal skipped: %s", e)
            return []
        raw = data.get("candidates")
        out: list[CandidateWorkflow] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict) or len(out) >= limit:
                continue
            g = _graph_from(item, pc, specs)
            if g is None:
                continue
            name = re.sub(r"[^\w\-]+", "_", str(item.get("name") or f"llm_{len(out) + 1}"))[:40] or f"llm_{len(out) + 1}"
            while name in taken_names:
                name += "_llm"
            taken_names.add(name)
            out.append(CandidateWorkflow(name=name, graph=g, source="llm", rationale=str(item.get("rationale") or "")))
        return out
