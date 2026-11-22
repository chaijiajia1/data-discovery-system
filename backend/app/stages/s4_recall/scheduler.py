"""按工作流图并行执行角色：**ASAP**——每个节点在它自己的全部前驱完成后立刻开始（不等同层的其他节点），与 ③ 评分时的排程模型一致；
峰值并发由 A_max 的信号量限制。单角色失败 / 超时不拖垮整体，也不会卡住它的下游。

每个节点是一个独立的 agent 副本（同一角色多个节点 = 多个副本并行）；所有副本共享一个协作中枢（黑板 + 能力画像注册表 + 路由）：同一需求只探查一次，其余节点复用证据卡或等待它。
"""
from __future__ import annotations

import asyncio
import time

from ...collab.hub import CollabHub, build_hub
from ...deps import Deps
from ...roles import Role, RoleContext, build_agents
from ...schemas.knowledge import KnowledgeItem
from ...schemas.recall import RoleReport
from ...schemas.task import TaskContract
from ...schemas.workflow import WorkflowGraph
from ...structured_log import log_event


async def _run_node(role: Role, rc: RoleContext, timeout: float) -> RoleReport:
    t0 = time.perf_counter()
    try:
        rep = await asyncio.wait_for(role.run(rc), timeout=timeout)
    except asyncio.TimeoutError:
        rep = RoleReport(node_id=rc.node.id, role=role.spec.name, status="timeout", error=f">{timeout}s")
    except Exception as e:  # noqa: BLE001 - 角色失败隔离
        rep = RoleReport(node_id=rc.node.id, role=role.spec.name, status="failed", error=str(e))
    rep.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    log_event("s4_recall.role_done", role=rep.role, status=rep.status, n=len(rep.candidates), elapsed_ms=rep.elapsed_ms)
    return rep


async def run_workflow(
    graph: WorkflowGraph, *, contract: TaskContract, knowledge: list[KnowledgeItem], deps: Deps, max_parallel: int,
    seed: RoleReport | None = None, cookie: str | None = None, recall_params: dict | None = None, known_assets: list[str] | None = None,
    mode: str = "recall_and_select",
) -> tuple[list[RoleReport], CollabHub]:
    """seed：L0/L1 子目标的已知资产（由 ④ 物化），作为选择类节点的额外上游。"""
    agents = build_agents(graph.nodes)        # 每个节点一个独立的 agent 副本；同一角色多个节点 = 多个副本并行
    hub = build_hub(agents, deps, contract=contract, knowledge=knowledge, cookie=cookie, recall_params=recall_params or {}, known_assets=known_assets)
    if not graph.nodes:
        return [], hub
    graph.layers()                           # 只为校验：有环 / 悬空边在这里抛出，不会在运行中死锁
    nodes = {n.id: n for n in graph.nodes}
    reports: dict[str, RoleReport] = {}
    done = {nid: asyncio.Event() for nid in nodes}
    sem = asyncio.Semaphore(max_parallel)

    async def run_node(nid: str) -> None:
        try:
            preds = [p for p in graph.predecessors(nid) if p in nodes]
            for p in preds:                              # 等自己的前驱，而不是等整层
                await done[p].wait()
            node = nodes[nid]
            rc = RoleContext(
                node=node, contract=contract, knowledge=knowledge, hub=hub,
                upstream=({"seed": seed} if seed else {}) | {p: reports[p] for p in preds if p in reports}, mode=mode,
            )
            if node.phase == "select":           # 选择阶段的节点启动时，上游节点把候选交付给它
                for p in preds:
                    if p in reports:
                        hub.deliver(p, nid, len(reports[p].candidates))
            async with sem:
                rep = await _run_node(agents[nid], rc, deps.settings.RECALL_ROLE_TIMEOUT_SEC)
            reports[nid] = rep
            if nid in hub.info:
                hub.info[nid].status = rep.status
        finally:
            done[nid].set()                              # 无论成败都放行下游

    await asyncio.gather(*(run_node(n.id) for n in graph.nodes))
    hub.finalize()
    return [reports[n.id] for n in graph.nodes], hub
