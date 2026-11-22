"""带核验与回滚的执行循环。

多指标场景下可能有多个主资产（各自负责一部分指标）：逐个执行，各自独立走
「执行 → 核验 → 归因回滚 → 换资产」循环；全部成功才算整体成功。
"""
from __future__ import annotations

from ...schemas.execution import ExecutionAttempt, ExecutionResult, StepResult
from ...schemas.recall import CandidateAsset
from ...structured_log import log_event
from ..base import RunContext
from .attribution import attribute
from .feedback import build_pair, write_back
from .verifier import verify


def _responsible_metrics(asset: CandidateAsset, metrics: list[str]) -> list[str]:
    fields = {f.lower() for f in asset.meta.fields}
    own = [m for m in metrics if str(m).lower() in fields]
    return own or metrics     # 一个都没命中则按全部指标核验（会被 schema_match 拦下）


async def _execute_target(ctx: RunContext, target: CandidateAsset, fallbacks: list[CandidateAsset], result: ExecutionResult) -> bool:
    eng, max_attempts = ctx.deps.engine, ctx.deps.settings.EXEC_MAX_ATTEMPTS
    metrics_all = list(ctx.contract.metrics.value or [])
    chain = [target, *fallbacks]
    order = eng.steps()
    asset_idx, start = 0, 0
    steps: dict[str, StepResult] = {}       # 当前资产下已保留的步骤结果
    failures: dict[str, int] = {}

    for attempt in range(1, max_attempts + 1):
        asset = chain[asset_idx]
        state = {sid: r.output for sid, r in steps.items() if r.status == "ok" and r.verified}
        for sid in order[start:]:
            r = await eng.run_step(sid, asset=asset, contract=ctx.contract, state=state)
            steps[sid] = r
            if r.status == "failed":
                break
            state[sid] = r.output
        report = verify(steps, asset, ctx.contract, _responsible_metrics(asset, metrics_all))
        att = ExecutionAttempt(attempt=attempt, asset_id=asset.asset_id, verification=report)
        result.attempts.append(att)

        if report.passed:
            for r in steps.values():
                r.verified = True
            att.steps = [steps[s].model_copy() for s in order if s in steps]
            pair = build_pair(ctx.query, ctx.contract, ctx.selection, ctx.preinjection, report, asset.asset_id)
            result.verified_pairs.append(pair)
            result.data_preview = (result.data_preview or {}) | {asset.asset_id: steps[order[-1]].output}
            log_event("s6.target_ok", asset=asset.asset_id, attempts=attempt)
            return True

        # 已通过核验的步骤标记为 verified（上游证据保留）
        bad = {c.step_id for c in report.checks if not c.passed}
        for sid, r in steps.items():
            r.verified = r.status == "ok" and sid not in bad
        att.steps = [steps[s].model_copy() for s in order if s in steps]

        att.attribution = a = attribute(
            step_order=order, report=report, step_failures=failures, has_fallback_asset=asset_idx + 1 < len(chain),
        )
        failures[a.suspect_step] = failures.get(a.suspect_step, 0) + 1
        log_event("s6.failure", asset=asset.asset_id, suspect=a.suspect_step, action=a.next_action)
        if a.next_action == "give_up":
            break
        if a.next_action == "switch_asset":
            asset_idx += 1
            steps, start, failures = {}, 0, {}
        else:  # retry_step：回滚到可疑步骤，保留其上游已验证结果
            for sid in order[order.index(a.rollback_to):]:
                steps.pop(sid, None)
            start = order.index(a.rollback_to)
    return False


async def execute_with_rollback(ctx: RunContext) -> ExecutionResult:
    assert ctx.selection is not None and ctx.contract is not None and ctx.recall is not None
    cmap = {c.asset_id: c for c in ctx.recall.candidates}
    primaries = [cmap[d.asset_id] for d in ctx.selection.primary if d.asset_id in cmap]
    supporting = [cmap[d.asset_id] for d in ctx.selection.alternates if d.asset_id in cmap and not d.needs_permission]    # 备选里有权限的可作兜底资产
    result = ExecutionResult()
    if not primaries:
        write_back(ctx.deps.pair_store, None, ctx.preinjection, False)
        return result

    result.asset_id = primaries[0].asset_id
    oks = []
    for p in primaries:
        metrics = _responsible_metrics(p, list(ctx.contract.metrics.value or []))
        # 兜底资产：佐证资产里同样包含该主资产负责的指标字段者
        fb = [s for s in supporting if {m.lower() for m in map(str, metrics)} <= {f.lower() for f in s.meta.fields}]
        oks.append(await _execute_target(ctx, p, fb, result))

    success = all(oks)
    result.status = "success" if success else "failed"
    if result.verified_pairs:
        result.verified_pair = result.verified_pairs[0]
    # 仅整体通过核验才回流新 pair；预注入 pair 的命中/成功统计无论成败都回写
    if success:
        for pair in result.verified_pairs:
            result.feedback_written = write_back(ctx.deps.pair_store, pair, None, True) or result.feedback_written
    write_back(ctx.deps.pair_store, None, ctx.preinjection, success)
    return result
