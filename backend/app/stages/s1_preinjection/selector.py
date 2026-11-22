"""知识块选择：带交互效用的贪心边际收益。

模型（每个知识块 k）：
  u_k  独立贡献 = 相关度（已含一致性折扣），clamp 到 [0,1]
  c_k  token 成本（至少为 1）
  w_ij 两块之间的交互效用：<0 冗余/冲突，>0 协同

  V(S)      = Σ_{k∈S} u_k + Σ_{i<j∈S} w_ij
  Δ(k | S)  = u_k + Σ_{j∈S} w_jk

每轮选 Δ(k|S)/c_k 最大的块；最大值 < τ 时停止。容量 L 只作为硬上界：
放不下的块直接剔除（over_budget），不参与比较，也不改变停止条件。
"""
from __future__ import annotations

from dataclasses import dataclass

from ...schemas.knowledge import DroppedKnowledge, KnowledgeItem, KnowledgeKind, SelectionStep
from ...text_utils import text_similarity


@dataclass(frozen=True)
class SelectorParams:
    tau: float = 0.001           # 停止阈值 τ（Δ/token）
    redundancy: float = 1.0      # 冗余：w = -redundancy · sim · min(u_i,u_j)
    conflict: float = 1.5        # 冲突：w = -conflict · min(u_i,u_j)（>1 保证后到者 Δ<0）
    synergy: float = 0.2         # 协同：w = +synergy · min(u_i,u_j)
    sim_floor: float = 0.3       # 文本相似度低于此值不算冗余
    dup_sim: float = 0.6         # 同主题块相似度 >= 此值视为重复（冗余），否则视为内容冲突


@dataclass
class SelectionOutcome:
    selected: list[KnowledgeItem]
    dropped: list[DroppedKnowledge]
    trace: list[SelectionStep]
    used_tokens: int


def utility(item: KnowledgeItem) -> float:
    return max(0.0, min(1.0, item.relevance))


def cost(item: KnowledgeItem) -> int:
    return max(1, item.token_cost)


def _topic(item: KnowledgeItem) -> str:
    return item.title.strip().casefold()


def interaction(a: KnowledgeItem, b: KnowledgeItem, p: SelectorParams) -> tuple[float, str]:
    """w_ab 及其类型标签：conflict | redundant | synergy | ""。"""
    m = min(utility(a), utility(b))
    sim = text_similarity(f"{a.title} {a.content}", f"{b.title} {b.content}")
    if a.kind == b.kind != KnowledgeKind.PAIR and _topic(a) == _topic(b) and sim < p.dup_sim:
        return -p.conflict * m, "conflict"           # 同一概念，内容却对不上（如同名指标两套口径）
    if sim >= p.sim_floor:
        return -p.redundancy * sim * m, "redundant"  # 说的是同一件事
    if a.kind != b.kind and set(a.asset_ids) & set(b.asset_ids):
        return p.synergy * m, "synergy"              # 同一资产的不同侧面（口径 + 元数据 / 历史选择）
    return 0.0, ""


def _interaction_matrix(items: list[KnowledgeItem], p: SelectorParams) -> dict[tuple[int, int], tuple[float, str]]:
    n = len(items)
    return {(i, j): interaction(items[i], items[j], p) for i in range(n) for j in range(i + 1, n)}


def _w(mat, i: int, j: int) -> tuple[float, str]:
    return mat[(i, j)] if i < j else mat[(j, i)]


def greedy_select(items: list[KnowledgeItem], capacity: int, p: SelectorParams) -> SelectionOutcome:
    mat = _interaction_matrix(items, p)
    u = [utility(it) for it in items]
    c = [cost(it) for it in items]
    remaining = set(range(len(items)))
    chosen: list[int] = []
    dropped: list[DroppedKnowledge] = []
    trace: list[SelectionStep] = []
    used = 0

    def delta(k: int) -> float:
        return u[k] + sum(_w(mat, j, k)[0] for j in chosen)

    while remaining:
        # 容量 L 为硬上界：放不下的块永远放不下（used 只增不减），直接剔除
        for k in sorted(k for k in remaining if used + c[k] > capacity):
            remaining.discard(k)
            dropped.append(DroppedKnowledge(item_id=items[k].id, title=items[k].title, reason="over_budget",
                                            detail=f"cost={c[k]} > remaining={capacity - used}"))
        if not remaining:
            break
        best = max(remaining, key=lambda k: (delta(k) / c[k], u[k], -k))
        d = delta(best)
        if d / c[best] < p.tau:
            break
        trace.append(SelectionStep(
            step=len(trace) + 1, item_id=items[best].id, utility=round(u[best], 4), cost=c[best],
            delta=round(d, 4), ratio=round(d / c[best], 6),
            interactions={items[j].id: round(_w(mat, j, best)[0], 4) for j in chosen if _w(mat, j, best)[0]},
        ))
        chosen.append(best)
        used += c[best]
        remaining.discard(best)

    # 未入选原因：被已选集合压低的主因（冲突 / 冗余），否则就是单位成本收益不足
    for k in sorted(remaining):
        pen: dict[str, float] = {}
        for j in chosen:
            w, label = _w(mat, j, k)
            if w < 0:
                pen[label] = pen.get(label, 0.0) - w
        label, amt = max(pen.items(), key=lambda x: x[1], default=("", 0.0))
        reason = label if amt >= 0.5 * u[k] and label else "low_gain"
        dropped.append(DroppedKnowledge(item_id=items[k].id, title=items[k].title, reason=reason,
                                        detail=f"Δ={delta(k):.4f} cost={c[k]} Δ/c={delta(k) / c[k]:.6f} < τ={p.tau}"))
    return SelectionOutcome([items[k] for k in chosen], dropped, trace, used)
