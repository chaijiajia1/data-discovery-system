"""发现驱动的转交：角色产出证据卡后，按发现物生成后续需求，主动交给其他角色继续查。

    口径查找 → 找到相关口径（口径文档里关联的其他指标）   →  交给 BI（找它的资产）+ 血缘（以其关联资产为种子沿血缘找）
    血缘发现 → 找到沿血缘关联的表 / 数据集 / 图表          →  交给 BI（解析这些资产的元信息 / 认证数据集）+ 口径查找（这些资产的口径定义）
    BI 发现  → 找到基础资产                              →  交给口径查找（该资产的口径定义）

转交的结果（新的证据卡）会再触发下一跳，直到深度上限（COLLAB_HANDOFF_MAX_DEPTH）、总预算（COLLAB_HANDOFF_MAX_TOTAL）或没有新发现。
每个后续需求只处理一次（以 need key 去重，黑板上已有证据卡的不再转交）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..schemas.collab import EvidenceCard, Need

if TYPE_CHECKING:  # pragma: no cover
    from .hub import CollabHub

DEFAULT_ASSET_TYPE = {"assets": "dataset", "lineage": "table", "caliber": "knowledge", "dimension": "table"}


def _label(a) -> str:
    return (a.name or a.asset_id).strip()


def _resolved(hub: "CollabHub", asset_id: str) -> bool:
    """资产已在候选池里，且分区 / 粒度元信息完整——不必再交给 BI 重新解析。"""
    a = hub.board.pool.get(asset_id)
    return a is not None and bool(a.meta.grain) and bool(a.meta.partition_latest)


def follow_ups(hub: "CollabHub", card: EvidenceCard, need: Need) -> list[Need]:
    """由一张证据卡派生的后续需求（去重 / 预算 / 谁来接是中枢的事）。"""
    fan = hub.settings.COLLAB_HANDOFF_FANOUT
    domains = need.domains or hub.env.domains
    hop = need.hop + 1
    subj = need.subject.casefold()

    def mk(kind: str, subject: str, hints: list[str]) -> Need:
        return Need(kind=kind, subject=subject.strip(), domains=list(domains), asset_type=DEFAULT_ASSET_TYPE[kind],  # type: ignore[arg-type]
                    hint_assets=list(dict.fromkeys(hints)), via=card.key, hop=hop)

    out: list[Need] = []
    if card.kind == "caliber":
        for e in [e for e in card.entities if e.kind == "metric" and e.name.casefold() != subj][:fan]:
            out += [mk("assets", e.name, e.asset_ids), mk("lineage", e.name, e.asset_ids)]
    elif card.kind == "lineage":
        found = sorted((a for a in hub.board.assets_of(card) if a.asset_id not in card.seeds), key=lambda a: (-a.retrieval_score, a.asset_id))[:fan]
        for a in found:
            if not _resolved(hub, a.asset_id):                  # 元信息已完整的不必再交给 BI 重新解析
                out.append(mk("assets", _label(a), [a.asset_id]))
            out.append(mk("caliber", _label(a), [a.asset_id]))
    elif card.kind == "assets":
        for a in sorted(hub.board.assets_of(card), key=lambda a: (-a.retrieval_score, a.asset_id))[:1]:
            out.append(mk("caliber", _label(a), [a.asset_id]))
    return [n for n in out if n.subject and not (n.kind == card.kind and n.subject.casefold() == subj)]
