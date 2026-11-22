"""角色注册表。新增角色：实现 Role 子类并在 REGISTRY 注册，规划阶段自动可见。"""
from __future__ import annotations

from .base import Role, RoleContext, RoleSpec
from .builtin import BIAssetDiscovery, CaliberLookup, DimValueResolver, LineageDiscovery, OmniDiscovery

REGISTRY: dict[str, type[Role]] = {
    c.spec.name: c for c in (
        BIAssetDiscovery, LineageDiscovery, CaliberLookup, OmniDiscovery, DimValueResolver,
    )
}


def list_role_specs() -> list[RoleSpec]:
    return [c.spec for c in REGISTRY.values()]


def build_agents(nodes) -> dict[str, Role]:
    """工作流图的每个节点一个独立的 agent 副本（key = node id）。同一角色出现 n 次就是 n 个副本，编号 1..n。"""
    total: dict[str, int] = {}
    for n in nodes:
        total[n.role] = total.get(n.role, 0) + 1
    seq: dict[str, int] = {}
    out: dict[str, Role] = {}
    for n in nodes:
        seq[n.role] = seq.get(n.role, 0) + 1
        out[n.id] = REGISTRY[n.role](agent_id=n.id, targets=list(n.targets), replica=seq[n.role], replicas=total[n.role])
    return out


def build_roles(names: list[str] | None = None) -> dict[str, Role]:
    return {n: REGISTRY[n]() for n in (REGISTRY if names is None else names)}


__all__ = ["REGISTRY", "Role", "RoleContext", "RoleSpec", "build_agents", "build_roles", "list_role_specs"]
