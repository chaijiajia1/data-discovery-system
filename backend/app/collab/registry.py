"""角色能力画像注册表：离线构建、在线更新、可检索。

画像 = 覆盖域 / 可服务的需求类型 / 擅长资产类型 / 关键词 + 历史响应质量（EMA）与采纳统计。
- 离线：`ensure(specs)` 以 RoleSpec 为种子建档；重复调用只刷新静态字段，保留学到的 quality 与统计；
- 检索：`search(need, roles)` 对画像做路由打分排序（见 router.py）；
- 在线：每次响应被 采纳 / 忽略 / 婉拒 都 `record` 一次，更新 quality 与统计；证据卡被复用按较低权重算采纳；
- 持久化：Json / 内存存储；画像文件损坏回退种子，不影响主流程。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ..schemas.collab import Need, ProfileUpdate, RoleProfile, RoutingScore
from .router import score_profile

logger = logging.getLogger(__name__)

# 一次响应对 quality 的目标值：被采纳 = 1，被忽略 = 0，婉拒 = 0.35（自主权，轻微下调，让之后更少被点名）
OUTCOME_TARGET = {"adopted": 1.0, "ignored": 0.0, "declined": 0.35}


class ProfileStore(Protocol):
    def load(self) -> dict[str, dict]: ...
    def save(self, data: dict[str, dict]) -> None: ...


class MemoryProfileStore:
    def __init__(self) -> None:
        self._d: dict[str, dict] = {}

    def load(self) -> dict[str, dict]:
        return json.loads(json.dumps(self._d))

    def save(self, data: dict[str, dict]) -> None:
        self._d = json.loads(json.dumps(data))


class JsonProfileStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def load(self) -> dict[str, dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("role profile store unreadable, fallback to seeds: %s", e)
            return {}

    def save(self, data: dict[str, dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as e:  # 画像持久化失败不应影响召回
            logger.warning("role profile store write failed: %s", e)


class ProfileRegistry:
    def __init__(self, store: ProfileStore | None = None, alpha: float = 0.2):
        self.store = store or MemoryProfileStore()
        self.alpha = alpha
        self._p: dict[str, RoleProfile] = {}
        for role, raw in self.store.load().items():
            try:
                self._p[role] = RoleProfile.model_validate(raw)
            except Exception:  # noqa: BLE001 - 单条画像损坏只丢这一条
                logger.warning("skip invalid role profile: %s", role)
        self._dirty = False

    def ensure(self, specs) -> None:
        """离线种子：新角色建档；已有角色只刷新静态字段（保留 quality / stats）。"""
        for sp in specs:
            seed = dict(domains=sorted(sp.domains), serves=sorted(sp.serves), asset_types=sorted(sp.asset_types),
                        keywords=list(sp.keywords), channels=sorted(sp.channels))
            cur = self._p.get(sp.name)
            if cur is None:
                self._p[sp.name] = RoleProfile(role=sp.name, **seed)
                self._dirty = True
            elif any(getattr(cur, k) != v for k, v in seed.items()):
                self._p[sp.name] = cur.model_copy(update=seed)
                self._dirty = True

    def get(self, role: str) -> RoleProfile | None:
        return self._p.get(role)

    def all(self) -> list[RoleProfile]:
        return list(self._p.values())

    def search(self, need: Need, roles: list[str] | None = None, top: int | None = None) -> list[RoutingScore]:
        ps = [p for r, p in self._p.items() if roles is None or r in roles]
        out = sorted((score_profile(need, p) for p in ps), key=lambda s: (-s.score, s.role))
        return out[:top] if top else out

    def record(self, role: str, outcome: str, *, weight: float = 1.0) -> ProfileUpdate:
        """一次响应的结果：quality ← (1−a)·quality + a·target，a = α·weight；并累计统计。"""
        p = self._p[role]
        before = p.quality
        a = self.alpha * weight
        p.quality = round(max(0.0, min(1.0, (1 - a) * before + a * OUTCOME_TARGET[outcome])), 6)
        setattr(p.stats, outcome, getattr(p.stats, outcome) + 1)
        if outcome in ("adopted", "ignored"):
            p.stats.answered += 1
        if outcome != "adopted" or weight >= 1.0:
            p.stats.asks += 1
        p.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._dirty = True
        return ProfileUpdate(role=role, outcome=outcome, quality_before=before, quality_after=p.quality)  # type: ignore[arg-type]

    def flush(self) -> None:
        if self._dirty:
            self.store.save({r: p.model_dump(mode="json") for r, p in self._p.items()})
            self._dirty = False


def build_profile_registry(backend: str, path: Path | str, alpha: float = 0.2) -> ProfileRegistry:
    return ProfileRegistry(store=JsonProfileStore(path) if backend == "json" else MemoryProfileStore(), alpha=alpha)
