"""历史 query 存储（每条 = query + 用到的知识 + 最终选择的 asset）——⑥ → ① 闭环的回流点。

文件格式：JSONL，每行一个 VerifiedPair（默认 data/history_queries.jsonl，可手工编辑 / 追加种子）。

- ⑥ 通过核验后 `add()`；
- ① 知识预注入时 `search()` 作为"来源 A"；
- ⑥ 结束后 `record_outcome()` 回写被预注入 pair 的命中/成功统计，供边际收益排序。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from ..schemas.knowledge import VerifiedPair
from ..text_utils import dice


class VerifiedPairStore(Protocol):
    def add(self, pair: VerifiedPair) -> None: ...
    def get(self, pair_id: str) -> VerifiedPair | None: ...
    def search(self, terms: list[str], topk: int) -> list[tuple[VerifiedPair, float]]: ...
    def record_outcome(self, pair_ids: list[str], success: bool) -> None: ...
    def all(self) -> list[VerifiedPair]: ...


class InMemoryPairStore:
    def __init__(self) -> None:
        self._pairs: dict[str, VerifiedPair] = {}
        self._lock = threading.Lock()

    def add(self, pair: VerifiedPair) -> None:
        with self._lock:
            old = self._pairs.get(pair.pair_id)
            if old:  # 同 pair 再次验证：累积计数
                pair.hit_count, pair.success_count = old.hit_count, old.success_count
            self._pairs[pair.pair_id] = pair

    def get(self, pair_id: str) -> VerifiedPair | None:
        return self._pairs.get(pair_id)

    def search(self, terms: list[str], topk: int) -> list[tuple[VerifiedPair, float]]:
        # TODO: 换成 embedding / 向量近邻；这里先用 term Dice 系数占位
        q = set(terms)
        scored = [(p, dice(q, set(p.terms))) for p in self._pairs.values()]
        scored = [(p, s) for p, s in scored if s > 0]
        scored.sort(key=lambda x: (x[1], x[0].success_count), reverse=True)
        return scored[:topk]

    def record_outcome(self, pair_ids: list[str], success: bool) -> None:
        with self._lock:
            for pid in pair_ids:
                p = self._pairs.get(pid)
                if p:
                    p.hit_count += 1
                    p.success_count += int(success)

    def all(self) -> list[VerifiedPair]:
        return list(self._pairs.values())


class JsonlPairStore(InMemoryPairStore):
    """JSONL 落盘（开发/单机用）；生产替换为 DB / 向量库实现同一协议即可。"""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    p = VerifiedPair.model_validate_json(line)
                    self._pairs[p.pair_id] = p

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            "\n".join(json.dumps(p.model_dump(mode="json"), ensure_ascii=False) for p in self._pairs.values()) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def add(self, pair: VerifiedPair) -> None:
        super().add(pair)
        self._flush()

    def record_outcome(self, pair_ids: list[str], success: bool) -> None:
        super().record_outcome(pair_ids, success)
        self._flush()


class ReadOnlyJsonlPairStore(JsonlPairStore):
    """加载 JSONL 种子，之后只在内存里更新、**不写回文件**（公开演示：公开接口写入的配对不能污染种子，重启即恢复）。"""

    def _flush(self) -> None:
        return None


def build_pair_store(backend: str, path: Path) -> VerifiedPairStore:
    if backend == "jsonl_ro":
        return ReadOnlyJsonlPairStore(path)
    return JsonlPairStore(path) if backend == "jsonl" else InMemoryPairStore()
