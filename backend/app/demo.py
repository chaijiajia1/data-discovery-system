"""外网演示模式（DEMO_MODE）：对外公开部署时的防护——限流、客户端地址、页面处理。

演示模式只用假数据，不连任何内网上游、不用 LLM、不处理用户 cookie；这里负责公开暴露面上的几件事：
- `RateLimiter`：按客户端地址的令牌桶（对 /api/stage* 等较重的接口限流，防止被刷）；
- `client_ip`：只信反向代理写进 X-Forwarded-For 的**最右**一个地址（左边的是客户端自己写的，可伪造）；
- `render_html`：前端页面是「另存为网页」得到的，带着三个遥测脚本和来源注释，外网部署前去掉，并注入演示标记。
"""
from __future__ import annotations

import re
import time
from typing import Callable, Mapping

DEMO_BLOCKED_PREFIXES = ("/api/cookie", "/api/me", "/api/pairs", "/api/prompts")
RATE_LIMITED_PREFIXES = ("/api/stage", "/api/discover", "/api/plan")

# Reference primary asset for each metric in the three published examples.
# These labels describe the demo catalog and never participate in selection.
DEMO_CASE_ANSWERS = {
    "last 7 days dau and ctr by platform": {"DAU": "app.user_active_d", "CTR": "app.ad_perf_board_d"},
    "last 30 days ctr by platform": {"CTR": "app.ad_perf_board_d"},
    "last 7 days dau and gmv by channel": {"DAU": "app.user_active_d", "GMV": "app.trade_overview_d"},
}


class RateLimiter:
    """令牌桶：每个 key 容量 burst，每分钟补 per_minute 个。表有上限，超过时丢最久没用的 key。"""

    def __init__(self, per_minute: float, burst: int, clock: Callable[[], float] = time.monotonic, max_keys: int = 10_000) -> None:
        self.rate, self.burst, self.clock, self.max_keys = max(per_minute, 0.001) / 60.0, max(1, burst), clock, max_keys
        self._buckets: dict[str, tuple[float, float]] = {}      # key → (tokens, last)

    def allow(self, key: str) -> tuple[bool, int]:
        """(是否放行, 被拒时建议的等待秒数)。"""
        now = self.clock()
        tokens, last = self._buckets.pop(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.rate)
        ok = tokens >= 1.0
        if ok:
            tokens -= 1.0
        self._buckets[key] = (tokens, now)                        # pop + 重新插入：dict 顺序即「最近使用」
        while len(self._buckets) > self.max_keys:
            self._buckets.pop(next(iter(self._buckets)))
        return ok, 0 if ok else max(1, int((1.0 - tokens) / self.rate + 0.999))


def client_ip(headers: Mapping[str, str], peer: str | None, trust_proxy: bool = True) -> str:
    """trust_proxy=False：前面没有可信反向代理（直接暴露端口），X-Forwarded-For 全是客户端自己写的，只认连接对端地址。"""
    if not trust_proxy:
        return peer or "unknown"
    h = {k.lower(): v for k, v in headers.items()}
    xff = [p.strip() for p in (h.get("x-forwarded-for") or "").split(",") if p.strip()]
    return xff[-1] if xff else (peer or "unknown")


_TELEMETRY_SRC = re.compile(r"<script\b[^>]*\bsrc=\"[^\"]*_files/[^\"]*\"[^>]*>\s*</script>", re.I)
_SAVED_FROM = re.compile(r"<!--\s*saved from url=[^>]*-->\s*", re.I)


def render_html(text: str, *, demo: bool) -> str:
    if not demo:
        return text
    text = _SAVED_FROM.sub("", text)
    text = _TELEMETRY_SRC.sub("", text)
    flag = "<script>window.DD_DEMO=true;</script>"
    return text.replace("</head>", flag + "</head>", 1) if "</head>" in text else flag + text
