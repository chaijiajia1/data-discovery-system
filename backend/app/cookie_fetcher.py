"""从本机浏览器读取 DRR 相关 cookie。

支持：Chrome / Chrome Canary / Safari / Firefox / Edge（依赖 browser_cookie3）。
macOS Chrome 首次读取会触发 Keychain 授权弹窗。
"""
from __future__ import annotations

from typing import Iterable

from .auth import cookie_names, is_sso_name

# DRR 域名清单（按优先级）
DEFAULT_DOMAINS: tuple[str, ...] = (
    "aimi.devops.xiaohongshu.com",
    "redbi.devops.xiaohongshu.com",
    "metric.devops.xiaohongshu.com",
    "devops.xiaohongshu.com",
    "xiaohongshu.com",
)

SUPPORTED_BROWSERS = ("chrome", "safari", "firefox", "edge", "chromium")


class CookieFetchError(Exception):
    """cookie 抓取失败统一异常。"""


def _filter_devops(cookies: Iterable) -> list:
    """保留所有 *.xiaohongshu.com 域 cookie。

    放宽策略：SSO/统一登录 token（如 auth-token, Admin-Token）domain 常为
    `.xiaohongshu.com` 父级，devops 子接口校验时可能依赖。把 xiaohongshu.com
    全域 cookie 一并保留，由服务端按需识别。
    """
    out = []
    for c in cookies:
        domain = (getattr(c, "domain", "") or "").lstrip(".").lower()
        if domain == "xiaohongshu.com" or domain.endswith(".xiaohongshu.com"):
            out.append(c)
    return out


def _cookies_to_header(cookies: Iterable) -> str:
    """拼成 HTTP Cookie 头字符串。"""
    pairs = []
    seen_keys = set()
    for c in cookies:
        name = getattr(c, "name", "") or ""
        if not name or name in seen_keys:
            continue
        seen_keys.add(name)
        pairs.append(f"{name}={c.value}")
    return "; ".join(pairs)


def fetch(browser: str = "chrome") -> dict:
    """从指定浏览器读取并过滤 devops cookie，返回 `{cookie_header, count, browser, sample_names}`。

    Raises:
        CookieFetchError on browser_cookie3 缺失 / 浏览器不支持 / 权限失败。
    """
    if browser not in SUPPORTED_BROWSERS:
        raise CookieFetchError(f"unsupported browser: {browser}")

    try:
        import browser_cookie3  # type: ignore
    except ImportError as e:
        raise CookieFetchError(
            "需要 pip install browser-cookie3 才能拉取 cookie"
        ) from e

    loader = getattr(browser_cookie3, browser, None)
    if loader is None:
        raise CookieFetchError(f"browser_cookie3 无 loader: {browser}")

    try:
        # 不指定 domain_name 避免被库内 filter 过严；交给我们自己 filter
        cj = loader()
    except Exception as e:  # noqa: BLE001
        raise CookieFetchError(
            f"读取浏览器 cookie 失败：{type(e).__name__}: {e}（macOS 可能需要授权 Keychain）"
        ) from e

    filtered = _filter_devops(cj)
    header = _cookies_to_header(filtered)
    return {
        "cookie_header": header,
        "count": len(filtered),
        "browser": browser,
        "sample_names": [getattr(c, "name", "") for c in filtered[:5]],
        "domains_seen": sorted({(getattr(c, "domain", "") or "").lstrip(".") for c in filtered}),
    }


def _has_sso(result: dict) -> bool:
    return any(is_sso_name(n) for n in cookie_names(result.get("cookie_header")))


def fetch_first_available(browsers: Iterable[str] = SUPPORTED_BROWSERS) -> dict:
    """按顺序尝试多个浏览器，优先返回含 SSO access token 的；全部失败抛 CookieFetchError。

    只要有 xiaohongshu.com 普通 cookie（a1/gid 等）就会 count>0，但那并不代表已登录内网；
    因此遇到带 token 的浏览器立即返回，否则退回第一个 count>0 的并附 warning。
    """
    errors: list[str] = []
    fallback: dict | None = None
    tried: list[str] = []
    for b in browsers:
        try:
            result = fetch(b)
        except CookieFetchError as e:
            errors.append(f"{b}: {e}")
            continue
        if result["count"] <= 0:
            errors.append(f"{b}: 0 cookies matched devops domain")
            continue
        if _has_sso(result):
            return result
        tried.append(b)
        fallback = fallback or result
    if fallback is not None:
        return {**fallback, "warning": f"已检查 {', '.join(tried)}，均未发现 SSO access-token cookie；请在其中一个浏览器登录内网站点后重试"}
    raise CookieFetchError(
        "所有浏览器都未能取到 cookie:\n  - " + "\n  - ".join(errors)
    )
