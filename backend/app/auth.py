"""cookie 解析与当前用户（对齐 old-version：body > X-Forward-Cookie > Cookie 头 > env）。

- 同域部署：浏览器原生 Cookie 头自动带上；
- 本地 dev（前端 file:// 或 localhost，浏览器不会带 *.xiaohongshu.com cookie）：
  前端「同步 cookie」后经 body.cookie / X-Forward-Cookie 透传。
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Mapping, TypedDict

from .settings import settings

logger = logging.getLogger(__name__)

USER_INFO_URL = "https://redbi.devops.xiaohongshu.com/api/system/user-info"


def resolve_cookie(body_cookie: str | None, headers: Mapping[str, str]) -> str | None:
    """优先级（高→低）：body 显式 > X-Forward-Cookie 头 > 请求 Cookie 头 > env CONTEXT_QUERY_COOKIE。"""
    h = {k.lower(): v for k, v in headers.items()}
    return body_cookie or h.get("x-forward-cookie") or h.get("cookie") or settings.CONTEXT_QUERY_COOKIE or None


# SSO access token 的 cookie 名随环境变化（common-internal-access-token-{prod,sit,beta}、
# access-token-redbi.devops.*），按前缀识别，不写死后缀
_SSO_PREFIXES = ("common-internal-access-token", "access-token-")


def is_sso_name(name: str) -> bool:
    return name.startswith(_SSO_PREFIXES)


def cookie_names(cookie: str | None) -> list[str]:
    return [p.split("=", 1)[0].strip() for p in (cookie or "").split(";") if "=" in p]


def sso_cookie(cookie: str | None) -> str:
    """只保留 SSO access token。透传整个浏览器 cookie 可能触发 Nginx 'Request Header Fields Too Large'。"""
    out = []
    for part in (cookie or "").split(";"):
        part = part.strip()
        if "=" in part and is_sso_name(part.split("=", 1)[0].strip()):
            out.append(part)
    return "; ".join(out)


def slim_for_upstream(cookie: str | None) -> str | None:
    """有 SSO token 就只发 token；一个都认不出时退回原 cookie（不替用户做错误假设）。"""
    return sso_cookie(cookie) or cookie or None


def filter_cookie(cookie_header: str | None, keep: frozenset[str]) -> str:
    if not cookie_header:
        return ""
    out = []
    for part in cookie_header.split(";"):
        part = part.strip()
        if "=" in part and part.split("=", 1)[0].strip() in keep:
            out.append(part)
    return "; ".join(out)


class UserInfo(TypedDict, total=False):
    name_en: str
    email: str
    display: str
    dept: str


_name_cache: dict[str, str] = {}


def username_from_cookie(cookie: str | None) -> str | None:
    """cookie → 用户名（用作 context-query 的 owner）。成功结果按 cookie 缓存。"""
    if not cookie:
        return None
    if cookie not in _name_cache:
        name = fetch_user_info(cookie).get("name_en")
        if not name:
            return None
        _name_cache[cookie] = name
    return _name_cache[cookie]


def fetch_user_info_ex(cookie: str | None, *, timeout: float = 5.0) -> tuple[UserInfo, str]:
    """返回 (info, reason)。成功时 reason 为空；失败时 reason 说明卡在哪一步（给前端徽标提示）。"""
    names = cookie_names(cookie)
    if not names:
        return {}, "no cookie"
    slim = sso_cookie(cookie)
    if not slim:
        hint = [n for n in names if any(k in n.lower() for k in ("token", "sso", "auth", "session"))]
        return {}, f"no SSO access-token cookie found among {len(names)} cookies (token-like names: {hint[:6] or 'none'}) — log in to the internal site in this browser, then Re-sync"
    try:
        req = urllib.request.Request(USER_INFO_URL, headers={"Accept": "application/json", "Cookie": slim})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {}, f"user-info HTTP {e.code}" + (" (token expired? log in again, then Re-sync)" if e.code in (401, 403) else "")
    except Exception as e:  # noqa: BLE001
        logger.warning("fetch_user_info failed: %s", e)
        return {}, f"user-info unreachable: {type(e).__name__}"
    if d.get("code") != 0 or not d.get("data"):
        return {}, f"user-info rejected: code={d.get('code')} msg={d.get('message') or d.get('msg')}"
    data = d["data"]
    email = data.get("userEmail") or ""
    name = email.split("@")[0] if "@" in email else ""
    if not name:
        return {}, "user-info has no userEmail"
    return {
        "name_en": name, "email": email, "display": data.get("displayName") or "",
        "dept": " / ".join(data.get("departmentNames") or []) or (data.get("departmentName") or ""),
    }, ""


def fetch_user_info(cookie: str | None, *, timeout: float = 5.0) -> UserInfo:
    """失败返回空 dict（容错优先）。context-query 的 owner/userId 可由此回填。"""
    return fetch_user_info_ex(cookie, timeout=timeout)[0]
