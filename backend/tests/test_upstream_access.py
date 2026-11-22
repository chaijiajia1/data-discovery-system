"""上游接入（对齐 old-version）：cookie 优先级 / gzip / 诊断 / cookie 同步 / 当前用户。"""
import gzip
import json
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from app import cookie_fetcher
from app.auth import filter_cookie, resolve_cookie
from app.deps import Deps
from app.main import create_app
from app.services.context_query import ContextQueryClient
from app.settings import settings
from app.upstream_diagnose import diagnose_all, evaluate


def test_cookie_priority_body_over_forward_over_cookie_over_env():
    h = {"X-Forward-Cookie": "fwd=1", "Cookie": "hdr=1"}
    with mock.patch.object(settings, "CONTEXT_QUERY_COOKIE", "env=1"):
        assert resolve_cookie("body=1", h) == "body=1"
        assert resolve_cookie(None, h) == "fwd=1"
        assert resolve_cookie(None, {"Cookie": "hdr=1"}) == "hdr=1"
        assert resolve_cookie(None, {}) == "env=1"
    with mock.patch.object(settings, "CONTEXT_QUERY_COOKIE", ""):
        assert resolve_cookie(None, {}) is None


def test_filter_cookie_keeps_only_listed_names():
    assert filter_cookie("a=1; b=2; c=3", frozenset({"a", "c"})) == "a=1; c=3"
    assert filter_cookie(None, frozenset({"a"})) == ""


def test_client_decodes_gzip_response():
    c = ContextQueryClient(base_url="https://x", source_types=[])
    payload = gzip.compress(json.dumps({"items": []}).encode())
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = payload
    resp.__enter__.return_value.headers = {"Content-Encoding": "gzip"}
    with mock.patch("urllib.request.urlopen", return_value=resp) as uo:
        assert c._post({"query": "q"}, "sso=1") == {"items": []}
    assert uo.call_args[0][0].get_header("Accept-encoding") == "gzip"


@pytest.mark.parametrize("status,parsed,ok", [
    (200, {"items": []}, True), (200, {"success": False, "code": 50003, "msg": "x"}, False),
    (401, None, False), (404, None, False), (502, None, False), (200, None, False),
])
def test_evaluate(status, parsed, ok):
    assert evaluate(status, parsed)[0] is ok


def test_diagnose_reports_cookie_problem():
    import urllib.error
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, mock.MagicMock(read=lambda: b""))
    with mock.patch("urllib.request.urlopen", side_effect=err):
        r = diagnose_all(cookie="x=1")
    assert r["summary"]["failed"] == ["context_query"] and "cookie" in r["context_query"]["hint"]


def test_http_cookie_endpoints():
    c = TestClient(create_app(Deps()))
    assert c.get("/api/me").status_code == 401
    with mock.patch.object(settings, "CONTEXT_QUERY_COOKIE", ""):
        assert c.get("/api/me").status_code == 401
    with mock.patch("app.main.fetch_user_info_ex", return_value=({"name_en": "zhang", "email": "zhang@x.com"}, "")):
        assert c.get("/api/me", headers={"X-Forward-Cookie": "a=1"}).json()["name_en"] == "zhang"
    with mock.patch.object(cookie_fetcher, "fetch_first_available", return_value={"cookie_header": "a=1", "count": 1, "browser": "chrome"}):
        assert c.get("/api/cookie/fetch").json()["browser"] == "chrome"
    with mock.patch.object(cookie_fetcher, "fetch_first_available", side_effect=cookie_fetcher.CookieFetchError("denied")):
        assert c.get("/api/cookie/fetch").status_code == 500
    with mock.patch("app.main.upstream_diagnose.diagnose_all", return_value={"summary": {"ok_count": 1}}) as d:
        assert c.post("/api/cookie/diagnose", json={"cookie": "z=1"}).json()["summary"]["ok_count"] == 1
    assert d.call_args.kwargs["cookie"] == "z=1"


def _fetch_result(browser, header):
    names = [p.split("=")[0] for p in header.split("; ") if p]
    return {"cookie_header": header, "count": len(names), "browser": browser, "sample_names": names}


def test_fetch_first_available_prefers_browser_with_sso_token():
    """chrome 只有普通 xhs cookie、edge 才有 SSO token：应选 edge，而不是第一个 count>0 的 chrome。"""
    results = {
        "chrome": _fetch_result("chrome", "a1=1; gid=2"),
        "edge": _fetch_result("edge", "a1=1; common-internal-access-token-prod=t"),
    }
    with mock.patch.object(cookie_fetcher, "fetch", side_effect=lambda b: results[b]):
        r = cookie_fetcher.fetch_first_available(("chrome", "edge"))
    assert r["browser"] == "edge"


def test_fetch_first_available_falls_back_to_non_sso_when_none_has_token():
    results = {"chrome": _fetch_result("chrome", "a1=1"), "edge": _fetch_result("edge", "b=2")}
    with mock.patch.object(cookie_fetcher, "fetch", side_effect=lambda b: results[b]):
        r = cookie_fetcher.fetch_first_available(("chrome", "edge"))
    assert r["browser"] == "chrome"
    assert "SSO" in r["warning"] and "chrome" in r["warning"] and "edge" in r["warning"]


def test_fetch_first_available_skips_failing_browser():
    def fake(b):
        if b == "chrome":
            raise cookie_fetcher.CookieFetchError("locked")
        return _fetch_result("edge", "access-token-x=t")
    with mock.patch.object(cookie_fetcher, "fetch", side_effect=fake):
        assert cookie_fetcher.fetch_first_available(("chrome", "edge"))["browser"] == "edge"


def test_stage1_forwards_cookie_to_retrieval():
    seen = {}

    class KH:
        async def retrieve(self, query, *, user_id=None, cookie=None, max_tokens=None):
            seen["cookie"] = cookie
            from app.schemas.knowledge import RetrievalPack
            return RetrievalPack()
        async def search(self, query, *, limit=10): return []

    c = TestClient(create_app(Deps(kh=KH())))
    with mock.patch.object(settings, "CONTEXT_QUERY_COOKIE", ""):
        c.post("/api/stage1", json={"query": "DAU"}, headers={"X-Forward-Cookie": "sso=abc"})
    assert seen["cookie"] == "sso=abc"


def test_ui_served_at_root_without_exposing_parent_dir(tmp_path):
    html = tmp_path / "Page.html"
    html.write_text("<html>ui</html>", encoding="utf-8")
    (tmp_path / "Page_files").mkdir()
    (tmp_path / "Page_files" / "a.js").write_text("x=1", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    with mock.patch.object(settings, "UI_HTML_PATH", html):
        c = TestClient(create_app(Deps()))
    assert c.get("/").text == "<html>ui</html>"
    assert c.get("/Page_files/a.js").text == "x=1"
    assert c.get("/secret.txt").status_code == 404
    assert c.get("/Page.html").status_code == 404


# ── 400 业务错误（实测：HTTP 400 + {success:false, errorCode:50003, errorMsg}）──────────
def _http_400(body: dict):
    import urllib.error
    return urllib.error.HTTPError("u", 400, "Bad Request", {}, mock.MagicMock(read=lambda: json.dumps(body).encode()))


async def test_400_surfaces_business_error_body():
    from app.errors import UpstreamError
    c = ContextQueryClient(base_url="https://x", source_types=[])
    err = _http_400({"success": False, "errorCode": 50003, "errorMsg": "无法确定 user_id"})
    with mock.patch("urllib.request.urlopen", side_effect=err), pytest.raises(UpstreamError) as ei:
        await c.retrieve("q")
    msg = str(ei.value)
    assert "HTTP 400" in msg and "50003" in msg and "caller identity missing" in msg and "sync the cookie" in msg


def test_parse_pack_accepts_both_error_shapes():
    from app.errors import UpstreamError
    from app.services.context_query import parse_pack
    for body in ({"success": False, "errorCode": 50003, "errorMsg": "x"}, {"success": False, "code": 50003, "msg": "x"}):
        with pytest.raises(UpstreamError, match="50003"):
            parse_pack(body)


def test_partial_without_failed_sources_is_not_degradation():
    from app.services.context_query import parse_pack
    assert parse_pack({"items": [], "partial": True, "failedSources": [], "failedReasons": []}).partial is False
    assert parse_pack({"items": [], "partial": True, "failedSources": ["memory"]}).partial is True


def test_owner_filled_from_cookie_when_user_id_missing():
    seen = {}

    class KH:
        async def retrieve(self, query, *, user_id=None, cookie=None, max_tokens=None):
            seen["user_id"] = user_id
            from app.schemas.knowledge import RetrievalPack
            return RetrievalPack()
        async def search(self, query, *, limit=10): return []

    import app.auth as auth
    auth._name_cache.clear()
    c = TestClient(create_app(Deps(kh=KH())))
    with mock.patch("app.auth.fetch_user_info", return_value={"name_en": "zhang"}):
        c.post("/api/stage1", json={"query": "DAU"}, headers={"X-Forward-Cookie": "sso=1"})
    assert seen["user_id"] == "zhang"
    with mock.patch("app.auth.fetch_user_info", return_value={"name_en": "zhang"}):
        c.post("/api/stage1", json={"query": "DAU", "user_id": "explicit"}, headers={"X-Forward-Cookie": "sso=1"})
    assert seen["user_id"] == "explicit"


# ── SSO token 识别 / 姓名失败原因 ──────────────────────────────────────────
def test_sso_cookie_keeps_token_prefix_only():
    from app.auth import slim_for_upstream, sso_cookie
    ck = "a=1; common-internal-access-token-prod=T1; sid=2; access-token-redbi.devops.beta.xiaohongshu.com=T2"
    assert sso_cookie(ck) == "common-internal-access-token-prod=T1; access-token-redbi.devops.beta.xiaohongshu.com=T2"
    assert slim_for_upstream(ck) == sso_cookie(ck)
    assert slim_for_upstream("a=1; b=2") == "a=1; b=2"      # 认不出 token 时退回原 cookie
    assert slim_for_upstream(None) is None


def test_user_info_reason_when_no_token():
    from app.auth import fetch_user_info_ex
    info, reason = fetch_user_info_ex("a=1; b=2; web_session=x")
    assert info == {} and "no SSO access-token cookie" in reason and "web_session" in reason
    assert fetch_user_info_ex(None)[1] == "no cookie"


def test_user_info_success_and_rejected():
    from app.auth import fetch_user_info_ex

    def fake(body):
        r = mock.MagicMock()
        r.__enter__.return_value.read.return_value = json.dumps(body).encode()
        return r
    ck = "common-internal-access-token-prod=T"
    ok = {"code": 0, "data": {"userEmail": "zhang@xiaohongshu.com", "displayName": "翼德(张飞)"}}
    with mock.patch("urllib.request.urlopen", return_value=fake(ok)):
        info, reason = fetch_user_info_ex(ck)
    assert info["name_en"] == "zhang" and info["display"] == "翼德(张飞)" and reason == ""
    with mock.patch("urllib.request.urlopen", return_value=fake({"code": 401, "message": "expired"})):
        assert "rejected" in fetch_user_info_ex(ck)[1]


def test_me_endpoint_returns_reason_in_401_detail():
    c = TestClient(create_app(Deps()))
    r = c.get("/api/me", headers={"X-Forward-Cookie": "a=1; b=2"})
    assert r.status_code == 401 and "no SSO access-token cookie" in r.json()["detail"]


def test_client_sends_only_sso_token_upstream():
    c = ContextQueryClient(base_url="https://x", source_types=[])
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = b"{}"
    resp.__enter__.return_value.headers = {}
    with mock.patch("urllib.request.urlopen", return_value=resp) as uo:
        c._post({"query": "q"}, "junk=1; common-internal-access-token-prod=T; big=" + "x" * 5000)
    assert uo.call_args[0][0].get_header("Cookie") == "common-internal-access-token-prod=T"
