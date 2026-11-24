"""FastAPI 入口：`uvicorn app.main:app --host 0.0.0.0 --port 8080`"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __service_name__, __version__, cookie_fetcher, upstream_diagnose
from .auth import fetch_user_info_ex, resolve_cookie, username_from_cookie
from .demo import DEMO_BLOCKED_PREFIXES, RATE_LIMITED_PREFIXES, RateLimiter, client_ip, render_html
from .deps import Deps, build_default_deps
from .pipeline import stream_pipeline
from .prompts import load_prompt
from .roles import list_role_specs
from .settings import Settings, settings
from .stages import RunContext
from .stages.s1_preinjection import PreInjectionStage
from .roles import REGISTRY as ROLE_REGISTRY
from .schemas.workflow import EdgeKind, WorkflowEdge, WorkflowGraph, WorkflowNode
from .stages.s2_coordination import CoordinationStage
from .stages.s3_planning import PlanningStage, evaluate_custom
from .stages.s4_recall import RecallStage
from .stages.s5_selection import SelectionStage
from .structured_log import set_run, set_session, setup_logging

setup_logging(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DiscoverRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str = "default"
    user_id: str | None = None
    cookie: str | None = None        # 可选；缺省时透传请求的 Cookie 头（SSO）
    # 用户主动补充的条件（可选，不会被追问）：{"metrics": ["DAU"], "caliber": "..."}，优先级最高
    clarifications: dict[str, Any] = Field(default_factory=dict)
    max_parallel: int | None = Field(default=None, ge=1, le=16)    # A_max：前端输入的最大并行 Agent 数
    confirmed_assets: dict[str, list[str]] = Field(default_factory=dict)   # 会话内已确认资产：指标 -> asset_id 列表
    run_id: str | None = None        # 续跑：复用同一 query 之前阶段的产物，不重复调用上游
    graph: dict[str, Any] | None = None   # 仅 /api/stage4：按这张（用户在 ③ 确认的）工作流图运行，而不是规划选出的那张


class DiagnoseRequest(BaseModel):
    cookie: str | None = None


class EvaluateRequest(BaseModel):
    run_id: str
    graph: dict[str, Any]            # {"nodes":[{id,role,targets}], "edges":[{src,dst,kind}]}
    max_parallel: int | None = Field(default=None, ge=1, le=16)


RUN_CACHE_SIZE = 64


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def create_app(deps: Deps | None = None) -> FastAPI:
    deps = deps or build_default_deps()
    deps.profiles.ensure(list_role_specs())          # 能力画像注册表：以全部已注册角色为种子建档（离线），运行中在线更新
    ds = deps.settings
    demo = ds.DEMO_MODE
    app = FastAPI(title=__service_name__, version=__version__, **({"docs_url": None, "redoc_url": None, "openapi_url": None} if demo else {}))
    origins = [o.strip() for o in ds.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    if demo:        # 公开部署：屏蔽 cookie / 调试端点，对较重的接口按客户端限流
        limiter = RateLimiter(ds.DEMO_RATE_LIMIT_PER_MIN, ds.DEMO_RATE_BURST)

        @app.middleware("http")
        async def demo_guard(request: Request, call_next):
            path = request.url.path
            if path.startswith(DEMO_BLOCKED_PREFIXES):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            if request.method != "OPTIONS" and path.startswith(RATE_LIMITED_PREFIXES):
                ok, retry = limiter.allow(client_ip(request.headers, request.client.host if request.client else None, ds.DEMO_TRUST_PROXY))
                if not ok:
                    return JSONResponse({"detail": "Rate limit exceeded — please slow down and retry shortly."}, status_code=429, headers={"Retry-After": str(retry)})
            return await call_next(request)

    async def _ctx(req: DiscoverRequest, request: Request) -> RunContext:
        if demo:        # 演示模式不处理任何 cookie / 身份：假数据不需要，也不去查内网的用户信息接口
            return RunContext(query=req.query, deps=deps, user_id=None, session_id=req.session_id, clarifications=req.clarifications, cookie=None,
                              max_parallel=req.max_parallel, confirmed_assets=req.confirmed_assets)
        cookie = resolve_cookie(req.cookie, request.headers)
        # 身份：显式 user_id > cookie 解析出的用户名 > CONTEXT_QUERY_OWNER（客户端兜底）
        user_id = req.user_id or await asyncio.to_thread(username_from_cookie, cookie)
        return RunContext(query=req.query, deps=deps, user_id=user_id, session_id=req.session_id,
                          clarifications=req.clarifications, cookie=cookie, max_parallel=req.max_parallel,
                          confirmed_assets=req.confirmed_assets)

    runs: OrderedDict[str, RunContext] = OrderedDict()      # 阶段间续跑 / 自定义图评分用的 run 缓存（LRU，进程内）

    def _remember(ctx: RunContext) -> None:
        runs[ctx.run_id] = ctx
        runs.move_to_end(ctx.run_id)
        while len(runs) > RUN_CACHE_SIZE:
            runs.popitem(last=False)

    async def _resume(req: DiscoverRequest, request: Request) -> RunContext:
        """同一 query 带 run_id 续跑：复用已有的 ① / ② 产物（② 仅在补充信息未变时复用）。"""
        ctx = await _ctx(req, request)
        prev = runs.get(req.run_id) if req.run_id else None
        if prev is not None and prev.query == req.query:
            ctx.run_id, ctx.preinjection = prev.run_id, prev.preinjection
            if prev.contract is not None and prev.clarifications == req.clarifications:
                ctx.contract = prev.contract
        return ctx

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": __service_name__, "version": __version__, "demo": demo}

    @app.post("/api/discover")
    async def discover(req: DiscoverRequest, request: Request):
        """SSE 流：stage1..stage6 / done / error"""
        ctx = await _ctx(req, request)

        async def gen():
            async for event, data in stream_pipeline(ctx):
                yield _sse(event, data)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/discover/sync")
    async def discover_sync(req: DiscoverRequest, request: Request):
        """非流式：收集全部事件一次性返回（调试 / 测试用）。"""
        events: list[dict[str, Any]] = []
        async for event, data in stream_pipeline(await _ctx(req, request)):
            events.append({"event": event, **data})
        return {"events": events}

    @app.post("/api/stage1")
    async def stage1(req: DiscoverRequest, request: Request):
        """只跑 ① 知识预注入（前端阶段 1 使用）。两路任一失败只降级，见 route_status。"""
        ctx = await _ctx(req, request)
        set_session(ctx.session_id)
        set_run(ctx.run_id)
        ctx.preinjection = await PreInjectionStage().run(ctx)
        _remember(ctx)
        return {"run_id": ctx.run_id, "result": ctx.preinjection.model_dump(mode="json")}

    async def _through_stage2(req: DiscoverRequest, request: Request) -> RunContext:
        ctx = await _resume(req, request)
        set_session(ctx.session_id)
        set_run(ctx.run_id)
        if ctx.preinjection is None:
            ctx.preinjection = await PreInjectionStage().run(ctx)
        if ctx.contract is None:
            ctx.contract = await CoordinationStage().run(ctx)
        return ctx

    @app.post("/api/stage2")
    async def stage2(req: DiscoverRequest, request: Request):
        """跑 ① → ②：用户问题 + 预注入知识 → 协调 agent → intent（含 contract.intent）。"""
        ctx = await _through_stage2(req, request)
        _remember(ctx)
        return {"run_id": ctx.run_id, "preinjection": ctx.preinjection.model_dump(mode="json"),
                "contract": ctx.contract.model_dump(mode="json")}

    @app.post("/api/stage3")
    async def stage3(req: DiscoverRequest, request: Request):
        """跑 ① → ② → ③：入口定级 / 覆盖单元 / 候选图与评分。"""
        ctx = await _through_stage2(req, request)
        ctx.plan = None
        if ctx.contract.status == "ready":
            ctx.plan = await PlanningStage().run(ctx)
        _remember(ctx)
        return {"run_id": ctx.run_id, "status": ctx.contract.status, "preinjection": ctx.preinjection.model_dump(mode="json"),
                "contract": ctx.contract.model_dump(mode="json"),
                "plan": ctx.plan.model_dump(mode="json") if ctx.plan else None}

    def _graph_from_request(raw: dict[str, Any]) -> WorkflowGraph:
        """前端的图（{nodes:[{id,role,targets}], edges:[{src,dst,kind}]}）→ WorkflowGraph；角色未注册 / 有环 / 悬空边 → 400。"""
        specs = {n: c.spec for n, c in ROLE_REGISTRY.items()}
        nodes = []
        for n in raw.get("nodes") or []:
            sp = specs.get(n.get("role"))
            if sp is None:
                raise HTTPException(400, f"角色 {n.get('role')!r} 未注册")
            nodes.append(WorkflowNode(id=str(n.get("id")), role=sp.name, subtask=sp.description.split("：")[0], phase=sp.phase,  # type: ignore[arg-type]
                                      channels=sorted(sp.channels), targets=[str(t) for t in (n.get("targets") or [])]))
        edges = [WorkflowEdge(src=str(e.get("src")), dst=str(e.get("dst")), kind=EdgeKind.THEN if e.get("kind") == "then" else EdgeKind.DEP)
                 for e in raw.get("edges") or []]
        g = WorkflowGraph(nodes=nodes, edges=edges)
        try:
            g.layers()
        except Exception as e:  # noqa: BLE001 - 有环 / 悬空边
            raise HTTPException(400, f"工作流图不合法：{getattr(e, 'message', e)}") from e
        return g

    @app.post("/api/stage4")
    async def stage4(req: DiscoverRequest, request: Request):
        """跑 ① → ④：按工作流图多 agent 并行检索，返回候选池 / 证据卡 / 协作轨迹（含事件流）。
        带 `graph` 时按它运行（每个节点一个 agent 副本），Roles / Communication 面板因此与 ③ 确认的图一致。"""
        graph = _graph_from_request(req.graph) if req.graph is not None else None
        ctx = await _through_stage2(req, request)
        ctx.plan = ctx.recall = None
        if ctx.contract.status == "ready":
            ctx.plan = await PlanningStage().run(ctx)
            if graph is not None:
                ctx.plan = ctx.plan.model_copy(update={"chosen": ctx.plan.chosen.model_copy(update={"graph": graph})})
            ctx.recall = await RecallStage().run(ctx)
        _remember(ctx)
        return {"run_id": ctx.run_id, "status": ctx.contract.status, "contract": ctx.contract.model_dump(mode="json"),
                "plan": ctx.plan.model_dump(mode="json") if ctx.plan else None,
                "recall": ctx.recall.model_dump(mode="json") if ctx.recall else None}

    @app.post("/api/stage5")
    async def stage5(req: DiscoverRequest, request: Request):
        """跑 ① → ⑤：为每个口径选出主资产，其余分入备选 / 丢弃（理由结构化）。
        带 `run_id` 且同一 query / 补充信息 / 已确认资产时，**复用 run 缓存里 ④ 的结果**（含用户确认的图），不重跑 ③④——
        前端刚跑完 ④ 再跑 ⑤，Phase 4 与 Phase 5 的候选池因此一致、上游也不会被调用两遍。没有缓存才自己跑 ③④。"""
        prev = runs.get(req.run_id) if req.run_id else None
        ctx = await _through_stage2(req, request)
        reused = (prev is not None and prev.query == req.query and prev.contract is ctx.contract and prev.confirmed_assets == req.confirmed_assets
                  and prev.plan is not None and prev.recall is not None)
        ctx.plan, ctx.recall = (prev.plan, prev.recall) if reused else (None, None)      # type: ignore[union-attr]
        ctx.selection = None
        if ctx.contract.status == "ready":
            if not reused:
                ctx.plan = await PlanningStage().run(ctx)
                ctx.recall = await RecallStage().run(ctx)
            ctx.selection = await SelectionStage().run(ctx)
        _remember(ctx)
        return {"run_id": ctx.run_id, "status": ctx.contract.status, "recall_reused": reused, "contract": ctx.contract.model_dump(mode="json"),
                "recall": ctx.recall.model_dump(mode="json") if ctx.recall else None,
                "selection": ctx.selection.model_dump(mode="json") if ctx.selection else None}

    @app.get("/api/profiles")
    async def profiles():
        """角色能力画像（覆盖域 / 资产类型 / 历史响应质量与采纳统计）。"""
        return [p.model_dump(mode="json") for p in deps.profiles.all()]

    @app.post("/api/plan/evaluate")
    async def plan_evaluate(req: EvaluateRequest):
        """用与 ③ 相同的上下文、可行性约束与评分公式评估用户自定义的图。"""
        prev = runs.get(req.run_id)
        if prev is None or prev.plan is None:
            raise HTTPException(404, "run 不存在或尚未规划，请先调用 /api/stage3")
        pc = prev.plan.context
        if req.max_parallel:
            pc = pc.model_copy(update={"max_parallel": req.max_parallel})
        specs = {n: c.spec for n, c in ROLE_REGISTRY.items()}
        nodes = []
        for n in req.graph.get("nodes") or []:
            sp = specs.get(n.get("role"))
            nodes.append(WorkflowNode(id=str(n.get("id")), role=str(n.get("role")), subtask=sp.description.split("：")[0] if sp else "",
                                      phase=sp.phase if sp else "recall", channels=sorted(sp.channels) if sp else [],   # type: ignore[arg-type]
                                      targets=[str(t) for t in (n.get("targets") or [])]))
        edges = [WorkflowEdge(src=str(e.get("src")), dst=str(e.get("dst")), kind=EdgeKind.THEN if e.get("kind") == "then" else EdgeKind.DEP)
                 for e in req.graph.get("edges") or []]
        return evaluate_custom(pc, WorkflowGraph(nodes=nodes, edges=edges)).model_dump(mode="json")

    @app.get("/api/cookie/fetch")
    async def cookie_fetch():
        """从本机浏览器读取 *.xiaohongshu.com cookie（仅本地开发；macOS 首次可能弹 Keychain 授权）。"""
        try:
            return await asyncio.to_thread(cookie_fetcher.fetch_first_available)
        except cookie_fetcher.CookieFetchError as e:
            raise HTTPException(500, str(e))

    @app.post("/api/cookie/diagnose")
    async def cookie_diagnose(req: DiagnoseRequest, request: Request):
        """用给定 / 透传的 cookie ping 上游，报告通断。"""
        return await asyncio.to_thread(upstream_diagnose.diagnose_all, cookie=resolve_cookie(req.cookie, request.headers))

    @app.get("/api/me")
    async def me(request: Request):
        cookie = resolve_cookie(None, request.headers)
        if not cookie:
            raise HTTPException(401, "缺少 cookie")
        info, reason = await asyncio.to_thread(fetch_user_info_ex, cookie)
        if not info.get("name_en"):
            raise HTTPException(401, reason or "user-info 解析失败")
        return info

    @app.get("/api/roles")
    async def roles():
        return [{"name": s.name, "description": s.description, "phase": s.phase, "channels": sorted(s.channels),
                 "dims": sorted(s.dims), "max_concurrency": s.max_concurrency} for s in list_role_specs()]

    @app.get("/api/pairs")
    async def pairs():
        """已验证 pair（闭环回流内容），调试用。"""
        return [p.model_dump(mode="json") for p in deps.pair_store.all()]

    @app.get("/api/prompts/{name}", response_class=PlainTextResponse)
    async def get_prompt(name: str):
        try:
            return load_prompt(name)
        except FileNotFoundError:
            raise HTTPException(404, f"prompt not found: {name}")

    _mount_ui(app, ds)
    return app


def _mount_ui(app: FastAPI, s: Settings) -> None:
    """同域托管前端：GET / 返回 HTML；HTML 引用的 `<名>_files/` 资源目录单独挂载（不暴露上层目录）。
    演示模式：页面去掉遥测脚本并注入演示标记，且不挂载资源目录。"""
    html = s.UI_HTML_PATH
    if not html.is_file():
        logger.warning("UI html not found: %s (GET / will be 404)", html)
        return
    if s.DEMO_MODE:
        page = render_html(html.read_text(encoding="utf-8"), demo=True)

        @app.get("/", include_in_schema=False)
        async def index_demo():
            return HTMLResponse(page)

        return
    files_dir = html.with_name(html.stem + "_files")
    if files_dir.is_dir():
        app.mount("/" + files_dir.name, StaticFiles(directory=files_dir), name="ui_files")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(html, media_type="text/html")


app = create_app()
