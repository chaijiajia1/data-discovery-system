"""外网演示用的假数据（只在 DEMO_MODE 下装配，见 deps.build_default_deps；单测用的默认假数据仍是 fakes.py）。

围绕页面上的三条示例问句编排，让每个面板都有内容：
  Q1 Last 7 days DAU and CTR by platform   L2 完整召回：DAU 有「应用层 vs 明细层」同覆盖度平局（歧义）、CTR 有「天 / 小时」两张表平局
  Q2 Last 30 days CTR by platform          L2 完整召回：候选含月表（粒度太粗被丢弃）、明细日志（无 CTR 字段）、未达标的图表
  Q3 Last 7 days DAU and GMV by channel    L0 直接执行（知识已覆盖全部子目标，跳过 Phase 3 工作流与 Phase 4 召回）：月表被丢弃、财务审计表无权限 → 降级到备选
所有资产、表名、指标口径均为示意占位，不对应任何真实系统。

注意：一个知识块的 terms 只放一个概念——多个概念会被 ⑤ 当成同义词桥接（如 GMV ↔ CHANNEL）。
注意：平台 / 渠道维度刻意不在注入知识里绑定资产字段（Q1、Q2 的 platform 要保持「未覆盖」，才会进入 L2 完整召回）。
"""
from __future__ import annotations

from ..schemas.knowledge import KnowledgeItem, KnowledgeKind, KnowledgeOrigin
from ..schemas.recall import AssetMeta, CandidateAsset
from ..text_utils import estimate_tokens


def _asset(aid, name, grain, fields, *, main, heat, refs, retrieval_score=0.0, pks=(), typ="table", scope="public", latest="2026-11-23", desc=None, **meta):
    return CandidateAsset(
        asset_id=aid, name=name, asset_type=typ, retrieval_score=retrieval_score,
        meta=AssetMeta(grain=grain, partition_field="dt", partition_latest=latest, fields=list(fields), primary_keys=list(pks), main_table=main,
                       heat_30d=heat, ref_count=refs, permission_scope=scope, description=desc, **meta),
    )


# (资产, 检索关键词)。关键词只放指标名：维度词（platform / channel）不进关键词，否则每个指标的检索都会带出无关资产。
DEMO_ASSETS: list[tuple[CandidateAsset, list[str]]] = [
    # 合成检索分反映目录中的可发现性：常用 DAU 表、常引用的月报和财务表容易召回。
    # 选择阶段另行评估统计粒度和访问权限。
    # ── DAU ──
    (_asset("app.user_active_d", "App-layer user activity (daily)", "day", ["dt", "platform", "channel", "dau"], main="app_user_active_d", retrieval_score=0.86, heat=9000, refs=150,
            pks=["dt", "platform", "channel"], desc="Certified daily active users by platform and channel"), ["DAU"]),
    (_asset("dws.user_active_d", "User daily-active detail", "day", ["dt", "platform", "channel", "dau"], main="dws_user_active_d", heat=6000, refs=80,
            pks=["dt", "platform", "channel"]), ["DAU"]),
    (_asset("dws.user_active_valid_d", "Valid daily-active detail", "day", ["dt", "platform", "channel", "edau"], main="dws_user_active_valid_d", heat=1200, refs=20,
            pks=["dt", "platform", "channel"]), ["DAU", "EDAU"]),
    (_asset("dws.user_active_m", "User monthly-active summary", "month", ["dt", "platform", "channel", "dau"], main="dws_user_active_m", heat=2500, refs=30,
            pks=["dt", "platform", "channel"]), ["DAU"]),
    (_asset("chart.growth_core_dau", "Growth core board: DAU", "day", [], main=None, heat=4000, refs=60, typ="chart", is_strict=False,
            metric_ids=["DAU"], supported_dims=["platform", "channel"]), ["DAU"]),
    # ── CTR ──
    (_asset("dws.ad_ctr_d", "Ad CTR / conversion daily table", "day", ["dt", "platform", "ctr", "cvr"], main="dws_ad_ctr_d", heat=5200, refs=70,
            pks=["dt", "platform"]), ["CTR"]),
    (_asset("app.ad_perf_board_d", "App-layer ad performance board (daily)", "day", ["dt", "platform", "ctr", "impressions", "clicks"], main="app_ad_perf_board_d", heat=7600, refs=110,
            pks=["dt", "platform"], desc="Certified ad performance board"), ["CTR"]),
    (_asset("dws.ad_ctr_h", "Ad CTR hourly table", "hour", ["dt", "hour", "platform", "ctr"], main="dws_ad_ctr_h", heat=900, refs=12,
            pks=["dt", "hour", "platform"]), ["CTR"]),
    (_asset("dws.ad_ctr_m", "Ad CTR monthly table", "month", ["dt", "platform", "ctr"], main="dws_ad_ctr_m", retrieval_score=0.97, heat=1800, refs=25,
            pks=["dt", "platform"]), ["CTR"]),
    (_asset("dwd.ad_click_log", "Ad click detail log", "day", ["dt", "ad_id", "platform", "is_click"], main="dwd_ad_click_log", heat=400, refs=8), ["CTR"]),
    (_asset("chart.ad_dashboard_ctr", "Ad dashboard: CTR trend", "day", [], main=None, heat=3000, refs=40, typ="chart", is_strict=False,
            metric_ids=["CTR"], supported_dims=["platform"]), ["CTR"]),
    # ── GMV ──
    (_asset("app.trade_overview_d", "App-layer trade overview (daily)", "day", ["dt", "channel", "gmv", "orders"], main="app_trade_overview_d", heat=7000, refs=130,
            pks=["dt", "channel"], desc="Certified trade overview"), ["GMV"]),
    (_asset("dws.trade_gmv_d", "Trade GMV daily detail", "day", ["dt", "channel", "gmv"], main="dws_trade_gmv_d", heat=5200, refs=90, pks=["dt", "channel"]), ["GMV"]),
    (_asset("dws.trade_gmv_m", "Trade GMV monthly summary", "month", ["dt", "channel", "gmv"], main="dws_trade_gmv_m", heat=2100, refs=35, pks=["dt", "channel"]), ["GMV"]),
    (_asset("ads.finance_gmv_audit", "Finance GMV audit table (restricted)", "day", ["dt", "channel", "gmv", "gmv_net"], main="ads_finance_gmv_audit", retrieval_score=0.96, heat=600, refs=5,
            pks=["dt", "channel"], scope="finance"), ["GMV"]),
]


def _k(i, kind, title, content, terms, assets, *, consistency="none"):
    return KnowledgeItem(id=i, kind=kind, title=title, content=content, origin=KnowledgeOrigin.RETRIEVAL, terms=terms, asset_ids=list(assets),
                         token_cost=estimate_tokens(content), consistency=consistency)


K = KnowledgeKind
# 注意：没有任何 DIMENSION / ASSET_META 块同时提到 platform 且绑定资产（见模块说明）。
DEMO_KNOWLEDGE: list[KnowledgeItem] = [
    _k("k_dau", K.METRIC_DEF, "DAU", "Daily active users, deduplicated by user_id; caliber = any app launch that day",
       ["DAU"], ["app.user_active_d", "dws.user_active_d", "dws.user_active_m"]),
    _k("k_edau", K.METRIC_DEF, "eDAU", "Valid daily active users: DAU after removing cheating devices; caliber = eDAU",
       ["EDAU"], ["dws.user_active_valid_d"]),
    _k("k_dau_doc", K.BUSINESS_TERM, "DAU", "Daily active users, deduplicated by user_id; any app launch that day counts",
       ["DAU"], []),                                                                                  # 与 k_dau 几乎重复：被冗余扣分、最终丢弃
    _k("k_dau_meta", K.ASSET_META, "app.user_active_d metadata", "Daily partition dt, grain = day, certified app-layer table",
       ["DAU"], ["app.user_active_d"]),
    _k("k_ctr", K.METRIC_DEF, "CTR", "Click-through rate = clicks / impressions",
       ["CTR"], ["dws.ad_ctr_m"]),
    _k("k_ctr_meta", K.ASSET_META, "Monthly CTR archive", "Frequently cited in executive reports; only calendar-month totals are retained",
       ["CTR"], ["dws.ad_ctr_m"]),
    _k("k_gmv", K.METRIC_DEF, "GMV", "Gross merchandise value: paid orders, refunds not deducted",
       ["GMV"], ["app.trade_overview_d", "dws.trade_gmv_d", "dws.trade_gmv_m", "ads.finance_gmv_audit"]),
    _k("k_gmv_meta", K.ASSET_META, "dws.trade_gmv_d metadata", "Daily partition dt, grain = day, fields channel / gmv",
       ["GMV"], ["dws.trade_gmv_d"]),
    _k("k_channel", K.DIMENSION, "Channel", "The channel field is the acquisition channel of the user",
       ["CHANNEL"], []),
]
