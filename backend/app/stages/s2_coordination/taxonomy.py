"""② 协调 · 业务域词表（单一事实源）。

- 五域 / sub_domain 封闭枚举：对齐 old-version/prompts/intent_analysis.md「sub_domain 识别表」第一列；
- 强信号词：对齐 old-version `intent_validate.py` 与 prompt「五域强信号词表」；
- 触发词：prompt 触发词表的精简版（规则兜底用；LLM 路径整表渲染进 prompt，保证校验与提示同源）。

改动此处即同时改变：prompt 里给 LLM 的枚举、校验用的枚举、规则兜底的识别。
"""
from __future__ import annotations

import re
from typing import Iterable

# ── 封闭枚举 ───────────────────────────────────────────────────────────────
SUB_DOMAIN_ENUM: dict[str, set[str]] = {
    "transaction": {
        "行业运营", "买手产运", "商家数据", "商品数据", "营销工具",
        "C端场域", "用户增长", "交易治理", "体验履约", "结算", "支付&资金",
        "活动相关", "海外", "交易驾驶舱",
    },
    "community": {
        "增长", "推荐", "热点", "点点", "互动直播", "视频", "社交", "评论",
        "搜索", "发布", "生态", "专业号", "创作者", "社区横向", "垂类",
    },
    "commercialization": {
        "common", "品牌广告", "种草广告", "搜索广告", "客资广告", "闭环电商广告",
        "外循环广告", "薯条", "蒲公英", "行业赛道", "产品平台", "广告审核",
        "客户体验与实验", "横向基建/领域层", "横向基建/中间层",
        "产品平台/聚光", "产品平台/灵犀", "产品平台/Insight",
        "产品平台/Onedash", "产品平台/RedBI",
    },
    "resource_cost": {"GPU资源", "Token消耗"},
    "overseas": {"用户规模与留存", "内容消费与互动", "搜索", "翻译", "Push触达", "注册监控"},
}
DOMAIN_ENUM = set(SUB_DOMAIN_ENUM)

INTENT_TYPES = ("definition", "data_fetch", "discovery", "analysis", "other")
INTENT_SUB_TYPES = ("attribution", "trend", "prediction", "compare")
KB_INTENTS = ("discovery", "definition", "data_fetch_bi", "write_sql", "diagnose", "attribute", "interpret")

# 资产/知识缺省 sub_domain：domain 已识别但无触发词命中时的语义兜底（prompt：domain 已识别 sub_domain 不得为空）
DEFAULT_SUB_DOMAIN: dict[str, str] = {
    "transaction": "交易驾驶舱",       # 「整体/大盘/总量」且无明确子域
    "community": "社区横向",           # 「大盘 DAU / 全站日活」
    "commercialization": "common",    # 泛商业化兜底
    "resource_cost": "GPU资源",       # token 语境另行判定为 Token消耗
    "overseas": "用户规模与留存",
}

# ── 强信号词（domain 判定 / 冲突检测）────────────────────────────────────────
COMMERCIALIZATION_STRONG_TOKENS = (
    "千帆推广", "千帆推", "乘风", "闭环广告", "内循环广告", "外循环广告",
    "聚光", "灵犀", "蒲公英", "薯条", "信息流广告", "搜索广告", "种草广告",
    "广告消耗", "广告流水", "ROAS", "CPC", "CPA", "ADLoad",
    "客资广告", "线索广告", "品牌广告", "GD",
)
INNER_LOOP_TOKENS = ("千帆推广", "千帆推", "乘风", "闭环广告", "内循环", "闭环电商")
GMV_TOKENS = ("DGMV", "TGMV", "RGMV", "GMV")
TRANSACTION_STRONG_TOKENS = GMV_TOKENS + (
    "店播", "K播", "买手", "商家", "店铺",
    "电商Load", "综搜", "商搜", "关注页", "购物车", "市集", "MF双列",
    "电商直播", "直播带货", "商笔", "商业笔记",
)
SOCIAL_LIVE_TOKENS = ("打赏", "ACU", "主播营收", "REDGALA")
COMMUNITY_STRONG_TOKENS = (
    "DAU", "MAU", "互动直播", "打赏", "ACU", "连麦", "主播营收",
    "REDGALA", "笔记发布", "完播率", "VV", "DAP", "eDAP", "WAP", "MAP", "YAP",
)
ECOM_LIVE_TOKENS = ("电商直播", "店播", "K播", "直播带货", "直播GMV", "直播DGMV")
LIVE_FLOW_TOKENS = ("观播", "停留", "进房", "直播间流量", "直播间观看")
# 不含「卡」「算力卡」等过短易撞词；「薯条」「小红卡」不是资源成本词
RESOURCE_COST_STRONG_TOKENS = (
    "GPU", "H200", "token消耗", "token用量", "大模型用量", "大模型成本", "模型调用量", "算力",
    "卡型", "用卡", "供需缺口", "待分配卡", "供应计划", "云资源成本", "带宽成本", "资源成本",
)
OVERSEAS_UNIQUE_TOKENS = (
    "翻译DAU", "自动翻译", "主动翻译", "视频字幕", "直播字幕", "图片翻译", "翻译渗透率",
    "register_user_num", "户口国", "常驻国",
)
OVERSEAS_SIGNAL_TOKENS = ("Rednote", "海外版", "oversea")
OVERSEAS_CONTEXT_TOKENS = ("DAU", "DNU", "MAU", "Feed", "Push", "搜索", "注册", "留存", "新设备", "用户")
OVERSEAS_TRADE_TOKENS = GMV_TOKENS + ("跨境", "买手", "商家", "交易", "电商")

# ── sub_domain 触发词（精简版）──────────────────────────────────────────────
SUB_DOMAIN_TRIGGERS: dict[str, dict[str, tuple[str, ...]]] = {
    "transaction": {
        "行业运营": ("商笔", "商品笔记", "笔记GMV", "笔记带货", "店播", "好货市场", "行业运营", "商家运营", "KA", "一级类目", "二级类目"),
        "买手产运": ("买手", "K播", "MCN", "场次", "买手分层", "买手画像", "蓝链分销", "买手激励"),
        "商家数据": ("商家", "店铺", "商家分层", "新商", "商家入驻", "服务商", "商家经营"),
        "商品数据": ("商品", "SPU", "爆品", "类目", "比价", "选品", "货品"),
        "营销工具": ("营销工具", "优惠券", "发券", "核销", "品牌会员", "促销", "津贴"),
        "C端场域": ("商详页", "购物车", "加购", "订单页", "小红卡", "市集", "商搜", "场域", "MF双列", "观播", "直播间", "关注页", "流量归因"),
        "用户增长": ("核心人群", "新客", "LTV", "首购", "MAB", "核心用户数"),
        "交易治理": ("风控", "刷单", "违规", "判责", "虚假交易", "黑灰产", "差评治理"),
        "体验履约": ("客服", "售后", "差评率", "履约", "发货", "退款", "物流", "退货"),
        "结算": ("结算", "打款", "账期", "对账"),
        "支付&资金": ("支付", "支付成功率", "收银台", "花呗", "分期"),
        "活动相关": ("大促", "星光夜市", "星火计划", "活动专项", "凑单"),
        "海外": ("Rednote", "出海", "跨境", "海外GMV", "海外买手", "海外商家"),
        "交易驾驶舱": ("OneDash", "千帆", "鹰眼", "大盘", "交易大盘", "DGMV", "TGMV", "GPM"),
    },
    "community": {
        "增长": ("增长", "投放", "获客", "拉新", "召回", "拉活", "留存", "ROI", "CPA", "签到", "唤端", "渠道归因"),
        "推荐": ("推荐", "分发", "个性化", "Feed", "CTR", "曝光", "点击"),
        "热点": ("热点", "热搜", "热榜", "榜单"),
        "点点": ("点点", "dots"),
        "互动直播": ("互动直播", "主播", "开播", "打赏", "连麦", "ACU", "主播营收", "REDGALA"),
        "视频": ("视频", "播放", "完播", "短视频", "Vlog", "VV"),
        "社交": ("社交", "关注", "粉丝", "私信", "群聊", "双关"),
        "评论": ("评论", "发评", "看评", "评论率"),
        "搜索": ("搜索", "SAU", "社搜", "搜索率"),
        "发布": ("发布", "DAP", "eDAP", "YAP", "发布漏斗", "创作"),
        "生态": ("生态", "TnS", "VVR", "举报", "负反馈", "Spam"),
        "专业号": ("专业号", "个人售卖", "C2C", "KOS"),
        "创作者": ("创作者", "万粉", "KOL", "变现", "涨粉", "活跃作者"),
        "社区横向": ("eDAU", "有效设备DAU", "全站日活", "一级看板", "社区核心指标", "社区onedash"),
        "垂类": ("垂类", "CES", "旅游", "美食探店", "内容垂类"),
    },
    "commercialization": {
        "common": ("商业化", "竞价", "广告", "消耗"),
        "品牌广告": ("品牌广告", "GD", "开屏", "品专", "硬广", "CPM", "合约"),
        "种草广告": ("种草广告", "种草", "信息流", "小红星", "小红盟", "KFS", "KOS"),
        "搜索广告": ("搜索广告", "搜索竞价", "竞价词", "ADLoad", "搜索词", "PVR"),
        "客资广告": ("客资", "线索广告", "表单广告", "留资", "询盘", "私信广告"),
        "闭环电商广告": ("闭环电商", "电商闭环", "千帆推广", "乘风", "闭环广告", "ROAS", "商家投广"),
        "外循环广告": ("外循环", "外链", "CID", "应用下载", "应用唤起", "游戏广告"),
        "薯条": ("薯条", "Chips", "笔记推广", "笔记加热"),
        "蒲公英": ("蒲公英", "商单", "博主合作", "达人营销", "众测"),
        "行业赛道": ("行业赛道", "行业投放", "赛道洞察", "AM", "在投客户"),
        "产品平台": ("产品平台", "广告平台", "ARPU", "大商业"),
        "广告审核": ("广审", "CQC", "机审", "人审", "质检", "申诉"),
        "客户体验与实验": ("客户体验", "广告体验", "变现率", "实验", "场域效率", "DAC", "MAC"),
        "横向基建/领域层": ("领域层", "领域宽表", "商业数仓", "创意宽表"),
        "横向基建/中间层": ("中间层", "DWD", "DWS", "流量域", "转化域", "收入域"),
        "产品平台/聚光": ("聚光", "Aurora"),
        "产品平台/灵犀": ("灵犀", "一方数据", "人群圈选"),
        "产品平台/Insight": ("Insight",),
        "产品平台/Onedash": ("Onedash", "AdsLoad", "经营看板"),
        "产品平台/RedBI": ("RedBI", "目标追踪"),
    },
    "resource_cost": {
        "GPU资源": ("GPU", "H200", "算力", "卡型", "用卡", "缺口", "到货", "供应计划", "利用率", "PAI", "QS"),
        "Token消耗": ("token", "大模型用量", "大模型成本", "模型调用量", "问一问"),
    },
    "overseas": {
        "用户规模与留存": ("DNU", "EDAU", "海外DAU", "新设备", "次留", "LT7", "LT30", "DAU", "MAU"),
        "内容消费与互动": ("Feed", "曝光", "阅读", "互动", "点赞", "CTR", "时长"),
        "搜索": ("搜索", "search", "search_session_num"),
        "翻译": ("翻译", "translate", "字幕", "图片翻译"),
        "Push触达": ("push", "推送", "送达", "可触达率", "Firebase"),
        "注册监控": ("注册", "register", "register_user_num", "户口国"),
    },
}

# 商业化知识库粗召 kb_intents 触发词（仅 domain=commercialization 使用）
KB_INTENT_TRIGGERS: dict[str, tuple[str, ...]] = {
    "discovery": ("哪个数据集", "哪里能看", "哪里看", "有哪些数据", "有什么数据", "推荐数据集", "找数据", "数据来源", "哪个看板", "怎么找"),
    "definition": ("是什么", "什么是", "什么意思", "怎么算", "如何计算", "怎么定义", "口径", "含义", "指的是", "区别"),
    "data_fetch_bi": ("查", "拉取", "导出", "统计", "走势", "明细", "多少", "近7天", "近N天", "消耗", "流水", "趋势", "排行"),
    "write_sql": ("写SQL", "SQL怎么写", "报错", "优化SQL", "建表", "DDL", "JOIN", "分区怎么筛", "Hive底表"),
    "diagnose": ("对不上", "数据偏大", "数据偏小", "数据错了", "数据重复", "数值异常", "不一致", "跟预期不符", "数字不对"),
    "attribute": ("为什么", "原因", "归因", "影响因素", "对比", "预测", "下降", "波动", "跌了", "涨了", "拆解"),
    "interpret": ("看板逻辑", "帮我确认", "是否正确", "这个指标什么意思", "这个数对吗", "代表什么", "正常吗"),
}

# intent_type 触发词（按 discovery → definition → analysis → data_fetch 顺序判定）
DISCOVERY_TRIGGERS = ("哪个数据集", "哪里能看", "哪里看", "有哪些", "有什么数据", "找数据", "数据来源", "哪个看板", "推荐数据集")
DEFINITION_TRIGGERS = ("是什么", "什么是", "什么意思", "怎么算", "如何计算", "怎么定义", "定义", "口径", "含义", "指的是", "区别")
ANALYSIS_TRIGGERS = ("为什么", "原因", "归因", "影响", "对比", "预测", "下降", "上涨", "跌了", "涨了", "波动", "拆解", "拆下", "拆一下")
DATA_FETCH_TRIGGERS = ("查", "取", "拉", "多少", "走势", "趋势", "明细", "列出", "统计", "消耗", "流水")


# ── 匹配工具 ───────────────────────────────────────────────────────────────
_CJK = "\u4e00-\u9fff"
_SQUASH = re.compile(rf"(?<=[{_CJK}])\s+|\s+(?=[{_CJK}])")


def squash(text: str) -> str:
    """去掉紧邻中文字符的空白：'token 消耗' → 'token消耗'、'K 播' → 'K播'（用户常在中英文间加空格）。"""
    return _SQUASH.sub("", text or "")


def _contains(norm_lower: str, token: str) -> bool:
    t = squash(token).lower()
    if t.isascii():
        return re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", norm_lower) is not None
    return t in norm_lower


def find_hits(text: str, tokens: Iterable[str]) -> list[str]:
    """命中的词（大小写不敏感、去重）。纯 ASCII 词要求单词边界，避免 GD 命中 GDP、GMV 命中 DGMV 内部。"""
    norm = squash(text).lower()
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens:
        k = tok.lower()
        if k in seen:
            continue
        if _contains(norm, tok):
            seen.add(k)
            out.append(tok)
    return out


def has(text: str, *tokens: str) -> bool:
    return bool(find_hits(text, tokens))


def first_pos(text: str, tokens: Iterable[str]) -> int:
    """tokens 中最早出现位置（无命中返回很大的数），用于多域平票时取先出现者。"""
    norm = squash(text).lower()
    pos = [norm.find(squash(t).lower()) for t in tokens]
    pos = [p for p in pos if p >= 0]
    return min(pos) if pos else 10**9


# ── 渲染进 prompt（词表与校验同源）──────────────────────────────────────────
def render_taxonomy() -> str:
    lines: list[str] = []
    for domain, subs in SUB_DOMAIN_TRIGGERS.items():
        lines.append(f"### domain = `{domain}`")
        for sub, trig in subs.items():
            lines.append(f"- {sub}：{'、'.join(trig)}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_signals() -> str:
    rows = {
        "transaction": TRANSACTION_STRONG_TOKENS,
        "commercialization": COMMERCIALIZATION_STRONG_TOKENS,
        "community": COMMUNITY_STRONG_TOKENS,
        "resource_cost": RESOURCE_COST_STRONG_TOKENS,
        "overseas": OVERSEAS_UNIQUE_TOKENS + OVERSEAS_SIGNAL_TOKENS,
    }
    return "\n".join(f"- {d}：{' / '.join(toks)}" for d, toks in rows.items())
