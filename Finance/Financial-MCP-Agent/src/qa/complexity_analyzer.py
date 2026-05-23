"""
高灵敏复杂度分析器 — 三层识别架构

Layer 1: 规则触发器（硬判断，速度快）
Layer 2: 加权打分模型（未命中硬触发时使用）
Layer 3: 运行时升级器（Phase 2 实现，此处保留接口）
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ComplexityResult:
    """复杂度分析结果"""
    level: str            # "L1" | "L2" | "L3" | "L4"
    score: int            # 0-100
    triggers: List[str]   # 触发的规则列表
    score_detail: dict    # 各维度得分明细
    need_clarify: bool    # 是否需要澄清
    recommended_model: str  # "mimo-v2.5" | "mimo-v2.5-pro"
    recommended_thinking: bool
    recommended_react: bool
    recommended_template: str    # "quick" | "standard" | "deep"


# ── Layer 1: 规则触发器 ──────────────────────────

HARD_TRIGGERS_L4 = [
    # 跨标的比较（需明确涉及两个实体）
    r"(?:比较|对比|vs|VS|versus|优劣|差异|区别).*(?:和|与|跟|及)",
    r"(?:和|与|跟).*(?:比|比较|对比|区别|差异)",
    # 因果归因
    r"为什么", r"原因", r"驱动", r"归因", r"本质", r"背后.*逻辑",
    # 情景推演
    r"如果.*会|假设.*会|情景.*分析",
    r"预期.*会|展望.*前景",
    # 深度判断
    r"还能不能", r"值不值得", r"是不是机会", r"是不是陷阱", r"还能拿",
    r"要不要", r"该不该",
    # 综合报告
    r"全面分析", r"深度分析", r"展开讲讲", r"写.*报告",
    r"详细.*分析", r"系统.*分析",
    # 多标的对比
    r"(?:宁德|比亚迪|茅台|五粮液|平安|美的|格力).*(?:宁德|比亚迪|茅台|五粮液|平安|美的|格力)",
    # 筛选排序
    r"筛选.*股|排序.*股|打分.*股|优选", r"推荐.*股",
    # 策略判断
    r"交易策略|操作策略|配置.*建议|仓位.*建议",
]

HARD_TRIGGERS_L3 = [
    r"估值.*合理", r"贵不贵", r"便不便宜", r"高不高",
    r"走势.*强", r"走势.*弱", r"趋势.*判断",
    r"板块.*轮动", r"行业.*景气", r"行业.*前景",
    r"资金.*流向", r"主力.*动向", r"北向.*资金",
]


def _check_hard_triggers(question: str) -> tuple:
    """Layer 1: 规则触发器，返回 (level, triggers)"""
    triggers = []

    for pattern in HARD_TRIGGERS_L4:
        if re.search(pattern, question):
            triggers.append(f"L4硬触发: {pattern}")

    for pattern in HARD_TRIGGERS_L3:
        if re.search(pattern, question):
            triggers.append(f"L3硬触发: {pattern}")

    if triggers:
        has_l4 = any("L4硬触发" in t for t in triggers)
        return ("L4" if has_l4 else "L3", triggers)
    return (None, [])


# ── L0 检测：无需数据的常识性问题 ─────────────────

NON_FINANCIAL_PATTERNS = [
    r"^你是谁", r"^你好", r"^嗨", r"^hello", r"^hi\b",
    r"^\d+[\+\-\*\/]\d+", r"^\d+\s*[＋－×÷]\s*\d+",  # 数学计算
    r"地球.*形状|太阳.*(大|温度|距离)|月亮.*(大|距离)",
    r"什么.*动物|什么.*植物|什么.*颜色",
    r"天气.*怎么|今天.*天气",
    r"翻译|翻译.*英文|英文.*怎么说",
    r"讲.*笑话|说.*笑话|笑话",
    r"几点了|现在.*时间|今天.*几号|今天.*星期",
]

NO_DATA_NEEDED_PATTERNS = [
    r"^你是谁", r"你能.*什么", r"你.*功能", r"你.*能力",
    r"怎么.*用|如何.*使用", r"帮助|help",
    r"谢谢|感谢|多谢|再见|拜拜|bye",
    r"什么是(?:股票|PE|PB|ROE|估值|K线|均线)",
]


INVESTMENT_DECISION_PATTERNS = [
    r"该不该买", r"该不该卖", r"值得买", r"值得卖", r"值得投资",
    r"能.*买", r"能.*卖", r"要不要.*买", r"要不要.*卖",
    r"建议.*买", r"建议.*卖", r"能不能.*买", r"能不能.*卖",
    r"现在.*买|现在.*卖|现在.*入场|现在.*进场",
]


def _needs_clarify_for_investment(question: str) -> bool:
    """投资决策类问题但无股票代码时，需要反问澄清"""
    for pat in INVESTMENT_DECISION_PATTERNS:
        if re.search(pat, question):
            # 检查是否包含股票代码
            codes = re.findall(r'(?<!\d)(?:[36]\d{5}|0\d{5}|8\d{5})(?!\d)', question)
            if not codes:
                return True
    return False


def _is_l0_question(question: str) -> bool:
    """检测是否为L0问题（无需调用数据工具）"""
    q = question.strip().lower()
    # 非财经类常识问题
    for pat in NON_FINANCIAL_PATTERNS:
        if re.search(pat, q):
            return True
    # 无数据需求的问题
    for pat in NO_DATA_NEEDED_PATTERNS:
        if re.search(pat, q):
            return True
    # 极短问题（≤5字且无股票代码/行业关键词）
    if len(q) <= 5:
        has_finance_kw = any(
            kw in q for kw in ["股", "涨", "跌", "PE", "PB", "估值", "财报",
                               "利润", "营收", "行业", "板块", "基金", "ETF"]
        )
        if not has_finance_kw:
            return True
    return False


# ── Layer 2: 加权打分模型 ──────────────────────────

def _score_question(question: str) -> ComplexityResult:
    """Layer 2: 对未命中硬触发的问题做加权打分"""
    score = 0
    detail = {}

    # 1. 主体数量 (0-15)
    # 只匹配疑似股票代码（6开头=沪市, 0/3开头=深市, 688=科创板, 8开头=北交所）
    codes = re.findall(r'(?<!\d)(?:[36]\d{5}|0\d{5}|8\d{5})(?!\d)', question)
    stock_count = len(set(codes))  # 去重
    names = re.findall(r'[一-鿿]{2,4}(?:股票|股份|集团|银行|证券|保险|科技|医药|汽车|能源)', question)
    entity_count = max(stock_count, len(names) if names else 1)
    if entity_count >= 3:
        detail["主体数量"] = 15
    elif entity_count == 2:
        detail["主体数量"] = 10
    elif entity_count == 1:
        detail["主体数量"] = 3
    else:
        detail["主体数量"] = 0
    score += detail["主体数量"]

    # 2. 时间跨度 (0-10)
    time_keywords_long = ["年", "长期", "三年", "五年", "历史", "历年", "跨周期"]
    time_keywords_mid = ["季度", "月", "中期", "半年", "今年以来"]
    time_keywords_short = ["周", "最近", "近期", "今天", "昨天", "明天", "本周", "本月"]
    if any(kw in question for kw in time_keywords_long):
        detail["时间跨度"] = 10
    elif any(kw in question for kw in time_keywords_mid):
        detail["时间跨度"] = 6
    elif any(kw in question for kw in time_keywords_short):
        detail["时间跨度"] = 3
    else:
        detail["时间跨度"] = 1
    score += detail["时间跨度"]

    # 3. 分析维度数量 (0-20)
    dimension_keywords = {
        "行情": ["价格", "涨", "跌", "走势", "行情", "趋势", "K线", "均线"],
        "估值": ["PE", "PB", "市盈", "市净", "估值", "贵", "便宜", "分位"],
        "财务": ["ROE", "利润", "收入", "毛利", "现金", "负债", "财报", "业绩", "盈利"],
        "资金": ["资金", "主力", "北向", "融资", "融券", "流入", "流出", "成交"],
        "行业": ["行业", "板块", "赛道", "同行", "竞品", "龙头"],
        "事件": ["新闻", "公告", "分红", "回购", "减持", "增持", "业绩预告"],
    }
    dims_found = 0
    for dim, kws in dimension_keywords.items():
        if any(kw in question for kw in kws):
            dims_found += 1
    detail["分析维度"] = min(dims_found * 5, 20)
    score += detail["分析维度"]

    # 4. 推理深度 (0-20)
    deep_patterns = [r"为什么", r"怎么.*变化", r"影响", r"判断", r"预测", r"推测"]
    compare_patterns = [r"对比", r"比较", r"区别", r"vs", r"优劣"]
    desc_patterns = [r"是多少", r"什么.*是", r"查询", r"看看", r"了解"]
    if any(re.search(p, question) for p in deep_patterns):
        detail["推理深度"] = 18
    elif any(re.search(p, question) for p in compare_patterns):
        detail["推理深度"] = 12
    elif any(re.search(p, question) for p in desc_patterns):
        detail["推理深度"] = 5
    else:
        detail["推理深度"] = 3
    score += detail["推理深度"]

    # 5. 计算复杂度 (0-15)
    calc_keywords = ["分位", "同比", "环比", "TTM", "相对强弱", "超额", "回撤", "波动"]
    calc_count = sum(1 for kw in calc_keywords if kw in question)
    detail["计算复杂度"] = min(calc_count * 5, 15)
    score += detail["计算复杂度"]

    # 6. 歧义程度 (0-10)
    ambiguity = 0
    if re.search(r'(?:它|这个|那个|这只|刚才|上次).*(?:股票|公司|股)', question) and not codes:
        ambiguity = 8
    elif not codes and not names and len(question) < 10:
        ambiguity = 5
    detail["歧义程度"] = ambiguity
    score += ambiguity

    # 7. 输出要求 (0-10)
    output_keywords = ["报告", "详细", "全面", "深度", "展开", "分析一下"]
    if any(kw in question for kw in output_keywords):
        detail["输出要求"] = 8
    else:
        detail["输出要求"] = 2
    score += detail["输出要求"]

    # 分级
    if score <= 24:
        level = "L1"
    elif score <= 49:
        level = "L2"
    elif score <= 69:
        level = "L3"
    else:
        level = "L4"

    return ComplexityResult(
        level=level,
        score=score,
        triggers=[f"评分={score}"],
        score_detail=detail,
        need_clarify=(detail.get("歧义程度", 0) >= 8),
        recommended_model="mimo-v2.5-pro" if level in ("L3", "L4") else "mimo-v2.5",
        recommended_thinking=(level == "L4"),
        recommended_react=(level in ("L3", "L4")),
        recommended_template="quick" if level == "L1" else ("standard" if level == "L2" else "deep"),
    )


# ── 公共接口 ──────────────────────────────────────

def analyze_complexity(question: str, history_depth: int = 0) -> ComplexityResult:
    """
    分析问题的复杂度。

    Args:
        question: 用户问题
        history_depth: 多轮对话深度（追问链中自动提升复杂度）

    Returns:
        ComplexityResult
    """
    # L0 检测：无需数据的常识/闲聊问题
    if _is_l0_question(question):
        return ComplexityResult(
            level="L0", score=0,
            triggers=["L0: 无需数据"],
            score_detail={"L0检测": "常识/闲聊问题"},
            need_clarify=False,
            recommended_model="mimo-v2.5",
            recommended_thinking=False,
            recommended_react=False,
            recommended_template="l0",
        )

    # 投资决策类问题无股票代码 → 需要澄清
    if _needs_clarify_for_investment(question):
        return ComplexityResult(
            level="L1", score=0,
            triggers=["需要澄清: 投资决策需指定标的"],
            score_detail={"澄清": "请提供股票代码或名称"},
            need_clarify=True,
            recommended_model="mimo-v2.5",
            recommended_thinking=False,
            recommended_react=False,
            recommended_template="quick",
        )

    # Layer 1: 硬触发规则
    hard_level, triggers = _check_hard_triggers(question)

    if hard_level:
        score = 75 if hard_level == "L4" else 55
        # 追问链提升（硬触发也适用）
        current_idx = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
        if history_depth >= 3:
            bump = min(history_depth, 3)
            new_idx = min(current_idx.get(hard_level, 2) + bump, 3)
            levels = ["L1", "L2", "L3", "L4"]
            if new_idx > current_idx.get(hard_level, 2):
                hard_level = levels[new_idx]
                triggers.append(f"追问链提升: depth={history_depth} -> {hard_level}")

        return ComplexityResult(
            level=hard_level,
            score=score,
            triggers=triggers,
            score_detail={"硬触发": hard_level},
            need_clarify=False,
            recommended_model="mimo-v2.5-pro" if hard_level in ("L3", "L4") else "mimo-v2.5",
            recommended_thinking=(hard_level == "L4"),
            recommended_react=(hard_level in ("L3", "L4")),
            recommended_template="deep" if hard_level == "L4" else "standard",
        )

    # Layer 2: 加权打分
    result = _score_question(question)

    # 追问链自动提升
    if history_depth >= 3:
        bump = min(history_depth, 3)
        current_idx = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
        levels = ["L1", "L2", "L3", "L4"]
        new_idx = min(current_idx.get(result.level, 0) + bump, 3)
        if new_idx > current_idx.get(result.level, 0):
            result.level = levels[new_idx]
            result.triggers.append(f"追问链提升: depth={history_depth} -> {result.level}")

    # 用户显式要求深度分析
    if any(kw in question for kw in ["详细", "深度", "全面", "展开"]):
        current_idx = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
        if current_idx.get(result.level, 0) < 2:
            result.level = "L3"
            result.triggers.append("用户显式要求深度分析 -> L3")

    return result


# ── Layer 3: 运行时升级器 ───────────────────────

RUNTIME_UPGRADE_TRIGGERS = {
    "data_missing_critical": {
        "description": "关键数据域缺失（超过40%的工具调用失败）",
        "upgrade_action": "upgrade_model",
    },
    "evidence_conflict": {
        "description": "证据之间出现显著矛盾",
        "upgrade_action": "upgrade_to_pro",
    },
    "multi_domain_required": {
        "description": "实际需要超过4个数据域",
        "upgrade_action": "upgrade_to_pro_with_thinking",
    },
    "tool_calls_excessive": {
        "description": "工具调用次数超过预期2倍",
        "upgrade_action": "upgrade_model",
    },
    "answer_low_confidence": {
        "description": "首轮回答置信度不足",
        "upgrade_action": "retry_with_thinking",
    },
}


def try_runtime_upgrade(
    current: ComplexityResult,
    tool_success_rate: float,
    evidence_missing_count: int,
    contradictory_signals: bool,
    actual_domain_count: int,
) -> ComplexityResult:
    """
    Layer 3: 运行时升级器。

    在数据获取后发现实际情况比预判更复杂时，升级模型和策略。

    Args:
        current: 初始复杂度结果
        tool_success_rate: 工具调用成功率 (0.0~1.0)
        evidence_missing_count: 缺失数据数
        contradictory_signals: 是否有矛盾证据
        actual_domain_count: 实际涉及的数据域数

    Returns:
        可能升级后的 ComplexityResult
    """
    result = ComplexityResult(
        level=current.level,
        score=current.score,
        triggers=list(current.triggers),
        score_detail=dict(current.score_detail),
        need_clarify=current.need_clarify,
        recommended_model=current.recommended_model,
        recommended_thinking=current.recommended_thinking,
        recommended_react=current.recommended_react,
        recommended_template=current.recommended_template,
    )

    level_idx = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
    current_idx = level_idx.get(result.level, 0)

    # 触发1: 工具成功率过低 → 升级模型
    if tool_success_rate < 0.6 and evidence_missing_count > 3:
        if current_idx < 2:
            result.level = "L3"
            result.triggers.append("运行时升级: 工具成功率过低 → L3")
        result.recommended_model = "mimo-v2.5-pro"

    # 触发2: 证据矛盾 → 升级到 Pro + reasoning
    if contradictory_signals:
        result.recommended_model = "mimo-v2.5-pro"
        result.recommended_thinking = True
        if current_idx < 2:
            result.level = "L3"
        result.triggers.append("运行时升级: 证据矛盾 → Pro + thinking")

    # 触发3: 实际数据域超出预期 → 升级
    warranted = "L4" if actual_domain_count >= 5 else ("L3" if actual_domain_count >= 4 else None)
    if warranted and level_idx.get(warranted, 0) > current_idx:
        result.level = warranted
        result.recommended_model = "mimo-v2.5-pro"
        result.recommended_thinking = (warranted == "L4")
        result.recommended_react = (warranted == "L4")
        result.triggers.append(f"运行时升级: 实际{actual_domain_count}个数据域 → {warranted}")

    # 触发4: L2+但关键数据缺失 → 开启 thinking 弥补
    if evidence_missing_count >= 3 and current_idx >= 1:
        result.recommended_thinking = True
        result.triggers.append("运行时升级: 数据缺失较多 → 开启thinking")

    # 触发5: L4但thinking未开 → 修正
    if result.level == "L4" and not result.recommended_thinking:
        result.recommended_thinking = True
        result.triggers.append("运行时修正: L4必须开启thinking")

    # 触发6: L4但ReAct未开 → 修正
    if result.level == "L4" and not result.recommended_react:
        result.recommended_react = True
        result.triggers.append("运行时修正: L4必须启用ReAct")

    # 触发7: Pro被推荐但model未升级 → 修正
    if result.recommended_model == "mimo-v2.5-pro" and current_idx >= 2:
        if not result.recommended_thinking and result.level == "L4":
            result.recommended_thinking = True

    return result

