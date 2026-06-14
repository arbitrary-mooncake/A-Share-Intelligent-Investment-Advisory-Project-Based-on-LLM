"""
Fund Merge Node: Non-LLM data transformation node for fund analysis pipeline.

Reads 7 fund analysis agent outputs from state.data and normalizes them into a
unified fund_analysis_package for downstream report/scoring agents.

This is a pure data transformation node — no LLM calls are made.
Extraction relies on regex patterns and keyword-based section parsing
from unstructured agent output text.
"""

import re
from typing import Dict, Any, List, Optional

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_KEYS = [
    "fund_product_doc",
    "fund_perf_risk",
    "fund_holdings",
    "fund_manager",
    "fund_benchmark",
    "fund_fee",
    "fund_event",
]

AGENT_LABELS = {
    "fund_product_doc": "产品文档分析",
    "fund_perf_risk": "业绩风险分析",
    "fund_holdings": "持仓结构分析",
    "fund_manager": "基金经理分析",
    "fund_benchmark": "基准风格分析",
    "fund_fee": "费率流动性分析",
    "fund_event": "事件风险分析",
}

DEFAULT_SCORE = 50
DEFAULT_CONFIDENCE = 0.70


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_score(text: str) -> int:
    """Extract a numeric score (0-100) from unstructured agent output text.

    Attempts multiple regex patterns in priority order. Returns DEFAULT_SCORE
    if no score can be reliably extracted.
    """
    if not text or not isinstance(text, str):
        return DEFAULT_SCORE

    # Priority-ordered patterns — most specific first
    patterns = [
        # "综合评分: 78/100" or "综合评分：78（满分100）"
        r'(?:综合)?(?:评分|得分|score)[:：]\s*(\d{1,3})\s*(?:/100|分|（满分\d+）)?',
        # "得分: 78分"
        r'(?:得分|评分|打分)[:：]\s*(\d{1,3})\s*分',
        # "Score: 78"
        r'(?:score|rating)[:：]\s*(\d{1,3})',
        # "78 分" or "78/100" near evaluation keywords
        r'(?:总体|综合|整体).*?(\d{1,3})\s*(?:分|/100)',
        # Bare number near "分" in last 500 chars (summary score)
        r'(\d{1,3})\s*分\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            return min(max(score, 0), 100)

    return DEFAULT_SCORE


def _extract_confidence(text: str) -> float:
    """Extract confidence score (0.0-1.0) from agent output text."""
    if not text or not isinstance(text, str):
        return DEFAULT_CONFIDENCE

    patterns = [
        r'(?:置信度|confidence)[:：]\s*(0?\.\d+|[01](?:\.\d+)?)',
        r'(?:可信度|确定性)[:：]\s*(0?\.\d+|[01](?:\.\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                conf = float(match.group(1))
                return min(max(conf, 0.0), 1.0)
            except ValueError:
                pass
    return DEFAULT_CONFIDENCE


def _extract_bullet_list(
    text: str,
    section_keywords: List[str],
    stop_keywords: Optional[List[str]] = None,
) -> List[str]:
    """Extract bullet-point items from a section of unstructured text.

    Scans for a section header matching a keyword, then collects all
    subsequent bullet/numbered items until the next section header
    (matched by stop_keywords) or end of text.

    Args:
        text: The full agent output text.
        section_keywords: Lines containing any of these trigger section start.
        stop_keywords: Lines containing any of these end the section.
                       Defaults to common section boundary words.

    Returns:
        List of extracted bullet string items (no bullet markers), up to 5.
    """
    if not text or not isinstance(text, str):
        return []

    if stop_keywords is None:
        stop_keywords = [
            "风险", "劣势", "不足", "隐患", "缺点", "警示",
            "risk", "weakness", "threat", "缺点",
            "评级", "结论", "总结", "建议", "评分",
            "小标题", "---", "===", "===",
        ]

    items: List[str] = []
    in_section = False
    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for section start
        if not in_section:
            if any(kw in stripped for kw in section_keywords):
                in_section = True
                # If the header line itself contains a bullet item after the keyword,
                # try to capture it.
                # e.g. "优势: 低费率" → extract after colon
                for sep in ["：", ":", "—", "- "]:
                    if sep in stripped:
                        remainder = stripped.split(sep, 1)[1].strip()
                        if remainder and len(remainder) > 2:
                            items.append(remainder)
                continue

        if in_section:
            # Stop if we hit the next section
            if any(
                kw in stripped
                for kw in stop_keywords
                if kw not in section_keywords
            ):
                in_section = False
                continue

            # Match bullet/numbered lines
            bullet_match = re.match(
                r'^[-*•●○✓✔\d]+\s*[\.\)、．\s]\s*(.+)',
                stripped,
            )
            if bullet_match:
                item_text = bullet_match.group(1).strip()
                if item_text and len(item_text) > 1:
                    items.append(item_text)
                continue

            # Also catch lines that look like structured items:
            # "1）xxx"  "（1）xxx"  "(1) xxx"
            paren_match = re.match(
                r'^(?:[（(]?\d+[）)])|(?:[（(]\d+[）)])\s*(.+)',
                stripped,
            )
            if paren_match:
                item_text = paren_match.group(1).strip()
                if item_text and len(item_text) > 1:
                    items.append(item_text)

    return items[:5]


def _extract_strengths(text: str) -> List[str]:
    """Extract strengths / positive points from agent output."""
    section_kw = [
        "优势", "加分项", "亮点", "正面因素", "看好理由",
        "strength", "bull", "positive", "advantage", "利好",
        "积极因素", "有利因素", "推荐理由",
    ]
    return _extract_bullet_list(text, section_kw)


def _extract_risks(text: str) -> List[str]:
    """Extract risks / negative points from agent output."""
    section_kw = [
        "风险", "劣势", "不足", "隐患", "负面因素",
        "risk", "weakness", "threat", "negative", "drawback",
        "利空", "警示", "担心", "不确定性", "要注意",
        "警惕", "避免", "缺陷", "短板",
    ]
    return _extract_bullet_list(text, section_kw)


def _extract_profile(text: str) -> Dict[str, str]:
    """Extract structured fund profile fields from fund_product_doc output.

    Parses the product doc agent's output for standard fields like fund code,
    name, type, benchmark, risk level, inception date, and management company.
    """
    profile: Dict[str, str] = {
        "fund_code": "",
        "fund_name": "",
        "fund_type": "",
        "benchmark": "",
        "risk_level": "",
        "inception_date": "",
        "management_company": "",
    }

    if not text or not isinstance(text, str):
        return profile

    # Field extraction patterns: (key, [regex patterns])
    field_patterns = {
        "fund_code": [
            r'(?:基金代码|fund.?code|代码)[:：]\s*([a-zA-Z0-9_.]+)',
            r'代码[:：]?\s*(\d{6})',
        ],
        "fund_name": [
            r'(?:基金名称|fund.?name|名称)[:：]\s*([^\n]{2,50})',
        ],
        "fund_type": [
            r'(?:基金类型|fund.?type|类型)[:：]\s*([^\n]{2,30})',
            r'(?:属于|是)[^\n]{0,10}([^\n]{0,10}(?:ETF|LOF|QDII|货币|债券|混合|股票|指数|联接|FOF))',
        ],
        "benchmark": [
            r'(?:业绩基准|benchmark|基准|比较基准)[:：]\s*([^\n]{2,50})',
        ],
        "risk_level": [
            r'(?:风险等级|risk.?level|风险)[:：]\s*([^\n]{2,15})',
            r'(R[1-5]|低风险|中低风险|中等风险|中高风险|高风险)',
        ],
        "inception_date": [
            r'(?:成立日|inception|成立日期|设立日)[:：]\s*([^\n]{4,15})',
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
        ],
        "management_company": [
            r'(?:管理人|基金公司|management.?company|发行方)[:：]\s*([^\n]{2,30})',
            r'(?:基金公司|管理人|发行)[:：]\s*([^\n]{2,30})',
        ],
    }

    for field, patterns in field_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = match.group(1).strip()
                except IndexError:
                    continue
                if value and len(value) > 1:
                    profile[field] = value
                    break

    return profile


# ---------------------------------------------------------------------------
# Conflict detection & tag generation
# ---------------------------------------------------------------------------

def _detect_conflicts(agent_outputs: Dict[str, str]) -> List[str]:
    """Detect conflicting conclusions between agent outputs.

    Checks for common contradiction patterns across agent pairs:
    - perf_agent says "业绩优异" vs holdings_agent says "持仓不合理"
    - manager_agent says "经验丰富" vs perf_agent says "业绩平庸"
    - etc.
    """
    conflicts: List[str] = []

    # Get key outputs (fallback to empty string)
    perf = agent_outputs.get("fund_perf_risk", "")
    manager = agent_outputs.get("fund_manager", "")
    holdings = agent_outputs.get("fund_holdings", "")
    benchmark = agent_outputs.get("fund_benchmark", "")
    fee = agent_outputs.get("fund_fee", "")
    product = agent_outputs.get("fund_product_doc", "")
    event = agent_outputs.get("fund_event", "")

    # Conflict 1: Performance vs Manager quality
    if perf and manager:
        perf_positive = any(
            kw in perf for kw in ["优秀", "优异", "突出", "领先", "稳健", "好"]
        )
        perf_negative = any(
            kw in perf for kw in ["不佳", "平庸", "落后", "差", "波动大", "回撤"]
        )
        mgr_positive = any(
            kw in manager for kw in ["经验丰富", "优秀", "能力强", "稳定", "明星"]
        )
        mgr_negative = any(
            kw in manager for kw in ["经验不足", "频繁更换", "新手", "能力弱"]
        )
        if perf_negative and mgr_positive:
            conflicts.append(
                "业绩风险分析认为表现不佳，但基金经理分析评价正面"
                "——建议核查管理人能力与业绩的匹配度"
            )
        if perf_positive and mgr_negative:
            conflicts.append(
                "业绩风险分析认为表现优秀，但基金经理评价偏负面"
                "——可能存在策略漂移或运气成分"
            )

    # Conflict 2: Portfolio vs Benchmark alignment
    if holdings and benchmark:
        hold_style = ""
        bench_style = ""
        if "集中" in holdings:
            hold_style = "集中"
        elif "分散" in holdings:
            hold_style = "分散"
        if "集中" in benchmark:
            bench_style = "集中"
        elif "分散" in benchmark:
            bench_style = "分散"
        if hold_style and bench_style and hold_style != bench_style:
            conflicts.append(
                f"持仓分析显示持仓{hold_style}，但基准分析显示风格{bench_style}"
                "——可能存在风格漂移"
            )

    # Conflict 3: Fee structure vs Product positioning
    if fee and product:
        fee_high = any(kw in fee for kw in ["高费率", "费率较高", "管理费高", "费率偏高"])
        product_simple = any(
            kw in product for kw in ["指数", "被动", "ETF", "联接", "低费率"]
        )
        if fee_high and product_simple:
            conflicts.append(
                "产品为被动/指数型但费率偏高——费率与产品复杂度不匹配"
            )

    # Conflict 4: Event risk contradicts holdings assessment
    if event and holdings:
        event_high = any(
            kw in event for kw in ["重大", "预警", "风险事件", "负面", "诉讼", "违约"]
        )
        hold_clean = any(
            kw in holdings for kw in ["健康", "合理", "优质", "良好", "无异常"]
        )
        if event_high and hold_clean:
            conflicts.append(
                "事件风险分析提示重大风险，但持仓分析未反映——建议关注事件影响范围"
            )

    # Conflict 5: Benchmark style vs stated fund type
    if benchmark and product:
        bench_type_hint = ""
        if any(kw in benchmark for kw in ["300", "沪深", "大盘"]):
            bench_type_hint = "大盘"
        elif any(kw in benchmark for kw in ["500", "中盘"]):
            bench_type_hint = "中盘"
        elif any(kw in benchmark for kw in ["1000", "小盘", "创业"]):
            bench_type_hint = "小盘"
        prod_type = ""
        if any(kw in product for kw in ["大盘"]):
            prod_type = "大盘"
        elif any(kw in product for kw in ["小盘"]):
            prod_type = "小盘"
        if bench_type_hint and prod_type and bench_type_hint != prod_type:
            conflicts.append(
                f"产品文档显示{prod_type}风格，但业绩基准指向{bench_type_hint}风格"
                "——可能存在风格漂移"
            )

    return conflicts


def _generate_frontend_tags(
    profile: Dict[str, str],
    strengths: List[str],
    risks: List[str],
    holding_hints: List[Dict[str, str]],
) -> List[str]:
    """Generate concise, frontend-ready display tags for UI rendering."""
    tags: List[str] = []

    # Risk level tag
    risk = profile.get("risk_level", "")
    if risk:
        tags.append(risk)

    # Fund type tag
    ftype = profile.get("fund_type", "")
    if ftype:
        # Simplify long type names
        if any(kw in ftype for kw in ["ETF", "交易型"]):
            tags.append("ETF")
        elif "LOF" in ftype:
            tags.append("LOF")
        elif any(kw in ftype for kw in ["封闭", "定开"]):
            tags.append("封闭式")
        elif any(kw in ftype for kw in ["股票", "权益"]):
            tags.append("主动权益")
        elif any(kw in ftype for kw in ["混合"]):
            tags.append("混合型")
        elif any(kw in ftype for kw in ["债券", "债"]):
            tags.append("债券型")
        elif any(kw in ftype for kw in ["货币"]):
            tags.append("货币型")
        elif any(kw in ftype for kw in ["指数"]):
            tags.append("指数基金")
        elif any(kw in ftype for kw in ["QDII"]):
            tags.append("QDII")
        elif any(kw in ftype for kw in ["FOF"]):
            tags.append("FOF")

    # Holding period hint tag
    if holding_hints:
        shortest = holding_hints[0]  # fee_agent hint typically first
        label = shortest.get("label", "")
        if label:
            tags.append(label)

    # Volatility hint from strengths/risks pool
    all_text = " ".join(strengths + risks)
    if any(kw in all_text for kw in ["高波动", "波动大", "回撤大"]):
        tags.append("中高波动")
    elif any(kw in all_text for kw in ["低波动", "波动小", "稳健"]):
        tags.append("低波动")

    # Active vs passive hint
    if any(kw in ftype for kw in ["指数", "被动", "ETF"]) or any(
        kw in all_text for kw in ["被动管理", "跟踪指数"]
    ):
        tags.append("被动管理")
    elif "主动" in ftype or any(kw in all_text for kw in ["主动管理", "选股"]):
        tags.append("主动管理")

    # Deduplicate while preserving order
    seen: set = set()
    unique: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _build_holding_hints(
    agent_outputs: Dict[str, str],
) -> List[Dict[str, str]]:
    """Derive holding period guidance from fee_agent, product_doc, and perf_risk.

    Priority order: fee_agent (structural), product_doc (design intent),
    perf_risk (empirical volatility).
    """
    hints: List[Dict[str, str]] = []

    fee_text = agent_outputs.get("fund_fee", "")
    product_text = agent_outputs.get("fund_product_doc", "")
    perf_text = agent_outputs.get("fund_perf_risk", "")

    # From fee_agent: redemption fee structure implies holding period
    if fee_text:
        if any(kw in fee_text for kw in ["7天", "7日", "短期赎回", "7天内"]):
            hints.append({
                "label": "短期灵活",
                "reason": "赎回费7天后降低，支持短线操作",
                "source_agent": "fund_fee",
            })
        if any(kw in fee_text for kw in ["1年", "一年", "持有1年", "365天"]):
            hints.append({
                "label": "1年以上",
                "reason": "持有满1年免赎回费，适合中线持有",
                "source_agent": "fund_fee",
            })
        if any(kw in fee_text for kw in ["2年", "两年", "3年", "长期"]):
            hints.append({
                "label": "长期持有",
                "reason": "赎回费结构鼓励长期持有（2年+）",
                "source_agent": "fund_fee",
            })
        if any(kw in fee_text for kw in ["封闭", "定开", "封闭期"]):
            hints.append({
                "label": "封闭期内",
                "reason": "封闭/定开产品，无法提前赎回",
                "source_agent": "fund_fee",
            })

    # From product_doc: fund type implies minimum holding
    if product_text:
        if any(kw in product_text for kw in ["货币", "短债", "超短"]):
            if not any(h["label"] == "短期灵活" for h in hints):
                hints.append({
                    "label": "短期灵活",
                    "reason": "货币/短债型基金，流动性高",
                    "source_agent": "fund_product_doc",
                })
        if any(kw in product_text for kw in ["定开", "封闭", "持有期"]):
            if not any(h["label"] == "封闭期内" for h in hints):
                hints.append({
                    "label": "锁定持有",
                    "reason": "产品设有封闭期/持有期限制",
                    "source_agent": "fund_product_doc",
                })

    # From perf_risk: empirical volatility implies required holding period
    if perf_text:
        if any(kw in perf_text for kw in ["高波动", "波动大", "回撤超过", "大幅回撤"]):
            hints.append({
                "label": "1-3年",
                "reason": "历史波动率较高，需较长持有期平滑波动",
                "source_agent": "fund_perf_risk",
            })
        elif any(kw in perf_text for kw in ["低波动", "波动小", "稳健"]):
            hints.append({
                "label": "6个月+",
                "reason": "历史波动率较低，中等持有即可",
                "source_agent": "fund_perf_risk",
            })

    return hints


def _compute_confidence_summary(
    agent_outputs: Dict[str, str], sub_scores: Dict[str, int]
) -> float:
    """Compute an overall confidence score for the analysis package.

    Combines:
    - Average confidence extracted from each agent's output text
    - How many agent outputs are non-empty
    - Score dispersion (high variance = lower confidence)
    """
    confidences: List[float] = []
    present_agents = 0

    for key in AGENT_KEYS:
        text = agent_outputs.get(key, "")
        if text and isinstance(text, str) and len(text.strip()) > 20:
            present_agents += 1
            conf = _extract_confidence(text)
            confidences.append(conf)

    if not confidences:
        return 0.0

    # Average confidence * coverage ratio
    avg_conf = sum(confidences) / len(confidences)
    coverage = present_agents / len(AGENT_KEYS)

    # Score dispersion penalty
    score_values = [v for v in sub_scores.values() if v != DEFAULT_SCORE]
    if len(score_values) >= 2:
        max_s = max(score_values)
        min_s = min(score_values)
        dispersion = (max_s - min_s) / 100.0
        # Reduce confidence if scores are wildly different
        dispersion_penalty = max(0, dispersion - 0.3) * 0.3
    else:
        dispersion_penalty = 0

    return round(
        min(max(avg_conf * coverage * (1 - dispersion_penalty), 0.0), 1.0),
        2,
    )


def _detect_missing_data(agent_outputs: Dict[str, str]) -> List[str]:
    """Identify known data gaps from agent outputs."""
    missing: List[str] = []

    for key in AGENT_KEYS:
        text = agent_outputs.get(key, "")
        label = AGENT_LABELS.get(key, key)
        if not text or not isinstance(text, str):
            missing.append(f"{label}：无输出")
        elif len(text.strip()) < 20:
            missing.append(f"{label}：输出过短（可能数据不足）")
        elif any(
            kw in text
            for kw in ["无法获取", "数据缺失", "暂无数据", "不可用", "no data"]
        ):
            missing.append(f"{label}：报告中存在数据缺失提示")

    return missing


# ---------------------------------------------------------------------------
# Main node function
# ---------------------------------------------------------------------------


async def fund_merge_node(state: AgentState) -> Dict[str, Any]:
    """Merge node: combines 7 fund agent outputs into a unified analysis package.

    Non-LLM node — pure data transformation. Reads unstructured agent output
    strings from state.data and produces a structured fund_analysis_package
    dict suitable for downstream report generation and scoring.

    Input (reads from state.data):
        - fund_product_doc  : Fund product documentation analysis
        - fund_perf_risk    : Performance and risk analysis
        - fund_holdings     : Portfolio holdings analysis
        - fund_manager      : Fund manager analysis
        - fund_benchmark    : Benchmark / style consistency analysis
        - fund_fee          : Fee structure and liquidity analysis
        - fund_event        : Event risk analysis

    Output (writes to state.data.fund_analysis_package):
        Structured dict with normalized scores, aggregated strengths/risks,
        conflict detection, holding period guidance, and frontend tags.
    """
    data: Dict[str, Any] = state.get("data", {})
    fund_code = data.get("fund_code", "") or data.get("stock_code", "")
    fund_name = data.get("fund_name", "") or data.get("company_name", "")

    logger.info(
        f"{WAIT_ICON} FundMergeNode: 开始合并基金 {fund_name}({fund_code}) 的7项分析结果"
    )

    try:
        # ---- Step 1: Collect agent outputs ----
        agent_outputs: Dict[str, str] = {}
        present_count = 0
        for key in AGENT_KEYS:
            val = data.get(key, "")
            if isinstance(val, str):
                agent_outputs[key] = val
            else:
                agent_outputs[key] = str(val) if val else ""
            if agent_outputs[key] and len(agent_outputs[key].strip()) > 20:
                present_count += 1

        logger.info(
            f"  FundMergeNode: {present_count}/{len(AGENT_KEYS)} 个基金分析Agent有实质输出"
        )

        # ---- Step 2: Extract normalized sub-scores ----
        sub_scores: Dict[str, int] = {}
        score_key_map = {
            "fund_product_doc": "product_positioning",
            "fund_perf_risk": "performance_risk",
            "fund_holdings": "portfolio_structure",
            "fund_manager": "manager_team",
            "fund_benchmark": "benchmark_style_consistency",
            "fund_fee": "fee_liquidity",
            "fund_event": "event_risk",
        }

        for agent_key, norm_key in score_key_map.items():
            score = _extract_score(agent_outputs.get(agent_key, ""))
            sub_scores[norm_key] = score
            logger.debug(
                f"  FundMergeNode: {AGENT_LABELS[agent_key]} → {norm_key} = {score}"
            )

        # ---- Step 3: Aggregate strengths and risks pools ----
        strengths_pool: List[str] = []
        risks_pool: List[str] = []

        for key, label in AGENT_LABELS.items():
            text = agent_outputs.get(key, "")
            if text:
                for s in _extract_strengths(text):
                    formatted = f"[{label}] {s}"
                    if formatted not in strengths_pool:
                        strengths_pool.append(formatted)
                for r in _extract_risks(text):
                    formatted = f"[{label}] {r}"
                    if formatted not in risks_pool:
                        risks_pool.append(formatted)

        # Deduplicate and limit
        strengths_pool = strengths_pool[:10]
        risks_pool = risks_pool[:10]

        # ---- Step 4: Build fund profile from product_doc ----
        profile = _extract_profile(agent_outputs.get("fund_product_doc", ""))

        # Backfill from state metadata if profile came up empty
        if not profile.get("fund_code") and fund_code:
            profile["fund_code"] = fund_code
        if not profile.get("fund_name") and fund_name:
            profile["fund_name"] = fund_name

        # ---- Step 5: Detect conflicts ----
        conflicts = _detect_conflicts(agent_outputs)

        # ---- Step 6: Build holding period hints ----
        holding_hints = _build_holding_hints(agent_outputs)

        # ---- Step 7: Compute confidence summary ----
        confidence_summary = _compute_confidence_summary(agent_outputs, sub_scores)

        # ---- Step 8: Detect missing data ----
        missing_data_summary = _detect_missing_data(agent_outputs)

        # ---- Step 9: Generate frontend tags ----
        frontend_tags = _generate_frontend_tags(
            profile, strengths_pool, risks_pool, holding_hints
        )

        # ---- Step 10: Assemble final package ----
        fund_analysis_package: Dict[str, Any] = {
            "fund_profile": profile,
            "normalized_subscores": sub_scores,
            "strengths_pool": strengths_pool,
            "risks_pool": risks_pool,
            "conflicts": conflicts,
            "holding_period_hints": holding_hints,
            "confidence_summary": confidence_summary,
            "missing_data_summary": missing_data_summary,
            "frontend_ready_tags": frontend_tags,
            "raw_agent_outputs": agent_outputs,  # Preserve original text for report agent
        }

        # ---- Step 11: Log summary ----
        avg_score = (
            sum(sub_scores.values()) / len(sub_scores) if sub_scores else 0
        )
        logger.info(
            f"{SUCCESS_ICON} FundMergeNode: 合并完成 | "
            f"平均分={avg_score:.1f} | "
            f"优势={len(strengths_pool)}条 | "
            f"风险={len(risks_pool)}条 | "
            f"冲突={len(conflicts)}条 | "
            f"置信度={confidence_summary:.0%} | "
            f"标签={frontend_tags}"
        )

        return {"data": {"fund_analysis_package": fund_analysis_package}}

    except Exception as e:
        logger.error(
            f"{ERROR_ICON} FundMergeNode: 合并失败: {e}",
            exc_info=True,
        )
        # Return a minimal error package so downstream nodes don't crash
        error_package = {
            "fund_profile": {
                "fund_code": fund_code,
                "fund_name": fund_name,
                "fund_type": "",
                "benchmark": "",
                "risk_level": "",
                "inception_date": "",
                "management_company": "",
            },
            "normalized_subscores": {k: DEFAULT_SCORE for k in [
                "product_positioning", "performance_risk", "portfolio_structure",
                "manager_team", "benchmark_style_consistency",
                "fee_liquidity", "event_risk",
            ]},
            "strengths_pool": [],
            "risks_pool": [],
            "conflicts": [f"合并节点异常: {str(e)}"],
            "holding_period_hints": [],
            "confidence_summary": 0.0,
            "missing_data_summary": ["合并节点执行失败，无法生成摘要"],
            "frontend_ready_tags": ["数据异常"],
            "raw_agent_outputs": {key: data.get(key, "") for key in AGENT_KEYS},
            "merge_error": str(e),
        }
        return {"data": {"fund_analysis_package": error_package}}
