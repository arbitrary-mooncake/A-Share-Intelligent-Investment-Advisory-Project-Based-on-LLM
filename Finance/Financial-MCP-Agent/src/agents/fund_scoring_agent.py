"""
FundScoringAgent: 基金综合打分Agent（Agent 9: 基金打分）

基于7维度基金分析结果，输出结构化评分卡，包含：
- 加权总分与评级标签
- 7项子维度得分
- 持有期建议
- 投资者适配性
- 核心优势/风险提炼
- 前端展示块

模型：Model 1 (MiMo-V2.5-Pro)，thinking=enabled, max_tokens=8000
"""
import os
import re
import json as json_mod
import time
from datetime import datetime, timezone
from typing import Dict, Any

from langchain_openai import ChatOpenAI

from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from dotenv import load_dotenv

load_dotenv(override=True)

logger = setup_logger(__name__)

# ── 7维度评分权重 ──
SCORING_WEIGHTS = {
    "product_positioning": 0.15,       # 产品定位与策略清晰度
    "performance_risk": 0.25,          # 业绩与风险表现
    "portfolio_structure": 0.20,       # 组合结构与持仓质量
    "manager_team": 0.15,              # 基金经理与团队稳定性
    "benchmark_style_consistency": 0.10,  # 基准一致性与风格稳定性
    "fee_liquidity": 0.10,             # 费用、流动性与持有体验
    "event_risk": 0.05,               # 事件风险与近期变化
}

# ── 评分标签映射 ──
def _score_to_label(score: int) -> str:
    """总分 → 评级标签"""
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "较优"
    elif score >= 70:
        return "中等偏上"
    elif score >= 60:
        return "一般"
    else:
        return "谨慎"


def _score_to_investment_view(score: int) -> str:
    """总分 → 投资观点"""
    if score >= 85:
        return "可重点关注"
    elif score >= 75:
        return "可关注"
    elif score >= 65:
        return "谨慎关注"
    else:
        return "暂不建议配置"


# ── 维度中文名映射 ──
SUBSOCRE_LABELS = {
    "product_positioning": "产品定位与策略清晰度",
    "performance_risk": "业绩与风险表现",
    "portfolio_structure": "组合结构与持仓质量",
    "manager_team": "基金经理与团队稳定性",
    "benchmark_style_consistency": "基准一致性与风格稳定性",
    "fee_liquidity": "费用、流动性与持有体验",
    "event_risk": "事件风险与近期变化",
}

# ── JSON 输出模板（用于指导 LLM 格式） ──
OUTPUT_TEMPLATE = {
    "score_meta": {
        "score_version": "fund_score_v1",
        "generated_at": "ISO8601时间戳",
        "as_of_date": "分析基准日期"
    },
    "fund_identity": {
        "fund_code": "sh.510050",
        "fund_name": "华夏上证50ETF",
        "fund_type": "ETF"
    },
    "overall_score": {
        "score": 82,
        "rating_label": "较优",
        "investment_view": "可关注"
    },
    "subscores": {
        "product_positioning": 85,
        "performance_risk": 78,
        "portfolio_structure": 84,
        "manager_team": 80,
        "benchmark_style_consistency": 79,
        "fee_liquidity": 76,
        "event_risk": 83
    },
    "holding_period_suggestion": {
        "label": "建议持有1年以上",
        "min_days": 365,
        "preferred_days": 730,
        "not_recommended_for": ["短期频繁交易", "高流动性资金停泊"]
    },
    "fit_for_user": {
        "suitability": "适合中高风险承受能力投资者",
        "matched": True,
        "reason": "基金波动与用户风险承受能力基本匹配"
    },
    "highlights": {
        "strengths": ["产品定位清晰", "组合风格相对稳定", "基金经理任职连续性较好"],
        "risks": ["回撤控制能力一般", "行业集中度偏高"]
    },
    "score_explanation": {
        "why_this_score": "该基金整体质量中上...",
        "why_this_holding_period": "基金特征更适合中长期持有..."
    },
    "frontend_blocks": {
        "score_badge": "82分",
        "rating_tag": "较优",
        "holding_tag": "1年以上",
        "risk_tag": "中高波动",
        "cta": "查看详细报告"
    }
}

# ── 7维度权重说明（注入 prompt 的详细文档） ──
WEIGHT_DOC = """
### 1. 产品定位与策略清晰度（15%）
评估标准：
- 投资目标是否明确，策略是否可理解、可复现
- 指数基金：跟踪的指数是否具有代表性，跟踪方式（完全复制/抽样）是否合理
- 主动基金：投资理念是否清晰，策略是否有持续竞争优势
- 产品设计：费率结构、申赎机制、分红政策是否合理
打分参考：90-100定位极其清晰；80-89明确；70-79基本清晰；60-69模糊；<60定位混乱

### 2. 业绩与风险表现（25%）—— 最高权重
评估标准：
- 收益表现：近1年/3年/成立以来收益率 vs 同类平均 vs 基准
- 风险调整收益：夏普比率、信息比率、卡玛比率
- 最大回撤：历史最大回撤及其恢复时间
- 波动率：年化波动率 vs 同类均值
- 胜率：月度/季度正收益概率
打分参考：90-100业绩优异且风险可控；80-89业绩良好；70-79中等；60-69偏弱；<60长期低迷

### 3. 组合结构与持仓质量（20%）
评估标准：
- 集中度：前十大重仓占比、行业集中度
- 持仓质量：重仓股的基本面质量（ROE、盈利稳定性、成长性）
- 组合均衡性：行业分散程度、市值风格分布
- 调仓行为：换手率水平、调仓逻辑是否合理
打分参考：90-100组合高质量且均衡；80-89较均衡；70-79基本合理；60-69偏集中或质量一般；<60过度集中或持仓质量差

### 4. 基金经理与团队稳定性（15%）
评估标准：
- 任职稳定性：基金经理任职年限、团队核心人员变动
- 历史业绩：基金经理管理同类产品的历史业绩
- 专业背景：团队研究覆盖面、专业资质
- 激励机制：基金公司实力、投研文化
打分参考：90-100团队卓越且稳定；80-89较稳定；70-79基本稳定；60-69有不确定性；<60不稳定

### 5. 基准一致性与风格稳定性（10%）
评估标准：
- 跟踪误差（指数基金）/ 风格一致性（主动基金）
- 信息比率：相对基准的超额收益稳定性
- 风格漂移：是否存在明显的风格偏移
- Barra因子暴露：规模、价值、动量等因子的暴露是否稳定
打分参考：90-100高度一致；80-89基本一致；70-79轻微漂移；60-69明显漂移；<60严重漂移

### 6. 费用、流动性与持有体验（10%）
评估标准：
- 综合费率：管理费+托管费+销售服务费 vs 同类均值
- 交易成本：申赎费率、买卖价差
- 流动性：日均成交额、基金规模、大额申赎冲击
- 持有体验：最大回撤修复时间、滚动收益分布
打分参考：90-100费率低+流动性好；80-89较优；70-79适中；60-69偏贵或流动性一般；<60费率高或流动性差

### 7. 事件风险与近期变化（5%）
评估标准：
- 近期公告：基金经理变更、策略调整、规模异动
- 合规风险：是否有监管处罚、违规记录
- 市场事件：近期是否有大额申赎、清盘预警
- 持仓暴雷：重仓股是否有重大负面事件
打分参考：90-100无不利事件；80-89轻微影响；70-79有一定风险；60-69明显风险；<60严重事件
"""

# ── 持有期建议映射规则 ──
HOLDING_PERIOD_DOC = """
持有期建议推导规则：
- 基金以长期资产配置为主（如宽基ETF、养老FOF）→ 1年以上
- 基金适合中长期持有但需关注市场周期 → 6个月-1年
- 基金适合中短期波段操作（如行业ETF、主题基金）→ 1-6个月
- 基金波动极大或不适合普通投资者长期持有 → <1个月（仅短线）

⚠️ **持有期与投资观点的对齐规则（必须遵守）**：
- 如果 investment_view 为 "暂不建议配置"（score < 65）：holding_period.label 必须设为 "当前不建议持有"，preferred_days 设为 0，not_recommended_for 设为 ["中长线配置", "短期交易"]——不可给出正面的持有期建议，否则与"不建议配置"自相矛盾。
- 如果 investment_view 为 "谨慎关注"（score 65-74）：holding_period 可以给出持有期建议，但必须在 why_this_holding_period 中说明"当前评分偏低，建议小仓位试仓"。
- 如果 investment_view 为 "可关注"或更好（score >= 75）：正常给出持有期建议。

对每类持有期，标注 not_recommended_for（不适合的情况）：
- 1年以上 → ["短期频繁交易", "高流动性资金停泊"]
- 6个月-1年 → ["超短线交易", "急需变现的资金"]
- 1-6个月 → ["短期投机", "不能承受波动的资金"]
- <1个月 → ["中长线配置", "风险规避型投资者"]
"""


async def fund_scoring_agent(
    fund_analysis_package: Dict[str, Any],
    fund_code: str = "",
    fund_name: str = "",
    fund_type: str = "ETF",
    current_date: str = "",
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
    """
    对基金进行7维度综合评分。

    Args:
        fund_analysis_package: 基金统一分析包（来自 fund_merge_node），
                               包含各子维度的分析结果、normalized_subscores 等
        fund_code: 基金代码（如 sh.510050）
        fund_name: 基金名称（如 华夏上证50ETF）
        fund_type: 基金类型（如 ETF、LOF、主动管理基金）
        current_date: 分析基准日期（YYYY-MM-DD）
        thinking_enabled: 是否启用 thinking 模式

    Returns:
        Dict 结构化评分卡，包含 score_meta, fund_identity, overall_score,
        subscores, holding_period_suggestion, fit_for_user, highlights,
        score_explanation, frontend_blocks
    """
    agent_name = "fund_scoring_agent"
    logger.info(f"{WAIT_ICON} FundScoringAgent: 开始对 {fund_name}({fund_code}) 进行基金综合打分")

    execution_logger = get_execution_logger()
    agent_start_time = time.time()

    execution_logger.log_agent_start(agent_name, {
        "fund_code": fund_code,
        "fund_name": fund_name,
        "fund_type": fund_type,
        "has_analysis_package": bool(fund_analysis_package),
        "package_keys": list(fund_analysis_package.keys()) if fund_analysis_package else [],
    })

    if not fund_analysis_package:
        raise ValueError("fund_analysis_package 为空，无法进行基金打分")

    try:
        # ── 模型配置：Model 1 (MiMo-V2.5-Pro) ──
        model_cfg = get_model_config_for_agent("fund_scoring_agent")
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        resolved_model = model_cfg["model_name"]

        if not all([api_key, base_url, resolved_model]):
            raise ValueError("缺少OpenAI环境变量")

        llm = ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
            request_timeout=360,
            max_tokens=8000,
            extra_body=get_thinking_body(base_url, thinking_enabled)
        )

        # ── 构建评分 Prompt ──
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        as_of = current_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        analysis_json = json_mod.dumps(fund_analysis_package, ensure_ascii=False, indent=2)
        weights_json = json_mod.dumps(SCORING_WEIGHTS, ensure_ascii=False, indent=2)
        template_json = json_mod.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)

        system_prompt = f"""你是一位资深基金评价分析师，拥有10年以上FOF投资和基金研究经验。请基于以下7维度分析结果，对基金进行综合评分。

**重要时间信息**
- 生成时间：{now_utc}
- 分析基准日期：{as_of}

## 7维度评分体系（总分为加权总分，满分100）

{WEIGHT_DOC}

## 持有期建议

{HOLDING_PERIOD_DOC}

## 评分标签映射

总分 → 评级标签（rating_label）：
- 90-100: 优秀
- 80-89: 较优
- 70-79: 中等偏上
- 60-69: 一般
- <60: 谨慎

总分 → 投资观点（investment_view）：
- 85+: 可重点关注
- 75-84: 可关注
- 65-74: 谨慎关注
- <65: 暂不建议配置

## 评分要求

1. **加权计算**：基于分析包中的 normalized_subscores 数据，结合你对各项数据质量的判断，逐维度给出0-100的整数得分，然后按权重计算加权总分。
2. **总分必须是整数**：round(weighted_sum) 四舍五入取整。
3. **映射标签**：根据总分映射到正确的 rating_label 和 investment_view。
4. **持有期建议与投资观点必须一致**：investment_view 和 holding_period 不可自相矛盾。具体规则见持有期建议映射规则中的对齐规则。如果 investment_view 是"暂不建议配置"，holding_period.label 必须是"当前不建议持有"，绝对不能给出正面持有期建议。
5. **核心提炼**：从分析包中提取并重构 top 3 优势和 top 3 风险（用简洁中文表达，每项≤20字）。风险和优势的表述必须具体（如"重仓股估值偏高"优于"估值风险"），不得使用模糊的概括性风险。
6. **score_explanation**：why_this_score 解释为什么得到这个分数（1-2句），why_this_holding_period 解释为什么建议这个持有期（1-2句）。
7. **frontend_blocks**：精确按要求格式生成前端展示块。
8. **fit_for_user**：根据基金的波动率水平、最大回撤、投资策略特征，给出适配性建议。
9. **如果分析包中某项数据缺失或质量低**：该维度得分应当偏低（≤60），并在 score_explanation 的 why_this_score 中说明原因。
10. **输出严格JSON**：不要包含任何Markdown代码块标记，不要包含解释性文字，只输出纯JSON。
11. **数据缺失的表述规范**：生成 risks 和 score_explanation 时，对数据缺失程度的描述必须使用分级词汇，禁止滥用"严重不足"等最高级表述：
  - 仅个别次要维度缺失（如仅有新闻/舆情缺失）→ 使用"新闻舆情数据暂不可得"、"个别辅助数据有限"，**不得**使用"数据不足"等全局性措辞
  - 有2-3个维度部分缺失但核心数据（净值/持仓/基本面）齐全 → 使用"部分维度数据欠缺"、"部分辅助数据有限"
  - 核心分析数据（净值/持仓/基本面）缺失50%以上 → 使用"核心数据覆盖不足"、"数据缺口较大影响分析深度"
  - 只有几乎全部数据缺失时才可使用"数据严重不足"

## ⛔ 输出格式（严格遵循）

{template_json}
"""

        user_prompt = f"""请对以下基金进行综合评分：

基金代码：{fund_code}
基金名称：{fund_name}
基金类型：{fund_type}

## 统一分析包

{analysis_json}

## 评分权重

{weights_json}

请严格按照系统提示中的评分体系和JSON格式输出打分结果。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ── 调用 LLM ──
        llm_start = time.time()
        response = await llm.ainvoke(messages)
        llm_time = time.time() - llm_start

        content = response.content.strip()

        # ── JSON 解析（含 fallback：处理 LLM 可能包裹的 ```json 标记） ──
        score_data = _parse_json_response(content)

        # ── 后处理：补全 score_meta ──
        if "score_meta" not in score_data:
            score_data["score_meta"] = {}
        score_data["score_meta"].update({
            "score_version": "fund_score_v1",
            "generated_at": now_utc,
            "as_of_date": as_of,
        })

        # ── 后处理：补全 fund_identity ──
        if "fund_identity" not in score_data:
            score_data["fund_identity"] = {}
        score_data["fund_identity"].update({
            "fund_code": fund_code if fund_code else score_data["fund_identity"].get("fund_code", ""),
            "fund_name": fund_name if fund_name else score_data["fund_identity"].get("fund_name", ""),
            "fund_type": fund_type if fund_type else score_data["fund_identity"].get("fund_type", "ETF"),
        })

        # ── 验证必要字段 ──
        _validate_score_output(score_data)

        # ── 如果 LLM 未返回 frontend_blocks，自动生成 ──
        if "frontend_blocks" not in score_data or not score_data["frontend_blocks"]:
            score_data["frontend_blocks"] = _generate_frontend_blocks(score_data)

        total_time = time.time() - agent_start_time

        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="fund_scoring",
            input_messages=messages,
            output_content=json_mod.dumps(score_data, ensure_ascii=False),
            model_config={
                "model": resolved_model,
                "temperature": 0.6,
                "max_tokens": 8000,
                "thinking": "enabled" if thinking_enabled else "disabled"
            },
            execution_time=llm_time
        )

        execution_logger.log_agent_complete(agent_name, score_data, total_time, True)

        overall = score_data.get("overall_score", {})
        logger.info(
            f"{SUCCESS_ICON} FundScoringAgent: {fund_name} "
            f"综合评分={overall.get('score', 'N/A')} "
            f"({overall.get('rating_label', 'N/A')})"
        )

        return score_data

    except Exception as e:
        logger.error(f"{ERROR_ICON} FundScoringAgent 打分失败: {e}", exc_info=True)
        total_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {}, total_time, False, str(e))
        raise


def _parse_json_response(content: str) -> Dict[str, Any]:
    """
    解析 LLM 返回的 JSON，支持以下格式：
    1. 纯 JSON 字符串
    2. ```json ... ``` 包裹
    3. ``` ... ``` 包裹
    """
    # 尝试1：纯 JSON
    try:
        return json_mod.loads(content)
    except json_mod.JSONDecodeError:
        pass

    # 尝试2：去除 ```json ... ``` 包裹
    json_patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]
    for pattern in json_patterns:
        match = re.search(pattern, content)
        if match:
            try:
                return json_mod.loads(match.group(1).strip())
            except json_mod.JSONDecodeError:
                continue

    # 尝试3：提取最外层 { ... }
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        try:
            return json_mod.loads(json_match.group(0))
        except json_mod.JSONDecodeError:
            pass

    raise ValueError(f"LLM未返回有效JSON，原始内容前500字符: {content[:500]}")


def _validate_score_output(data: Dict[str, Any]) -> None:
    """
    验证评分输出的必要字段。
    """
    # 检查 overall_score
    if "overall_score" not in data:
        raise ValueError("打分结果缺少 overall_score 字段")

    overall = data["overall_score"]
    for key in ["score", "rating_label", "investment_view"]:
        if key not in overall:
            raise ValueError(f"overall_score 缺少必要字段: {key}")

    if not isinstance(overall.get("score"), (int, float)):
        raise ValueError(f"overall_score.score 必须是数值类型，实际: {type(overall.get('score'))}")

    # 检查 subscores
    subscores = data.get("subscores", {})
    expected_dimensions = list(SCORING_WEIGHTS.keys())
    for dim in expected_dimensions:
        if dim not in subscores:
            logger.warning(f"subscores 缺少维度: {dim} ({SUBSOCRE_LABELS.get(dim, dim)})")

    # 检查 highlights
    if "highlights" not in data:
        raise ValueError("打分结果缺少 highlights 字段")

    highlights = data["highlights"]
    for key in ["strengths", "risks"]:
        if key not in highlights:
            raise ValueError(f"highlights 缺少必要字段: {key}")


def _generate_frontend_blocks(score_data: Dict[str, Any]) -> Dict[str, str]:
    """
    当 LLM 未返回 frontend_blocks 时，自动从已有数据生成。
    """
    overall = score_data.get("overall_score", {})
    score = overall.get("score", 0)
    rating = overall.get("rating_label", "N/A")
    holding = score_data.get("holding_period_suggestion", {})
    holding_tag = holding.get("label", "N/A")

    # 根据波动或风险提示推断 risk_tag
    highlights = score_data.get("highlights", {})
    risks = highlights.get("risks", [])
    risk_text = " ".join(risks)
    if any(kw in risk_text for kw in ["波动", "回撤", "风险"]):
        risk_tag = "中高波动"
    else:
        risk_tag = "中等波动"

    return {
        "score_badge": f"{score}分",
        "rating_tag": rating,
        "holding_tag": holding_tag,
        "risk_tag": risk_tag,
        "cta": "查看详细报告",
    }
