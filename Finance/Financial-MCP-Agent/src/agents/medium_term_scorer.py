"""
MediumTermScorer: 中线投资打分Agent (以月为单位，1-3个月持仓)

本Agent是产品的核心竞争力和主要卖点。
必须经过4个分析Agent(fundamental/technical/value/news)的完整输出后才进行打分。

核心特性：
- 行业自适应打分：自动识别行业，注入行业估值基准和打分指引
- 跨行业可比：使用相对行业均值的百分位打分，而非绝对PE/PB数值
- 权重框架统一不变，但每个维度内的打分参考随行业调整

评分体系（v2，7维度100分）：
1. 基本面质量(20分) - ROE/ROIC持续性、盈利质量、资产负债健康度、成长持续性
2. 估值(15分) - PE/PB/EV-EBITDA相对行业、历史分位、安全边际
3. 财务质量/治理风险(20分) - 利润现金含量、应收/存货/商誉风险、质押/减持/关联交易、审计与监管风险
4. 事件持续性(15分) - 催化剂持续性、事件对1-3个月窗口的实质影响
5. 技术与量价确认(10分) - 中期趋势方向、量价配合、中线进场/出场时机
6. 行业/估值适配(10分) - 行业景气周期、跨行业可比性
7. 新闻叙事(10分) - 市场预期合理性、一致预期、题材持续性
风险扣分：由risk_gate统一后处理

使用thinking模式(max_tokens=16000)，最大化打分准确性。
"""
import os
import re
import json as json_mod
import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI

from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.industry_knowledge import (
    identify_industry,
    generate_industry_context_prompt,
    INDUSTRY_BENCHMARKS,
)
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from dotenv import load_dotenv

load_dotenv(override=True)

logger = setup_logger(__name__)


async def medium_term_scorer(
    stock_code: str,
    company_name: str,
    fundamental_analysis: str = "",
    technical_analysis: str = "",
    value_analysis: str = "",
    news_analysis: str = "",
    event_analysis: str = "",
    quality_risk_analysis: str = "",
    moneyflow_analysis: str = "",
    analysis_package = None,
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
    model_name: str = "",
    model_api_key: str = "",
    model_base_url: str = "",
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
    """
    中线投资打分 (1-3个月持仓)

    评分体系（v2，7维度总分100分）：
    - 基本面质量(20分)
    - 估值(15分)
    - 财务质量/治理风险(20分)
    - 事件持续性(15分)
    - 技术与量价确认(10分)
    - 行业/估值适配(10分)
    - 新闻叙事(10分)
    风险扣分：由risk_gate统一后处理
    """
    agent_name = "medium_term_scorer"
    logger.info(f"{WAIT_ICON} MediumTermScorer: 开始对 {company_name}({stock_code}) 进行中线打分")

    execution_logger = get_execution_logger()
    agent_start_time = time.time()

    # 自动识别行业
    all_analysis_text = f"{fundamental_analysis} {value_analysis} {news_analysis}"
    detected_industry = identify_industry(company_name, all_analysis_text)

    execution_logger.log_agent_start(agent_name, {
        "stock_code": stock_code,
        "company_name": company_name,
        "scoring_type": "medium_term",
        "detected_industry": detected_industry,
        "has_fundamental": bool(fundamental_analysis),
        "has_technical": bool(technical_analysis),
        "has_value": bool(value_analysis),
        "has_news": bool(news_analysis),
        "has_event": bool(event_analysis),
        "has_quality_risk": bool(quality_risk_analysis),
        "has_moneyflow": bool(moneyflow_analysis),
        "has_analysis_package": analysis_package is not None,
    })

    try:
        # 模型配置：优先显式参数（快筛覆盖），否则使用 agent 分配的模型 (MiMo-V2.5-Pro)
        model_cfg = get_model_config_for_agent("medium_term_scorer")
        api_key = model_api_key or model_cfg["api_key"]
        base_url = model_base_url or model_cfg["base_url"]
        resolved_model = model_name or model_cfg["model_name"]

        if not all([api_key, base_url, resolved_model]):
            raise ValueError("缺少OpenAI环境变量")

        llm = ChatOpenAI(
            model=resolved_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.6,
            request_timeout=360,
            max_tokens=16000,
            extra_body=get_thinking_body(base_url, thinking_enabled)
        )

        # 生成行业上下文指引
        industry_context = generate_industry_context_prompt(detected_industry)

        system_prompt = f"""你是一位资深A股中线投资专家，专注于1-3个月的中线波段操作策略。
你所在的券商研究所以"基本面+技术面+估值"三维共振为核心方法论，
在过往的中线推荐中取得了显著的超额收益。

**重要时间信息：当前实际时间是 {current_time_info}**
**分析基准日期：{current_date}**

你的任务是综合分析数据，对股票进行中线量化打分（1-3个月持仓周期）。

{industry_context}

## 评分体系（v2，7维度总分100分）

请严格按照以下7个维度进行打分，每个维度必须给出明确的数据支撑和打分依据：

### 1. 基本面质量（满分20分）
ROE/ROIC持续性、盈利质量(经营现金流/净利润)、资产负债健康度、成长持续性

打分参考：
- 17-20分：基本面优秀，各项指标均显著优于行业平均水平
- 13-16分：基本面良好，个别指标略逊但不影响整体质量
- 8-12分：基本面一般，存在一定瑕疵但不致命
- 0-7分：基本面恶化或存在明显风险

### 2. 估值（满分15分）
PE/PB/EV-EBITDA相对行业、历史分位、估值对增长的反映程度、安全边际

打分参考：
- 13-15分：显著低估（相对行业均值和历史分位），安全边际充足
- 10-12分：估值合理，未明显高估也未明显低估
- 6-9分：估值略高，但可由增长预期支撑
- 0-5分：明显高估，缺乏安全边际

重要：请使用**相对行业均值的百分位**进行估值打分，不要使用绝对PE/PB数值。

### 3. 财务质量/治理风险（满分20分）
利润现金含量、应收/存货/商誉风险、质押/减持/关联交易、审计与监管风险

打分参考：
- 17-20分：财务质量优异，无重大治理风险，现金流健康
- 13-16分：财务质量良好，风险可控
- 8-12分：存在一定财务瑕疵或治理隐患
- 0-7分：显著的财务风险或治理缺陷

### 4. 事件持续性（满分15分）
催化剂是否可持续(一次性vs持续性)、事件对1-3个月窗口的实质影响

打分参考：
- 13-15分：存在持续性强的催化剂，对中线走势有显著正面推动
- 10-12分：催化剂有一定持续性，正面影响可期
- 6-9分：催化剂偏一次性，中线影响有限
- 0-5分：无明显催化或面临持续性利空

### 5. 技术与量价确认（满分10分）
中期趋势方向(20/60日均线)、量价配合、是否为中线进场/出场时机

打分参考：
- 9-10分：明确上升中期趋势，量价健康，中线进场时机佳
- 6-8分：趋势偏多但面临关键阻力
- 3-5分：趋势不明确或横盘整理
- 0-2分：中期下降趋势，中线不宜进场

### 6. 行业/估值适配（满分10分）
行业景气周期、行业相对估值容忍度、跨行业可比性

打分参考：
- 9-10分：行业景气上行，估值在行业中具备优势
- 6-8分：行业中性，估值合理
- 3-5分：行业景气下行或估值相对偏高
- 0-2分：行业衰退或估值严重偏离行业中枢

### 7. 新闻叙事（满分10分）
市场预期是否合理、是否形成一致预期、题材持续性

打分参考：
- 9-10分：市场预期合理且积极，一致预期形成，题材持续性强
- 6-8分：市场预期偏乐观，有一定关注度
- 3-5分：市场预期中性或分歧较大
- 0-2分：市场预期悲观，负面叙事主导

风险扣分：由risk_gate统一后处理

## 综合评级标准

根据总分给出投资评级：
- 85-100分：强烈推荐（中线买入，相对沪深300显著跑赢概率大）
- 70-84分：推荐（中线增持，预期跑赢大盘）
- 55-69分：谨慎推荐（中线持有，预期与大盘同步）
- 40-54分：中性（观望为主，下行风险与上行空间相当）
- 25-39分：谨慎减持（下行风险大于上行空间）
- 0-24分：减持（建议规避）

## 输出要求

请以JSON格式返回打分结果，必须包含以下所有字段：

{{
    "score": 总分(0-100的整数),
    "sub_scores": {{
        "fundamental_quality": 基本面质量得分(0-20的整数),
        "valuation": 估值得分(0-15的整数),
        "quality_governance": 财务质量/治理风险得分(0-20的整数),
        "event_sustainability": 事件持续性得分(0-15的整数),
        "technical_confirmation": 技术与量价确认得分(0-10的整数),
        "industry_fit": 行业/估值适配得分(0-10的整数),
        "sentiment_narrative": 新闻叙事得分(0-10的整数)
    }},
    "rating": "投资评级（强烈推荐/推荐/谨慎推荐/中性/谨慎减持/减持）",
    "reasoning": "打分核心理由（3-5句话，指出最强的2-3个因素和最弱的1-2个因素）",
    "risk_warning": "中线风险提示（1-2句话，指出最大1-2个风险点）",
    "data_basis": {{
        "fundamental_quality": "列出本维度评分依据的具体数据点",
        "valuation": "列出本维度评分依据的具体数据点",
        "quality_governance": "列出本维度评分依据的具体数据点",
        "event_sustainability": "列出本维度评分依据的具体数据点",
        "technical_confirmation": "列出本维度评分依据的具体数据点",
        "industry_fit": "列出本维度评分依据的具体数据点",
        "sentiment_narrative": "列出本维度评分依据的具体数据点"
    }},
    "data_reliability": "数据可靠性(高/中/低)，注明哪些基于直接数据，哪些基于推断",
    "suggested_action": "具体操作建议（2-3句话，含建议买入价位区间和目标价位区间）",
    "time_horizon": "建议持仓周期（如：建议持有1-3个月，关注XX事件催化）"
}}

## 重要约束

1. 所有分数必须是**整数**，不能是小数
2. 各子项分数之和必须严格等于总分
3. **评分前必须先列出每个维度的数据依据**（见data_basis字段），数据不充分时必须在data_reliability中标注"低"
4. 打分必须基于提供的分析数据，不能凭空臆断；如果某项数据缺失，在data_basis中标注"无直接数据，基于已有信息推断"
4. 每个维度的打分必须有具体数据引用（如"ROE为18%，属于优秀水平，得7/8分"）
5. 中线打分应**综合权衡**，不能过度偏重某一个维度
6. 如果某个维度的分析数据缺失，请在reasoning中说明并酌情调整
7. 输出必须是纯JSON，不要包含markdown代码块标记
"""

        if analysis_package and hasattr(analysis_package, 'compact_prompt_context'):
            structured_context = analysis_package.compact_prompt_context
        else:
            structured_context = ""

        user_prompt = f"""请对以下股票进行中线投资打分：

公司名称：{company_name}
股票代码：{stock_code}
{"行业：" + detected_industry if detected_industry else "（行业待识别，请根据分析内容自行判断）"}
"""
        if structured_context:
            user_prompt += f"## 结构化分析摘要（优先参考）\n{structured_context}\n\n"

        user_prompt += f"""## 基本面分析
{fundamental_analysis if fundamental_analysis else "（暂无基本面分析数据）"}

## 技术分析
{technical_analysis if technical_analysis else "（暂无技术分析数据）"}

## 估值分析
{value_analysis if value_analysis else "（暂无估值分析数据）"}

## 新闻分析
{news_analysis if news_analysis else "（暂无新闻分析数据）"}
"""

        if event_analysis:
            user_prompt += f"## 事件分析数据\n{event_analysis}\n\n"
        if quality_risk_analysis:
            user_prompt += f"## 质量风险数据\n{quality_risk_analysis}\n\n"
        if moneyflow_analysis:
            user_prompt += f"## 资金面数据\n{moneyflow_analysis}\n\n"

        user_prompt += """请严格按照系统提示中的评分体系、JSON格式和打分参考进行打分。
每个维度的打分都要有具体的数据引用作为依据。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        llm_start = time.time()
        response = await llm.ainvoke(messages)
        llm_time = time.time() - llm_start

        content = response.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group(0)
            score_data = json_mod.loads(json_str)
        else:
            raise ValueError(f"LLM未返回有效JSON: {content[:500]}")

        total_time = time.time() - agent_start_time

        # 验证数据结构
        required_keys = ["score", "sub_scores", "rating", "reasoning", "risk_warning", "suggested_action", "time_horizon"]
        for key in required_keys:
            if key not in score_data:
                raise ValueError(f"打分结果缺少必要字段: {key}")

        sub_keys = ["fundamental_quality", "valuation", "quality_governance", "event_sustainability", "technical_confirmation", "industry_fit", "sentiment_narrative"]
        for key in sub_keys:
            if key not in score_data.get("sub_scores", {}):
                raise ValueError(f"子评分缺少必要字段: {key}")

        # 附加行业信息到输出
        score_data["detected_industry"] = detected_industry

        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="medium_term_scoring",
            input_messages=messages,
            output_content=json_mod.dumps(score_data, ensure_ascii=False),
            model_config={"model": resolved_model, "temperature": 0.6, "max_tokens": 16000, "thinking": "enabled" if thinking_enabled else "disabled"},
            execution_time=llm_time
        )

        execution_logger.log_agent_complete(agent_name, score_data, total_time, True)
        logger.info(f"{SUCCESS_ICON} MediumTermScorer: {company_name} 中线评分={score_data['score']} ({score_data['rating']}) 行业={detected_industry}")

        return score_data

    except Exception as e:
        logger.error(f"{ERROR_ICON} MediumTermScorer 打分失败: {e}", exc_info=True)
        total_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {}, total_time, False, str(e))
        raise
