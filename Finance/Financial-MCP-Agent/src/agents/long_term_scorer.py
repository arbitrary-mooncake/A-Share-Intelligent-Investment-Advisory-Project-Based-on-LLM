"""
LongTermScorer: 长线投资打分Agent (以年为单位，1-3年以上持仓)

必须经过4个分析Agent(fundamental/technical/value/news)的完整输出后才进行打分。

核心特性：
- 行业自适应打分：自动识别行业，注入行业估值基准和打分指引
- 长线专属逻辑：护城河+行业景气度为核心，技术面权重极低
- 跨行业可比：使用相对行业均值的百分位打分

与中线打分的核心区别：
- 中线关注"未来1-3个月的走势判断"，长线关注"3年以上的内在价值增长"
- 长线大幅降低技术面权重(5分 vs 中线15分)，大幅提高护城河和行业权重
- 新增"行业景气度"维度(20分)，这是中线没有的独立维度
- 估值判断侧重历史5年分位和绝对估值(DCF)，而非短期相对估值
- 风险评估更侧重公司治理、大股东行为、ESG等长期因素

评分体系：
1. 商业护城河(30分) - 护城河深度、ROE持续性、现金流质量、管理层
2. 行业景气度(20分) - 行业空间、竞争格局、政策支持、技术替代风险
3. 估值水平(20分) - 历史分位、相对估值、绝对估值、股息率
4. 成长性(15分) - 3-5年CAGR、第二增长曲线、增长质量
5. 技术面(5分) - 月线级别大趋势（权重极低）
6. 风险与治理(10分) - 治理风险、行业风险、ESG风险

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
)
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from dotenv import load_dotenv

load_dotenv(override=True)

logger = setup_logger(__name__)


async def long_term_scorer(
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
    长线投资打分 (1-3年以上持仓) — v2

    评分体系（总分100分）：
    - 基本面与资本回报(25分)
    - 财务质量/治理风险(20分)
    - 估值安全边际(15分)
    - 行业地位/商业质量(15分)
    - 资本配置/股东回报(10分)
    - 事件与政策风险(10分)
    - 技术确认(5分)
    """
    agent_name = "long_term_scorer"
    logger.info(f"{WAIT_ICON} LongTermScorer: 开始对 {company_name}({stock_code}) 进行长线打分")

    execution_logger = get_execution_logger()
    agent_start_time = time.time()

    # 自动识别行业
    all_analysis_text = f"{fundamental_analysis} {value_analysis} {news_analysis}"
    detected_industry = identify_industry(company_name, all_analysis_text)

    execution_logger.log_agent_start(agent_name, {
        "stock_code": stock_code,
        "company_name": company_name,
        "scoring_type": "long_term",
        "detected_industry": detected_industry,
        "has_fundamental": bool(fundamental_analysis),
        "has_technical": bool(technical_analysis),
        "has_value": bool(value_analysis),
        "has_news": bool(news_analysis),
        "has_event": bool(event_analysis),
        "has_quality_risk": bool(quality_risk_analysis),
        "has_moneyflow": bool(moneyflow_analysis),
    })

    try:
        # 模型配置：优先显式参数（快筛覆盖），否则使用 agent 分配的模型 (MiMo-V2.5-Pro)
        model_cfg = get_model_config_for_agent("long_term_scorer")
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

        system_prompt = f"""你是一位资深A股长线价值投资专家，参考巴菲特、芒格的价值投资理念，
专注于1-3年以上的长线投资。你擅长识别具有持久竞争优势和内在价值增长潜力的公司。

**重要时间信息：当前实际时间是 {current_time_info}**
**分析基准日期：{current_date}**

你的任务是综合基本面、技术面、估值和新闻四个维度的分析数据，
从长线价值投资的视角对股票进行量化打分。

{industry_context}

## 中线与长线视角的核心区别

请注意：同样的数据在中线和长线视角下的权重和解读完全不同。
- **长线不看短期波动**：一个季度业绩不及预期、短期技术面走弱，对长线评分影响极小
- **长线看重"不变的"**：商业模式是否持久、护城河是否加深、行业空间是否扩大
- **长线容忍"暂时的"**：短期估值可以略高（如果增长确定），但长期必须有安全边际
- **长线关心"管理层"**：管理层的能力和诚信在长线中至关重要，而在中线中可以忽略

## 评分体系（总分100分）

请严格按照以下7个维度进行打分：

### 1. 基本面与资本回报（满分25分）
ROE/ROIC长期持续性、资本配置能力(再投资vs分红vs回购)、自由现金流累积能力

评估子项：
- ROE/ROIC持续性(10分)：过去3-5年ROE是否持续>15%，趋势是上升还是下降，ROIC是否持续高于WACC
- 资本回报质量(8分)：增量资本回报率(ROIC)，是否每一元再投资产生超过一元的价值增长
- 自由现金流累积(7分)：自由现金流/净利润持续>1、自由现金流累计值、现金转化效率

打分参考：
- 22-25分：ROE持续>20%，ROIC>15%，自由现金流充沛，资本回报卓越
- 16-21分：ROE持续>15%，ROIC高于行业平均，自由现金流健康
- 9-15分：ROE 10-15%，资本回报中等，自由现金流一般
- 0-8分：ROE<10%或波动大，资本回报差，自由现金流持续为负

### 2. 财务质量/治理风险（满分20分）
财务真实性(利润现金含量)、减值/商誉/表外风险、治理结构与股东行为

评估子项：
- 财务真实性(8分)：经营性现金流/净利润的一致性、应收账款/营收比例是否异常、存货周转是否健康
- 减值风险(6分)：商誉占比、应收账款坏账风险、大额担保/表外负债
- 治理结构(6分)：大股东行为(质押比例、减持历史)、关联交易透明度、独立董事有效性

打分参考：
- 18-20分：财务真实可靠，利润现金含量>100%，无商誉减值风险，治理优良
- 14-17分：财务质量较好，存在少量瑕疵但不影响整体判断
- 8-13分：存在中等风险(商誉较高、关联交易较多等)，需持续关注
- 0-7分：存在重大风险(财务造假嫌疑、大股东高比例质押、商誉减值风险高等)

### 3. 估值安全边际（满分15分）
3-5年历史估值分位、跨周期估值区间、DCF/剩余收益等绝对估值锚

评估子项：
- 历史估值分位(6分)：PE/PB在过去3-5年的百分位，当前处于估值周期的什么位置
- 跨周期估值区间(5分)：历史估值中枢与波动区间，当前估值相对中枢的偏离程度
- 绝对估值锚(4分)：DCF/剩余收益模型的内在价值估算，市值与内在价值的折扣率

打分参考：
- 13-15分：历史分位<20%，显著低于估值中枢，绝对估值有30%+折扣
- 10-12分：历史分位20-40%，估值合理偏低，有一定安全边际
- 6-9分：历史分位40-70%，估值合理或略高
- 0-5分：历史分位>70%，或绝对估值显示高估

重要：请使用**相对行业均值的百分位**进行估值打分。不同行业的合理估值区间天差地别。

### 4. 行业地位/商业质量（满分15分）
护城河深度与类型、市场份额与定价权、行业生命周期阶段

评估子项：
- 护城河深度与类型(6分)：技术壁垒/品牌壁垒/规模效应/转换成本/网络效应/牌照壁垒的宽度与持久性
- 市场份额与定价权(5分)：市占率变化趋势、产品/服务的定价权、客户粘性
- 行业生命周期(4分)：行业处于成长期/成熟期/衰退期，未来3-5年行业CAGR预期

打分参考：
- 13-15分：拥有2种以上深厚护城河，市场份额领先且持续扩大，行业成长期
- 9-12分：护城河明确(至少1种类型)，市场份额稳定，行业成熟期
- 5-8分：有一定竞争优势但不突出，市场份额面临挑战
- 0-4分：无明显护城河，市场份额持续下滑，行业衰退

### 5. 资本配置/股东回报（满分10分）
分红历史与可持续性、回购意愿、资本开支纪律

评估子项：
- 分红历史与可持续性(4分)：连续分红年数、分红率、股息率与无风险利率的对比
- 回购意愿(3分)：是否有回购历史、回购规模、回购是否注销
- 资本开支纪律(3分)：管理层是否理性使用自由现金流（偏好再投资vs分红vs回购），是否存在盲目扩张

打分参考：
- 9-10分：连续5年+分红，分红率合理，有回购历史，资本开支理性
- 6-8分：分红稳定，资本配置基本合理
- 3-5分：分红不稳定或少，资本配置存在问题
- 0-2分：长期不分红，资本开支粗放或存在利益输送

### 6. 事件与政策风险（满分10分）
长期政策方向(产业政策/监管框架)、宏观系统性风险、地缘风险

评估子项：
- 产业政策风险(4分)：行业是否受政策支持，是否存在被政策打压的风险
- 监管框架(3分)：监管环境是否稳定，合规成本变化趋势
- 宏观/地缘风险(3分)：宏观周期位置、地缘政治对业务的可能影响

打分参考（分数越高风险越低）：
- 9-10分：政策明确支持，监管环境稳定，无宏观/地缘风险
- 6-8分：政策中性，监管环境可预期，有一定宏观风险但可控
- 3-5分：面临一定政策/监管不确定性，宏观风险中等
- 0-2分：政策打压风险高，监管框架剧烈变化，地缘风险大

### 7. 技术确认（满分5分）
仅用月线级别趋势判断极端高/低位，不过度放大短期价格行为

评估子项：
- 月线级别趋势(3分)：长期是上升通道/横盘/下降通道
- 极端位置判断(2分)：是否处于历史极端高位/低位（月线RSI超买/超卖、布林带极值）

打分参考：
- 5分：月线明确上升通道，且非历史极端高位
- 3-4分：月线横盘/上升初期，或处于合理估值区间
- 1-2分：月线下降通道，或处于历史极端高位
- 0分：月线明确下降且处于历史极端低位（需警惕价值陷阱）
（技术确认权重极低，仅用于辅助判断极端区域，不应主导长线决策）

风险扣分：由risk_gate统一后处理

## 综合评级标准

根据总分给出长线投资评级：
- 85-100分：强烈买入（长线核心标的，建议重仓持有）
- 70-84分：买入（长线优质标的，建议配置）
- 55-69分：持有（基本面不错但估值或行业有待观察）
- 40-54分：观望（存在较多不确定性，建议等待更好时机）
- 25-39分：规避（长线逻辑不够强，不建议作为核心持仓）
- 0-24分：卖出（基本面恶化或存在重大风险）

## 输出要求

请以JSON格式返回打分结果，必须包含以下所有字段：

{{
    "score": 总分(0-100的整数),
    "sub_scores": {{
        "fundamental_returns": 基本面与资本回报得分(0-25的整数),
        "quality_governance": 财务质量/治理风险得分(0-20的整数),
        "valuation_margin": 估值安全边际得分(0-15的整数),
        "business_quality": 行业地位/商业质量得分(0-15的整数),
        "capital_allocation": 资本配置/股东回报得分(0-10的整数),
        "event_policy_risk": 事件与政策风险得分(0-10的整数，分数越高风险越低),
        "technical_confirmation": 技术确认得分(0-5的整数)
    }},
    "rating": "长线投资评级（强烈买入/买入/持有/观望/规避/卖出）",
    "reasoning": "打分核心理由（3-5句话，指出最强的2-3个因素和最弱的1-2个因素，特别要说明护城河和基本面资本回报的判断依据）",
    "risk_warning": "长线风险提示（1-2句话，指出未来3年最大的1-2个风险点）",
    "data_basis": {{
        "fundamental_returns": "列出本维度评分依据的具体数据点",
        "quality_governance": "列出本维度评分依据的具体数据点",
        "valuation_margin": "列出本维度评分依据的具体数据点",
        "business_quality": "列出本维度评分依据的具体数据点",
        "capital_allocation": "列出本维度评分依据的具体数据点",
        "event_policy_risk": "列出本维度评分依据的具体数据点",
        "technical_confirmation": "列出本维度评分依据的具体数据点"
    }},
    "data_reliability": "数据可靠性(高/中/低)，注明哪些基于直接数据，哪些基于推断",
    "suggested_action": "具体操作建议（2-3句话，含建议建仓策略如"分批建仓，首次建仓30%，回调加仓"）",
    "time_horizon": "建议持仓周期（如：建议持有3年以上，核心逻辑是XX）",
    "moat_type": "护城河类型（如：技术壁垒+品牌壁垒 / 规模效应+转换成本 / 无明显护城河）"
}}

## 重要约束

1. 所有分数必须是**整数**，不能是小数
2. 各子项分数之和必须严格等于总分
3. **评分前必须先列出每个维度的数据依据**（见data_basis字段），数据不充分时必须在data_reliability中标注"低"
4. 打分必须基于提供的各维分析数据，不能凭空臆断；如果某项数据缺失，在data_basis中标注"无直接数据，基于已有信息推断"
5. 每个维度的打分必须有具体数据引用
6. 长线打分应**以基本面和资本回报为核心**，技术确认权重极低
7. 如果某个维度的分析数据缺失，请在reasoning中说明
8. 输出必须是纯JSON，不要包含markdown代码块标记
"""

        # 优先注入结构化分析上下文
        if analysis_package and hasattr(analysis_package, 'compact_prompt_context'):
            structured_context = analysis_package.compact_prompt_context
        else:
            structured_context = ""

        user_prompt = f"""请对以下股票进行长线投资打分：

公司名称：{company_name}
股票代码：{stock_code}
{"行业：" + detected_industry if detected_industry else "（行业待识别，请根据分析内容自行判断）"}

"""
        if structured_context:
            user_prompt += f"## 结构化分析摘要（优先参考）\n{structured_context}\n\n"

        if fundamental_analysis:
            user_prompt += f"## 基本面分析\n{fundamental_analysis}\n\n"
        else:
            user_prompt += "## 基本面分析\n（暂无基本面分析数据）\n\n"

        if technical_analysis:
            user_prompt += f"## 技术分析\n{technical_analysis}\n\n"
        else:
            user_prompt += "## 技术分析\n（暂无技术分析数据）\n\n"

        if value_analysis:
            user_prompt += f"## 估值分析\n{value_analysis}\n\n"
        else:
            user_prompt += "## 估值分析\n（暂无估值分析数据）\n\n"

        if news_analysis:
            user_prompt += f"## 新闻分析\n{news_analysis}\n\n"
        else:
            user_prompt += "## 新闻分析\n（暂无新闻分析数据）\n\n"

        if event_analysis:
            user_prompt += f"## 事件分析数据\n{event_analysis}\n\n"

        if quality_risk_analysis:
            user_prompt += f"## 财务质量/治理风险分析数据\n{quality_risk_analysis}\n\n"

        if moneyflow_analysis:
            user_prompt += f"## 资金面分析数据\n{moneyflow_analysis}\n\n"

        user_prompt += """
请从长线价值投资视角，严格按照系统提示中的评分体系、JSON格式和打分参考进行打分。
特别注意：
1. 基本面和资本回报是长线最核心的判断维度
2. 技术确认权重极低，不要因为短期走势否定长线逻辑
3. 每个维度的打分都要有具体数据引用作为依据
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
        required_keys = ["score", "sub_scores", "rating", "reasoning", "risk_warning", "suggested_action", "time_horizon", "moat_type"]
        for key in required_keys:
            if key not in score_data:
                raise ValueError(f"打分结果缺少必要字段: {key}")

        sub_keys = ["fundamental_returns", "quality_governance", "valuation_margin", "business_quality", "capital_allocation", "event_policy_risk", "technical_confirmation"]
        for key in sub_keys:
            if key not in score_data.get("sub_scores", {}):
                raise ValueError(f"子评分缺少必要字段: {key}")

        score_data["detected_industry"] = detected_industry

        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="long_term_scoring",
            input_messages=messages,
            output_content=json_mod.dumps(score_data, ensure_ascii=False),
            model_config={"model": resolved_model, "temperature": 0.6, "max_tokens": 16000, "thinking": "enabled" if thinking_enabled else "disabled"},
            execution_time=llm_time
        )

        execution_logger.log_agent_complete(agent_name, score_data, total_time, True)
        logger.info(f"{SUCCESS_ICON} LongTermScorer: {company_name} 长线评分={score_data['score']} ({score_data['rating']}) 行业={detected_industry}")

        return score_data

    except Exception as e:
        logger.error(f"{ERROR_ICON} LongTermScorer 打分失败: {e}", exc_info=True)
        total_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {}, total_time, False, str(e))
        raise
