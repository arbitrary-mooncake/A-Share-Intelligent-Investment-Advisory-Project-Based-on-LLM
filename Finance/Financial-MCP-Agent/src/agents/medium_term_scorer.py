"""
MediumTermScorer: 中线投资打分Agent (以月为单位，1-3个月持仓)

本Agent是产品的核心竞争力和主要卖点。
必须经过4个分析Agent(fundamental/technical/value/news)的完整输出后才进行打分。

核心特性：
- 行业自适应打分：自动识别行业，注入行业估值基准和打分指引
- 跨行业可比：使用相对行业均值的百分位打分，而非绝对PE/PB数值
- 权重框架统一不变，但每个维度内的打分参考随行业调整

评分体系参照券商金工多因子量化选股框架：
1. 基本面质量(25分) - ROE质量、盈利能力、现金流、资产负债
2. 成长性(15分) - 营收/利润增速、行业对比、增长持续性
3. 估值水平(20分) - PE/PB/PS vs 行业、历史分位、安全边际
4. 技术趋势(15分) - 中期趋势、均线系统、关键价位
5. 情绪资金(10分) - 新闻情绪、机构关注度、资金流向
6. 风险评估(15分) - 政策/财务/市场/流动性风险

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
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """
    中线投资打分 (1-3个月持仓)

    评分体系（总分100分）：
    - 基本面质量(25分)
    - 成长性(15分)
    - 估值水平(20分)
    - 技术趋势(15分)
    - 情绪资金(10分)
    - 风险评估(15分)
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
    })

    try:
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        model_name = os.getenv("OPENAI_COMPATIBLE_MODEL")

        if not all([api_key, base_url, model_name]):
            raise ValueError("缺少OpenAI环境变量")

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=1.0,
            max_tokens=16000,
            extra_body={"thinking": {"type": "enabled"}}
        )

        # 生成行业上下文指引
        industry_context = generate_industry_context_prompt(detected_industry)

        system_prompt = f"""你是一位资深A股中线投资专家，专注于1-3个月的中线波段操作策略。
你所在的券商研究所以"基本面+技术面+估值"三维共振为核心方法论，
在过往的中线推荐中取得了显著的超额收益。

**重要时间信息：当前实际时间是 {current_time_info}**
**分析基准日期：{current_date}**

你的任务是综合基本面、技术面、估值和新闻四个维度的分析数据，
对股票进行中线量化打分。

{industry_context}

## 评分体系（总分100分）

请严格按照以下6个维度进行打分，每个维度必须给出明确的数据支撑和打分依据：

### 1. 基本面质量（满分25分）— 核心维度
评估子项：
- ROE质量(8分)：ROE水平及其质量、杜邦分析拆解(净利率×资产周转率×权益乘数)
- 盈利能力(6分)：毛利率、净利率趋势及行业对比
- 现金流质量(6分)：经营性现金流/净利润比率、自由现金流状况
- 负债与偿债(5分)：资产负债率、流动比率、有息负债占比

打分参考：
- 22-25分：基本面优秀，各项指标均显著优于行业平均水平
- 16-21分：基本面良好，个别指标略逊但不影响整体质量
- 9-15分：基本面一般，存在一定瑕疵但不致命
- 0-8分：基本面恶化或存在明显风险

### 2. 成长性（满分15分）
评估子项：
- 营收增速(5分)：同比/环比增速，是否加速
- 利润增速(5分)：净利润增速，与营收增速对比
- 行业对比(3分)：增速是否高于行业中枢
- 增长持续性(2分)：过去4个季度是否持续增长

打分参考：
- 13-15分：增速显著高于行业平均水平且持续加速
- 9-12分：增速符合或略高于行业平均水平
- 5-8分：增速低于行业平均或增长停滞
- 0-4分：负增长或业绩下滑

注意：不同行业的"高增长"标准不同。银行5-8%就算不错，但科技股需要20%+才算高增长。

### 3. 估值水平（满分20分）
评估子项：
- PE估值(7分)：当前PE TTM vs 行业均值 vs 历史3-5年分位
- PB估值(5分)：当前PB vs 行业均值，是否处于合理区间
- 安全边际(5分)：当前股价 vs 内在价值，是否有足够安全边际
- 股息率(3分)：股息率与无风险利率及行业分红的对比

打分参考：
- 18-20分：显著低估（相对行业均值和历史分位），安全边际充足
- 13-17分：估值合理，未明显高估也未明显低估
- 7-12分：估值略高，但可由增长预期支撑
- 0-6分：明显高估，缺乏安全边际

重要：请使用**相对行业均值的百分位**进行估值打分，不要使用绝对PE/PB数值。
- 银行PE 7倍可能是合理估值（行业中位数6-7倍）
- 科技PE 60倍可能是低估（行业中位数55倍，且增速快）

### 4. 技术趋势（满分15分）
评估子项：
- 中期趋势(5分)：20日/60日均线的排列方向，趋势是否明确
- 关键价位(5分)：当前股价与关键支撑/阻力位的关系
- 量价配合(3分)：中期量价关系是否健康
- MACD趋势(2分)：日线/周线MACD方向

打分参考：
- 13-15分：明确上升中期趋势，均线多头排列，量价健康
- 9-12分：趋势偏多但面临关键阻力
- 5-8分：趋势不明确或横盘整理
- 0-4分：中期下降趋势

### 5. 情绪资金（满分10分）
评估子项：
- 新闻情绪(4分)：近期新闻整体是利好还是利空
- 机构关注度(3分)：券商研报覆盖数量、评级变化趋势
- 资金流向(3分)：主力资金近期净流入/流出

打分参考：
- 9-10分：极度乐观，机构持续上调评级，资金大幅流入
- 6-8分：情绪偏多，有一定资金关注
- 3-5分：情绪中性，资金进出平衡
- 0-2分：情绪悲观，资金持续流出

### 6. 风险评估（满分15分）— 注意：此维度得分越高表示风险越低
评估子项：
- 政策风险(3分)：行业政策是否支持，有无重大政策风险
- ST/*ST风险(4分)：公司是否已被或可能被实施ST/*ST。连续两年净利润为负、净资产为负、审计报告被出具无法表示意见等均可能触发。ST/*ST股票中线操作应坚决规避
- 财务风险(3分)：有无应收账款暴雷、商誉减值、大股东质押等风险
- 市场风险(3分)：行业景气度变化、竞争加剧风险
- 流动性风险(2分)：日均成交额是否充足（ST/*ST股票流动性可能骤降）

打分参考：
- 13-15分：各项风险均低，安全垫厚
- 9-12分：存在一定风险但可控
- 5-8分：存在中等风险，需关注
- 0-4分：存在显著风险，可能影响中线走势

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
        "fundamental_quality": 基本面质量得分(0-25的整数),
        "growth": 成长性得分(0-15的整数),
        "valuation": 估值水平得分(0-20的整数),
        "technical_trend": 技术趋势得分(0-15的整数),
        "sentiment_flow": 情绪资金得分(0-10的整数),
        "risk_assessment": 风险评估得分(0-15的整数，分数越高表示风险越低)
    }},
    "rating": "投资评级（强烈推荐/推荐/谨慎推荐/中性/谨慎减持/减持）",
    "reasoning": "打分核心理由（3-5句话，指出最强的2-3个因素和最弱的1-2个因素）",
    "risk_warning": "中线风险提示（1-2句话，指出最大1-2个风险点）",
    "suggested_action": "具体操作建议（2-3句话，含建议买入价位区间和目标价位区间）",
    "time_horizon": "建议持仓周期（如：建议持有1-3个月，关注XX事件催化）"
}}

## 重要约束

1. 所有分数必须是**整数**，不能是小数
2. 各子项分数之和必须严格等于总分
3. 打分必须基于提供的四维分析数据，不能凭空臆断
4. 每个维度的打分必须有具体数据引用（如"ROE为18%，属于优秀水平，得7/8分"）
5. 中线打分应**综合权衡**，不能过度偏重某一个维度
6. 如果某个维度的分析数据缺失，请在reasoning中说明并酌情调整
7. 输出必须是纯JSON，不要包含markdown代码块标记
"""

        user_prompt = f"""请对以下股票进行中线投资打分：

公司名称：{company_name}
股票代码：{stock_code}
{"行业：" + detected_industry if detected_industry else "（行业待识别，请根据分析内容自行判断）"}

## 基本面分析
{fundamental_analysis if fundamental_analysis else "（暂无基本面分析数据）"}

## 技术分析
{technical_analysis if technical_analysis else "（暂无技术分析数据）"}

## 估值分析
{value_analysis if value_analysis else "（暂无估值分析数据）"}

## 新闻分析
{news_analysis if news_analysis else "（暂无新闻分析数据）"}

请严格按照系统提示中的评分体系、JSON格式和打分参考进行打分。
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

        sub_keys = ["fundamental_quality", "growth", "valuation", "technical_trend", "sentiment_flow", "risk_assessment"]
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
            model_config={"model": model_name, "temperature": 1.0, "max_tokens": 16000, "thinking": "enabled"},
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
