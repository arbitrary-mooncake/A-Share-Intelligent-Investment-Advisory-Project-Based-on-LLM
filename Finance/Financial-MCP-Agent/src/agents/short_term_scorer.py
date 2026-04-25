"""
ShortTermScorer: 短线投资打分Agent (以天为单位，1-5个交易日)

特点：
- 不经过fundamental_agent和value_agent，仅依赖technical_agent和news_agent的输出
- 行业自适应：不同行业的短线波动特征、资金行为、情绪催化完全不同
- 核心权重集中在：量价关系、技术信号、趋势动量、情绪资金
- 使用thinking模式(max_tokens=16000)保证打分准确性
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
    INDUSTRY_BENCHMARKS,
)
from dotenv import load_dotenv

load_dotenv(override=True)

logger = setup_logger(__name__)


async def short_term_scorer(
    stock_code: str,
    company_name: str,
    technical_analysis: str = "",
    news_analysis: str = "",
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """
    短线投资打分 (1-5个交易日持仓)

    评分体系（总分100分）：
    - 量价关系(30分)
    - 技术信号(25分)
    - 趋势动量(20分)
    - 情绪资金(25分)
    """
    agent_name = "short_term_scorer"
    logger.info(f"{WAIT_ICON} ShortTermScorer: 开始对 {company_name}({stock_code}) 进行短线打分")

    execution_logger = get_execution_logger()
    agent_start_time = time.time()

    # 自动识别行业
    all_analysis_text = f"{technical_analysis} {news_analysis}"
    detected_industry = identify_industry(company_name, all_analysis_text)

    execution_logger.log_agent_start(agent_name, {
        "stock_code": stock_code,
        "company_name": company_name,
        "scoring_type": "short_term",
        "detected_industry": detected_industry,
        "has_technical": bool(technical_analysis),
        "has_news": bool(news_analysis),
    })

    # 根据行业生成短线特征指引
    industry_short_guidance = _get_short_term_industry_guidance(detected_industry)

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

        system_prompt = f"""你是一位资深A股短线交易专家，专注于1-5个交易日的短线操作策略。

**重要时间信息：当前实际时间是 {current_time_info}**
**分析基准日期：{current_date}**

## A股短线交易规则

以下规则直接影响短线操作策略，请在打分时充分考虑：

### 1. T+1交易制度
- 当日买入的股票，次一交易日才能卖出（不可当日卖出）
- 这意味着：当日追涨买入后即使判断正确，也无法在当日兑现收益
- 打板策略必须考虑次日开盘承接力，避免T+1导致次日低开无法止损

### 2. 涨跌停板制度
- **主板**（沪市60开头/深市00开头）：±10%
- **创业板**（深市30开头）：±20%
- **科创板**（沪市68开头）：±20%
- **北交所**（8开头/4开头）：±30%
- **ST股**：±10%（2026年4月新规，此前为±5%）
- *ST股：±10%（2026年4月新规，此前为±5%）

涨跌停板直接影响短线策略：
- 涨停板股票次日可能继续高开，但也可能是诱多陷阱
- 跌停板股票次日可能继续低开，T+1制度下当日无法止损
- 涨停/跌停次数越多，短线情绪越强/弱

### 3. ST/*ST股票特别风险提示
- ST股票表示公司存在其他风险状况异常（如连续两年亏损）
- *ST股票表示存在退市风险
- ST/*ST股票虽然涨跌幅放宽到10%，但流动性可能骤降
- 短线操作ST/*ST股票需特别谨慎，最大风险是突发退市

你的任务是综合已有分析数据，对股票进行短线量化打分。

{industry_short_guidance}

## 评分体系（总分100分）

请严格按照以下4个维度进行打分，每个维度必须有具体数据支撑：

### 1. 量价关系（满分30分）
评估标准：
- 成交量异动：近期是否出现明显放量/缩量（对比5/10日均量）
- 量价配合：上涨放量、下跌缩量为健康；量价背离为隐患
- 换手率：是否处于异常水平
- 主力资金流向：近期主力是净流入还是净流出

打分参考：
- 25-30分：量价关系健康，放量上涨/缩量回调，主力资金持续净流入
- 18-24分：量价关系尚可，偶有背离但不严重
- 10-17分：量价关系一般，存在一定背离
- 0-9分：量价关系恶化，放量下跌/缩量上涨，资金持续流出

### 2. 技术信号（满分25分）
评估标准：
- MACD：金叉/死叉状态，红绿柱趋势，DIF/DEA数值关系
- RSI：是否在超买(>70)或超卖(<30)区域，趋势方向
- K线形态：是否出现关键K线形态（如吞没、锤子线、十字星、突破形态）
- 布林带：股价在布林带中的位置，是否面临突破

打分参考：
- 22-25分：多个技术指标发出明确买入信号
- 16-21分：技术指标偏多，但信号不够强烈
- 8-15分：技术指标中性或存在矛盾信号
- 0-7分：技术指标发出明确卖出信号

### 3. 趋势动量（满分20分）
评估标准：
- 均线排列：5日/10日/20日均线的排列方向（多头/空头）
- 趋势方向：短期趋势是上升/下降/横盘
- 动量强度：近3日/5日涨跌幅，是否加速/减速
- 关键价位：是否突破关键支撑/阻力位

打分参考：
- 18-20分：明确上升趋势，均线多头排列，动量强劲
- 13-17分：趋势偏多但不够强势
- 7-12分：趋势不明确或横盘
- 0-6分：下降趋势，均线空头排列

### 4. 情绪资金（满分25分）
评估标准：
- 新闻情绪：近期新闻是利好还是利空
- 市场关注度：投资者情绪是否积极
- 板块效应：所属板块整体表现如何
- 短期催化剂：是否有即将到来的事件驱动（如业绩公告、政策发布）

打分参考：
- 22-25分：情绪极度乐观，资金大量流入，有明确催化剂
- 16-21分：情绪偏多，有一定资金关注
- 8-15分：情绪中性，资金进出平衡
- 0-7分：情绪悲观，资金持续流出

## 综合评级标准

根据总分给出短线建议：
- 85-100分：强烈买入（短线极佳机会，果断入场）
- 70-84分：买入（短线偏多，可以参与）
- 55-69分：谨慎买入（有一定机会但需控制仓位）
- 40-54分：观望（方向不明确，等待更明确信号）
- 25-39分：谨慎卖出（偏空信号，考虑减仓）
- 0-24分：卖出（短线风险大，建议离场）

## 输出要求

请以JSON格式返回打分结果，必须包含以下字段：

{{
    "score": 总分(0-100的整数),
    "sub_scores": {{
        "volume_price": 量价关系得分(0-30的整数),
        "technical_signal": 技术信号得分(0-25的整数),
        "trend_momentum": 趋势动量得分(0-20的整数),
        "sentiment_flow": 情绪资金得分(0-25的整数)
    }},
    "recommendation": "短线建议（强烈买入/买入/谨慎买入/观望/谨慎卖出/卖出/强烈卖出）",
    "reasoning": "打分核心理由（2-3句话，指出最关键的1-2个因素）",
    "risk_warning": "短线风险提示（1句话，指出最大风险点）",
    "suggested_action": "具体操作建议（1-2句话，含止损价位参考）"
}}

## 重要约束

1. 所有分数必须是**整数**，不能是小数
2. 各子项分数之和必须等于总分
3. 打分必须基于提供的分析数据，不能凭空臆断
4. 如果某些数据缺失（如没有新闻分析），请说明并在其他维度中适当调整权重
5. 短线打分应以**价格行为和量价关系**为核心
6. 输出必须是纯JSON，不要包含markdown代码块标记
"""

        user_prompt = f"""请对以下股票进行短线投资打分：

公司名称：{company_name}
股票代码：{stock_code}
{"行业：" + detected_industry if detected_industry else "（行业待识别）"}

"""
        if technical_analysis:
            user_prompt += f"## 技术分析数据\n{technical_analysis}\n\n"
        else:
            user_prompt += "## 技术分析数据\n（暂无数据）\n\n"

        if news_analysis:
            user_prompt += f"## 新闻分析数据\n{news_analysis}\n\n"
        else:
            user_prompt += "## 新闻分析数据\n（暂无数据）\n\n"

        user_prompt += f"""
请严格按照系统提示中的评分体系和JSON格式输出打分结果。
如果某些数据缺失，请在reasoning中说明，但仍需给出合理评分。
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
        required_keys = ["score", "sub_scores", "recommendation", "reasoning", "risk_warning", "suggested_action"]
        for key in required_keys:
            if key not in score_data:
                raise ValueError(f"打分结果缺少必要字段: {key}")

        score_data["detected_industry"] = detected_industry

        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="short_term_scoring",
            input_messages=messages,
            output_content=json_mod.dumps(score_data, ensure_ascii=False),
            model_config={"model": model_name, "temperature": 1.0, "max_tokens": 16000, "thinking": "enabled"},
            execution_time=llm_time
        )

        execution_logger.log_agent_complete(agent_name, score_data, total_time, True)
        logger.info(f"{SUCCESS_ICON} ShortTermScorer: {company_name} 短线评分={score_data['score']} ({score_data['recommendation']})")

        return score_data

    except Exception as e:
        logger.error(f"{ERROR_ICON} ShortTermScorer 打分失败: {e}", exc_info=True)
        total_time = time.time() - agent_start_time
        execution_logger.log_agent_complete(agent_name, {}, total_time, False, str(e))
        raise


def _get_short_term_industry_guidance(industry: str) -> str:
    """
    生成短线打分的行业特征指引

    不同行业的短线特征完全不同：
    - 银行/消费：波动小、资金行为以机构为主、情绪催化慢
    - 科技/军工：波动大、游资偏好、情绪催化快
    - 周期股：受商品价格影响大
    """
    if not industry:
        return """
## 短线行业特征提醒

不同行业的短线波动和资金行为差异很大：
- **银行/大盘蓝筹**：波动小(日涨跌幅多在±2%以内)，换手率低，机构主导，短线机会少
- **消费/医药**：波动中等，趋势相对稳定，适合波段操作
- **科技/军工/AI**：波动大，游资活跃，情绪催化强，短线机会多但风险也大
- **周期股**：受商品价格和事件驱动影响大
- **小盘/题材股**：换手率极高，游资主导，情绪是主要驱动力

请根据股票所属行业特征调整打分判断。
"""

    guidance_map = {
        "银行": """
## 行业短线特征：银行
- 波动极小：日涨跌幅多在±2%以内，极少出现涨停/跌停
- 换手率低：通常<1%，流动性主要靠机构调仓
- 资金行为：以公募、险资等长期资金为主，游资极少参与
- 短线机会：主要来自政策利好（如降息、降准）或分红季
- 评分注意：银行短线评分>70已属极高，不要期望短线暴涨
""",
        "计算机": """
## 行业短线特征：计算机/科技
- 波动大：日涨跌幅常见±3-5%，题材炒作活跃
- 换手率高：中小盘科技股换手率常>5%
- 资金行为：游资主导，机构也有但比例不如银行
- 短线机会：AI/信创等政策催化时容易出现快速拉升
- 评分注意：高波动意味着高分和低分都可能出现，情绪面权重应提高
""",
        "电子": """
## 行业短线特征：电子/半导体
- 波动大：受产业周期和全球半导体政策影响大
- 换手率高：热门半导体股日换手率可达10%+
- 资金行为：游资+机构混合，北向资金也较活跃
- 短线机会：半导体周期拐点、国产替代政策催化
- 评分注意：半导体短线看产业催化和板块共振
""",
        "医药生物": """
## 行业短线特征：医药
- 波动中等偏大：集采/创新药审批等事件驱动
- 换手率中等：热门创新药/医药概念股活跃
- 短线机会：新药获批、集采结果公布、医保谈判
- 评分注意：医药短线看政策事件驱动，平时趋势性不强
""",
        "食品饮料": """
## 行业短线特征：消费/食品饮料
- 波动中等：大盘蓝筹为主，但白酒等子板块波动较大
- 资金行为：机构主导为主，北向资金影响大
- 短线机会：业绩超预期、消费政策利好、白酒提价
- 评分注意：消费短线偏趋势性操作，突发暴涨概率低
""",
        "有色金属": """
## 行业短线特征：有色金属/周期
- 波动大：受国际大宗商品价格影响
- 短线机会：商品期货价格异动、供需格局变化
- 评分注意：周期股短线看商品价格走势和供给侧消息
""",
        "国防军工": """
## 行业短线特征：军工
- 波动极大：题材属性强，容易出现连续涨停
- 换手率极高：游资高度参与
- 短线机会：军费公布、军演、装备采购、地缘政治事件
- 评分注意：军工短线情绪权重极高，技术信号次之
""",
        "汽车": """
## 行业短线特征：汽车/新能源
- 波动中等偏大：新能源子板块波动更大
- 短线机会：销量数据超预期、新车型发布、政策补贴
- 评分注意：新能源看月度销量和智能化进展催化
""",
    }

    return guidance_map.get(industry, """
## 短线行业特征提醒
请根据该行业的特点（波动率、换手率、资金结构、催化事件类型）调整短线打分判断。
""")
