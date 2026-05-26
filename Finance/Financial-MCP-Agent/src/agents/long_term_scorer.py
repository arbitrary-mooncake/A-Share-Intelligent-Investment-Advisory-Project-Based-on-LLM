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
from src.utils.model_config import get_model_config_for_agent
from dotenv import load_dotenv

load_dotenv(override=True)

logger = setup_logger(__name__)


def _build_extra_body(base_url: str, thinking_enabled: bool) -> dict:
    """根据 API 提供商返回正确的 thinking 参数格式"""
    if not thinking_enabled:
        return {}
    if "dashscope" in base_url:
        return {"enable_thinking": True}
    return {"thinking": {"type": "enabled"}}


async def long_term_scorer(
    stock_code: str,
    company_name: str,
    fundamental_analysis: str = "",
    technical_analysis: str = "",
    value_analysis: str = "",
    news_analysis: str = "",
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
    model_name: str = "",
    model_api_key: str = "",
    model_base_url: str = "",
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
    """
    长线投资打分 (1-3年以上持仓)

    评分体系（总分100分）：
    - 商业护城河(30分)
    - 行业景气度(20分)
    - 估值水平(20分)
    - 成长性(15分)
    - 技术面(5分)
    - 风险与治理(10分)
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
            temperature=1.0,
            request_timeout=360,
            max_tokens=16000,
            extra_body=_build_extra_body(base_url, thinking_enabled)
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

请严格按照以下6个维度进行打分：

### 1. 商业护城河（满分30分）— 长线最核心维度
评估子项：
- 护城河类型与深度(12分)：
  * 技术壁垒：是否拥有专利、核心技术领先优势（如芯片设计、创新药）
  * 品牌壁垒：品牌认知度和忠诚度（如茅台、片仔癀）
  * 规模效应：成本优势、市场份额（如海天味业）
  * 转换成本：客户粘性强弱（如SaaS、工业软件）
  * 网络效应：用户越多价值越大（如微信、支付宝）
  * 牌照/特许经营权：政策准入壁垒（如机场、高速公路）
- ROE持续性(8分)：过去3年ROE是否持续>15%，趋势是上升还是下降
- 现金流质量(6分)：经营现金流/净利润持续>1、自由现金流持续为正
- 资本配置能力(4分)：管理层是否理性使用自由现金流（再投资、分红、回购，而非盲目扩张）

打分参考：
- 26-30分：拥有2种以上类型的深厚护城河，ROE持续优秀，现金流卓越 — 伟大公司标准
- 20-25分：护城河明确(至少1种类型)，ROE持续>15%，现金流健康
- 12-19分：有一定竞争优势但不突出，ROE 10-15%
- 0-11分：缺乏持久竞争优势，ROE<10%或波动大

### 2. 行业景气度（满分20分）— 长线独有维度
评估子项：
- 行业成长空间(7分)：TAM大小、渗透率阶段、未来3-5年行业CAGR预期
- 竞争格局(6分)：CR5集中度、公司市占率变化、新进入者威胁
- 政策环境(4分)：国家政策方向、产业政策支持度
- 技术替代风险(3分)：是否存在颠覆性技术威胁

打分参考：
- 18-20分：行业高速成长，竞争格局利好龙头，政策支持，无替代风险
- 13-17分：行业稳健增长，竞争格局稳定，政策中性
- 7-12分：行业增速放缓，竞争加剧
- 0-6分：行业萎缩或面临重大替代风险

### 3. 估值水平（满分20分）— 长线侧重历史分位和绝对估值
评估子项：
- 历史估值分位(8分)：PE/PB在过去3-5年的百分位
- 相对估值(5分)：与行业均值对比、PEG
- 绝对估值/安全边际(5分)：基于自由现金流的内在价值估算
- 股息率(2分)：长期持有的现金回报

打分参考：
- 18-20分：显著低估（相对行业均值和历史分位），安全边际充足
- 13-17分：合理估值，有一定安全边际
- 7-12分：略高估，但增长可消化
- 0-6分：明显高估，缺乏安全边际

重要：请使用**相对行业均值的百分位**进行估值打分。不同行业的合理估值区间天差地别。

### 4. 成长性（满分15分）— 长线看3-5年的持续增长
评估子项：
- 营收/利润3-5年CAGR(6分)
- 第二增长曲线(5分)：新产品/新业务/新市场拓展
- 增长质量(4分)：内生vs并购、是否伴随利润率提升

打分参考：
- 13-15分：高CAGR，第二曲线清晰且已起量，内生增长为主
- 9-12分：稳健CAGR，有第二曲线潜力
- 5-8分：低速CAGR，增长依赖单一业务
- 0-4分：增长停滞或负增长

注意：不同行业的增速标准不同。银行5-8%就算稳健增长，科技股需要20%+。

### 5. 技术面（满分5分）— 长线权重极低，仅看大趋势
评估子项：
- 月线级别趋势(3分)：长期是上升通道还是下降通道
- 长期均线(2分)：股价是否在年线(250日)上方

打分参考：
- 5分：月线明确上升，股价在年线上方
- 3分：月线横盘或刚转多
- 1分：月线仍在下降通道
（技术面对长线影响极小，即使技术面差也不应否决基本面优秀的长线标的）

### 6. 风险与治理（满分10分）— 分数越高表示风险越低
评估子项：
- 公司治理(3分)：大股东行为、管理层稳定性、关联交易
- ST/*ST风险(4分)：长线投资必须排除ST/*ST股票。关注连续亏损、净资产为负、审计报告无法表示意见、重大违法等退市指标。ST/*ST股票长线评分应直接给出极低分数
- 监管/合规风险(3分)：未来1-3年是否面临重大政策变化、环保处罚、财务造假嫌疑等

打分参考：
- 9-10分：治理优秀，无ST风险，无重大合规隐患
- 6-8分：有一定风险但可控，不存在ST风险
- 3-5分：存在中等风险，可能面临ST风险需持续关注
- 0-2分：存在重大风险隐患，已触发或即将触发ST/*ST

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
        "business_moat": 商业护城河得分(0-30的整数),
        "industry_outlook": 行业景气度得分(0-20的整数),
        "valuation": 估值水平得分(0-20的整数),
        "growth": 成长性得分(0-15的整数),
        "technical": 技术面得分(0-5的整数),
        "risk_governance": 风险与治理得分(0-10的整数，分数越高表示风险越低)
    }},
    "rating": "长线投资评级（强烈买入/买入/持有/观望/规避/卖出）",
    "reasoning": "打分核心理由（3-5句话，指出最强的2-3个因素和最弱的1-2个因素，特别要说明护城河和行业景气的判断依据）",
    "risk_warning": "长线风险提示（1-2句话，指出未来3年最大的1-2个风险点）",
    "data_basis": {{
        "business_moat": "列出本维度评分依据的具体数据点",
        "industry_outlook": "列出本维度评分依据的具体数据点",
        "valuation": "列出本维度评分依据的具体数据点",
        "growth": "列出本维度评分依据的具体数据点",
        "technical": "列出本维度评分依据的具体数据点",
        "risk_governance": "列出本维度评分依据的具体数据点"
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
4. 打分必须基于提供的四维分析数据，不能凭空臆断；如果某项数据缺失，在data_basis中标注"无直接数据，基于已有信息推断"
4. 每个维度的打分必须有具体数据引用
5. 长线打分应**以护城河和行业为核心**，技术面权重极低
6. 如果某个维度的分析数据缺失，请在reasoning中说明
7. 输出必须是纯JSON，不要包含markdown代码块标记
"""

        user_prompt = f"""请对以下股票进行长线投资打分：

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

请从长线价值投资视角，严格按照系统提示中的评分体系、JSON格式和打分参考进行打分。
特别注意：
1. 护城河和行业景气度是长线最核心的判断维度
2. 技术面权重极低，不要因为短期走势否定长线逻辑
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

        sub_keys = ["business_moat", "industry_outlook", "valuation", "growth", "technical", "risk_governance"]
        for key in sub_keys:
            if key not in score_data.get("sub_scores", {}):
                raise ValueError(f"子评分缺少必要字段: {key}")

        score_data["detected_industry"] = detected_industry

        execution_logger.log_llm_interaction(
            agent_name=agent_name,
            interaction_type="long_term_scoring",
            input_messages=messages,
            output_content=json_mod.dumps(score_data, ensure_ascii=False),
            model_config={"model": resolved_model, "temperature": 1.0, "max_tokens": 16000, "thinking": "enabled" if thinking_enabled else "disabled"},
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
