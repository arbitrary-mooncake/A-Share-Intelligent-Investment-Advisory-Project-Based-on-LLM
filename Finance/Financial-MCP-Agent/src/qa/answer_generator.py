"""
回答生成器 — LLM调用 + 回答模板 + 流式输出

三种模板：quick(L1), standard(L2), deep(L3/L4)
严格分离「数据事实」和「分析判断」，强制标注数据截至时间。
"""
import asyncio
import time
from typing import AsyncGenerator, Dict, Any
from openai import AsyncOpenAI
import httpx

from src.qa.complexity_analyzer import ComplexityResult
from src.qa.evidence_assembler import EvidencePackage
from src.utils.model_config import get_model_config_for_agent
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

LLM_TIMEOUT = 60  # 快路径LLM读超时
DEEP_LLM_TIMEOUT = 120  # 深度分析LLM读超时


def _build_system_prompt(template: str, current_date: str) -> str:
    """构建系统提示词，按模板分层"""

    base = (
        f"你是一位资深A股证券分析师，拥有10年以上卖方研究经验，擅长用通俗语言解释复杂的金融问题。\n\n"
        f"**当前日期：{current_date}**\n"
        f"**分析基准时间：{current_date}**\n\n"
        f"你的核心原则：\n"
        f"1. **数据优先**：所有数字类结论必须来源于工具提供的证据数据，绝不编造\n"
        f"2. **事实与判断分离**：回答中严格区分「数据事实」和「分析判断」\n"
        f"3. **专业但不晦涩**：让非专业投资者也能听懂，但保持专业深度\n"
        f"4. **有结论不绕**：先回答用户最关心的问题，再给证据\n"
        f"5. **风险提示具体**：风险提示要针对具体问题，不要泛泛而谈\n"
        f"6. **明确边界**：无法获取的数据说\"无法获取\"，不做猜测性补充\n"
        f"7. **对话感**：像投研分析师在回答同事问题一样，自然流畅但不失专业"
    )

    if template == "quick":
        return base + f"""

**输出格式（快答模板）：**
1. 先给核心结论（2-3句话，直接回答用户问题）
2. 关键证据（2-4条，每条用【数据】开头标注数据来源）
3. 风险提示（1-2句话，用【判断】开头，针对性强）
4. 标注数据截至时间：{current_date}

⛔ 防幻觉规则：
- 数字结论前必须加【数据】，非数字推断前必须加【判断】
- 数据不可用时标注「数据不可用」，严禁编造

要求：回答简洁有力，控制在300-500字。"""
    elif template == "standard":
        return base + f"""

**输出格式（标准分析模板）：**
1. 核心结论（3-5句话，【判断】开头）
2. 分维度分析（根据实际情况选2-4个相关维度：行情面/估值面/财务面/资金面/行业面/消息面）
3. 综合判断（【判断】开头）
4. 风险提示
5. 数据截至时间：{current_date}
6. 可继续追问的方向（1-2个）

⛔ 防幻觉规则：
- 具体数字前必须加【数据】，推断结论前必须加【判断】
- 引用数据时标注来源工具名
- 数据不可用时标注「数据不可用」，严禁编造

要求：回答控制在600-1000字，先结论后证据。"""
    else:
        return base + f"""

**输出格式（深度分析模板 — 两段式输出）：**
⚠️ 重要：先输出分析框架（100-150字），再输出完整分析。两段之间用 `---` 分隔线隔开。

**第一段：分析框架**
- 我将从哪几个维度分析（2-4个）
- 初步核心判断（1-2句话）
- 一句话说明数据获取情况

**第二段：完整深度分析**
1. 核心结论
2. 分维度深度分析（每个维度包含数据事实+分析判断）
3. 与可比对象/行业对比（如适用）
4. 关键矛盾点分析（如有）
5. 情景判断（多情景推演）
6. 风险与反证
7. 数据截至时间：{current_date}
8. 后续观察点

要求：全面深入但不冗长，总字数控制在1500-2500字。第一段快速让用户了解分析方向，第二段给出完整论证。"""


def _build_user_prompt(
    question: str,
    evidence: EvidencePackage,
    history_text: str,
) -> str:
    """构建用户提示词"""
    parts = [f"## 用户问题\n{question}\n"]

    if history_text:
        parts.append(f"## 历史对话\n{history_text}\n")

    parts.append(
        f"## 证据数据\n{evidence.raw_text}\n\n"
        f"## 数据获取摘要\n"
        f"- 成功获取: {evidence.tool_call_summary}\n"
        f"- 缺失数据: {', '.join(evidence.missing) if evidence.missing else '无'}\n"
        f"- 数据获取耗时: {evidence.elapsed_seconds:.1f}秒"
    )

    return "\n".join(parts)


async def generate_answer_stream(
    question: str,
    evidence: EvidencePackage,
    complexity: ComplexityResult,
    history_text: str,
    current_date: str,
) -> AsyncGenerator[str, None]:
    """
    流式生成回答（SSE）。
    Yields: "data: {chunk}\\n\\n" 或 "data: [DONE]\\n\\n"
    """
    model_cfg = get_model_config_for_agent("qa_engine")
    api_key = model_cfg["api_key"]
    base_url = model_cfg["base_url"]
    model_name = model_cfg["model_name"]

    # 复杂问题升级模型 → Model 1 (MiMo-V2.5-Pro)
    if complexity.recommended_model == "mimo-v2.5-pro":
        pro_cfg = get_model_config_for_agent("qa_engine_pro")
        if all([pro_cfg["api_key"], pro_cfg["base_url"], pro_cfg["model_name"]]):
            api_key = pro_cfg["api_key"]
            base_url = pro_cfg["base_url"]
            model_name = pro_cfg["model_name"]
        else:
            logger.warning(
                f"QA Answer: 复杂问题需要Pro模型但Model 1未配置，"
                f"降级使用{model_name}（回答可能不够深入）"
            )

    if not all([api_key, base_url, model_name]):
        yield _sse_error("模型配置缺失，请检查 .env 文件")
        return

    system_prompt = _build_system_prompt(complexity.recommended_template, current_date)
    user_prompt = _build_user_prompt(question, evidence, history_text)

    read_timeout = DEEP_LLM_TIMEOUT if complexity.recommended_template == "deep" else LLM_TIMEOUT

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(connect=15.0, read=float(read_timeout), write=30.0, pool=10.0),
        max_retries=1,
    )

    extra_body = {"thinking": {"type": "enabled" if complexity.recommended_thinking else "disabled"}}

    logger.info(
        f"{WAIT_ICON} QA Answer: 调用LLM (model={model_name}, "
        f"template={complexity.recommended_template}, "
        f"thinking={'ON' if complexity.recommended_thinking else 'OFF'})"
    )

    try:
        max_tokens = 4096 if complexity.recommended_template == "quick" else (
            8192 if complexity.recommended_template == "standard" else 16384
        )

        stream = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0 if complexity.recommended_thinking else 0.6,
            max_tokens=max_tokens,
            extra_body=extra_body,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {content}\n\n"

        yield "data: [DONE]\n\n"

    except asyncio.TimeoutError:
        yield _sse_error("回答生成超时，请重试")
    except Exception as e:
        logger.error(f"{ERROR_ICON} QA Answer: LLM 调用失败: {e}")
        yield _sse_error(f"回答生成失败: {e}")


def _sse_error(message: str) -> str:
    """SSE 错误格式"""
    return f"data: [ERROR] {message}\n\ndata: [DONE]\n\n"
