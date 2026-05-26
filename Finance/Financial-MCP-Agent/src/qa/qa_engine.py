"""
QA引擎 — 智能问答总编排器

协调复杂度分析 → 任务规划 → 证据装配 → 运行时升级 → 回答生成的完整流程。
Phase 3: 增加结构化监控日志。
"""
import os
import time
import json
from datetime import datetime
from typing import AsyncGenerator, Optional

from src.qa.session_manager import get_session_manager, QASession
from src.qa.complexity_analyzer import (
    analyze_complexity, try_runtime_upgrade, ComplexityResult,
)
from src.qa.task_planner import plan_task, extract_stock_from_question, match_topic
from src.qa.evidence_assembler import (
    assemble_evidence_fast,
    assemble_evidence_react,
    EvidencePackage,
)
from src.qa.answer_generator import generate_answer_stream, format_answer
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

# 监控日志路径
_QA_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs", "qa"
)
os.makedirs(_QA_LOG_DIR, exist_ok=True)


async def process_question(
    question: str,
    session_id: Optional[str] = None,
    current_date: str = "",
    current_time_info: str = "",
) -> AsyncGenerator[str, None]:
    """
    处理用户问题的主流程（Phase 2：含运行时升级和降级）。

    Args:
        question: 用户问题
        session_id: 会话ID（None时自动创建新会话）
        current_date: 当前日期 YYYY-MM-DD
        current_time_info: 完整时间信息

    Yields:
        SSE格式字符串
    """
    start_time = time.time()
    session_mgr = get_session_manager()
    session = session_mgr.get_or_create(session_id)
    actual_session_id = session.session_id

    # Step 1: 复杂度分析
    logger.info(f"{WAIT_ICON} QA Engine: 分析问题复杂度...")
    history_depth = len(session.history) // 2
    complexity = analyze_complexity(question, history_depth)

    logger.info(
        f"QA Engine: 复杂度={complexity.level}, 评分={complexity.score}, "
        f"触发={complexity.triggers}, ReAct={complexity.recommended_react}"
    )

    # Step 2: 提取股票信息（优先于澄清检查，因会话上下文可消歧）
    stock_code, company_name = extract_stock_from_question(
        question,
        session_stock_code=session.last_stock_code or "",
        session_company_name=session.last_company_name or "",
    )

    # 无股票代码时尝试主题匹配
    topic_context = ""
    matched_topic_name = ""
    topic_rep_stocks: list = []  # [(code, name), ...] 代表性个股供并行拉取数据
    if not stock_code and not company_name:
        topic_info = match_topic(question)
        if topic_info:
            matched_topic_name, topic_data = topic_info
            etf = topic_data["etfs"][0]
            stock_code = etf[0]
            company_name = f"{matched_topic_name}主题({etf[1]})"
            topic_rep_stocks = topic_data["stocks"][:5]  # 最多5只代表股
            # 构建主题上下文供LLM参考
            related = [f"{c}({n})" for c, n in topic_rep_stocks]
            topic_context = (
                f"[主题匹配] 用户询问{matched_topic_name}板块/赛道。"
                f"使用代表性ETF {stock_code} 作为基准数据源，"
                f"代表性个股: {', '.join(related)}。"
                f"请优先使用已获取的ETF行情数据和板块成分股数据进行客观分析，"
                f"辅以行业常识，避免仅凭旧知识即兴发挥。"
                f"如板块成分股数据已获取，请从中选取龙头标的重点分析。"
            )
            logger.info(f"QA Engine: 主题匹配 → {matched_topic_name}, 使用ETF {stock_code}")

    # 澄清检查：已尝试所有消歧手段（问题文本+会话上下文+主题匹配）仍无标的时才反问
    if complexity.need_clarify and not stock_code and not company_name:
        clarify_msg = "我需要确认一下您具体想了解哪只股票或哪个板块，请提供股票名称，我会自动为您查找对应的信息。"
        yield f"event: clarify\ndata: {clarify_msg}\n\n"
        yield "data: [DONE]\n\n"
        return

    if stock_code:
        session_mgr.update_context(
            actual_session_id,
            last_stock_code=stock_code,
            last_company_name=company_name,
            last_complexity_level=complexity.level,
        )

    yield _sse_event("meta", {
        "session_id": actual_session_id,
        "complexity": complexity.level,
        "stock_code": stock_code,
        "company_name": company_name,
    })

    # Step 3: 任务规划
    history_text = _build_history_text(session)
    task_plan = plan_task(question, complexity.level, history_text,
                          topic_name=matched_topic_name,
                          stock_code=stock_code or "", company_name=company_name or "")

    logger.info(
        f"QA Engine: 任务规划 — 数据域={task_plan.domains}, "
        f"工具数={len(task_plan.tools)}, ReAct={task_plan.need_react}"
    )

    # Step 4: 证据装配（L0跳过）
    if complexity.level == "L0":
        evidence = EvidencePackage(
            subject=company_name or question,
            stock_code=stock_code or "",
            company_name=company_name or "",
            raw_text="（无需数据查询）",
            tool_call_summary="L0: 跳过数据获取",
        )
    else:
        yield _sse_event("status", {
            "message": f"正在获取数据（涉及{len(task_plan.domains)}个数据域）..."
        })

        evidence = await _assemble_with_fallback(
        task_plan, complexity, stock_code or "", company_name or "",
        question, current_date, current_time_info, actual_session_id,
        topic_name=matched_topic_name,
        representative_stocks=topic_rep_stocks,
    )

    # ── 运行时升级（Layer 3） ──
    tool_labels = task_plan.tools
    total_tools = max(len(tool_labels), 1)
    success_count = total_tools - len(evidence.missing)
    tool_success_rate = success_count / total_tools
    contradictory = _detect_contradictions(evidence)
    actual_domains = max(len(task_plan.domains), 1)

    complexity = try_runtime_upgrade(
        complexity,
        tool_success_rate=tool_success_rate,
        evidence_missing_count=len(evidence.missing),
        contradictory_signals=contradictory,
        actual_domain_count=actual_domains,
    )

    if complexity.triggers and any("运行时" in t for t in complexity.triggers):
        logger.info(
            f"QA Engine: 运行时升级 → {complexity.level}, "
            f"model={complexity.recommended_model}, "
            f"thinking={complexity.recommended_thinking}, "
            f"react={complexity.recommended_react}"
        )
        upgrade_msg = (
            f"检测到复杂问题，已自动升级分析策略（{complexity.level}级/"
            f"{'深度推理' if complexity.recommended_thinking else '标准'}模式）"
        )
        yield _sse_event("status", {"message": upgrade_msg})

    # ── 降级：数据完全缺失 ──
    if not evidence.raw_text or evidence.raw_text == "(无数据)":
        yield _sse_event("status", {
            "message": "数据获取受限，将基于可用信息和专业知识进行回答..."
        })
        evidence.raw_text = (
            f"（系统说明：部分数据源当前不可用。以下回答将基于已有信息和行业知识进行分析，"
            f"所有推断会明确标注。）\n"
            f"可用信息：股票={company_name or '未指定'}，代码={stock_code or '未指定'}，"
            f"日期={current_date}"
        )

    # 注入主题上下文
    if topic_context:
        evidence.raw_text = topic_context + "\n\n" + evidence.raw_text

    yield _sse_event("status", {
        "message": (
            f"数据获取完成（{evidence.tool_call_summary}，"
            f"耗时{evidence.elapsed_seconds:.1f}秒），正在生成回答..."
        )
    })

    # Step 5: 流式回答生成（含LLM降级）
    yield _sse_event("answer_start", {"template": complexity.recommended_template})

    full_answer = ""
    llm_success = False

    try:
        async for chunk in generate_answer_stream(
            question=question,
            evidence=evidence,
            complexity=complexity,
            history_text=history_text,
            current_date=current_date,
        ):
            # 拦截 [DONE]：先格式化排版再保存，确保前端拉取时数据已落地
            if chunk.strip() == "data: [DONE]":
                if full_answer.strip():
                    formatted = format_answer(full_answer.strip())
                    session.add_message("user", question)
                    session.add_message("assistant", formatted)
                    session_mgr.save_session(actual_session_id)
                yield chunk
                llm_success = True
                break
            yield chunk
            if chunk.startswith("data: ") and not chunk.startswith("data: [ERROR]"):
                full_answer += chunk[6:].rstrip("\n")
        else:
            llm_success = True
    except Exception as llm_err:
        logger.error(f"{ERROR_ICON} QA Engine: LLM流式生成异常: {llm_err}")

    # ── 降级：LLM 失败时返回证据数据 ──
    if not llm_success or not full_answer.strip():
        fallback = _build_fallback_answer(evidence, company_name, stock_code, current_date)
        yield f"data: {fallback}\n\n"
        yield "data: [DONE]\n\n"
        full_answer = fallback
        if full_answer.strip():
            formatted = format_answer(full_answer.strip())
            session.add_message("user", question)
            session.add_message("assistant", formatted)
            session_mgr.save_session(actual_session_id)

    total_time = time.time() - start_time
    logger.info(
        f"{SUCCESS_ICON} QA Engine: 回答完成 "
        f"({total_time:.1f}s, 复杂度={complexity.level}, "
        f"LLM成功={llm_success})"
    )

    # 写入结构化监控日志
    _write_qa_log(
        session_id=actual_session_id,
        question=question[:200],
        complexity=complexity,
        evidence=evidence,
        total_time=total_time,
        llm_success=llm_success,
        answer_length=len(full_answer) if full_answer else 0,
    )



# ── 辅助函数 ──────────────────────────────────────

async def _assemble_with_fallback(
    task_plan, complexity, stock_code, company_name,
    question, current_date, current_time_info, session_id,
    topic_name: str = "",
    representative_stocks: list = None,
) -> EvidencePackage:
    """证据装配 + 降级保护"""
    try:
        if task_plan.need_react and complexity.level == "L4":
            logger.info(f"{WAIT_ICON} QA Engine: 使用 ReAct 路径...")
            evidence = await assemble_evidence_react(
                stock_code, company_name,
                question, current_date, current_time_info,
            )
        else:
            logger.info(f"{WAIT_ICON} QA Engine: 使用快路径并行拉取数据...")
            evidence = await assemble_evidence_fast(
                stock_code, company_name,
                task_plan.tools, question, current_date,
                session_id=session_id,
                topic_name=topic_name,
                representative_stocks=representative_stocks,
            )
    except Exception as e:
        logger.error(f"{ERROR_ICON} QA Engine: 证据装配失败: {e}")
        # 降级：返回空证据包
        evidence = EvidencePackage(
            subject=company_name or stock_code or question,
            stock_code=stock_code,
            company_name=company_name,
            missing=["证据装配异常: " + str(e)],
            tool_call_summary="装配失败",
        )
    return evidence


def _detect_contradictions(evidence: EvidencePackage) -> bool:
    """简单检测证据中是否存在矛盾信号"""
    text = evidence.raw_text.lower()
    # 常见矛盾模式
    patterns = [
        ("上涨" in text or "增长" in text or "上升" in text) and
        ("下跌" in text or "下降" in text or "下滑" in text),
        ("盈利" in text or "利润.*正" in text) and
        ("亏损" in text or "利润.*负" in text),
    ]
    return any(patterns)


def _build_fallback_answer(
    evidence: EvidencePackage,
    company_name: str,
    stock_code: str,
    current_date: str,
) -> str:
    """LLM失败时的降级回答：直接返回已获取的数据事实"""
    parts = [
        f"## 数据查询结果\n",
        f"**查询对象**: {company_name or stock_code or '未指定'}",
        f"**数据截至**: {current_date}",
        f"**数据获取状态**: {evidence.tool_call_summary}",
        f"",
        f"⚠️ LLM分析服务暂时不可用，以下为已获取的原始数据：",
        f"",
        evidence.raw_text if evidence.raw_text else "（无可用数据）",
        f"",
        f"---",
        f"*数据缺失: {', '.join(evidence.missing) if evidence.missing else '无'}*",
        f"",
        f"*请稍后重试以获取完整分析，或简化问题重新提问。*",
    ]
    return "\n".join(parts)


def _sse_event(event_type: str, data: dict) -> str:
    """生成SSE事件格式"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_history_text(session: QASession) -> str:
    """构建历史对话文本（用于上下文注入）"""
    if not session.history:
        return ""
    recent = session.history[-6:]
    lines = []
    for msg in recent:
        role = "用户" if msg.role == "user" else "分析师"
        lines.append(f"{role}: {msg.content[:200]}")
    return "\n".join(lines)


def save_answer_to_session(session_id: str, question: str, answer: str):
    """外部调用：将问答保存到会话历史"""
    session_mgr = get_session_manager()
    session = session_mgr.get_session(session_id)
    if session:
        session.add_message("user", question)
        session.add_message("assistant", answer)


def _write_qa_log(
    session_id: str,
    question: str,
    complexity: ComplexityResult,
    evidence: EvidencePackage,
    total_time: float,
    llm_success: bool,
    answer_length: int,
):
    """写入结构化QA监控日志"""
    try:
        now = datetime.now()
        log_entry = {
            "timestamp": now.isoformat(),
            "session_id": session_id,
            "question": question,
            "complexity_level": complexity.level,
            "complexity_score": complexity.score,
            "triggers": complexity.triggers,
            "model": complexity.recommended_model,
            "thinking": complexity.recommended_thinking,
            "react": complexity.recommended_react,
            "template": complexity.recommended_template,
            "tool_summary": evidence.tool_call_summary,
            "tools_missing": evidence.missing,
            "evidence_elapsed": round(evidence.elapsed_seconds, 1),
            "total_elapsed": round(total_time, 1),
            "llm_success": llm_success,
            "answer_length": answer_length,
        }
        log_file = os.path.join(_QA_LOG_DIR, f"{now.strftime('%Y%m%d')}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"写入QA监控日志失败: {e}")

