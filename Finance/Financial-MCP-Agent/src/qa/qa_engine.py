"""
QA引擎 — 智能问答总编排器

协调复杂度分析 → 任务规划 → 证据装配 → 回答生成的完整流程。
"""
import time
import json
from typing import AsyncGenerator, Optional

from src.qa.session_manager import get_session_manager, QASession
from src.qa.complexity_analyzer import analyze_complexity, ComplexityResult
from src.qa.task_planner import plan_task, extract_stock_from_question
from src.qa.evidence_assembler import (
    assemble_evidence_fast,
    assemble_evidence_react,
    EvidencePackage,
)
from src.qa.answer_generator import generate_answer_stream
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


async def process_question(
    question: str,
    session_id: Optional[str] = None,
    current_date: str = "",
    current_time_info: str = "",
) -> AsyncGenerator[str, None]:
    """
    处理用户问题的主流程。

    Args:
        question: 用户问题
        session_id: 会话ID（None时自动创建新会话）
        current_date: 当前日期 YYYY-MM-DD
        current_time_info: 完整时间信息

    Yields:
        SSE格式字符串: "event: {type}\\ndata: {json}\\n\\n" 或 "data: {chunk}\\n\\n"
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

    # 如果需要澄清
    if complexity.need_clarify:
        yield _sse_event("clarify", {
            "message": "我需要确认一下：您指的是哪只股票或哪个板块？请补充股票代码或名称。",
            "session_id": actual_session_id,
        })
        yield "data: [DONE]\n\n"
        return

    # Step 2: 提取股票信息
    stock_code, company_name = extract_stock_from_question(
        question,
        session_stock_code=session.last_stock_code or "",
        session_company_name=session.last_company_name or "",
    )

    if stock_code:
        session_mgr.update_context(
            actual_session_id,
            last_stock_code=stock_code,
            last_company_name=company_name,
        )

    yield _sse_event("meta", {
        "session_id": actual_session_id,
        "complexity": complexity.level,
        "stock_code": stock_code,
        "company_name": company_name,
    })

    # Step 3: 任务规划
    history_text = _build_history_text(session)
    task_plan = plan_task(question, complexity.level, history_text)

    logger.info(
        f"QA Engine: 任务规划 — 数据域={task_plan.domains}, "
        f"工具数={len(task_plan.tools)}, ReAct={task_plan.need_react}"
    )

    # Step 4: 证据装配
    yield _sse_event("status", {
        "message": f"正在获取数据（涉及{len(task_plan.domains)}个数据域）..."
    })

    if task_plan.need_react and complexity.level == "L4":
        logger.info(f"{WAIT_ICON} QA Engine: 使用 ReAct 路径...")
        evidence = await assemble_evidence_react(
            stock_code or "", company_name or "",
            question, current_date, current_time_info
        )
    else:
        logger.info(f"{WAIT_ICON} QA Engine: 使用快路径并行拉取数据...")
        evidence = await assemble_evidence_fast(
            stock_code or "", company_name or "",
            task_plan.tools, question, current_date
        )

    yield _sse_event("status", {
        "message": (
            f"数据获取完成（{evidence.tool_call_summary}，"
            f"耗时{evidence.elapsed_seconds:.1f}秒），正在生成回答..."
        )
    })

    # Step 5: 流式回答生成
    yield _sse_event("answer_start", {"template": complexity.recommended_template})

    full_answer = ""
    async for chunk in generate_answer_stream(
        question=question,
        evidence=evidence,
        complexity=complexity,
        history_text=history_text,
        current_date=current_date,
    ):
        yield chunk
        # 收集完整回答用于保存到会话历史（仅收集data块）
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]") and not chunk.startswith("data: [ERROR]"):
            full_answer += chunk[6:].rstrip("\n")

    # Step 6: 保存到会话历史
    if full_answer.strip():
        session.add_message("user", question)
        session.add_message("assistant", full_answer.strip())

    total_time = time.time() - start_time
    logger.info(
        f"{SUCCESS_ICON} QA Engine: 回答完成 "
        f"({total_time:.1f}s, 复杂度={complexity.level})"
    )


def _sse_event(event_type: str, data: dict) -> str:
    """生成SSE事件格式"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_history_text(session: QASession) -> str:
    """构建历史对话文本（用于上下文注入）"""
    if not session.history:
        return ""
    recent = session.history[-6:]  # 最近3轮
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
