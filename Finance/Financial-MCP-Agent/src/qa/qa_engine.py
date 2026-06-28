"""
QA引擎 — 智能问答总编排器

协调复杂度分析 → 任务规划 → 证据装配 → 运行时升级 → 回答生成的完整流程。
Phase 3: 增加结构化监控日志。
"""
import os
import time
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Optional

from src.qa.session_manager import get_session_manager, QASession
from src.qa.complexity_analyzer import (
    analyze_complexity, try_runtime_upgrade, ComplexityResult,
)
from src.qa.task_planner import plan_task, extract_stock_from_question, match_topic, normalize_stock_code, resolve_index_name
from src.qa.evidence_assembler import (
    assemble_evidence_fast,
    EvidencePackage,
)
from src.qa.answer_generator import generate_answer_stream, format_answer
from src.tools.mcp_client import get_mcp_tools
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

    # Step 2: 提取标的（结构化提取 → LLM 语义提取 → 名称反查 → 指数/主题兜底）
    stock_code, company_name = extract_stock_from_question(
        question,
        session_stock_code=session.last_stock_code or "",
        session_company_name=session.last_company_name or "",
    )

    # 结构化提取未命中 → LLM 语义提取（主力路径）
    if not stock_code and not company_name:
        llm_code, llm_name = await _extract_stock_by_llm(question)
        if llm_name:
            company_name = llm_name
        if llm_code:
            stock_code = normalize_stock_code(llm_code)

    topic_context = ""
    matched_topic_name = ""
    topic_rep_stocks: list = []

    # 名称反查：LLM 提取的名称 → tushare_search_stock 查代码
    if not stock_code and company_name and not _is_topic_keyword(company_name):
        resolved_code, resolved_name = await _resolve_stock_code_by_name(company_name)
        if resolved_code:
            stock_code = resolved_code
            company_name = resolved_name
        else:
            # 个股反查失败 → 尝试指数代码匹配
            index_code = resolve_index_name(company_name)
            if index_code:
                logger.info(f"QA Engine: 指数匹配 → '{company_name}' → {index_code}")
                stock_code = index_code

    # 主题匹配兜底：无标的时候尝试匹配投资主题（黄金/原油/半导体等）
    if not stock_code:
        topic_info = match_topic(question)
        if topic_info:
            matched_topic_name, topic_data = topic_info
            etf = topic_data["etfs"][0]
            stock_code = etf[0]
            company_name = f"{matched_topic_name}主题({etf[1]})"
            topic_rep_stocks = topic_data["stocks"][:5]
            related = [f"{c}({n})" for c, n in topic_rep_stocks]
            topic_context = (
                f"[主题匹配] 用户询问{matched_topic_name}板块/赛道。"
                f"使用代表性ETF {stock_code} 作为基准数据源，"
                f"代表性个股: {', '.join(related)}。"
                f"请优先使用已获取的ETF行情数据和板块成分股数据进行客观分析，"
                f"辅以行业常识，避免仅凭旧知识即兴发挥。"
                f"\n⚠️ ETF特殊说明：{stock_code}是ETF/指数基金，不是个股。"
                f"ETF没有ROE、毛利率、净利率、营收增速、负债率、分红、EV/EBITDA等个股财务指标。"
                f"这些字段缺失是正常现象，不代表'数据获取不足'。"
                f"请基于ETF的行情走势、成交量、资金流向、板块景气度进行分析。"
            )
            macro_topic_hints = {
                "黄金": (
                    "\n\n[国际数据源] 本主题涉及国际定价资产。"
                    "请优先使用 web_search 获取最新的美联储政策声明、地缘政治事件。"
                ),
                "白银": "\n\n[国际数据源] 白银定价同时受贵金属属性和工业需求影响。",
                "原油": "\n\n[国际数据源] 原油是全球定价大宗商品。",
            }
            if matched_topic_name in macro_topic_hints and complexity.level != "L1":
                topic_context += macro_topic_hints[matched_topic_name]
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

    # Step 3.5: LLM 工具过滤 — 裁掉问题不相关的工具，减少无关数据干扰
    if len(task_plan.tools) > 3 and len(task_plan.domains) > 1:
        task_plan.tools = await _filter_tools_by_llm(question, task_plan.tools)
        logger.info(
            f"QA Engine: LLM过滤后 — 工具数={len(task_plan.tools)}"
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

        # 后台运行证据装配，定期发送心跳SSE事件防止前端超时断连
        evidence_task = asyncio.create_task(
            _assemble_with_fallback(
                task_plan, complexity, stock_code or "", company_name or "",
                question, current_date, current_time_info, actual_session_id,
                topic_name=matched_topic_name,
                representative_stocks=topic_rep_stocks if complexity.level != "L1" else [],
            )
        )
        _heartbeat_secs = 0
        while True:
            try:
                evidence = await asyncio.wait_for(
                    asyncio.shield(evidence_task), timeout=15.0
                )
                break
            except asyncio.TimeoutError:
                _heartbeat_secs += 15
                yield _sse_event("status", {
                    "message": f"正在获取数据... ({_heartbeat_secs}s)"
                })
            except Exception as e:
                logger.error(f"{ERROR_ICON} QA Engine: 证据装配任务异常: {e}")
                if not evidence_task.done():
                    evidence_task.cancel()
                evidence = EvidencePackage(
                    subject=company_name or stock_code or question,
                    stock_code=stock_code or "",
                    company_name=company_name or "",
                    missing=["证据装配异常: " + str(e)],
                    tool_call_summary="装配失败",
                )
                break

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
        # 更新前端显示的复杂度等级
        yield _sse_event("meta", {
            "session_id": actual_session_id,
            "complexity": complexity.level,
            "stock_code": stock_code,
            "company_name": company_name,
            "upgraded": True,
        })

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

def _is_topic_keyword(name: str) -> bool:
    """检查提取的公司名是否实际是已知的投资主题关键词或其扩展（如'半导体板块'以'半导体'开头）。
    仅当前缀匹配时才认为是主题扩展（避免'贵州茅台'被'茅台'误匹配为白酒主题）。"""
    from src.qa.task_planner import TOPIC_STOCK_MAP
    name_lower = name.lower()
    for topic, info in TOPIC_STOCK_MAP.items():
        for kw in info["keywords"]:
            kw_lower = kw.lower()
            # 精确匹配
            if kw_lower == name_lower:
                return True
            # 前缀匹配：提取名以主题关键词开头（如"半导体板块"以"半导体"开头）
            if len(kw_lower) >= 2 and name_lower.startswith(kw_lower):
                return True
    return False


async def _resolve_stock_code_by_name(name: str) -> tuple:
    """通过 tushare_search_stock MCP 工具将公司名反查为股票代码。
    返回 (normalized_code, official_name) 或 (None, None)。"""
    if not name or len(name) < 2:
        return None, None
    try:
        all_tools = await get_mcp_tools(tool_filter=["tushare_search_stock"])
    except Exception as e:
        logger.warning(f"QA Engine: tushare_search_stock 工具获取失败: {e}")
        return None, None
    if not all_tools:
        logger.warning("QA Engine: tushare_search_stock 工具不可用")
        return None, None
    tool = all_tools[0]
    try:
        result = await asyncio.wait_for(tool.ainvoke({"keyword": name}), timeout=15.0)
    except Exception as e:
        logger.warning(f"QA Engine: tushare_search_stock 调用失败: {e}")
        return None, None
    text = str(result).strip()
    if not text or len(text) < 10:
        return None, None
    import re as _re
    rows = [ln for ln in text.splitlines() if ln.strip().startswith("|") and "-" not in ln]
    if len(rows) < 2:
        return None, None
    header = [c.strip() for c in rows[0].split("|") if c.strip()]
    data_row = [c.strip() for c in rows[1].split("|") if c.strip()]
    if len(header) != len(data_row) or len(header) < 2:
        return None, None
    row_dict = dict(zip(header, data_row))
    raw_code = row_dict.get("ts_code") or row_dict.get("code") or ""
    official_name = row_dict.get("name") or name
    if not raw_code:
        return None, None
    normalized = raw_code.lower().replace(".", "")
    if normalized.startswith(("sh", "sz", "bj")):
        pass
    elif raw_code.endswith((".SH", ".SZ", ".BJ")):
        exch = raw_code.split(".")[-1].lower()
        digits = raw_code.split(".")[0]
        normalized = f"{exch}.{digits}"
    else:
        digits = raw_code.replace(".", "")
        if digits.startswith(("6", "5")):
            normalized = f"sh.{digits}"
        elif digits.startswith(("0", "3", "1", "4")):
            normalized = f"sz.{digits}"
        elif digits.startswith(("430", "431", "8", "920")):
            normalized = f"bj.{digits}"
        else:
            normalized = digits
    logger.info(f"QA Engine: 名称反查 '{name}' → {normalized} ({official_name})")
    return normalized, official_name


async def _extract_stock_by_llm(question: str) -> tuple:
    """LLM 语义提取：从自然语言问题中识别A股股票名称和代码。
    作为正则提取的兜底，用语义理解替代硬编码边界词列表。
    使用 Model 2 (Qwen3.6-Flash)，最快最便宜。
    返回 (code, name) 或 (None, None)。
    """
    import re as _re
    from openai import AsyncOpenAI
    import httpx

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY_2", "")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_2", "")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL_2", "")

    if not all([api_key, base_url, model_name]):
        return None, None

    prompt = (
        '从用户问题中提取A股股票信息。返回纯JSON，不要任何其他内容。\n'
        f'问题：「{question}」\n'
        '格式：{"name":"公司简称或全称","code":"6位数字代码"}\n'
        '- 提到公司名但未提代码 → code填null\n'
        '- 都没提到 → name和code都填null\n'
        '- 示例: "茅台今天多少钱" → {"name":"贵州茅台","code":null}\n'
        '- 示例: "帮我看看万科A估值" → {"name":"万科A","code":null}\n'
        '- 示例: "600519估值高吗" → {"name":null,"code":"600519"}\n'
        '- 示例: "今天天气不错" → {"name":null,"code":null}'
    )

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(connect=5, read=5, write=5, pool=5),
        )
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = resp.choices[0].message.content.strip()
        json_match = _re.search(r'\{[^}]+\}', text)
        if json_match:
            import json as _json
            data = _json.loads(json_match.group())
            llm_name = data.get("name")
            llm_code = data.get("code")
            if isinstance(llm_name, str) and llm_name.strip() and llm_name.strip().lower() != "null":
                llm_name = llm_name.strip()
            else:
                llm_name = None
            if isinstance(llm_code, str) and llm_code.strip() and llm_code.strip().lower() != "null":
                llm_code = llm_code.strip()
            else:
                llm_code = None
            if llm_code or llm_name:
                logger.info(f"QA Engine: LLM语义提取 → code={llm_code}, name={llm_name}")
            return llm_code, llm_name
    except Exception as e:
        logger.warning(f"QA Engine: LLM股票提取失败（回退到正则结果）: {e}")

    return None, None


async def _filter_tools_by_llm(question: str, tools: list) -> list:
    """用轻量 LLM 过滤工具列表，裁掉与问题不相关的工具。
    减少无关数据（如CPI/PMI混入业绩预告查询）对LLM回答的干扰。
    使用 Model 2 (Qwen3.6-Flash)，~0.3s，极低成本。
    """
    if len(tools) <= 3:
        return tools

    import re as _re
    from openai import AsyncOpenAI
    import httpx

    api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY_2", "")
    base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL_2", "")
    model_name = os.getenv("OPENAI_COMPATIBLE_MODEL_2", "")
    if not all([api_key, base_url, model_name]):
        return tools

    tool_list = "\n".join(f"- {t}" for t in tools)
    prompt = (
        "你是一个工具过滤器。给定用户问题和可用工具列表，只保留与回答问题直接相关的工具。\n\n"
        f"用户问题：「{question}」\n\n"
        f"可用工具（{len(tools)}个）：\n{tool_list}\n\n"
        "规则：\n"
        "1. 保留所有与问题明确相关的工具\n"
        "2. 如果问题只涉及某个具体主题（如公司新闻、宏观经济、个股分析），裁掉其他主题的工具\n"
        "3. web_search 始终保留（万能兜底）\n"
        "4. 返回纯JSON数组，不要任何其他内容\n"
        "示例: [\"tushare_major_news\", \"web_search\"]"
    )

    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(connect=3, read=3, write=3, pool=3),
        )
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = resp.choices[0].message.content.strip()
        json_match = _re.search(r'\[.*?\]', text, _re.DOTALL)
        if json_match:
            import json as _json
            filtered = _json.loads(json_match.group())
            if isinstance(filtered, list) and len(filtered) > 0:
                # 只保留过滤后仍在原列表中的工具（防止LLM幻觉编造工具名）
                valid = [t for t in filtered if t in tools]
                if valid:
                    removed = set(tools) - set(valid)
                    if removed:
                        logger.info(
                            f"QA Engine: LLM工具过滤 — 裁掉 {len(removed)} 个无关工具: {sorted(removed)}"
                        )
                    return valid
    except Exception as e:
        logger.warning(f"QA Engine: 工具过滤LLM调用失败，使用原工具列表: {e}")

    return tools


async def _assemble_with_fallback(
    task_plan, complexity, stock_code, company_name,
    question, current_date, current_time_info, session_id,
    topic_name: str = "",
    representative_stocks: list = None,
) -> EvidencePackage:
    """证据装配 + 降级保护（全部复杂度统一使用两阶段快路径）"""
    try:
        logger.info(f"{WAIT_ICON} QA Engine: 使用快路径并行拉取数据 (复杂度={complexity.level})...")
        evidence = await assemble_evidence_fast(
            stock_code, company_name,
            task_plan.tools, question, current_date,
            session_id=session_id,
            topic_name=topic_name,
            representative_stocks=representative_stocks,
            complexity_level=complexity.level,
        )
    except Exception as e:
        logger.error(f"{ERROR_ICON} QA Engine: 证据装配失败: {e}")
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

