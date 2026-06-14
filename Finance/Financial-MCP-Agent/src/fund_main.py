"""
基金分析智能体系统主程序 (Fund Analysis AI Agent System Main Program)

本文件是基金分析智能体系统的核心入口点，实现了以下主要功能：

1. 多智能体工作流管理：使用LangGraph构建并行执行的智能体工作流
2. 命令行界面：提供用户友好的交互式命令行界面
3. 自然语言处理：自动识别和提取基金代码、基金名称
4. 日志系统：完整的执行日志记录和错误处理
5. 报告生成：生成综合性的基金分析报告

工作流程：
start_node → [fund_product_doc_agent, fund_perf_risk_agent, fund_holdings_agent,
              fund_manager_agent, fund_benchmark_agent, fund_fee_agent,
              fund_event_agent] (7 parallel)
         → fund_merge_node
              ├── fund_report_agent  (default: report mode)
              └── fund_scoring_agent (score mode: when intent="score")
                    └── END
"""

# ============================================================================
# 导入必要的模块和依赖
# ============================================================================

# 在导入其他模块之前设置环境变量，抑制无用输出
import os
import sys

# Windows: 使用 SelectorEventLoop 以支持子进程（MCP stdio 传输需要）
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 设置控制台UTF-8编码以支持中文和Emoji
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 设置环境变量来抑制transformers和其他库的冗余输出
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 只显示错误信息
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行化警告
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 减少CUDA相关输出
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 减少内存分配信息

# 设置日志级别，抑制第三方库的INFO级别输出
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# 日志和状态管理相关导入
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.utils.state_definition import AgentState
from src.utils.execution_logger import initialize_execution_logger, finalize_execution_logger, get_execution_logger

# 基金智能体模块导入 - 七个核心分析智能体
from src.agents.fund_product_doc_agent import fund_product_doc_agent  # 基金产品文档解析智能体
from src.agents.fund_perf_risk_agent import fund_perf_risk_agent       # 基金业绩风险评估智能体
from src.agents.fund_holdings_agent import fund_holdings_analysis     # 基金持仓分析智能体
from src.agents.fund_manager_agent import fund_manager_agent           # 基金经理分析智能体
from src.agents.fund_benchmark_agent import fund_benchmark_agent       # 基金基准对标智能体
from src.agents.fund_fee_agent import fund_fee_agent                   # 基金费用分析智能体
from src.agents.fund_event_agent import fund_event_agent               # 基金事件分析智能体
from src.agents.fund_merge_node import fund_merge_node                 # 基金合并节点
from src.agents.fund_report_agent import fund_report_agent             # 基金报告生成智能体
from src.agents.fund_scoring_agent import fund_scoring_agent           # 基金评分智能体

# LangGraph工作流框架导入
from langgraph.graph import StateGraph, END

# 环境变量和系统相关导入
from dotenv import load_dotenv
import argparse
import asyncio
import re
from datetime import datetime

# ============================================================================
# 初始化和配置
# ============================================================================

# 设置日志记录器
logger = setup_logger(__name__)

# 添加项目根目录到Python路径，确保模块导入正常工作
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 加载环境变量（从.env文件）
load_dotenv(override=True)

# 调试：打印关键环境变量以验证配置
logger.info(f"Environment Variables Loaded:")
logger.info(
    f"  OPENAI_COMPATIBLE_MODEL: {os.getenv('OPENAI_COMPATIBLE_MODEL', 'Not Set')}")
logger.info(
    f"  OPENAI_COMPATIBLE_BASE_URL: {os.getenv('OPENAI_COMPATIBLE_BASE_URL', 'Not Set')}")
logger.info(
    f"  OPENAI_COMPATIBLE_API_KEY: {'*' * 20 if os.getenv('OPENAI_COMPATIBLE_API_KEY') else 'Not Set'}")

# 重新设置日志记录器（确保正确配置）
logger = setup_logger(__name__)


# ============================================================================
# 基金代码检测和规范化工具函数
# ============================================================================

def _is_fund_code(code: str) -> bool:
    """Check if a 6-digit code looks like a fund/ETF code"""
    c = code.replace("sh.", "").replace("sz.", "").strip()
    if len(c) != 6:
        return False
    # Shanghai ETF/LOF: 51xxxx, 58xxxx, 50xxxx
    # Shenzhen ETF/LOF: 15xxxx, 16xxxx, 18xxxx
    return c.startswith(("51", "58", "50", "15", "16", "18"))


def _normalize_fund_code(code: str) -> str:
    """Normalize fund code with exchange prefix"""
    c = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").replace(".OF", "").strip()
    if c.startswith(("51", "58", "50", "5", "6", "8")):
        return f"sh.{c}"
    else:
        return f"sz.{c}"


def _detect_intent(query: str) -> str:
    """Detect user intent: 'report' or 'score' based on keywords in query."""
    score_keywords = ["打分", "评分", "score", "评级", "测评", "评估一下"]
    for kw in score_keywords:
        if kw.lower() in query.lower():
            return "score"
    return "report"


# ============================================================================
# 基金名称→代码查找（通过 MCP tushare_fund_search）
# ============================================================================

async def _lookup_fund_by_name(fund_name: str) -> str | None:
    """
    通过基金名称查找基金代码。
    使用 tushare_fund_search MCP 工具进行搜索。
    返回规范化后的基金代码（如 sh.510050），或 None。
    """
    try:
        from src.tools.mcp_client import get_mcp_tools

        all_tools = await get_mcp_tools(tool_filter=["tushare_fund_search"])
        tool_map = {t.name: t for t in all_tools}

        if "tushare_fund_search" not in tool_map:
            logger.warning(f"tushare_fund_search 工具不可用，无法按名称查找基金")
            return None

        tool = tool_map["tushare_fund_search"]
        result = await asyncio.wait_for(
            tool.ainvoke({"keyword": fund_name}),
            timeout=30
        )
        result_text = str(result).strip()

        if not result_text or "不可用" in result_text or len(result_text) < 5:
            logger.warning(f"tushare_fund_search 未找到匹配: {fund_name}")
            return None

        # 尝试从结果中提取基金代码
        # 结果可能是多行文本，提取第一个匹配的6位数字代码
        # 优先匹配基金类代码（51xxxx, 58xxxx, 50xxxx, 15xxxx, 16xxxx, 18xxxx）
        fund_code_pattern = r'(?:ts_code[=:]\s*["\']?)?(\d{6})\.?(?:OF|SH|SZ)?'
        matches = re.findall(fund_code_pattern, result_text)

        for m in matches:
            if _is_fund_code(m):
                normalized = _normalize_fund_code(m)
                logger.info(f"基金名称 '{fund_name}' → 基金代码: {normalized}")
                return normalized

        # 如果没找到基金类代码，尝试返回第一个6位数字
        if matches:
            normalized = _normalize_fund_code(matches[0])
            logger.info(f"基金名称 '{fund_name}' → (fallback) 基金代码: {normalized}")
            return normalized

        logger.warning(f"tushare_fund_search 结果中未提取到基金代码: {result_text[:200]}")
        return None

    except asyncio.TimeoutError:
        logger.warning(f"tushare_fund_search 超时: {fund_name}")
        return None
    except Exception as e:
        logger.warning(f"tushare_fund_search 调用失败 ({fund_name}): {e}")
        return None


# ============================================================================
# 基金信息提取函数
# ============================================================================

def extract_fund_info(query: str) -> tuple[str | None, str | None]:
    """从自然语言查询中提取基金代码和基金名称"""
    fund_code = None
    fund_name = None

    # 模式1: 包含"请帮我分析一下"的复杂查询，如"请帮我分析一下华夏上证50ETF(510050)"
    pattern1 = r'请帮我分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match1 = re.search(pattern1, query)
    if match1:
        fund_name = match1.group(1).strip()
        fund_code = match1.group(2)
        return fund_name, fund_code

    # 模式2: 包含"分析一下"的复杂查询，如"分析一下华夏上证50ETF(510050)的投资价值"
    pattern2 = r'分析一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match2 = re.search(pattern2, query)
    if match2:
        fund_name = match2.group(1).strip()
        fund_code = match2.group(2)
        return fund_name, fund_code

    # 模式3: 基金代码在括号内，如"分析华夏上证50ETF(510050)"
    pattern3 = r'分析\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match3 = re.search(pattern3, query)
    if match3:
        fund_name = match3.group(1).strip()
        fund_code = match3.group(2)
        return fund_name, fund_code

    # 模式4: 基金代码在括号内，如"分析(510050)华夏上证50ETF"
    pattern4 = r'分析\s*[（(](\d{5,6})[)）]\s*([^）)]+)'
    match4 = re.search(pattern4, query)
    if match4:
        fund_code = match4.group(1)
        fund_name = match4.group(2).strip()
        return fund_name, fund_code

    # 模式5: 包含"帮我看看"的查询，如"帮我看看510050这只基金"
    pattern5 = r'帮我看看\s*[（(]?(\d{5,6})[)）]?\s*([^）)]+?)(?:\s*这只|\s*这个)?\s*基金'
    match5 = re.search(pattern5, query)
    if match5:
        fund_code = match5.group(1)
        fund_name = match5.group(2).strip()
        return fund_name, fund_code

    # 模式6: 包含"我想了解一下"的查询，如"我想了解一下华夏上证50ETF(510050)的投资价值"
    pattern6 = r'我想了解一下\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match6 = re.search(pattern6, query)
    if match6:
        fund_name = match6.group(1).strip()
        fund_code = match6.group(2)
        return fund_name, fund_code

    # 模式7: 包含"帮我看看"的查询，如"帮我看看华夏上证50ETF(510050)这只基金值得投资吗"
    pattern7 = r'帮我看看\s*([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match7 = re.search(pattern7, query)
    if match7:
        fund_name = match7.group(1).strip()
        fund_code = match7.group(2)
        return fund_name, fund_code

    # 模式8: 直接基金名+括号格式，如"华夏上证50ETF(510050)值得买吗"
    pattern8 = r'^([^（(]+?)\s*[（(](\d{5,6})[)）]'
    match8 = re.search(pattern8, query)
    if match8:
        fund_name = match8.group(1).strip()
        fund_code = match8.group(2)
        return fund_name, fund_code

    # 模式9: 包含"分析一下"的查询，如"分析一下华夏上证50ETF的表现"
    pattern9 = r'分析一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)'
    match9 = re.search(pattern9, query)
    if match9:
        fund_name = match9.group(1).strip()

    # 模式10: 包含"分析"关键词，如"分析华夏上证50ETF"
    pattern10 = r'分析\s*([^0-9（）()\s]+)'
    match10 = re.search(pattern10, query)
    if match10 and not fund_name:
        fund_name = match10.group(1).strip()

    # 模式11: 包含"基金"关键词的查询，如"华夏上证50ETF这只基金怎么样"
    pattern11 = r'([^0-9（）()\s]+)\s*(?:这只|这个|的)?\s*基金'
    match11 = re.search(pattern11, query)
    if match11 and not fund_name:
        fund_name = match11.group(1).strip()

    # 模式12: 包含"投资价值"的查询，如"了解一下华夏上证50ETF的投资价值"
    pattern12 = r'了解一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)'
    match12 = re.search(pattern12, query)
    if match12 and not fund_name:
        fund_name = match12.group(1).strip()

    # 模式13: 包含"给我分析一下"的查询，如"给我分析一下华夏上证50ETF的表现"
    pattern13 = r'给我分析一下\s*([^0-9（）()\s]+?)(?:\s*的|\s|$)'
    match13 = re.search(pattern13, query)
    if match13 and not fund_name:
        fund_name = match13.group(1).strip()

    # 模式14: 包含"打分"或"评分"的查询，如"打分510050"、"评分华夏上证50ETF"
    pattern14 = r'(?:打分|评分|score)\s*(\d{5,6})'
    match14 = re.search(pattern14, query)
    if match14:
        fund_code = match14.group(1)

    # 模式15: 直接包含6位数字基金代码（更精确的基金代码模式）
    # 基金代码特征：51xxxx, 58xxxx, 50xxxx (上交所ETF), 15xxxx, 16xxxx, 18xxxx (深交所ETF)
    pattern15 = r'(?<!\d)(\d{6})(?!\d)'
    match15 = re.search(pattern15, query)
    if match15:
        raw_code = match15.group(1)
        if _is_fund_code(raw_code):
            fund_code = raw_code

    # 模式16: 包含"值得买"的查询，如"510050 这个基金值得买吗"
    pattern16 = r'(\d{5,6})\s*(?:这个|这只)?\s*基金\s*值得买'
    match16 = re.search(pattern16, query)
    if match16 and not fund_code:
        fund_code = match16.group(1)

    # 模式17: 包含"这个基金最近表现"的查询，如"510050这个基金最近表现怎么样"
    pattern17 = r'(\d{5,6})\s*这个\s*基金\s*最近表现'
    match17 = re.search(pattern17, query)
    if match17 and not fund_code:
        fund_code = match17.group(1)

    # 清理基金名称（移除常见的无意义词汇）
    if fund_name:
        stop_words = ['的', '这个', '这只', '一下', '看看', '了解', '分析', '帮我',
                       '我想', '给我', '投资价值', '基本情况', '这只基金', '这个基金',
                       '表现', '打分', '评分', 'score']
        for word in stop_words:
            fund_name = fund_name.replace(word, '').strip()

        # 如果基金名称太短（少于2个字符），可能是误匹配
        if len(fund_name) < 2:
            fund_name = None

    return fund_name, fund_code


async def main():
    """
    主函数：基金分析智能体系统的核心执行逻辑

    功能包括：
    1. 初始化执行日志系统
    2. 构建LangGraph工作流
    3. 处理命令行参数和用户输入
    4. 提取基金信息（代码、基金名称）
    5. 执行多智能体分析工作流
    6. 生成和保存分析报告
    7. 错误处理和日志记录
    """

    # 初始化执行日志系统
    execution_logger = initialize_execution_logger()
    logger.info(
        f"{SUCCESS_ICON} 执行日志系统已初始化，日志目录: {execution_logger.execution_dir}")

    try:
        # ============================================================================
        # 1. 定义LangGraph工作流
        # ============================================================================

        # 创建工作流图，使用AgentState作为状态类型
        workflow = StateGraph(AgentState)

        # 添加起始节点 - 作为并行分支的清晰起点
        workflow.add_node("start_node", lambda state: state)

        # 添加七个核心基金分析智能体节点
        workflow.add_node("fund_product_doc_agent", fund_product_doc_agent)  # 基金产品文档解析
        workflow.add_node("fund_perf_risk_agent", fund_perf_risk_agent)       # 基金业绩风险评估
        workflow.add_node("fund_holdings_agent", fund_holdings_analysis)     # 基金持仓分析
        workflow.add_node("fund_manager_agent", fund_manager_agent)           # 基金经理分析
        workflow.add_node("fund_benchmark_agent", fund_benchmark_agent)       # 基金基准对标
        workflow.add_node("fund_fee_agent", fund_fee_agent)                   # 基金费用分析
        workflow.add_node("fund_event_agent", fund_event_agent)               # 基金事件分析

        # 添加合并节点和报告/评分节点
        workflow.add_node("fund_merge_node", fund_merge_node)                 # 基金合并节点
        workflow.add_node("fund_report_agent", fund_report_agent)             # 基金报告生成
        # fund_scoring_agent 签名是 (fund_analysis_package, fund_code, ...) 而非 (state, ...)
        # 需要包装器将其适配为 LangGraph 节点
        async def _fund_scoring_wrapper(state: AgentState) -> dict:
            data = state.get("data", {})
            pkg = data.get("fund_analysis_package", {})
            fc = data.get("fund_code", "")
            fn = data.get("fund_name", "")
            cd = data.get("current_date", "")
            # 从分析包或数据中提取基金类型，而非硬编码为 ETF
            ft = (pkg.get("fund_profile", {}) or {}).get("fund_type", "") or data.get("fund_type", "") or "未知类型"
            score_result = await fund_scoring_agent(
                fund_analysis_package=pkg,
                fund_code=fc, fund_name=fn, fund_type=ft,
                current_date=cd,
            )
            return {"data": {"fund_score": score_result}}

        workflow.add_node("fund_scoring_agent", _fund_scoring_wrapper)           # 基金评分

        # 设置工作流入口点
        workflow.set_entry_point("start_node")

        # 七个基金分析智能体：定义并行执行边
        fund_agents = [
            "fund_product_doc_agent",
            "fund_perf_risk_agent",
            "fund_holdings_agent",
            "fund_manager_agent",
            "fund_benchmark_agent",
            "fund_fee_agent",
            "fund_event_agent",
        ]

        # start_node → 每个基金智能体（并行分支）
        for agent in fund_agents:
            workflow.add_edge("start_node", agent)

        # 每个基金智能体 → fund_merge_node（汇聚）
        for agent in fund_agents:
            workflow.add_edge(agent, "fund_merge_node")

        # 条件路由：根据 intent 决定走 report 还是 scoring
        def route_intent(state: AgentState) -> str:
            """根据用户意图路由到报告或评分路径"""
            current_data = state.get("data", {})
            intent = current_data.get("fund_intent", "report")
            if intent == "score":
                return "fund_scoring_agent"
            return "fund_report_agent"

        workflow.add_conditional_edges(
            "fund_merge_node",
            route_intent,
            {
                "fund_report_agent": "fund_report_agent",
                "fund_scoring_agent": "fund_scoring_agent",
            }
        )

        # 报告和评分都通向END
        workflow.add_edge("fund_report_agent", END)
        workflow.add_edge("fund_scoring_agent", END)

        # 编译工作流
        app = workflow.compile()

        # ============================================================================
        # 2. 实现命令行界面
        # ============================================================================

        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description="Fund Analysis Agent CLI")
        parser.add_argument(
            "--command",
            type=str,
            required=False,  # 改为非必需，支持交互式输入
            help="The user query for fund analysis (e.g., '分析基金510050')"
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["report", "score"],
            default=None,
            help="Analysis mode: 'report' for full report, 'score' for scoring (default: auto-detect)"
        )
        parser.add_argument(
            "--code",
            type=str,
            default=None,
            help="Fund code to analyze (e.g., 510050, 159919)"
        )
        args = parser.parse_args()

        # 确定分析模式
        explicit_mode = args.mode  # --mode 显式指定

        # 处理用户查询输入
        if args.command:
            # 如果通过命令行参数提供查询
            user_query = args.command
        else:
            # 显示ASCII艺术开屏图像和交互式界面
            print("\n")
            print(
                "╔══════════════════════════════════════════════════════════════════════════════╗")
            print(
                "║                                                                              ║")
            print(
                "║      ███████╗██╗   ██╗███╗   ██╗██████╗      █████╗ ███╗   ██╗ █████╗      ║")
            print(
                "║      ██╔════╝██║   ██║████╗  ██║██╔══██╗    ██╔══██╗████╗  ██║██╔══██╗     ║")
            print(
                "║      █████╗  ██║   ██║██╔██╗ ██║██║  ██║    ███████║██╔██╗ ██║███████║     ║")
            print(
                "║      ██╔══╝  ██║   ██║██║╚██╗██║██║  ██║    ██╔══██║██║╚██╗██║██╔══██║     ║")
            print(
                "║      ██║     ╚██████╔╝██║ ╚████║██████╔╝    ██║  ██║██║ ╚████║██║  ██║     ║")
            print(
                "║      ╚═╝      ╚═════╝ ╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝     ║")
            print(
                "║                                                                              ║")
            print(
                "║                █████╗  ██████╗ ███████╗███╗   ██╗████████╗                  ║")
            print(
                "║               ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝                  ║")
            print(
                "║               ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                     ║")
            print(
                "║               ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                     ║")
            print(
                "║               ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                     ║")
            print(
                "║               ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                     ║")
            print(
                "║                                                                              ║")
            print("║                          📊 基金分析智能体系统                              ║")
            print(
                "║                       Fund Analysis AI Agent System                        ║")
            print(
                "║                                                                              ║")
            print(
                "║    ┌───────────────────────────────────────────────────────────────────┐   ║")
            print("║    │  📋 产品解析  │  📈 业绩风险  │  🏢 持仓分析  │  👤 经理分析  │   ║")
            print(
                "║    │  🎯 基准对标  │  💰 费用分析  │  📰 事件分析  │  📝 智能报告  │   ║")
            print(
                "║    └───────────────────────────────────────────────────────────────────┘   ║")
            print(
                "║                                                                              ║")
            print(
                "╚══════════════════════════════════════════════════════════════════════════════╝")
            print("\n🔹 本系统可以对公募基金进行全面分析，包括：")
            print("  • 产品文档解析 - 基金类型、投资策略、风险等级等")
            print("  • 业绩风险评估 - 收益率、波动率、最大回撤、夏普比率等")
            print("  • 持仓分析 - 前十大重仓股、行业配置、集中度风险")
            print("  • 基金经理分析 - 从业经验、历史业绩、投资风格")
            print("  • 基准对标 - 与业绩基准、同类基金的对比分析")
            print("  • 费用分析 - 管理费、托管费、申赎费、综合持有成本")
            print("  • 事件分析 - 分红、拆分、限购、申赎状态等关键事件")
            print("\n🔹 支持多种自然语言查询方式：")
            print("  • 分析基金510050")
            print("  • 帮我看看华夏上证50ETF这只基金怎么样")
            print("  • 我想了解一下159919的投资价值")
            print("  • 510050 这个基金值得买吗？")
            print("  • 打分510050（评分模式）")
            print("\n🔹 系统支持两种分析模式：")
            print("  • 报告模式（默认）：生成完整的综合分析报告")
            print("  • 评分模式（输入「打分」/「评分」触发）：输出量化评分和投资建议")
            print("\n💡 提示：建议使用基金代码（如 510050、159919）以获得更准确的分析结果")
            print("\n" + "─" * 78 + "\n")

            # 获取用户输入
            user_query = input("💬 请输入您的分析需求: ")

            # 确保输入不为空
            while not user_query.strip():
                print(f"{ERROR_ICON} 输入不能为空，请重新输入！")
                user_query = input("请输入您的分析需求: ")

        # 记录用户查询到执行日志
        execution_logger.log_agent_start("main", {"user_query": user_query})

        # ============================================================================
        # 3. 自然语言处理和基金信息提取
        # ============================================================================

        # 从查询中提取基金代码和基金名称
        fund_code = None
        fund_name = None

        # 如果通过 --code 指定了基金代码，优先使用
        if args.code:
            fund_code = _normalize_fund_code(args.code)

        # 执行提取
        extracted_name, extracted_code = extract_fund_info(user_query)

        if not fund_name:
            fund_name = extracted_name
        if not fund_code and extracted_code:
            fund_code = _normalize_fund_code(extracted_code)

        # 确定分析模式
        if explicit_mode:
            fund_intent = explicit_mode
        else:
            fund_intent = _detect_intent(user_query)

        # 记录提取结果
        logger.info(f"从查询中提取 - 基金名称: {fund_name}, 基金代码: {fund_code}, 模式: {fund_intent}")

        # ============================================================================
        # 4. 基金名称→代码查找（当只有名称没有代码时）
        # ============================================================================

        if fund_name and not fund_code:
            print(f"\n{WAIT_ICON} 正在通过基金名称 '{fund_name}' 查找基金代码...")
            looked_up_code = await _lookup_fund_by_name(fund_name)
            if looked_up_code:
                fund_code = looked_up_code
                print(f"{SUCCESS_ICON} 已找到基金代码: {fund_code}")
            else:
                print(f"{ERROR_ICON} 无法通过名称 '{fund_name}' 找到对应的基金代码")
                logger.warning(f"基金名称 '{fund_name}' 查找失败，请手动输入基金代码")
                # 如果找不到代码，提示用户
                if not user_query:
                    manual_code = input("💬 请输入基金代码（如 510050）: ").strip()
                    if manual_code and _is_fund_code(manual_code):
                        fund_code = _normalize_fund_code(manual_code)

        # ============================================================================
        # 5. 时间信息处理
        # ============================================================================

        # 获取当前时间信息
        current_datetime = datetime.now()
        current_date_cn = current_datetime.strftime("%Y年%m月%d日")
        current_date_en = current_datetime.strftime("%Y-%m-%d")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                              "星期五", "星期六", "星期日"][current_datetime.weekday()]
        current_time = current_datetime.strftime("%H:%M:%S")

        # 格式化完整的时间信息
        current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

        logger.info(f"当前时间: {current_time_info}")

        # ============================================================================
        # 6. 准备初始状态数据
        # ============================================================================

        # 准备初始状态
        initial_data = {
            "query": user_query,
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat(),
            "fund_intent": fund_intent,
        }

        # 添加基金名称（如果提取到）
        if fund_name:
            initial_data["fund_name"] = fund_name

        # 添加基金代码（如果提取到），使用规范化格式
        if fund_code:
            initial_data["fund_code"] = _normalize_fund_code(fund_code)

        # 创建LangGraph工作流的初始状态
        initial_state = AgentState(
            messages=[],  # Langchain约定：消息列表
            data=initial_data,  # 应用特定数据，包含提取的信息
            metadata={}  # 其他运行时特定信息
        )

        # ============================================================================
        # 7. 执行工作流
        # ============================================================================

        # 显示分析开始信息
        mode_label = "评分" if fund_intent == "score" else "分析报告"
        print(f"\n{WAIT_ICON} 正在开始对 '{user_query}' 进行基金{mode_label}...")
        if fund_name:
            print(f"{WAIT_ICON} 基金名称: {fund_name}")
        if fund_code:
            print(f"{WAIT_ICON} 基金代码: {fund_code}")
        print(f"{WAIT_ICON} 分析模式: {mode_label}")
        logger.info(
            f"Starting fund analysis workflow for query: '{user_query}', intent: {fund_intent}")

        # 显示分析阶段提示
        print(f"\n{WAIT_ICON} 正在执行基金产品文档解析...")
        print(f"{WAIT_ICON} 正在执行基金业绩风险评估...")
        print(f"{WAIT_ICON} 正在执行基金持仓分析...")
        print(f"{WAIT_ICON} 正在执行基金经理分析...")
        print(f"{WAIT_ICON} 正在执行基金基准对标...")
        print(f"{WAIT_ICON} 正在执行基金费用分析...")
        print(f"{WAIT_ICON} 正在执行基金事件分析...")
        print(f"{WAIT_ICON} 这可能需要几分钟时间，请耐心等待...\n")

        # 调用工作流 - 这是阻塞调用，会等待所有智能体完成
        final_state = await app.ainvoke(initial_state)
        print(f"{SUCCESS_ICON} 分析完成！")
        logger.info("Fund workflow execution completed successfully")

        # ============================================================================
        # 8. 结果处理和报告生成
        # ============================================================================

        if final_state and final_state.get("data"):
            if fund_intent == "score" and "fund_score" in final_state["data"]:
                # 评分模式输出
                print("\n--- 基金评分报告 (Fund Score Report) ---\n")
                # print(final_state["data"]["fund_score"])

                if "fund_score_path" in final_state["data"]:
                    print(
                        f"\n{SUCCESS_ICON} 评分报告已保存到: {final_state['data']['fund_score_path']}")
                    logger.info(
                        f"Score report saved to: {final_state['data']['fund_score_path']}")

                    execution_logger.log_final_report(
                        final_state["data"]["fund_score"],
                        final_state["data"]["fund_score_path"]
                    )
                else:
                    execution_logger.log_final_report(
                        final_state["data"]["fund_score"],
                        "N/A (fund_score in state.data)"
                    )

            elif "fund_report" in final_state["data"]:
                # 报告模式输出
                print("\n--- 基金分析报告 (Fund Analysis Report) ---\n")
                # print(final_state["data"]["fund_report"])

                if "fund_report_path" in final_state["data"]:
                    print(
                        f"\n{SUCCESS_ICON} 报告已保存到: {final_state['data']['fund_report_path']}")
                    logger.info(
                        f"Report saved to: {final_state['data']['fund_report_path']}")

                    execution_logger.log_final_report(
                        final_state["data"]["fund_report"],
                        final_state["data"]["fund_report_path"]
                    )
                else:
                    execution_logger.log_final_report(
                        final_state["data"]["fund_report"],
                        "N/A (fund_report in state.data)"
                    )
            else:
                print(f"\n{ERROR_ICON} 错误: 无法从工作流中检索最终报告。")
                logger.error(
                    "Could not retrieve the final report from the workflow")
                print("调试信息 - 最终状态数据键:", list(final_state["data"].keys()))
        else:
            print(f"\n{ERROR_ICON} 错误: 无法从工作流中检索最终状态。")
            logger.error(
                "Could not retrieve the final state from the workflow")
            print("调试信息 - 最终状态内容:", final_state)

        # 完成执行日志记录
        finalize_execution_logger(success=True)
        print(f"{SUCCESS_ICON} 执行日志已保存到: {execution_logger.execution_dir}")

    except Exception as e:
        # ============================================================================
        # 9. 错误处理
        # ============================================================================

        print(f"\n{ERROR_ICON} 工作流执行期间发生错误: {e}")
        logger.error(f"Error during fund workflow execution: {e}", exc_info=True)

        # 记录错误并完成执行日志
        finalize_execution_logger(success=False, error=str(e))
        print(f"{ERROR_ICON} 错误日志已保存到: {get_execution_logger().execution_dir}")


# ============================================================================
# 程序入口点
# ============================================================================

if __name__ == "__main__":
    # 使用asyncio运行主函数
    asyncio.run(main())
