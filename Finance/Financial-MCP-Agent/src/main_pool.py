"""
金融分析智能体系统 - 股票池管理入口
提供股票池的增删改查、评分、报告生成功能

命令：
  python -m src.main_pool add <股票代码> <公司名称>   - 添加股票
  python -m src.main_pool remove <股票代码>             - 删除股票
  python -m src.main_pool list                          - 列出所有股票
  python -m src.main_pool score <股票代码>              - 对指定股票评分
  python -m src.main_pool score-all                     - 对所有待评分股票评分
  python -m src.main_pool report <股票代码>             - 查看股票分析报告
"""

import os
import sys
import asyncio
import re
from typing import Optional

# 抑制无关输出
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(override=True)

from src.stock_pool.stock_pool_manager import StockPoolManager
from src.stock_pool.scoring_engine import ScoringEngine
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


def format_stock_table(stocks):
    """格式化股票列表为表格输出"""
    if not stocks:
        print(f"\n  股票池为空")
        return

    # 表头
    header = f"{'序号':<5} {'股票代码':<12} {'公司名称':<12} {'评分':<7} {'评级':<8} {'状态':<10} {'更新时间':<22}"
    separator = "─" * 80

    print(f"\n{'=' * 80}")
    print("  股票池列表")
    print(f"{'=' * 80}")
    print(header)
    print(separator)

    for i, stock in enumerate(stocks, 1):
        score = str(stock.get("score", "-")) if stock.get("score") is not None else "-"
        recommendation = stock.get("recommendation", "-") or "-"
        status = stock.get("status", "pending")
        last_updated = stock.get("last_updated", "-")[:19] if stock.get("last_updated") else "-"
        company_name = stock.get("company_name", "-")[:10]
        stock_code = stock.get("stock_code", "-")

        print(f"{i:<5} {stock_code:<12} {company_name:<12} {score:<7} {recommendation:<8} {status:<10} {last_updated:<22}")

    print(f"{'=' * 80}")
    print(f"  共 {len(stocks)} 只股票")
    print()


def normalize_stock_code(code: str) -> str:
    """
    标准化股票代码，添加交易所前缀

    Args:
        code: 股票代码(可能带或不带sh./sz.前缀)

    Returns:
        标准化后的股票代码
    """
    code = code.strip().lower()
    # 如果已有前缀，直接返回
    if code.startswith("sh.") or code.startswith("sz."):
        return code
    # 根据数字判断交易所
    if code.startswith("6"):
        return f"sh.{code}"
    elif code.startswith("0") or code.startswith("3"):
        return f"sz.{code}"
    return code


async def cmd_add(pool: StockPoolManager, engine: ScoringEngine, args):
    """添加股票到池中"""
    if len(args) < 2:
        print(f"{ERROR_ICON} 用法: add <股票代码> <公司名称>")
        print(f"  示例: add 603871 嘉友国际")
        return

    stock_code = normalize_stock_code(args[0])
    company_name = args[1]

    stock = pool.add_stock(stock_code, company_name)
    print(f"{SUCCESS_ICON} 已添加: {company_name}({stock_code})")
    print(f"  状态: {stock['status']}")
    print(f"\n是否立即对该股票进行评分? (y/n, 默认n): ", end="")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer == "y":
        print(f"\n{WAIT_ICON} 正在启动完整评分Pipeline...")
        result = await engine.score_stock(stock_code, company_name)
        if result.get("score_data"):
            sd = result["score_data"]
            print(f"{SUCCESS_ICON} 评分完成!")
            print(f"  中线评分(主评分): {sd['score']} ({sd['recommendation']})")
            if sd.get("short_term_score"):
                sts = sd["short_term_score"]
                print(f"  短线评分: {sts.get('score', '-')} ({sts.get('recommendation', '-')})")
            if sd.get("long_term_score"):
                lts = sd["long_term_score"]
                print(f"  长线评分: {lts.get('score', '-')} ({lts.get('rating', '-')})")
        else:
            print(f"{ERROR_ICON} 评分失败: {result.get('error')}")


async def cmd_remove(pool: StockPoolManager, engine: ScoringEngine, args):
    """从池中删除股票"""
    if len(args) < 1:
        print(f"{ERROR_ICON} 用法: remove <股票代码>")
        return

    stock_code = normalize_stock_code(args[0])
    stock = pool.get_stock(stock_code)
    if stock:
        print(f"确认删除 {stock['company_name']}({stock_code})? (y/n): ", end="")
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            pool.remove_stock(stock_code)
            print(f"{SUCCESS_ICON} 已删除: {stock['company_name']}({stock_code})")
        else:
            print("  已取消删除")
    else:
        print(f"{ERROR_ICON} 股票 {stock_code} 不在池中")


async def cmd_list(pool: StockPoolManager, engine: ScoringEngine, args):
    """列出所有股票"""
    stocks = pool.list_stocks()
    format_stock_table(stocks)


async def cmd_score(pool: StockPoolManager, engine: ScoringEngine, args):
    """对指定股票评分"""
    if len(args) < 1:
        print(f"{ERROR_ICON} 用法: score <股票代码>")
        return

    stock_code = normalize_stock_code(args[0])
    stock = pool.get_stock(stock_code)
    if not stock:
        print(f"{ERROR_ICON} 股票 {stock_code} 不在池中，请先使用 add 命令添加")
        return

    print(f"{WAIT_ICON} 正在对 {stock['company_name']}({stock_code}) 运行完整评分Pipeline...")
    print(f"{WAIT_ICON} 这将运行7个Agent(4分析+3打分)并行，可能需要几分钟...\n")

    result = await engine.score_stock(stock_code, stock["company_name"])

    if result.get("score_data"):
        sd = result["score_data"]
        print(f"\n{SUCCESS_ICON} 评分完成!")
        print(f"  {'─' * 50}")

        # 中线评分（主评分）
        mt = sd.get("medium_term_score", {})
        if mt:
            print(f"  【中线】评分: {mt.get('score', '-')} ({mt.get('rating', '-')})")
            print(f"  理由: {mt.get('reasoning', '-')}")
            subs = mt.get("sub_scores", {})
            if subs:
                print(f"  分项: 基本面{subs.get('fundamental_quality', '-')}/25 成长{subs.get('growth', '-')}/15 "
                      f"估值{subs.get('valuation', '-')}/20 技术{subs.get('technical_trend', '-')}/15 "
                      f"情绪{subs.get('sentiment_flow', '-')}/10 风险{subs.get('risk_assessment', '-')}/15")

        # 短线评分
        st = sd.get("short_term_score", {})
        if st:
            print(f"\n  【短线】评分: {st.get('score', '-')} ({st.get('recommendation', '-')})")
            print(f"  理由: {st.get('reasoning', '-')}")

        # 长线评分
        lt = sd.get("long_term_score", {})
        if lt:
            print(f"\n  【长线】评分: {lt.get('score', '-')} ({lt.get('rating', '-')})")
            print(f"  护城河: {lt.get('moat_type', '-')}")
            print(f"  理由: {lt.get('reasoning', '-')}")

        print(f"  {'─' * 50}")
        print(f"  耗时: {result['execution_time']:.1f}秒")
    else:
        print(f"{ERROR_ICON} 评分失败: {result.get('error')}")


async def cmd_score_all(pool: StockPoolManager, engine: ScoringEngine, args):
    """对所有待评分股票逐一评分"""
    pending = pool.get_pending_stocks()
    if not pending:
        print(f"{SUCCESS_ICON} 没有待评分的股票")
        return

    print(f"\n{WAIT_ICON} 共有 {len(pending)} 只股票待评分:\n")
    for s in pending:
        print(f"  - {s['company_name']}({s['stock_code']})")
    print()

    print("确认对所有待评分股票逐一评分? (y/n): ", end="")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    if answer != "y":
        print("  已取消")
        return

    results = await engine.score_all_pending()

    print(f"\n{'=' * 80}")
    print("  评分结果汇总")
    print(f"{'=' * 80}")
    for r in results:
        name = r["company_name"]
        code = r["stock_code"]
        if r.get("score_data"):
            sd = r["score_data"]
            st = sd.get("short_term_score", {})
            mt = sd.get("medium_term_score", {})
            lt = sd.get("long_term_score", {})
            st_s = st.get("score", "-") if st else "-"
            mt_s = mt.get("score", "-") if mt else "-"
            lt_s = lt.get("score", "-") if lt else "-"
            print(f"  {SUCCESS_ICON} {name}({code}): 短线={st_s} 中线={mt_s} 长线={lt_s}")
        else:
            print(f"  {ERROR_ICON} {name}({code}): 评分失败 - {r.get('error')}")
    print(f"{'=' * 80}")


async def cmd_report(pool: StockPoolManager, engine: ScoringEngine, args):
    """查看股票评分详情"""
    if len(args) < 1:
        print(f"{ERROR_ICON} 用法: report <股票代码>")
        return

    stock_code = normalize_stock_code(args[0])
    stock = pool.get_stock(stock_code)

    if not stock:
        print(f"{ERROR_ICON} 股票 {stock_code} 不在池中")
        return

    if not stock.get("score"):
        print(f"{WAIT_ICON} {stock['company_name']} 暂无评分")
        print(f"是否现在生成评分? (y/n): ", end="")
        try:
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            result = await engine.score_stock(stock_code, stock["company_name"])
            if result.get("score_data"):
                stock = pool.get_stock(stock_code)  # 刷新数据
            else:
                print(f"{ERROR_ICON} 评分失败: {result.get('error')}")
                return

    print(f"\n{'=' * 80}")
    print(f"  {stock['company_name']}({stock['stock_code']}) 评分详情")
    print(f"  主评分: {stock.get('score', '-')}  |  主评级: {stock.get('recommendation', '-')}")
    print(f"{'=' * 80}\n")

    # 中线评分详情
    mt = stock.get("medium_term_score", {})
    if mt:
        print(f"  【中线 1-3个月】")
        print(f"  评分: {mt.get('score', '-')} ({mt.get('rating', '-')})")
        print(f"  理由: {mt.get('reasoning', '-')}")
        print(f"  风险: {mt.get('risk_warning', '-')}")
        print(f"  建议: {mt.get('suggested_action', '-')}")
        print(f"  周期: {mt.get('time_horizon', '-')}")
        print()

    # 短线评分详情
    st = stock.get("short_term_score", {})
    if st:
        print(f"  【短线 1-5天】")
        print(f"  评分: {st.get('score', '-')} ({st.get('recommendation', '-')})")
        print(f"  理由: {st.get('reasoning', '-')}")
        print(f"  风险: {st.get('risk_warning', '-')}")
        print(f"  建议: {st.get('suggested_action', '-')}")
        print()

    # 长线评分详情
    lt = stock.get("long_term_score", {})
    if lt:
        print(f"  【长线 1-3年+】")
        print(f"  评分: {lt.get('score', '-')} ({lt.get('rating', '-')})")
        print(f"  护城河: {lt.get('moat_type', '-')}")
        print(f"  理由: {lt.get('reasoning', '-')}")
        print(f"  风险: {lt.get('risk_warning', '-')}")
        print(f"  建议: {lt.get('suggested_action', '-')}")
        print(f"  周期: {lt.get('time_horizon', '-')}")
        print()

    print(f"{'=' * 80}")


COMMANDS = {
    "add": (cmd_add, "添加股票到股票池"),
    "remove": (cmd_remove, "从股票池中删除股票"),
    "list": (cmd_list, "列出所有股票"),
    "ls": (cmd_list, "列出所有股票"),
    "score": (cmd_score, "对指定股票运行完整Pipeline评分(短+中+长线)"),
    "score-all": (cmd_score_all, "对所有待评分股票逐一评分"),
    "report": (cmd_report, "查看股票评分详情"),
}


def print_help():
    """打印帮助信息"""
    print("\n" + "─" * 60)
    print("  股票池管理系统 - 帮助")
    print("─" * 60)
    print("\n命令列表:\n")
    for cmd, (_, desc) in sorted(COMMANDS.items()):
        print(f"  {cmd:<12} {desc}")
    print(f"  {'help':<12} 显示此帮助信息")
    print(f"  {'quit/exit':<12} 退出程序")
    print("\n示例:\n")
    print("  add 603871 嘉友国际          # 添加股票")
    print("  score 603871                 # 对该股票评分")
    print("  list                         # 查看股票池")
    print("  report 603871                # 查看分析报告")
    print("─" * 60 + "\n")


async def interactive_mode():
    """交互式命令行模式"""
    pool = StockPoolManager()
    engine = ScoringEngine(pool_manager=pool)

    print("\n╔══════════════════════════════════════════════════╗")
    print("║          股票池管理系统                          ║")
    print("║          Stock Pool Manager                      ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\n{SUCCESS_ICON} 股票池已加载: {pool.count()} 只股票")
    print("输入 'help' 查看可用命令\n")

    while True:
        try:
            user_input = input("pool> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        if user_input.lower() in ("help", "h", "?"):
            print_help()
            continue

        parts = user_input.split()
        cmd = parts[0].lower()
        cmd_args = parts[1:]

        if cmd in COMMANDS:
            handler, _ = COMMANDS[cmd]
            try:
                await handler(pool, engine, cmd_args)
            except Exception as e:
                logger.error(f"{ERROR_ICON} 命令执行失败: {e}", exc_info=True)
                print(f"{ERROR_ICON} 错误: {e}")
        else:
            print(f"{ERROR_ICON} 未知命令: {cmd}，输入 'help' 查看可用命令")


async def main():
    """主入口：支持CLI参数和交互模式"""
    if len(sys.argv) > 1:
        # CLI参数模式
        pool = StockPoolManager()
        engine = ScoringEngine(pool_manager=pool)

        cmd = sys.argv[1].lower()
        args = sys.argv[2:]

        if cmd in COMMANDS:
            handler, _ = COMMANDS[cmd]
            await handler(pool, engine, args)
        else:
            print(f"{ERROR_ICON} 未知命令: {cmd}")
            print_help()
    else:
        # 交互模式
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
