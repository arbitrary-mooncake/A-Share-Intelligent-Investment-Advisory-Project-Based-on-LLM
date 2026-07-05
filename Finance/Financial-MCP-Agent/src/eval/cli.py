"""
评测系统CLI入口 — python -m src.eval <command>
"""
import sys
import os
import argparse
from datetime import datetime


def cmd_check(args):
    """执行日常检查（收盘前调仓 + 收盘后结算）"""
    import asyncio
    from src.eval.check_runner import run_check_async

    print("=" * 60)
    print("  评分智能体 — 日常检查")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        result = asyncio.run(run_check_async())
        print(f"  批次: {result.get('batch_id', '?')[:16]}")
        settle = result.get("settlement", {})
        print(f"  结算: {settle.get('total_settled', 0)}笔")
        rebalance = result.get("rebalance", {})
        for term, info in rebalance.items():
            lines = info.get("lines", {})
            success = sum(1 for l in lines.values() if l.get("status") == "success")
            print(f"  {term}: {success}/{len(lines)}条线调仓成功")
        print("=" * 60)
    except Exception as e:
        print(f"  检查失败: {e}")
        print("  (如Tushare token未配置，请先配置.env中的TUSHARE_TOKEN)")


def cmd_rebalance(args):
    """收盘前调仓"""
    import asyncio
    from src.eval.check_runner import run_rebalance_async
    from src.eval.data_fetcher import build_market_data_map
    from src.eval.orchestrator import EvalOrchestrator

    term = args.term or "short"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {term} 收盘前调仓...")
    try:
        orch = EvalOrchestrator()
        pool = orch.pool_manager.get_pool(term)
        if not pool:
            print(f"  {term}精筛池为空，请先运行 python -m src.eval pool update --term {term}")
            return
        market_data = build_market_data_map(pool)
        result = asyncio.run(orch.run_daily_rebalance(term, datetime.now().strftime("%Y-%m-%d"), market_data))
        for line_id, info in result.get("lines", {}).items():
            print(f"  {line_id}: {info.get('status', '?')} | 持仓{info.get('holdings_count', 0)}只 | 现金¥{info.get('cash', 0):,.0f}")
    except Exception as e:
        print(f"  调仓失败: {e}")


def cmd_settle(args):
    """收盘后结算"""
    from src.eval.orchestrator import EvalOrchestrator

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 收盘后结算...")
    try:
        orch = EvalOrchestrator()
        orch.run_daily_settlement()
        lines = orch.line_manager.get_all_status()
        for l in lines[:5]:
            print(f"  {l['line_id']}: 市值¥{l.get('total_value', 0):,.0f} | 收益{l.get('cumulative_return_pct', 0):.1f}% | 回撤{l.get('max_drawdown_pct', 0):.1f}%")
        print(f"  ... 共{len(lines)}条线已结算")
    except Exception as e:
        print(f"  结算失败: {e}")


def cmd_status(args):
    """查看各线状态"""
    from src.eval.database import init_db
    from src.eval.repositories import get_latest_batch, get_recent_batches
    from src.eval.orchestrator import EvalOrchestrator

    init_db()

    latest = get_latest_batch()
    if latest:
        print(f"  最新批次: {latest['batch_id']}")
        print(f"  状态: {latest['status']}")
        print(f"  时间: {latest.get('started_at', 'N/A')}")

    recents = get_recent_batches(5)
    if recents:
        print(f"\n  最近{len(recents)}次批次:")
        for b in recents:
            print(f"    {b['batch_id'][:16]} | {b['status']} | {b.get('started_at', 'N/A')[:19]}")

    try:
        orch = EvalOrchestrator()
        pools = orch.pool_manager.get_all_summaries()
        print(f"\n  精筛池:")
        for term, s in pools.items():
            print(f"    {term}: {s['size']}只 | 均分{s['avg_score']} | 更新: {s['updated_at'][:19] if s['updated_at'] else '从未'}")
    except Exception:
        pass


def cmd_pool(args):
    """精筛池管理"""
    from src.eval.pool_manager import PoolManager

    action = args.pool_action
    term = args.term or "short"

    pm = PoolManager()

    if action == "status":
        summary = pm.get_pool_summary(term)
        print(f"  精筛池 [{term}]:")
        print(f"    股票数: {summary['size']}/{summary['target_size']}")
        print(f"    平均分: {summary['avg_score']}")
        print(f"    最近更新: {summary['updated_at'][:19] if summary['updated_at'] else '从未'}")
        print(f"    版本: {summary['version']}")
        need = pm.needs_update(term)
        if need['needs_update']:
            print(f"    ⚠ 已{need['days_since_update']}天未更新（阈值{need['max_days']}天），建议更新")
        # 显示前10只
        stocks = pm.get_pool_with_scores(term)[:10]
        if stocks:
            print(f"    Top 10:")
            for s in stocks:
                if isinstance(s, dict):
                    print(f"      {s.get('code', '?')} {s.get('name', '?')}: {s.get('score', '?')}分")

    elif action == "update":
        mode = getattr(args, "mode", "full") or "full"
        print(f"  更新模式: {mode}")
        print(f"  期限: {term}")
        from src.eval.orchestrator import EvalOrchestrator
        import asyncio
        orch = EvalOrchestrator()
        result = asyncio.run(orch.run_pool_update(term, mode=mode))
        if result.get("error"):
            print(f"  ❌ 错误: {result['error']}")
            if result.get("message"):
                print(f"  {result['message']}")
        else:
            pool_size = result.get("final_pool_size", 0)
            stats = result.get("stats", {})
            print(f"  ✅ 更新完成: {pool_size}只")
            if mode == "partial":
                print(f"  替换: {stats.get('removed', 0)}只, "
                      f"新增: {stats.get('added', 0)}只, "
                      f"保留: {stats.get('kept', 0)}只")


def cmd_backtest(args):
    """运行回测"""
    import asyncio
    from src.eval.replay_backtest_engine import ReplayBacktestEngine, BacktestConfig
    from src.eval.pool_manager import PoolManager

    term = args.term or "medium"
    start = args.start or "2024-01-01"
    end = args.end or "2025-12-31"

    print(f"  回测 [{term}] {start} ~ {end}")
    print(f"  正在生成anchor dates...")

    pm = PoolManager()
    pool = pm.get_pool(term)
    if not pool:
        print(f"  {term}精筛池为空！请先建立精筛池。")
        return

    cfg = BacktestConfig(term=term, start_date=start, end_date=end)
    engine = ReplayBacktestEngine(cfg)

    anchors = engine.generate_anchor_dates()
    print(f"  共{len(anchors)}个anchor dates")

    async def run():
        return await engine.run_full_backtest(pool, {})

    try:
        result = asyncio.run(run())
        print(f"  回测完成:")
        print(f"    执行anchor数: {result['config']['num_anchors_executed']}")
        print(f"    Agent贡献摘要:")
        for agent, info in result.get("contribution_summary", {}).items():
            print(f"      {agent}: ΔL={info['mean_delta_L']:.4f} ({info['direction']})")
        # 参考线结果 (SB-L6)
        ref = result.get("reference_line")
        if ref and "error" not in ref:
            print(f"\n  ── 长持对照线 (SB-L6, 现实性校验) ──")
            print(f"    累计收益: {ref.get('cumulative_return_pct', 0)}%")
            print(f"    年化收益: {ref.get('annualized_return_pct', 0)}%")
            print(f"    最大回撤: {ref.get('max_drawdown_pct', 0)}%")
            print(f"    Sharpe: {ref.get('sharpe_ratio', 0)}")
            print(f"    胜率: {ref.get('win_rate_pct', 0)}%")
            print(f"    最终持仓: {ref.get('final_holdings_count', 0)}只")
            print(f"    注: SB-L6不参与消融ΔLoss计算")
        elif ref and "error" in ref:
            print(f"\n  [警告] SB-L6参考线回测失败: {ref['error']}")
        print(f"\n  ⚠ 声明: {result['declarations'][0]}")
    except Exception as e:
        print(f"  回测失败: {e}")


def cmd_report(args):
    """查看报告"""
    from src.eval.repositories import get_latest_batch
    from src.eval.orchestrator import EvalOrchestrator
    from src.eval.report_builder import ReportBuilder

    try:
        orch = EvalOrchestrator()
        status = orch.get_status()
        latest = get_latest_batch()
        batch_id = args.batch or (latest["batch_id"] if latest else "unknown")

        rb = ReportBuilder()
        data = rb.build_batch_report_data(batch_id, status)
        path = rb.save_report(data)
        print(f"  报告已生成: {path}")
        print(f"  批次: {batch_id[:20]}")
        exec_sum = data.get("executive_summary", {})
        if exec_sum:
            print(f"  全线平均收益: {exec_sum.get('avg_return_all_lines', 0)}%")
    except Exception as e:
        print(f"  报告生成失败: {e}")


def cmd_trends(args):
    """查看趋势"""
    from src.eval.memory_manager import MemoryManager
    from src.eval.chart_service import ChartService

    metric = args.metric or "score"
    term = args.term or "medium"
    days = args.days or 90

    mm = MemoryManager()
    cs = ChartService(mm)

    if metric == "score":
        data = cs.get_score_trend_data(days)
    elif metric == "loss":
        data = cs.get_loss_trend_data(days)
    elif metric == "contribution":
        data = {"title": "Agent贡献趋势", "data": mm.get_agent_trend(term, days)}
    else:
        data = cs.get_score_trend_data(days)

    print(f"  {data['title']}")
    for entry in data.get("data", [])[-10:]:
        if isinstance(entry, dict):
            print(f"    {entry.get('date', '?')}: {entry.get('value', entry.get('delta_L_total', 0)):.4f}")


def cmd_agent_contribution(args):
    """查看Agent贡献数据"""
    from src.eval.memory_manager import MemoryManager
    from src.eval.database import init_db

    init_db()
    term = args.term or "medium"
    source = args.source or "backtest"

    mm = MemoryManager()
    data = mm.get_agent_trend(term, days=args.days or 90)

    print(f"  Agent贡献趋势 [{term}] (来源: {source})")
    print(f"  {'─' * 40}")
    if data:
        for entry in data[-10:]:
            if isinstance(entry, dict):
                agent = entry.get("agent_name", "?")
                delta = entry.get("delta_L_total", 0)
                stars = entry.get("stars", "")
                direction = "👍 正贡献" if delta > 0.01 else ("👎 负贡献" if delta < -0.01 else "➖ 中性")
                print(f"  {agent:20s} ΔL={delta:+.4f} {stars:4s} {direction}")
    else:
        print("  暂无贡献数据。运行回测或积累实盘数据后获取。")


def cmd_optimize(args):
    """优化操作"""
    from src.eval.optimizer.router import OptimizeRouter
    from src.eval.optimizer.manual_package_builder import ManualPackageBuilder
    from src.eval.repositories import get_tickets_by_batch, get_latest_batch

    action = args.opt_action or "analyze"

    if action == "analyze":
        print("  分析最新检查结果，生成优化建议...")
        latest = get_latest_batch()
        if not latest:
            print("  暂无检查批次，请先运行 python -m src.eval check")
            return

        router = OptimizeRouter()
        evidence = {
            "type": "parameter_suboptimal",
            "title": f"批次{latest['batch_id'][:16]}分析",
            "summary": "请运行完整检查后查看详细分析",
            "severity": "medium",
            "affected_files": ["config/eval/defaults.json"],
            "complexity": "low",
        }
        classification = router.classify_issue(evidence)
        ticket = router.generate_ticket(latest["batch_id"], evidence, classification)
        print(f"  类型: {ticket['ticket_type']} | 路由: {ticket['route']}")
        print(f"  建议操作: {ticket['suggested_actions'][0]}")

        if ticket["route"] == "manual":
            builder = ManualPackageBuilder()
            pkg = builder.build_package(ticket, evidence)
            path = builder.save_package(pkg)
            print(f"  人工优化包已生成: {path}")

    elif action == "apply":
        if args.ticket_id:
            print(f"  应用优化ticket: {args.ticket_id}")
            print(f"  请在Web UI中审核后执行，或使用优化包手动修改")
    elif action == "rollback":
        if args.ticket_id:
            print(f"  回滚优化ticket: {args.ticket_id}")
            print(f"  回滚功能需通过Web UI操作")


def main():
    parser = argparse.ArgumentParser(
        description="评分智能体 — 评估、归因、优化控制塔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.eval check              # 一键收盘前调仓+收盘后结算
  python -m src.eval status             # 查看当前各线状态
  python -m src.eval pool status --term short   # 查看短线池
  python -m src.eval pool update --term short --mode full  # 全量更新短线池
  python -m src.eval backtest --term medium --start 2024-01-01 --end 2025-12-31
  python -m src.eval report --latest
  python -m src.eval trends --metric score --term medium --days 90
  python -m src.eval optimize --analyze
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # check
    p_check = subparsers.add_parser("check", help="一键日常检查（调仓+结算）")
    p_check.set_defaults(func=cmd_check)

    # rebalance
    p_reb = subparsers.add_parser("rebalance", help="收盘前调仓")
    p_reb.add_argument("--term", choices=["short", "medium", "long"], default="short", help="期限")
    p_reb.set_defaults(func=cmd_rebalance)

    # settle
    p_set = subparsers.add_parser("settle", help="收盘后结算")
    p_set.set_defaults(func=cmd_settle)

    # status
    p_stat = subparsers.add_parser("status", help="查看当前各线状态")
    p_stat.set_defaults(func=cmd_status)

    # pool
    p_pool = subparsers.add_parser("pool", help="精筛池管理")
    p_pool.add_argument("pool_action", choices=["status", "update"], help="操作")
    p_pool.add_argument("--term", choices=["short", "medium", "long"], help="期限")
    p_pool.add_argument("--mode", choices=["status", "full", "partial"], help="更新模式")
    p_pool.set_defaults(func=cmd_pool)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="运行回测")
    p_bt.add_argument("--term", choices=["short", "medium", "long"], help="期限")
    p_bt.add_argument("--start", help="起始日期 YYYY-MM-DD")
    p_bt.add_argument("--end", help="结束日期 YYYY-MM-DD")
    p_bt.set_defaults(func=cmd_backtest)

    # report
    p_rpt = subparsers.add_parser("report", help="查看评测报告")
    p_rpt.add_argument("--batch", help="批次ID")
    p_rpt.add_argument("--latest", action="store_true", help="最新报告")
    p_rpt.set_defaults(func=cmd_report)

    # trends
    p_trd = subparsers.add_parser("trends", help="查看趋势")
    p_trd.add_argument("--metric", help="指标: score/loss/contribution")
    p_trd.add_argument("--term", choices=["short", "medium", "long"], help="期限")
    p_trd.add_argument("--days", type=int, help="天数")
    p_trd.set_defaults(func=cmd_trends)

    # agent-contribution
    p_ac = subparsers.add_parser("agent-contribution", help="查看Agent贡献")
    p_ac.add_argument("--term", choices=["short", "medium", "long"], help="期限")
    p_ac.add_argument("--source", choices=["backtest", "live"], default="backtest", help="数据来源")
    p_ac.add_argument("--days", type=int, help="统计天数")
    p_ac.set_defaults(func=cmd_agent_contribution)

    # optimize
    p_opt = subparsers.add_parser("optimize", help="优化操作")
    p_opt.add_argument("opt_action", nargs="?", choices=["analyze", "apply", "rollback"],
                        default="analyze", help="操作")
    p_opt.add_argument("--ticket-id", help="优化ticket ID")
    p_opt.set_defaults(func=cmd_optimize)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
