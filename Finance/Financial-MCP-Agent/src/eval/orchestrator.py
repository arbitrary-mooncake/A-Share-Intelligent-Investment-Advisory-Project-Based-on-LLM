"""
编排器 — 评测系统的中央调度引擎。
连接分析→评分→策略→仿真→结算→Loss的完整主循环。
"""
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from src.eval.database import init_db, generate_id
from src.eval.repositories import (
    create_batch, update_batch_status, get_latest_batch,
    create_snapshot, get_snapshots_by_batch
)
from src.eval.schemas import EvalBatch
from src.eval.line_manager import LineManager
from src.eval.pool_manager import PoolManager
from src.eval.market_simulator import MarketSimulator, MarketData
from src.eval.strategies.factory import get_strategy
from src.eval.config import get_config


class EvalOrchestrator:
    """评测编排器 — 管理完整的检查-结算-评分-调仓循环"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        init_db()

        self.line_manager = LineManager(
            initial_capital=self.config.get("initial_capital_per_line", 1000000.0)
        )
        self.pool_manager = PoolManager()
        self.market_simulator = MarketSimulator(config)
        self.current_batch_id = ""

    # ═══════════════════════════════════════════════
    # 阶段入口
    # ═══════════════════════════════════════════════

    def start_batch(self, trigger_source: str = "ui") -> str:
        """开始一个新的评测批次"""
        batch = EvalBatch(
            batch_id=f"eval_{generate_id()}",
            status="running",
            trigger_source=trigger_source,
            started_at=datetime.now().isoformat(),
            market_session="post_close",
        )
        self.current_batch_id = create_batch(batch)
        return self.current_batch_id

    def finish_batch(self, success: bool = True, error: str = ""):
        """完成当前批次"""
        if not self.current_batch_id:
            return
        status = "completed" if success else "failed"
        update_batch_status(
            self.current_batch_id, status,
            finished_at=datetime.now().isoformat(),
            error_message=error,
            optimize_ready=int(success),
        )

    # ═══════════════════════════════════════════════
    # 阶段1: 结算历史样本
    # ═══════════════════════════════════════════════

    async def settle_historical(self, current_date: str) -> Dict[str, Any]:
        """结算所有已成熟的历史snapshot"""
        from src.eval.settlement_engine import SettlementEngine
        engine = SettlementEngine(self.config)

        results = {}
        for term in ["short", "medium", "long"]:
            results[term] = await engine.settle_matured_snapshots(current_date, term)

        total_settled = sum(r["settled"] for r in results.values())
        return {"terms": results, "total_settled": total_settled}

    # ═══════════════════════════════════════════════
    # 阶段2: 评分与调仓
    # ═══════════════════════════════════════════════

    async def run_daily_rebalance(self, term: str, current_date: str,
                                   market_data_map: Dict[str, MarketData]) -> Dict[str, Any]:
        """
        执行一个期限的日常调仓（所有该期限的线）。

        流程:
          1. 获取精筛池
          2. 各线评分（V1使用模拟评分，Phase 3接入真实agent）
          3. 短线消融线交叉截面同步
          4. 各线策略选股+生成订单
          5. 市场仿真层执行订单
          6. 更新各线持仓状态
          7. 保存snapshot
        """
        pool = self.pool_manager.get_pool(term)
        if not pool:
            return {"status": "skipped", "reason": f"{term}精筛池为空"}

        lines = self.line_manager.get_lines_by_term(term)
        results = {}

        # 短线消融线：在评分前同步持仓
        if term == "short":
            self.line_manager.sync_ablation_holdings("short")

        for line in lines:
            # 每条线使用独立的MarketSimulator实例（T+1隔离）
            line_sim = MarketSimulator(self.config)
            line_sim.reset_daily_state()
            
            try:
                # 获取策略
                strategy_type = line.definition.get("strategy", "default")
                strategy = get_strategy(term, strategy_type, self.config)

                # V1: 生成模拟评分（Phase 3接入真实scorer）
                scores = self._get_simulated_scores(pool, term)

                # 周一全量重评：消化周末消息面变化，强制重评池内所有股票
                if term == "short" and EvalOrchestrator._get_schedule()["is_monday"]:
                    logger.info("Monday full run: scoring all short pool stocks (%d stocks)", len(pool))
                    # Force re-score all pool stocks (not just top-N)

                # 策略选股
                buy_orders = strategy.select_stocks(
                    pool, scores, line.holdings, line.cash,
                    {code: market_data_map.get(code) for code in pool}
                )

                # 策略卖出
                sell_orders = strategy.generate_sell_orders(
                    line.holdings, scores,
                    {code: market_data_map.get(code) for code in pool},
                    line.purchase_prices, line.hold_days
                )

                # 仓位调整
                sized_orders = strategy.size_positions(
                    buy_orders, line.total_value,
                    len(line.holdings), strategy.get_max_positions(),
                    strategy.get_single_weight_limit(), strategy.get_min_cash_ratio()
                )

                # 执行订单（通过市场仿真层）
                from src.eval.market_simulator import Order as SimOrder

                # 先执行卖出
                for so in sell_orders:
                    if so.stock_code in line.holdings and line.holdings[so.stock_code] > 0:
                        shares_to_sell = int(line.holdings[so.stock_code] * so.sell_ratio / 100) * 100
                        if shares_to_sell > 0:
                            md = market_data_map.get(so.stock_code)
                            if md:
                                order = SimOrder(stock_code=so.stock_code, direction="sell",
                                                quantity=shares_to_sell, target_value=0)
                                result = line_sim.execute_order(order, md)
                                if result.status in ("filled", "partial"):
                                    line.cash += result.net_cost  # net_cost is positive for sell
                                    line.holdings[so.stock_code] -= result.quantity
                                    if line.holdings[so.stock_code] <= 0:
                                        del line.holdings[so.stock_code]
                                        line.purchase_prices.pop(so.stock_code, None)
                                        line.hold_days.pop(so.stock_code, None)
                                    line.trade_count += 1

                # 再执行买入
                for bo in sized_orders:
                    if bo.target_value > 0 and bo.target_value <= line.cash:
                        md = market_data_map.get(bo.stock_code)
                        if md:
                            order = SimOrder(stock_code=bo.stock_code, direction="buy",
                                            target_value=bo.target_value)
                            result = line_sim.execute_order(order, md)
                            if result.status in ("filled", "partial"):
                                line.cash -= result.net_cost  # net_cost is positive for buy
                                line.holdings[bo.stock_code] = (
                                    line.holdings.get(bo.stock_code, 0) + result.quantity
                                )
                                if bo.stock_code in line.purchase_prices:
                                    old_qty = line.holdings[bo.stock_code] - result.quantity
                                    old_avg = line.purchase_prices[bo.stock_code]
                                    new_avg = (old_avg * old_qty + result.actual_price * result.quantity) / line.holdings[bo.stock_code]
                                    line.purchase_prices[bo.stock_code] = new_avg
                                else:
                                    line.purchase_prices[bo.stock_code] = result.actual_price
                                line.hold_days[bo.stock_code] = 0
                                line.trade_count += 1

                # 更新持有天数
                for code in list(line.hold_days.keys()):
                    if code in line.holdings and line.holdings[code] > 0:
                        line.hold_days[code] += 1
                    else:
                        del line.hold_days[code]

                # 重置市场仿真器日内状态
                self.market_simulator.reset_daily_state()

                results[line.line_id] = {
                    "status": "success",
                    "holdings_count": len(line.holdings),
                    "cash": line.cash,
                }

            except Exception as e:
                results[line.line_id] = {"status": "error", "error": str(e)}

        return {"term": term, "lines": results}

    def _get_simulated_scores(self, pool: List[str], term: str = "short") -> Dict[str, float]:
        """
        获取精筛池评分 — 优先使用池中已存储评分，缺失时通过Tushare实时数据估算。
        """
        scores = {}
        pool_data = self.pool_manager.get_pool_with_scores(term)
        # Build score lookup from pool
        score_lookup = {}
        for s in pool_data:
            if isinstance(s, dict):
                score_lookup[s.get("code", "")] = s.get("score", 50)

        for code in pool:
            if code in score_lookup:
                scores[code] = score_lookup[code]
            else:
                scores[code] = 50.0  # 默认中性

        return scores

    async def _fetch_market_data(self, pool: List[str],
                                  trade_date: str = "") -> Dict[str, 'MarketData']:
        """通过Tushare获取真实市场数据"""
        try:
            from src.eval.data_fetcher import build_market_data_map
            return build_market_data_map(pool, trade_date)
        except Exception:
            return {}

    async def run_pool_update(self, term: str = "short",
                               on_stage: callable = None) -> Dict[str, Any]:
        """
        精筛池更新 — 完整四层管线（总纲 §4.1）。

        管线:
          Layer 0: 硬筛 (去ST/BJ/B/新股/低成交额)
          Layer 1: M1/M3批量粗筛分档 (强烈推荐→白名单, 买入/谨慎买入/观望→初筛池, 卖出→黑名单)
          Layer 2: M2快筛过滤初筛池
          Layer 3: 1:1.2差额组建候选 → 正式7Agent+3Scorer → LLM动态阈值 → 最终精筛池

        Args:
            term: short/medium/long
            on_stage: 阶段回调(stage_name, message) — 供Streamlit实时显示进度

        Returns:
            完整结果字典
        """
        from src.eval.pool_screening import run_pool_update as _run
        return await _run(term=term, on_stage=on_stage)

    async def run_pool_update_light(self, term: str = "short") -> Dict[str, Any]:
        """[已废弃] 保留兼容旧接口，内部转调 run_pool_update（四层管线）"""
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning("run_pool_update_light is deprecated, using run_pool_update (4-layer pipeline) instead")
        return await self.run_pool_update(term=term)

    async def run_pool_update_formal(self, term: str = "short",
                                      max_stocks: int = 50) -> Dict[str, Any]:
        """[已废弃] 保留兼容旧接口，内部转调 run_pool_update。
        max_stocks 参数忽略——候选数量由四层管线内在的1:1.2差额机制自行决定。"""
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning("run_pool_update_formal is deprecated, using run_pool_update (4-layer pipeline) instead")
        return await self.run_pool_update(term=term)

    # ═══════════════════════════════════════════════
    # 阶段3: 收盘后结算
    # ═══════════════════════════════════════════════

    def run_daily_settlement(self, current_prices: Dict[str, float] = None):
        """根据收盘价更新所有线的持仓市值和收益。若未提供价格，从Tushare获取。"""
        if not current_prices:
            try:
                all_codes = set()
                for line in self.line_manager.lines.values():
                    all_codes.update(line.holdings.keys())
                from src.eval.data_fetcher import fetch_realtime_prices
                current_prices = fetch_realtime_prices(list(all_codes))
            except Exception:
                current_prices = {}
        self.line_manager.update_all_values(current_prices)

    # ═══════════════════════════════════════════════
    # 一键完整流程
    # ═══════════════════════════════════════════════

    async def run_full_check(self, current_date: str = "",
                              market_data: Dict[str, MarketData] = None) -> Dict[str, Any]:
        """
        一键完整检查：结算 → 评分调仓 → 收盘结算

        Args:
            current_date: YYYY-MM-DD
            market_data: {stock_code: MarketData} 当日市场数据

        Returns:
            完整批次结果
        """
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")

        # 阶段0: 开始批次
        self.start_batch("ui")

        # 自动获取市场数据（如未提供）
        if not market_data:
            all_pool_codes = (
                self.pool_manager.get_pool("short")
                + self.pool_manager.get_pool("medium")
                + self.pool_manager.get_pool("long")
            )
            market_data = await self._fetch_market_data(list(set(all_pool_codes)), current_date)

        # 阶段1: 结算历史
        settle_result = await self.settle_historical(current_date)

        # 阶段2: 各期限调仓（按星期调度）
        schedule = self._get_schedule()
        logger.info("Schedule: short=%s, medium=%s, long=%s, is_monday=%s, date=%s",
                     schedule["short"], schedule["medium"], schedule["long"],
                     schedule["is_monday"], schedule["date"])
        rebalance_results = {}
        for term in ["short", "medium", "long"]:
            if schedule[term]:
                rebalance_results[term] = await self.run_daily_rebalance(
                    term, current_date, market_data
                )
            else:
                rebalance_results[term] = {
                    "status": "skipped",
                    "reason": f"{term} not scheduled today"
                }

        # 阶段3: 收盘结算
        current_prices = {
            code: md.close for code, md in market_data.items() if md.close > 0
        }
        self.run_daily_settlement(current_prices)

        # 阶段4: 生成报告并记录记忆
        try:
            from src.eval.report_builder import ReportBuilder
            from src.eval.memory_manager import MemoryManager
            rb = ReportBuilder()
            status = self.get_status()
            report_data = rb.build_batch_report_data(self.current_batch_id, status)
            report_path = rb.save_report(report_data)

            mm = MemoryManager()
            mm.record_batch(self.current_batch_id, report_data)
        except Exception:
            report_path = ""

        # 阶段5: 完成批次
        self.finish_batch(success=True)

        return {
            "batch_id": self.current_batch_id,
            "date": current_date,
            "settlement": settle_result,
            "rebalance": rebalance_results,
            "lines_summary": self.line_manager.get_all_status(),
            "report_path": report_path,
        }

    @staticmethod
    def _get_schedule() -> dict:
        """Returns which terms should rebalance today based on weekday."""
        import calendar
        today = datetime.now()
        weekday = today.weekday()  # 0=Monday, 4=Friday
        day = today.day
        last_day = calendar.monthrange(today.year, today.month)[1]
        is_month_end = (day >= last_day - 2)  # Last 3 trading days of month

        return {
            "short": True,           # Short-term: every trading day
            "medium": weekday == 4,  # Medium-term: Friday only (weekly)
            "long": is_month_end,    # Long-term: month-end only
            "is_monday": weekday == 0,  # Monday full run flag
            "weekday": weekday,
            "date": today.strftime("%Y-%m-%d"),
        }

    def get_status(self) -> Dict[str, Any]:
        """获取当前系统状态"""
        return {
            "current_batch_id": self.current_batch_id,
            "lines": self.line_manager.get_all_status(),
            "pools": self.pool_manager.get_all_summaries(),
            "latest_batch": get_latest_batch(),
        }
