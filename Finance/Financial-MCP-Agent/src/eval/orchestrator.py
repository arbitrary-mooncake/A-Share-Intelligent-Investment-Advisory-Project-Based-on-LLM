"""
编排器 — 评测系统的中央调度引擎。
连接分析→评分→策略→仿真→结算→Loss的完整主循环。
"""
import json
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
from src.eval.schemas import EvalBatch, PredictionSnapshot
from src.eval.line_manager import LineManager, ABLATION_AGENTS
from src.eval.pool_manager import PoolManager
from src.eval.market_simulator import MarketSimulator, MarketData, Order as SimOrder
from src.eval.strategies.factory import get_strategy
from src.eval.config import get_config

# ---- 最后结算日配置路径 ----
_EVAL_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "eval"
)
_LAST_SETTLED_PATH = os.path.join(_EVAL_CONFIG_DIR, "last_settled.json")

# 补回时可用的分析Agent（news/event不可靠补回）
_CATCHUP_BACKFILLABLE_AGENTS = [
    "fundamental", "technical", "value", "quality_risk", "moneyflow"
]
_CATCHUP_DISABLED_AGENTS = ["news", "event"]


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
        self._strategy_cache = {}  # key: (line_id, term, strategy_type) -> strategy instance

    # ═══════════════════════════════════════════════
    # 阶段入口
    # ═══════════════════════════════════════════════

    def start_batch(self, trigger_source: str = "ui",
                    cache_namespace: Optional[str] = None) -> str:
        """
        开始一个新的评测批次。

        Args:
            trigger_source: ui/cli/catch_up
            cache_namespace: None=生产缓存共享(精筛池筛选), "eval"=评测缓存隔离(日常模拟盘/回测)
        """
        # 设置缓存命名空间 — 控制agent中间产物写入位置
        from src.utils.cache_utils import set_cache_namespace
        set_cache_namespace(cache_namespace)

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
        """完成当前批次，恢复缓存命名空间为生产默认"""
        from src.utils.cache_utils import set_cache_namespace
        if not self.current_batch_id:
            return
        status = "completed" if success else "failed"
        update_batch_status(
            self.current_batch_id, status,
            finished_at=datetime.now().isoformat(),
            error_message=error,
            optimize_ready=int(success),
        )
        # 恢复默认命名空间（生产缓存），防止评测命名空间泄漏到其他功能
        set_cache_namespace(None)

    # ═══════════════════════════════════════════════
    # 最后结算日追踪
    # ═══════════════════════════════════════════════

    @staticmethod
    def _load_last_settled_date() -> Optional[str]:
        """从配置文件读取最后结算日期。返回None表示从未运行过。"""
        os.makedirs(_EVAL_CONFIG_DIR, exist_ok=True)
        if not os.path.exists(_LAST_SETTLED_PATH):
            return None
        try:
            with open(_LAST_SETTLED_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            date_val = data.get("last_settled_date", "")
            if date_val and len(date_val) == 10 and date_val[4] == "-":
                return date_val
        except (json.JSONDecodeError, IOError):
            pass
        return None

    @staticmethod
    def _save_last_settled_date(date_str: str):
        """将最后结算日期写入配置文件。"""
        os.makedirs(_EVAL_CONFIG_DIR, exist_ok=True)
        try:
            with open(_LAST_SETTLED_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "last_settled_date": date_str,
                    "updated_at": datetime.now().isoformat(),
                }, f, ensure_ascii=False, indent=2)
        except IOError:
            logger.warning("无法写入last_settled.json: %s", _LAST_SETTLED_PATH)

    # ═══════════════════════════════════════════════
    # 交易日缺失检测
    # ═══════════════════════════════════════════════

    def detect_missed_days(self, current_date: str = "") -> Dict[str, Any]:
        """
        检测最后一次结算日至今之间缺失的交易日。

        对比 last_settled_date 与当前最新交易日，通过 Tushare trade_cal
        获取期间所有交易日，返回排序后的缺失日期列表。

        Args:
            current_date: 当前日期 YYYY-MM-DD（默认今天）

        Returns:
            {
                "missed_days": ["2026-06-18", "2026-06-19"],  # 缺失的交易日（升序）
                "non_trading_skipped": 2,       # 跳过的非交易日数
                "last_settled_date": "2026-06-15",
                "latest_trading_day": "2026-06-20",
                "has_missed": True,              # 是否有缺失
                "is_first_run": False,           # 是否首次运行
            }
        """
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")

        from src.eval.data_fetcher import fetch_trade_calendar, fetch_latest_trading_day

        # 获取最新交易日
        latest_trading_day = fetch_latest_trading_day()

        # 读取上次结算日
        last_settled = self._load_last_settled_date()

        # 首次运行：不追溯历史，只从最新交易日开始
        if last_settled is None:
            logger.info("首次运行：设置最后结算日为最新交易日 %s，不进行历史补回", latest_trading_day)
            self._save_last_settled_date(latest_trading_day)
            return {
                "missed_days": [],
                "non_trading_skipped": 0,
                "last_settled_date": latest_trading_day,
                "latest_trading_day": latest_trading_day,
                "has_missed": False,
                "is_first_run": True,
                "reset_triggered": False,
            }

        # 如果上次结算日已经 >= 最新交易日，无需补回
        if last_settled >= latest_trading_day:
            return {
                "missed_days": [],
                "non_trading_skipped": 0,
                "last_settled_date": last_settled,
                "latest_trading_day": latest_trading_day,
                "has_missed": False,
                "is_first_run": False,
                "reset_triggered": False,
            }

        # 查询 last_settled 次日至 latest_trading_day 之间的交易日历
        start_dt = datetime.strptime(last_settled, "%Y-%m-%d") + timedelta(days=1)
        start_str = start_dt.strftime("%Y-%m-%d")

        all_trading_days = fetch_trade_calendar(start_str, latest_trading_day)

        # 总纲 §7.1: 最大追赶天数限制（可配置，默认7天）
        max_catchup_days = int(self.config.get("catchup_max_missed_days", 7))
        if len(all_trading_days) > max_catchup_days:
            logger.warning(
                "缺失%d个交易日超过上限%d天，重置为最新交易日 %s",
                len(all_trading_days), max_catchup_days, latest_trading_day
            )
            self._save_last_settled_date(latest_trading_day)
            # 重置所有线持仓状态
            for line in self.line_manager.lines.values():
                line.holdings.clear()
                line.purchase_prices.clear()
                line.hold_days.clear()
                line.cash = self.config.get("initial_capital_per_line", 1000000.0)
                line.total_value = line.cash
                line.daily_returns.clear()
                line.cumulative_return = 0.0
                line.max_drawdown = 0.0
                line.peak_value = line.cash
            return {
                "missed_days": [],
                "non_trading_skipped": 0,
                "last_settled_date": latest_trading_day,
                "latest_trading_day": latest_trading_day,
                "has_missed": False,
                "is_first_run": False,
                "reset_triggered": True,
                "reset_reason": f"缺失{len(all_trading_days)}天超过上限{max_catchup_days}天，已重置为最新交易日",
            }

        # 计算期间总日历天数
        end_dt = datetime.strptime(latest_trading_day, "%Y-%m-%d")
        total_calendar_days = (end_dt - start_dt).days + 1
        non_trading_skipped = max(0, total_calendar_days - len(all_trading_days))

        # 过滤掉可能早于 start_str 的结果
        missed_days = [d for d in all_trading_days if d >= start_str and d <= latest_trading_day]

        return {
            "missed_days": missed_days,
            "non_trading_skipped": non_trading_skipped,
            "last_settled_date": last_settled,
            "latest_trading_day": latest_trading_day,
            "has_missed": len(missed_days) > 0,
            "is_first_run": False,
            "reset_triggered": False,
        }

    # ═══════════════════════════════════════════════
    # 交易日补回调度
    # ═══════════════════════════════════════════════

    @staticmethod
    def _get_schedule_for_date(date_str: str) -> dict:
        """
        根据指定日期返回该日的调仓排程。

        Args:
            date_str: YYYY-MM-DD 格式日期

        Returns:
            {"short": bool, "medium": bool, "long": bool, "is_monday": bool,
             "weekday": int, "date": str}
        """
        import calendar
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            dt = datetime.now()
        weekday = dt.weekday()  # 0=Monday
        day = dt.day
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        is_month_end = (day >= last_day - 2)  # 月末最后3个交易日

        # 简化处理：假设查询到的都是交易日（已由trade_cal保证）
        return {
            "short": True,    # 交易日短线每天都做
            "medium": weekday == 4,   # 周五做中线
            "long": is_month_end,     # 月末做长线
            "is_monday": weekday == 0,
            "weekday": weekday,
            "date": date_str,
        }

    @staticmethod
    def _build_partial_signal_pack(agent_name: str, stock_code: str,
                                    as_of_date: str) -> dict:
        """
        为补回时禁用的Agent构建pit_mode='partial'的signal pack。

        news和event的历史数据不可靠补回，补回期间标记为partial，
        与回测逻辑一致。
        """
        return {
            "agent_name": agent_name,
            "stock_code": stock_code,
            "as_of_date": as_of_date,
            "pit_mode": "partial",
            "bias": "neutral",
            "confidence": 0.0,
            "data_quality_score": 0.0,
            "signals": [],
            "risk_flags": [],
            "missing_data": [
                f"{agent_name} 历史数据不可靠补回，补回期间禁用"
            ],
            "notes": "Catch-up: agent disabled (same as backtest logic)"
        }

    async def catch_up_missed_days(self, missed_days: List[str]) -> Dict[str, Any]:
        """
        按时间顺序补回所有错过的交易日。

        每个补回日执行：
          1. 获取历史市场数据
          2. 为5个可补回Agent生成分析/信号包（news/event标记partial）
          3. 按该日排程运行各期限调仓
          4. 收盘结算（更新持仓市值与收益）
          5. 保存快照
          6. 更新最后结算日

        补回是顺序的（每天持仓依赖前一天），不是并行的。

        Args:
            missed_days: detect_missed_days返回的缺失交易日列表（升序）

        Returns:
            {
                "caught_up": 3,           # 成功补回天数
                "failed": 0,              # 失败天数
                "skipped": 1,             # 跳过天数（非交易日/无数据）
                "per_day_status": [
                    {"date": "2026-06-18", "status": "success", ...},
                    ...
                ]
            }
        """
        results = {
            "caught_up": 0,
            "failed": 0,
            "skipped": 0,
            "per_day_status": [],
        }

        if not missed_days:
            logger.info("无需补回：缺失交易日列表为空")
            return results

        logger.info("开始补回 %d 个交易日: %s", len(missed_days), missed_days)

        for date_str in missed_days:
            day_result = {
                "date": date_str,
                "status": "pending",
                "terms_rebalanced": [],
                "holdings_count": 0,
                "error": "",
            }
            try:
                # 阶段0: 为该补回日创建独立批次（评测缓存隔离）
                from src.utils.cache_utils import set_cache_namespace
                set_cache_namespace("eval")
                catchup_batch = EvalBatch(
                    batch_id=f"eval_catchup_{generate_id()}",
                    status="running",
                    trigger_source="catch_up",
                    started_at=datetime.now().isoformat(),
                    market_session="post_close",
                )
                prev_batch_id = self.current_batch_id
                self.current_batch_id = create_batch(catchup_batch)

                # 阶段1: 获取当日市场数据
                all_pool_codes = (
                    self.pool_manager.get_pool("short")
                    + self.pool_manager.get_pool("medium")
                    + self.pool_manager.get_pool("long")
                )
                market_data = await self._fetch_market_data(
                    list(set(all_pool_codes)), date_str
                )

                if not market_data:
                    day_result["status"] = "skipped"
                    day_result["error"] = "Tushare未返回市场数据"
                    results["skipped"] += 1
                    results["per_day_status"].append(day_result)
                    self.current_batch_id = prev_batch_id
                    continue

                # 阶段2: 按该日排程运行调仓
                schedule = self._get_schedule_for_date(date_str)
                rebalance_results = {}
                for term in ["short", "medium", "long"]:
                    if schedule[term]:
                        pool = self.pool_manager.get_pool(term)
                        if pool:
                            rb_result = await self.run_daily_rebalance(
                                term, date_str, market_data
                            )
                            rebalance_results[term] = rb_result
                            if rb_result.get("status") != "skipped":
                                day_result["terms_rebalanced"].append(term)
                        else:
                            rebalance_results[term] = {
                                "status": "skipped",
                                "reason": "精筛池为空"
                            }
                    else:
                        rebalance_results[term] = {
                            "status": "skipped",
                            "reason": f"{term} not scheduled on {date_str}"
                        }

                # 阶段3: 收盘结算（用当日收盘价更新所有线持仓市值）
                closing_prices = {
                    code: md.close for code, md in market_data.items()
                    if md.close > 0
                }
                self.run_daily_settlement(closing_prices)

                # 阶段4: 保存快照
                self._save_catchup_snapshots(date_str, rebalance_results)

                # 阶段5: 更新最后结算日
                self._save_last_settled_date(date_str)

                # 完成补回批次
                self.finish_batch(success=True)

                # 恢复原始批次ID
                self.current_batch_id = prev_batch_id

                day_result["status"] = "success"
                day_result["holdings_count"] = sum(
                    len(line.holdings)
                    for line in self.line_manager.lines.values()
                )
                results["caught_up"] += 1

            except Exception as e:
                logger.error("补回日 %s 失败: %s", date_str, str(e), exc_info=True)
                day_result["status"] = "failed"
                day_result["error"] = str(e)[:200]
                results["failed"] += 1
                try:
                    self.finish_batch(success=False, error=str(e)[:200])
                except Exception:
                    pass

            results["per_day_status"].append(day_result)

        logger.info(
            "补回完成: 成功=%d 失败=%d 跳过=%d",
            results["caught_up"], results["failed"], results["skipped"]
        )
        return results

    def _save_catchup_snapshots(self, date_str: str,
                                 rebalance_results: Dict[str, Any]):
        """
        为补回日保存各线持仓快照。

        遍历所有成功调仓的线，为每条线的每个持仓创建PredictionSnapshot。
        news/event的signal_pack标记为pit_mode='partial'。
        """
        for term, term_result in rebalance_results.items():
            if term_result.get("status") == "skipped":
                continue
            for line_id, line_result in term_result.get("lines", {}).items():
                if line_result.get("status") != "success":
                    continue
                line = self.line_manager.get_line(line_id)
                if not line:
                    continue
                for stock_code, shares in line.holdings.items():
                    if shares <= 0:
                        continue
                    # 构建signal_pack_bundle：5个可补回+2个partial
                    signal_bundle = {}
                    for agent in _CATCHUP_BACKFILLABLE_AGENTS:
                        signal_bundle[agent] = {
                            "agent_name": agent,
                            "stock_code": stock_code,
                            "as_of_date": date_str,
                            "pit_mode": "backfill",
                            "bias": "neutral",
                            "confidence": 0.5,
                            "data_quality_score": 0.6,
                            "signals": [],
                            "risk_flags": [],
                            "notes": "Catch-up backfill via Tushare historical data"
                        }
                    for agent in _CATCHUP_DISABLED_AGENTS:
                        signal_bundle[agent] = self._build_partial_signal_pack(
                            agent, stock_code, date_str
                        )

                    snap = PredictionSnapshot(
                        batch_id=self.current_batch_id,
                        line_id=line_id,
                        symbol=stock_code,
                        name="",
                        term=term,
                        as_of_date=date_str,
                        pit_mode="partial",  # 补回模式（news/event缺失）
                        action="hold",
                        score=50.0,
                        signal_pack_bundle_json=json.dumps(
                            signal_bundle, ensure_ascii=False, default=str
                        ),
                    )
                    try:
                        create_snapshot(snap)
                    except Exception:
                        logger.warning(
                            "保存补回快照失败: %s %s %s",
                            date_str, line_id, stock_code, exc_info=True
                        )

    def _get_or_create_strategy(self, line_id: str, term: str, strategy_type: str):
        """Get cached strategy instance or create new one. Preserves state across days."""
        cache_key = (line_id, term, strategy_type)
        if cache_key not in self._strategy_cache:
            self._strategy_cache[cache_key] = get_strategy(term, strategy_type, self.config)
        return self._strategy_cache[cache_key]

    def _cleanup_strategies(self, active_line_ids: set):
        """Remove strategy cache entries for lines that no longer exist."""
        stale = [k for k in self._strategy_cache if k[0] not in active_line_ids]
        for k in stale:
            del self._strategy_cache[k]

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

        # ── 分级打分频率（总纲 §14.4）──
        is_monday = EvalOrchestrator._get_schedule_for_date(current_date)["is_monday"]
        schedule_info = EvalOrchestrator._get_schedule_for_date(current_date)

        # 收集所有线的持仓用于分级判断
        all_holdings = set()
        for line in lines:
            all_holdings.update(code for code, shares in line.holdings.items() if shares > 0)

        # 获取池内评分用于分级（优先缓存，缺失用默认分）
        pool_scores_for_tier = {}
        pool_data = self.pool_manager.get_pool_with_scores(term)
        for s in pool_data:
            code = s.get("code", "") if isinstance(s, dict) else s
            score = s.get("score", 50) if isinstance(s, dict) else 50
            pool_scores_for_tier[code] = score

        # 计算每个股票的分析频率等级
        tiers = EvalOrchestrator._get_scoring_frequency_tier(
            pool_scores_for_tier, all_holdings, term
        )
        # 基于当天日期确定哪些等级需要分析
        day_of_year = datetime.strptime(current_date, "%Y-%m-%d").timetuple().tm_yday
        stocks_to_analyze = [
            code for code in pool
            if EvalOrchestrator._should_analyze_today(
                tiers.get(code, "daily"), day_of_year, is_monday
            )
        ]
        # 持仓股始终分析（不受分级限制）
        for code in all_holdings:
            if code in pool and code not in stocks_to_analyze:
                stocks_to_analyze.append(code)

        if len(stocks_to_analyze) < len(pool):
            logger.info("分级频率: %s池 %d/%d只需要分析%d只（持仓%d只 + 分级%d只）",
                       term, len(stocks_to_analyze), len(pool),
                       len(stocks_to_analyze), len(all_holdings),
                       len(stocks_to_analyze) - len(all_holdings))
        elif is_monday:
            logger.info("Monday full run: pool-wide analysis for %s (%d stocks)",
                        term, len(pool))

        for line in lines:
            # 每条线使用独立的MarketSimulator实例（T+1隔离）
            line_sim = MarketSimulator(self.config)
            line_sim.reset_daily_state()

            try:
                # 获取策略（缓存实例以保留跨日状态如_high_water_marks、_low_score_days）
                strategy_type = line.definition.get("strategy", "default")
                strategy = self._get_or_create_strategy(line.line_id, term, strategy_type)

                # 真实Agent评分（缓存驱动，跨线共享）
                # 分级频率：仅当天需要分析的股票走Agent管线，其余用池内缓存评分
                line_agents = line.definition.get("agents", "all")
                scores = await self._get_real_scores(
                    stocks_to_analyze, term, current_date, line_agents
                )
                # 补充非分析日股票的池内缓存评分
                for code in pool:
                    if code not in scores:
                        scores[code] = pool_scores_for_tier.get(code, 50.0)

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
                # SimOrder imported at top of file

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

    async def _get_real_scores(self, pool: List[str], term: str,
                               as_of_date: str,
                               line_agents: str = "all") -> Dict[str, float]:
        """
        通过真实Agent分析获取评分 — 缓存驱动，跨线跨期限共享。

        总纲 §14.1: 分析Agent只跑一次。每个agent独立缓存，利用各自TTL：
          - fundamental_analysis: 15天（财报数据低频变化）
          - value_analysis: 7天（估值框架稳定）
          - quality_risk_analysis: 7天（质量指标低频）
          - technical/moneyflow/news/event: 1天（日频市场数据）

        缓存隔离：评测专用 _eval 后缀，与生产缓存（data/intermediate_cache/）完全隔离。

        Args:
            pool: 股票代码列表
            term: short/medium/long
            as_of_date: 分析时点 YYYY-MM-DD
            line_agents: "all"（全agent）或 "-fundamental" 等消融标记

        Returns:
            {stock_code: score (0-100)}
        """
        from src.eval.cache import (
            read_cache, write_cache, read_signal_pack_cache, write_signal_pack_cache
        )
        from src.eval.adapters.stock_pipeline_adapter import run_stock_analysis
        import json

        scores = {}
        ablated_agent = None
        if line_agents != "all" and line_agents.startswith("-"):
            ablated_agent = line_agents.lstrip("-")

        # 所有7个agent及其TTL（天）
        _ALL_AGENTS = [
            ("fundamental", 15),
            ("value", 7),
            ("quality_risk", 7),
            ("technical", 1),
            ("news", 1),
            ("event", 1),
            ("moneyflow", 1),
        ]

        for code in pool:
            try:
                # ── Step 1: 检查每个agent的独立缓存 ──
                all_fresh = True
                cached_signal_packs = {}
                for agent_name, ttl_days in _ALL_AGENTS:
                    sp = read_signal_pack_cache(agent_name, code, as_of_date)
                    if sp:
                        cached_signal_packs[agent_name] = sp
                    else:
                        all_fresh = False

                # ── Step 2: 缓存全命中 → 跳过管线 ──
                if all_fresh and cached_signal_packs:
                    logger.debug("All %d agent caches hit for %s @ %s, skipping pipeline",
                                len(cached_signal_packs), code, as_of_date)
                    # 从独立缓存组装结果
                    result = self._assemble_from_agent_caches(
                        code, as_of_date, cached_signal_packs, term
                    )
                else:
                    # ── Step 3: 有缓存未命中 → 运行完整管线 ──
                    stale_agents = [
                        a for a, _ in _ALL_AGENTS
                        if a not in cached_signal_packs
                    ]
                    logger.info("Agent cache miss for %s @ %s (%d/%d fresh, stale: %s), "
                                "running pipeline...",
                                code, as_of_date, len(cached_signal_packs),
                                len(_ALL_AGENTS), ','.join(stale_agents))

                    result = await run_stock_analysis(
                        code, "", as_of_date, eval_mode=True
                    )

                    if not result.get("error"):
                        # ── Step 4: 分解管线结果，写每个agent的独立缓存 ──
                        signal_packs = result.get("signal_packs", {})
                        analysis_texts = result.get("analysis_texts", {})
                        for agent_name, _ in _ALL_AGENTS:
                            sp = signal_packs.get(agent_name)
                            if sp and isinstance(sp, dict):
                                write_signal_pack_cache(
                                    agent_name, code, as_of_date, sp
                                )
                            # 同时写文本分析缓存
                            txt = analysis_texts.get(agent_name, "")
                            if txt:
                                write_cache(agent_name, code, as_of_date, txt)

                        # 写完整结果缓存（1天TTL，作为快速回退）
                        write_cache("full_analysis", code, as_of_date,
                                   json.dumps(result, ensure_ascii=False, default=str))

                # ── Step 5: 提取评分 ──
                term_key = f"{term}_term_score"
                term_score = result.get(term_key, {})
                if isinstance(term_score, dict):
                    base_score = float(term_score.get("score", 50))
                else:
                    base_score = 50.0

                # ── Step 6: 消融线调整 ──
                if ablated_agent and ablated_agent in ABLATION_AGENTS:
                    signal_packs = result.get("signal_packs", cached_signal_packs)
                    adjusted = self._adjust_score_for_ablation(
                        base_score, signal_packs, ablated_agent, term
                    )
                    scores[code] = round(adjusted, 2)
                else:
                    scores[code] = round(base_score, 2)

            except Exception as e:
                logger.warning("Failed to get real score for %s: %s", code, str(e))
                if "Tushare" in str(e) or "token" in str(e).lower():
                    raise RuntimeError(
                        f"Tushare不可用且缓存中无 {code}@{as_of_date} 的数据，"
                        f"无法继续。请检查Tushare配置或网络连接。错误: {e}"
                    ) from e
                scores[code] = 50.0

        return scores

    @staticmethod
    def _assemble_from_agent_caches(stock_code: str, as_of_date: str,
                                     signal_packs: dict, term: str) -> dict:
        """
        从独立agent缓存组装结果。

        当所有7个agent的signal_pack都在各自TTL内时调用。
        注意：scorer结果不缓存（依赖完整signal_pack组合），
        因此返回占位评分，调用方应运行管线获取真实评分。
        此方法主要用于：
          1. 提供缓存的signal_packs给管线复用
          2. 统计缓存命中率
          3. 未来支持跳过管线（当scorer缓存也命中时）
        """
        return {
            "stock_code": stock_code,
            "as_of_date": as_of_date,
            "signal_packs": signal_packs,
            "analysis_texts": {},
            f"{term}_term_score": {"score": 50.0, "_cache_only": True},
            "_from_agent_cache": True,
        }

    @staticmethod
    def _get_scoring_frequency_tier(pool_scores: Dict[str, float],
                                     holdings: set,
                                     term: str) -> Dict[str, str]:
        """
        分级打分频率 — 总纲 §14.4。

        根据股票评分和持仓状态，将池内股票分为5个频率等级，
        减少对稳定股票的冗余分析。

        频率等级:
          - "daily":    持仓股 + 高波动候选（每日必跑）
          - "3day":     稳定高分区 score>75且近10日波动<5%
          - "5day":     中位区 score 45-65
          - "7day":     稳定低分区 score<45且无明显改善
          - "weekly":   周一全量（覆盖所有未在其他等级中的股票）

        Returns:
            {stock_code: frequency_tier}
        """
        tiers = {}
        for code, score in pool_scores.items():
            if code in holdings:
                tiers[code] = "daily"
            elif score >= 75:
                tiers[code] = "3day"
            elif 45 <= score < 65:
                tiers[code] = "5day"
            elif score < 45:
                tiers[code] = "7day"
            else:
                tiers[code] = "3day"  # score 65-75
        return tiers

    @staticmethod
    def _should_analyze_today(frequency_tier: str, day_index: int,
                               is_monday: bool = False) -> bool:
        """
        判断某频率等级的股票今天是否需要分析。

        Args:
            frequency_tier: "daily"/"3day"/"5day"/"7day"/"weekly"
            day_index: 从首次运行开始计的交易日序号（0-based）
            is_monday: 是否为周一

        Returns:
            True 如果今天应该分析
        """
        if frequency_tier == "daily":
            return True
        elif frequency_tier == "3day":
            return day_index % 3 == 0
        elif frequency_tier == "5day":
            return day_index % 5 == 0
        elif frequency_tier == "7day":
            return day_index % 7 == 0
        elif frequency_tier == "weekly":
            return is_monday
        return True  # 未知等级默认每天跑

    @staticmethod
    def _adjust_score_for_ablation(base_score: float,
                                    signal_packs: dict,
                                    ablated_agent: str,
                                    term: str) -> float:
        """
        消融评分调整：从全agent评分中移除指定agent的贡献。

        总纲 §3.2: 消融线每天从相同持仓出发，独立评分，收益差异归因到Agent配置差异。

        方法：计算被移除agent的"投票贡献"（bias × confidence × term_weight），
        从基础分中扣除。正bias的agent被移除 → 分数下降；负bias的agent被移除 → 分数上升。

        Args:
            base_score: 全7 agent的评分 (0-100)
            signal_packs: 所有agent的signal_pack dict
            ablated_agent: 被移除的agent名称
            term: short/medium/long

        Returns:
            调整后的评分 (0-100)
        """
        sp = signal_packs.get(ablated_agent)
        if not sp or not isinstance(sp, dict):
            return base_score  # 无该agent数据，不调整

        # 各agent在不同期限scorer中的权重（总纲 §3.2）
        _AGENT_TERM_WEIGHTS = {
            "short": {"fundamental": 0.12, "technical": 0.25, "value": 0.10,
                      "news": 0.15, "event": 0.15, "quality_risk": 0.10, "moneyflow": 0.13},
            "medium": {"fundamental": 0.20, "technical": 0.10, "value": 0.15,
                       "news": 0.10, "event": 0.10, "quality_risk": 0.20, "moneyflow": 0.15},
            "long": {"fundamental": 0.22, "technical": 0.05, "value": 0.15,
                     "news": 0.08, "event": 0.08, "quality_risk": 0.22, "moneyflow": 0.20},
        }

        weights = _AGENT_TERM_WEIGHTS.get(term, _AGENT_TERM_WEIGHTS["medium"])
        agent_weight = weights.get(ablated_agent, 0.10)

        # 解析bias: bullish→+1, bearish→-1, neutral→0
        bias_str = str(sp.get("bias", "neutral")).lower()
        bias_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        bias = bias_map.get(bias_str, 0.0)

        # 解析confidence: 0-1
        try:
            confidence = float(sp.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # 该agent对评分的贡献量（正=推高分数，负=拉低分数）
        contribution = bias * confidence * agent_weight * 100.0

        # 移除该agent后，剩余agent的权重和
        remaining_weight = 1.0 - agent_weight

        if remaining_weight <= 0:
            return base_score

        # 调整：从基础分中移除该agent的贡献，再按剩余权重重新缩放
        adjusted = (base_score - contribution) / remaining_weight

        return max(0.0, min(100.0, adjusted))

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
        一键完整检查：检测缺失日 → 补回 → 结算 → 评分调仓 → 收盘结算

        Args:
            current_date: YYYY-MM-DD
            market_data: {stock_code: MarketData} 当日市场数据

        Returns:
            完整批次结果（含补回摘要）
        """
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")

        # 阶段-1: 检测缺失交易日
        missed_info = self.detect_missed_days(current_date)
        catchup_summary = None

        # 阶段-1b: 如有缺失，执行补回
        if missed_info["has_missed"]:
            # 排除 current_date（当日由正常流程处理，避免重复结算）
            catchup_days = [d for d in missed_info["missed_days"]
                            if d < current_date]
            if catchup_days:
                logger.info(
                    "检测到 %d 个缺失交易日，跳过了 %d 个非交易日，开始补回...",
                    len(catchup_days), missed_info["non_trading_skipped"]
                )
                catchup_summary = await self.catch_up_missed_days(catchup_days)
                logger.info(
                    "补回完成: 成功=%d 失败=%d 跳过=%d",
                    catchup_summary["caught_up"],
                    catchup_summary["failed"],
                    catchup_summary["skipped"],
                )
            else:
                logger.info("缺失日为当日(%s)，交由正常流程处理", current_date)

        # 阶段0: 开始批次（评测缓存隔离模式 — M5产物不污染生产缓存）
        self.start_batch("ui", cache_namespace="eval")

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

        # 阶段2: 各期限调仓（按 current_date 的星期调度，而非today）
        schedule = self._get_schedule_for_date(current_date)
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

        # 更新最后结算日（当日结算完成即记录）
        self._save_last_settled_date(current_date)

        # 阶段4: 生成报告并记录记忆
        try:
            from src.eval.report_builder import ReportBuilder
            from src.eval.memory_manager import MemoryManager
            rb = ReportBuilder()
            status = self.get_status()
            report_data = rb.build_batch_report_data(self.current_batch_id, status)
            report_path = rb.save_report(report_data)

            # LLM增强报告（DeepSeek V4 Pro + 反幻觉验证）
            try:
                from src.eval.report_writer_agent import ReportWriterAgent
                rw = ReportWriterAgent()
                enhanced = rw.write_batch_report(report_data)
                if enhanced.passed_verification and enhanced.confidence != "LOW":
                    report_path = rb.save_report(report_data, suffix="_llm")
            except Exception:
                pass  # LLM报告失败不阻断主流程

            mm = MemoryManager()
            mm.record_batch(self.current_batch_id, report_data)

            # 记录保真度趋势（从report_data提取或从fidelity_engine计算）
            try:
                from src.eval.fidelity_engine import FidelityEngine
                fe = FidelityEngine()
                snapshots = report_data.get("snapshots", [])
                if snapshots:
                    fidelity_result = fe.compute_fidelity_loss(snapshots)
                    mm.record_fidelity(self.current_batch_id, fidelity_result)
            except Exception:
                pass

            # 记录运行耗时
            try:
                from src.eval.cache import get_cache_stats
                cache_stats = get_cache_stats()
                mm.record_runtime(self.current_batch_id, {
                    "total_duration_seconds": report_data.get("execution_time", 0),
                    "agent_calls": report_data.get("agent_calls", 0),
                    "cache_hits": cache_stats.get("l2_files", 0),
                    "cache_misses": report_data.get("cache_misses", 0),
                    "estimated_tokens": report_data.get("estimated_tokens", 0),
                    "line_count": len(self.line_manager.lines),
                })
            except Exception:
                pass
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
            # 补回摘要（如有）
            "catchup": catchup_summary,
            # 重置信息（缺失超过7天触发）
            "reset_triggered": missed_info.get("reset_triggered", False),
            "reset_reason": missed_info.get("reset_reason", ""),
        }

    @staticmethod
    def _get_schedule() -> dict:
        """Returns which terms should rebalance today based on weekday."""
        return EvalOrchestrator._get_schedule_for_date(
            datetime.now().strftime("%Y-%m-%d")
        )

    def get_status(self) -> Dict[str, Any]:
        """获取当前系统状态（含聚合评分指标）"""
        last_settled = self._load_last_settled_date()
        lines = self.line_manager.get_all_status()

        # 计算各期限/总Score（从线路收益反推，100*(1-Loss近似)）
        scores_by_term = {"short": [], "medium": [], "long": []}
        for l in lines:
            term = l.get("term", "")
            if term in scores_by_term:
                ret = l.get("cumulative_return_pct", 0) or 0
                dd = l.get("max_drawdown_pct", 100) or 0
                # 简化Score: 收益越好分越高，回撤越大扣分越多
                raw = 50 + ret - dd * 0.5
                scores_by_term[term].append(max(0, min(100, raw)))

        def _avg(lst):
            return round(sum(lst) / len(lst), 1) if lst else 0.0

        short_score = _avg(scores_by_term["short"])
        medium_score = _avg(scores_by_term["medium"])
        long_score = _avg(scores_by_term["long"])
        all_scores = scores_by_term["short"] + scores_by_term["medium"] + scores_by_term["long"]
        total_score = _avg(all_scores)

        return {
            "current_batch_id": self.current_batch_id,
            "lines": lines,
            "pools": self.pool_manager.get_all_summaries(),
            "latest_batch": get_latest_batch(),
            "last_settled_date": last_settled,
            "total_score": total_score if all_scores else "N/A",
            "short_score": short_score if scores_by_term["short"] else "N/A",
            "medium_score": medium_score if scores_by_term["medium"] else "N/A",
            "long_score": long_score if scores_by_term["long"] else "N/A",
        }
