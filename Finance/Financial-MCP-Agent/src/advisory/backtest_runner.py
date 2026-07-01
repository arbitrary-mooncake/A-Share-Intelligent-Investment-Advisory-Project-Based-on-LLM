"""日线历史回测执行器 — 模型3调仓时序（收盘信号->次日开盘执行）。

使用 Tushare 拉取日线数据，通过 StrategyEngine 计算策略信号，
在 PortfolioManager 上执行交易，回放历史行情并生成回测报告。

Usage:
    from src.advisory.backtest_runner import BacktestRunner
    from src.advisory.portfolio_manager import PortfolioManager

    pm = PortfolioManager()
    pf = pm.create("测试组合", initial_capital=100000.0)
    pf, _ = pm.add_holding(pf, "600519.SH", "贵州茅台", 100, 1680.0)

    runner = BacktestRunner(portfolio_manager=pm)
    result = runner.run(
        pf=pf,
        start_date="20240101",
        end_date="20241231",
        strategy_name="ma_cross",
    )
    print(f"总收益率: {result.total_return_pct:.2f}%")
    print(f"最大回撤: {result.max_drawdown_pct:.2f}%")
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.advisory.portfolio_manager import PortfolioManager
from src.advisory.schemas import AdvisoryPortfolio, BacktestResult, TradeRecord
from src.advisory.strategy_engine import StrategyEngine
from src.utils.tushare_client import _call, _items_to_dicts

# Tushare daily API request fields
_DAILY_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,"
    "change,pct_chg,vol,amount"
)


class BacktestRunner:
    """日线历史回测执行器，模型3 调仓时序（收盘信号->次日开盘执行）。

    核心流程:
        1. 拉取历史日线（含 400 天预热数据）。
        2. 逐日回放：收盘数据 -> StrategyEngine 计算信号。
        3. 次日开盘价执行交易（模型3 时序）。
        4. 计算总收益率、最大回撤、权益曲线。
        5. 返回 BacktestResult dataclass。

    Attributes:
        portfolio_manager: PortfolioManager 实例，用于执行持仓变更。
    """

    def __init__(
        self,
        portfolio_manager: Optional[PortfolioManager] = None,
    ) -> None:
        """初始化 BacktestRunner。

        Args:
            portfolio_manager: PortfolioManager 实例。为 None 时自动创建。
        """
        self.portfolio_manager = portfolio_manager or PortfolioManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        pf: AdvisoryPortfolio,
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """执行历史回测。

        模型3 调仓时序:
            T 日     收盘后依据收盘价计算策略信号。
            T+1 日   开盘价执行交易（买入/卖出）。

        Args:
            pf: 初始投资组合（会被原地修改，每个交易日自动保存）。
            start_date: 回测开始日期，YYYYMMDD 格式。
            end_date: 回测结束日期，YYYYMMDD 格式。
            strategy_name: 策略注册名称，如 ``"ma_cross"``。
                           None 时尝试使用 ``pf.bound_strategy``。
            strategy_params: 策略参数字典。None 时使用策略默认参数。

        Returns:
            BacktestResult 包含完整回测结果。

        Raises:
            ValueError: 持仓为空、策略未指定、回测区间无交易日。
        """
        # ---- 1. 确定策略 ----
        if strategy_name is None:
            strategy_name = pf.bound_strategy
        if not strategy_name:
            raise ValueError(
                "回测必须指定 strategy_name，或 portfolio 已绑定策略"
            )
        strategy_params = dict(strategy_params) if strategy_params else {}
        if pf.strategy_params:
            # 合并 portfolio 级默认参数（不覆盖显式参数）
            for k, v in pf.strategy_params.items():
                strategy_params.setdefault(k, v)

        # ---- 2. 收集持仓股票代码 ----
        stock_codes = list(pf.holdings.keys())
        if not stock_codes:
            raise ValueError("投资组合中无持仓，无法执行回测")

        # 保存公司名称索引（持有被清仓后仍可找回名称）
        name_map: Dict[str, str] = {
            code: h.company_name for code, h in pf.holdings.items()
        }

        # ---- 3. 拉取日线数据（含 400 天预热） ----
        warmup_start = _shift_date(start_date, -400)
        daily_data = self._fetch_daily_data(stock_codes, warmup_start, end_date)
        if daily_data.empty:
            raise ValueError("未能拉取到任何日线数据，请检查 Tushare 连接")

        # ---- 4. 获取回测区间内的交易日 ----
        all_dates = sorted(daily_data["trade_date"].unique())
        target_dates = [
            d for d in all_dates if start_date <= d <= end_date
        ]
        if not target_dates:
            raise ValueError(
                f"回测区间 {start_date}~{end_date} 内无交易日"
            )

        # ---- 5. 初始权益 ----
        initial_value = pf.cash + sum(
            h.quantity * h.current_price for h in pf.holdings.values()
        )

        # ---- 6. 逐日回放 ----
        equity_curve: List[float] = [initial_value]
        all_trades: List[TradeRecord] = []

        for i, date in enumerate(target_dates):
            # 最后一个交易日无法执行次日开盘交易
            if i >= len(target_dates) - 1:
                break
            next_date = target_dates[i + 1]

            # 逐股计算信号并交易
            for code in stock_codes:
                stock_df = _filter_stock_df(daily_data, code)
                if stock_df.empty:
                    continue

                # 截取到当前日期为止的数据用于信号计算
                data_up_to_date = stock_df[
                    stock_df["trade_date"] <= date
                ].copy()
                if len(data_up_to_date) < 2:
                    continue

                today_row = data_up_to_date.loc[
                    data_up_to_date["trade_date"] == date
                ]
                next_row = stock_df.loc[
                    stock_df["trade_date"] == next_date
                ]
                if today_row.empty or next_row.empty:
                    continue

                # 当前持仓信息 -> 策略上下文
                holding = pf.holdings.get(code)
                position = holding.quantity if holding else 0
                entry_price = holding.cost_price if holding else None

                # 计算 T 日收盘信号（模型3 时序）
                try:
                    sig, reason = StrategyEngine.compute_signal(
                        strategy_name,
                        data_up_to_date,
                        strategy_params,
                        context={
                            "position": position,
                            "entry_price": entry_price,
                            "cash": pf.cash,
                        },
                    )
                except Exception:
                    # 信号计算失败时跳过该股
                    continue

                # T+1 日开盘价执行
                exec_price = float(next_row.iloc[0]["open"])

                # T+1 日交易日期（TradeRecord 用）
                trade_date_str = _fmt_ymd(next_date)

                if sig == 1 and position == 0:
                    # 买入信号 & 无持仓 -> 开仓
                    company_name = name_map.get(code, code)
                    try:
                        pf, trade = self.portfolio_manager.add_holding(
                            pf, code, company_name, 100, exec_price,
                            trade_date=trade_date_str,
                        )
                        all_trades.append(trade)
                    except (ValueError, Exception):
                        # 资金不足等情况跳过
                        continue

                elif sig == -1 and position > 0:
                    # 卖出信号 & 有持仓 -> 清仓
                    company_name = name_map.get(code, code)
                    try:
                        pf, trade = self.portfolio_manager.remove_holding(
                            pf, code, position, exec_price,
                            trade_date=trade_date_str,
                        )
                        all_trades.append(trade)
                    except (ValueError, Exception):
                        continue

            # Mark-to-market: 将所有持仓按当日收盘价重估
            # (修复：无交易时 current_price 不更新的 bug)
            for code, h in list(pf.holdings.items()):
                close_rows = daily_data[
                    (daily_data["ts_code"] == code)
                    & (daily_data["trade_date"] == date)
                ]
                if not close_rows.empty:
                    h.current_price = float(close_rows.iloc[0]["close"])
            self.portfolio_manager._recalc_holdings(pf)

            # 记录当日权益
            daily_value = pf.cash + sum(
                h.quantity * h.current_price
                for h in pf.holdings.values()
            )
            equity_curve.append(daily_value)

        # ---- 7. 计算结果指标 ----
        final_value = equity_curve[-1]
        trading_days = len(target_dates)

        total_return_pct = (
            (final_value / initial_value - 1) * 100
            if initial_value > 0
            else 0.0
        )

        max_dd = self._calc_max_drawdown(equity_curve)

        annualized_return_pct = _calc_annualized_return(
            initial_value, final_value, trading_days
        )

        return BacktestResult(
            portfolio_id=pf.portfolio_id,
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=round(initial_value, 2),
            final_value=round(final_value, 2),
            total_return_pct=round(total_return_pct, 4),
            annualized_return_pct=round(annualized_return_pct, 4),
            max_drawdown_pct=round(max_dd, 4),
            sharpe_ratio=0.0,
            win_rate=0.0,
            trade_count=len(all_trades),
            equity_curve=equity_curve,
            trades=all_trades,
            settlement_records=[],
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_daily_data(
        stock_codes: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """从 Tushare 拉取多只股票的日线 OHLCV 数据。

        Args:
            stock_codes: ts_code 列表，如 ``["600519.SH", "000858.SZ"]``。
            start: 起始日期 YYYYMMDD。
            end: 结束日期 YYYYMMDD。

        Returns:
            包含 ``ts_code, trade_date, open, high, low, close, vol``
            列的 DataFrame，按 ts_code + trade_date 升序排列。
            无数据时返回空 DataFrame。
        """
        dfs: List[pd.DataFrame] = []

        for code in stock_codes:
            result = _call(
                "daily",
                {"ts_code": code, "start_date": start, "end_date": end},
                _DAILY_FIELDS,
            )
            items = _items_to_dicts(result)
            if not items:
                continue

            df_code = pd.DataFrame(items)
            # 确保数值类型
            for col in ("open", "high", "low", "close", "vol", "amount"):
                df_code[col] = pd.to_numeric(df_code[col], errors="coerce")
            dfs.append(df_code)

        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_max_drawdown(equity_curve: Sequence[float]) -> float:
        """从权益曲线计算最大回撤百分比（返回正数）。

        最大回撤 = max( (峰值 - 谷值) / 峰值 ) x 100

        Args:
            equity_curve: 每日权益值序列（首日为初始权益）。

        Returns:
            最大回撤百分比，例如 15.5 表示 15.5%。
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0

        peak = float(equity_curve[0])
        max_dd = 0.0

        for value in equity_curve[1:]:
            val = float(value)
            if val > peak:
                peak = val
            if peak > 0:
                dd = (peak - val) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd

        return max_dd


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _shift_date(date_str: str, days: int) -> str:
    """将 YYYYMMDD 日期字符串偏移指定天数。

    Args:
        date_str: 原始日期 YYYYMMDD。
        days: 偏移天数（负数向前）。

    Returns:
        偏移后的日期 YYYYMMDD。
    """
    dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=days)
    return dt.strftime("%Y%m%d")


def _fmt_ymd(date_str: str) -> str:
    """将 YYYYMMDD 转换为 YYYY-MM-DD 格式。"""
    if len(date_str) != 8 or not date_str.isdigit():
        return date_str
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def _filter_stock_df(df: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """从全量 DataFrame 中过滤出单只股票的日线并保证升序。"""
    sub = df[df["ts_code"] == ts_code].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("trade_date").reset_index(drop=True)
    return sub


def _calc_annualized_return(
    initial_value: float,
    final_value: float,
    trading_days: int,
    trading_days_per_year: float = 244.0,
) -> float:
    """计算年化收益率。

    Args:
        initial_value: 初始权益。
        final_value: 最终权益。
        trading_days: 回测交易天数。
        trading_days_per_year: 年化交易天数基准（A 股约 244 天）。

    Returns:
        年化收益率百分比（如 10.5 表示 10.5%）。
    """
    if (
        initial_value <= 0
        or final_value <= 0
        or trading_days < 1
    ):
        return 0.0

    years = trading_days / trading_days_per_year
    if years <= 0:
        return 0.0

    return ((final_value / initial_value) ** (1.0 / years) - 1) * 100
