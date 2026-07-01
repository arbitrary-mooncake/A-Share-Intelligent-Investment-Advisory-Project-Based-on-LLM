"""模拟盘执行器 — SimulationRunner 类。

实现日级模拟盘运行，包括单日结算和多日追赶。
从绑定策略计算信号 -> 执行调仓 -> 更新现价 -> 重算持仓 -> 保存结算记录。

纯 Python 执行，无 LLM 调用。策略信号与实时运行时完全一致。

Usage:
    from src.advisory.portfolio_manager import PortfolioManager
    from src.advisory.simulation_runner import SimulationRunner

    pm = PortfolioManager()
    runner = SimulationRunner(portfolio_manager=pm)
    result = runner.run_daily_settlement("pf_id", "2026-06-30")

    # 自动追赶
    catch_up_result = runner.run_catch_up("pf_id")
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src.advisory.catch_up import CatchUpDetector
from src.advisory.portfolio_manager import PortfolioManager
from src.advisory.strategy_engine import StrategyEngine
from src.utils.tushare_client import _call, _items_to_dicts


# Tushare daily API 请求字段
_DAILY_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,"
    "change,pct_chg,vol,amount"
)


class SimulationRunner:
    """模拟盘执行器。

    职责范围:
    - 单日结算：策略信号 -> 调仓执行 -> 更新现价 -> 重算 -> 保存。
    - 多日追赶：通过 CatchUpDetector 自动补跑缺失交易日。
    - 从 Tushare 拉取单只股票日线用于市价更新。
    """

    def __init__(
        self,
        portfolio_manager: Optional[PortfolioManager] = None,
        settlement_dir: Optional[str] = None,
    ):
        """初始化 SimulationRunner。

        Args:
            portfolio_manager: PortfolioManager 实例。
                               为 None 时自动创建。
            settlement_dir: 结算记录保存目录。
                            默认 ``<项目根>/data/advisory_settlements/``。
        """
        self.portfolio_manager = portfolio_manager or PortfolioManager()
        if settlement_dir is None:
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            settlement_dir = os.path.join(root, "data", "advisory_settlements")
        self._settlement_dir = settlement_dir
        os.makedirs(self._settlement_dir, exist_ok=True)

        self._catch_up_detector = CatchUpDetector(settlement_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_daily_settlement(
        self,
        portfolio_id: str,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行单日模拟盘结算。

        流程:
        1. 加载组合，获取持仓股票列表。
        2. 拉取持仓股票在 date 当日的日线行情。
        3. 使用绑定策略计算信号，执行调仓（如果有信号）。
        4. 以当日收盘价更新所有持仓的现价。
        5. 重算组合市值/权重/盈亏。
        6. 保存组合。
        7. 保存结算记录 JSON。

        Args:
            portfolio_id: 组合 ID。
            date: 结算日期 YYYY-MM-DD。默认使用系统当前日期。

        Returns:
            {
                "date": "2026-06-30",
                "portfolio_id": "...",
                "total_value": 123456.78,
                "daily_return_pct": 0.55,
                "trades": [...],
                "holdings": {...},
                "status": "ok"
            }

        Raises:
            ValueError: 组合不存在或未绑定策略。
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        # 1. 加载组合
        pf = self.portfolio_manager.load(portfolio_id)
        if pf is None:
            raise ValueError(f"组合 {portfolio_id} 不存在")

        # 2. 获取策略信息
        strategy_name = pf.bound_strategy
        strategy_params = dict(pf.strategy_params) if pf.strategy_params else {}

        stock_codes = list(pf.holdings.keys())
        name_map: Dict[str, str] = {
            code: h.company_name for code, h in pf.holdings.items()
        }

        # 获取上一结算日（用于计算日收益率）
        prev_date = self._catch_up_detector.get_last_settlement_date(portfolio_id)
        prev_total_value: Optional[float] = None
        if prev_date:
            prev_settlement = self._load_settlement(portfolio_id, prev_date)
            if prev_settlement:
                prev_total_value = prev_settlement.get("total_value")

        # 3. 拉取结算日行情（所有持仓股）
        daily_map = self._fetch_market_data(stock_codes, date)

        # 4. 策略信号 -> 调仓
        trades: List[Dict[str, Any]] = []

        if stock_codes and strategy_name:
            # 拉取含预热数据的日线用于信号计算
            warmup_df = self._fetch_warmup_data(stock_codes, date)
            if not warmup_df.empty:
                for code in stock_codes:
                    stock_df = warmup_df[warmup_df["ts_code"] == code].copy()
                    if stock_df.empty or len(stock_df) < 2:
                        continue

                    # 该结算日的行情数据
                    today_data = daily_map.get(code)
                    if today_data is None:
                        continue
                    today_close = today_data.get("close")
                    if today_close is None:
                        continue

                    holding = pf.holdings.get(code)
                    position = holding.quantity if holding else 0
                    entry_price = holding.cost_price if holding else None

                    try:
                        sig, reason = StrategyEngine.compute_signal(
                            strategy_name,
                            stock_df,
                            strategy_params,
                            context={
                                "position": position,
                                "entry_price": entry_price,
                                "cash": pf.cash,
                            },
                        )
                    except Exception:
                        sig, reason = 0, "信号计算失败"

                    # 模拟盘以当日开盘价执行（收盘后结算视角，简化处理）
                    if sig == 1 and position == 0:
                        company_name = name_map.get(code, code)
                        exec_price = self._safe_float(
                            today_data.get("open"), today_close
                        )
                        try:
                            pf, trade = self.portfolio_manager.add_holding(
                                pf, code, company_name, 100, exec_price
                            )
                            trades.append({
                                "date": date,
                                "action": "buy",
                                "stock_code": code,
                                "company_name": company_name,
                                "price": exec_price,
                                "shares": 100,
                                "commission": trade.commission,
                                "reason": reason,
                            })
                        except ValueError:
                            pass

                    elif sig == -1 and position > 0:
                        company_name = name_map.get(code, code)
                        exec_price = self._safe_float(
                            today_data.get("open"), today_close
                        )
                        try:
                            pf, trade = self.portfolio_manager.remove_holding(
                                pf, code, position, exec_price
                            )
                            trades.append({
                                "date": date,
                                "action": "sell",
                                "stock_code": code,
                                "company_name": company_name,
                                "price": exec_price,
                                "shares": position,
                                "commission": trade.commission,
                                "reason": reason,
                            })
                        except ValueError:
                            pass

        # 5. 以日收盘价更新所有持仓现价
        for code, h in pf.holdings.items():
            today_data = daily_map.get(code)
            if today_data is not None:
                close_val = today_data.get("close")
                if close_val is not None:
                    h.current_price = float(close_val)

        # 6. 重算组合
        self.portfolio_manager.recalc(pf)
        self.portfolio_manager.save(pf)

        # 7. 计算日收益率
        total_value = pf.cash + pf.total_market_value
        daily_return_pct = 0.0
        if prev_total_value is not None and prev_total_value > 0:
            daily_return_pct = round(
                (total_value - prev_total_value) / prev_total_value * 100, 4
            )

        # 8. 保存结算记录
        result: Dict[str, Any] = {
            "date": date,
            "portfolio_id": portfolio_id,
            "total_value": round(total_value, 2),
            "total_cost": round(pf.total_cost, 2),
            "cash": round(pf.cash, 2),
            "daily_return_pct": daily_return_pct,
            "trades": trades,
            "holdings": {
                code: {
                    "company_name": h.company_name,
                    "quantity": h.quantity,
                    "cost_price": h.cost_price,
                    "current_price": h.current_price,
                    "market_value": h.market_value,
                    "weight": h.weight,
                }
                for code, h in pf.holdings.items()
            },
            "status": "ok",
        }
        self._save_settlement(portfolio_id, date, result)

        return result

    def run_catch_up(self, portfolio_id: str) -> Dict[str, Any]:
        """自动追赶缺失交易日。

        调用 CatchUpDetector 检测缺失日，逐日回放。
        追赶不设上限，所有缺失日无论多少都会逐日处理。

        Args:
            portfolio_id: 组合 ID。

        Returns:
            CatchUpDetector.catch_up() 返回的完整追赶结果字典。
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return self._catch_up_detector.catch_up(
            portfolio_id,
            today,
            runner_fn=self.run_daily_settlement,
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_single_stock(
        self, code: str, date: str
    ) -> Optional[Dict[str, Any]]:
        """从 Tushare 拉取单只股票在指定日期的日线数据。

        Args:
            code: ts_code，如 ``"600519.SH"``。
            date: 日期 YYYY-MM-DD。

        Returns:
            包含 open, high, low, close, vol, amount 等字段的 dict，
            或 None（无数据/失败）。
        """
        ts_date = date.replace("-", "")
        result = _call(
            "daily",
            {"ts_code": code, "start_date": ts_date, "end_date": ts_date},
            _DAILY_FIELDS,
        )
        items = _items_to_dicts(result)
        if not items:
            return None
        row = items[0]
        # 数值类型转换
        for col in ("open", "high", "low", "close", "vol", "amount", "pre_close", "pct_chg"):
            if col in row and row[col] is not None:
                try:
                    row[col] = float(row[col])
                except (ValueError, TypeError):
                    pass
        return row

    def _fetch_market_data(
        self, stock_codes: List[str], date: str
    ) -> Dict[str, Dict[str, Any]]:
        """批量拉取多只股票在指定日期的行情。

        Args:
            stock_codes: ts_code 列表。
            date: 日期 YYYY-MM-DD。

        Returns:
            {ts_code: {open, high, low, close, ...}, ...}。
            拉取失败的股票不会出现在返回中。
        """
        result_map: Dict[str, Dict[str, Any]] = {}
        for code in stock_codes:
            data = self._fetch_single_stock(code, date)
            if data:
                result_map[code] = data
        return result_map

    def _fetch_warmup_data(
        self, stock_codes: List[str], end_date: str, warmup_days: int = 400
    ) -> pd.DataFrame:
        """拉取多只股票含预热数据的日线。

        用于策略信号计算，确保有足够的历史数据计算技术指标。

        Args:
            stock_codes: ts_code 列表。
            end_date: 截止日期 YYYY-MM-DD。
            warmup_days: 预热天数，默认 400 个自然日（约 270+ 交易日）。

        Returns:
            包含 ts_code, trade_date, open, close 等列的 DataFrame，
            按 ts_code + trade_date 升序排列。无数据时返回空 DataFrame。
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=warmup_days)
        start_str = start_dt.strftime("%Y%m%d")
        end_ts = end_date.replace("-", "")

        dfs: List[pd.DataFrame] = []
        for code in stock_codes:
            result = _call(
                "daily",
                {"ts_code": code, "start_date": start_str, "end_date": end_ts},
                _DAILY_FIELDS,
            )
            items = _items_to_dicts(result)
            if not items:
                continue
            df = pd.DataFrame(items)
            for col in ("open", "high", "low", "close", "vol", "amount"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["trade_date"] = df["trade_date"].astype(str)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        full = pd.concat(dfs, ignore_index=True)
        full = full.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        return full

    # ------------------------------------------------------------------
    # Settlement persistence
    # ------------------------------------------------------------------

    def _save_settlement(
        self, portfolio_id: str, date: str, data: Dict[str, Any]
    ) -> str:
        """保存日结算 JSON 文件。

        文件名格式: ``{portfolio_id}_{date}.json``

        Args:
            portfolio_id: 组合 ID。
            date: 结算日期 YYYY-MM-DD。
            data: 结算数据字典。

        Returns:
            保存的文件路径。
        """
        file_path = os.path.join(
            self._settlement_dir, f"{portfolio_id}_{date}.json"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return file_path

    def _load_settlement(
        self, portfolio_id: str, date: str
    ) -> Optional[Dict[str, Any]]:
        """加载指定日期的结算记录。

        Args:
            portfolio_id: 组合 ID。
            date: 日期 YYYY-MM-DD。

        Returns:
            结算字典，或 None（不存在或读取失败）。
        """
        file_path = os.path.join(
            self._settlement_dir, f"{portfolio_id}_{date}.json"
        )
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any, fallback: float) -> float:
        """安全地将值转为 float，失败时返回 fallback。

        Args:
            value: 待转换的值。
            fallback: 转换失败时的默认值。

        Returns:
            float 类型的值。
        """
        if value is None:
            return fallback
        try:
            return float(value)
        except (ValueError, TypeError):
            return fallback
