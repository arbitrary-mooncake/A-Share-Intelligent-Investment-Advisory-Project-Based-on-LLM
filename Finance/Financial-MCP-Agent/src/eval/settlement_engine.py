"""
结算引擎 — 批量结算历史预测快照。
对已成熟的snapshot，通过Tushare获取真实价格数据，计算实际收益。
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.eval.label_builder import build_realized_label
from src.eval.database import init_db
from src.eval.repositories import (
    get_unsettled_snapshots, create_label, get_snapshots_by_batch
)
from src.eval.schemas import RealizedLabel


class SettlementEngine:
    """
    结算引擎：查找已到期但未结算的snapshot，获取市场价格数据，结算收益。

    V1版本使用简化的价格查询（通过现有Tushare MCP工具）。
    完整历史价格回填将在Phase 3回测引擎中实现。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        init_db()

    async def settle_matured_snapshots(
        self, current_date: str, term: str = "short"
    ) -> Dict[str, Any]:
        """
        结算所有已成熟的snapshot。

        Args:
            current_date: 当前日期 YYYY-MM-DD
            term: short/medium/long

        Returns:
            {"settled": 5, "failed": 1, "skipped": 0, "total": 6}
        """
        horizon_days = {"short": 1, "medium": 20, "long": 240}.get(term, 1)

        unsettled = get_unsettled_snapshots(term, horizon_days, current_date)

        settled = 0
        failed = 0
        skipped = 0

        for snap in unsettled:
            try:
                label = await self._settle_single(snap, horizon_days)
                if label:
                    create_label(RealizedLabel(
                        snapshot_id=snap["snapshot_id"],
                        line_id=snap.get("line_id", ""),
                        term=term,
                        horizon_days=horizon_days,
                        outcome_date=label.get("outcome_date", current_date),
                        entry_price=label.get("entry_price", 0),
                        exit_price=label.get("exit_price", 0),
                        asset_return_pct=label.get("asset_return_pct", 0),
                        benchmark_return_pct=label.get("benchmark_return_pct", 0),
                        excess_return_pct=label.get("excess_return_pct", 0),
                        max_drawdown_pct=label.get("max_drawdown_pct", 0),
                        volatility_pct=label.get("volatility_pct", 0),
                        is_valid=label.get("is_valid", True),
                    ))
                    settled += 1
                else:
                    skipped += 1
            except Exception as e:
                failed += 1

        return {"settled": settled, "failed": failed, "skipped": skipped, "total": len(unsettled)}

    async def _settle_single(self, snap: Dict[str, Any], horizon_days: int) -> Optional[Dict[str, Any]]:
        """
        结算单个snapshot — 通过Tushare HTTP API获取真实价格数据。
        """
        as_of_date = snap.get("as_of_date", "")
        symbol = snap.get("symbol", "")

        if not as_of_date or not symbol:
            return None

        try:
            from src.eval.data_fetcher import (
                convert_code_to_ts, convert_date_to_ts,
                fetch_daily_prices, fetch_benchmark_prices
            )
            from src.eval.label_builder import build_realized_label, compute_drawdown

            ts_code = convert_code_to_ts(symbol)
            start_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=horizon_days + 5)  # +5 buffer for non-trading days

            ts_start = convert_date_to_ts(as_of_date)
            ts_end = convert_date_to_ts(end_dt.strftime("%Y-%m-%d"))

            # 获取股票价格序列
            daily_data = fetch_daily_prices(ts_code, ts_start, ts_end)

            if not daily_data or len(daily_data) < 2:
                return {
                    "snapshot_id": snap["snapshot_id"],
                    "outcome_date": end_dt.strftime("%Y-%m-%d"),
                    "entry_price": 0.0, "exit_price": 0.0,
                    "asset_return_pct": 0.0, "benchmark_return_pct": 0.0,
                    "excess_return_pct": 0.0, "max_drawdown_pct": 0.0,
                    "volatility_pct": 0.0,
                    "is_valid": False,
                    "settlement_notes": "Tushare未返回足够的价格数据",
                }

            entry_price = daily_data[0]["close"]
            exit_price = daily_data[-1]["close"]
            prices = [d["close"] for d in daily_data]

            # 构建label
            actual_days = len(prices) - 1
            price_info = {
                "entry_price": entry_price,
                "exit_price": exit_price,
                "prices": prices,
                "start_date": as_of_date,
                "end_date": end_dt.strftime("%Y-%m-%d"),
            }

            # 尝试获取基准（CSI 300）
            benchmark_info = {}
            try:
                benchmark_prices = fetch_benchmark_prices("000300.SH", as_of_date, end_dt.strftime("%Y-%m-%d"))
                if benchmark_prices and len(benchmark_prices) >= 2:
                    benchmark_info = {
                        "entry_price": benchmark_prices[0],
                        "exit_price": benchmark_prices[-1],
                    }
            except Exception:
                pass

            label = build_realized_label(snap, price_info, benchmark_info or None)
            label["settlement_notes"] = f"Tushare真实数据 | 持有{actual_days}个交易日"
            return label

        except Exception as e:
            start_dt = datetime.strptime(snap.get("as_of_date", "2020-01-01"), "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=horizon_days)
            return {
                "snapshot_id": snap["snapshot_id"],
                "outcome_date": end_dt.strftime("%Y-%m-%d"),
                "entry_price": 0.0, "exit_price": 0.0,
                "asset_return_pct": 0.0, "benchmark_return_pct": 0.0,
                "excess_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "volatility_pct": 0.0,
                "is_valid": False,
                "settlement_notes": f"Tushare查询异常: {str(e)[:100]}",
            }

    def get_settlement_status(self, batch_id: str) -> Dict[str, Any]:
        """获取某批次的结算进度"""
        snapshots = get_snapshots_by_batch(batch_id)

        total = len(snapshots)
        settled_count = 0
        for snap in snapshots:
            # Count those with realized_label
            from src.eval.repositories import get_labels_by_snapshot
            labels = get_labels_by_snapshot(snap["snapshot_id"])
            if labels:
                settled_count += 1

        return {
            "batch_id": batch_id,
            "total_snapshots": total,
            "settled": settled_count,
            "unsettled": total - settled_count,
            "progress_pct": round(settled_count / max(total, 1) * 100, 1),
        }
