"""模拟盘追赶机制 — CatchUpDetector 类。

检测指定组合从最后一次结算日到今日之间缺失的交易日，
按交易日历逐日回放，调用回调函数完成结算追赶。

纯 Python 实现，无 LLM 调用。追赶不设上限，所有缺失日逐日处理。

Usage:
    from src.advisory.catch_up import CatchUpDetector

    detector = CatchUpDetector()
    missed, days = detector.get_missed_days("pf_id", "2026-07-01")
    result = detector.catch_up("pf_id", "2026-07-01", my_runner_fn)
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from src.utils.tushare_client import _call


class CatchUpDetector:
    """模拟盘追赶检测器。

    职责范围:
    - 从 settlement JSON 文件名读取最后结算日期。
    - 通过 Tushare 交易日历获取两个日期之间的交易日列表。
    - 调用 runner_fn 逐日追赶。
    """

    def __init__(self, settlement_dir: Optional[str] = None):
        """初始化 CatchUpDetector。

        Args:
            settlement_dir: 结算记录目录。默认为
                            ``<项目根>/data/advisory_settlements/``。
        """
        if settlement_dir is None:
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            settlement_dir = os.path.join(root, "data", "advisory_settlements")
        self._settlement_dir = settlement_dir
        os.makedirs(self._settlement_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_last_settlement_date(self, portfolio_id: str) -> Optional[str]:
        """读取该组合最后结算日的日期。

        从 settlement 目录中查找符合 ``{portfolio_id}_YYYY-MM-DD.json``
        模式的所有文件，取日期最大的作为最后结算日。

        Args:
            portfolio_id: 组合 ID。

        Returns:
            最后结算日 YYYY-MM-DD 格式字符串。无结算记录时返回 None。
        """
        pattern = re.compile(
            rf"^{re.escape(portfolio_id)}_(\d{{4}}-\d{{2}}-\d{{2}})\.json$"
        )
        latest: Optional[str] = None
        if not os.path.isdir(self._settlement_dir):
            return None
        for fname in os.listdir(self._settlement_dir):
            m = pattern.match(fname)
            if m:
                d = m.group(1)
                if latest is None or d > latest:
                    latest = d
        return latest

    def get_missed_days(
        self,
        portfolio_id: str,
        today: Optional[str] = None,
    ) -> Tuple[int, List[str]]:
        """获取自上次结算以来缺失的交易日列表。

        通过 Tushare trade_cal 接口获取从最后结算日次日到今日
        （不含今日）之间的交易日，按升序排列。

        Args:
            portfolio_id: 组合 ID。
            today: 基准日期 YYYY-MM-DD。默认使用系统当前日期。

        Returns:
            (缺失天数, 缺失日期列表) 的元组。
            日期格式 YYYY-MM-DD，升序排列。
            无缺失时返回 (0, [])。
        """
        today = today or datetime.now().strftime("%Y-%m-%d")
        last_date = self.get_last_settlement_date(portfolio_id)

        if last_date is None:
            # 无结算记录则无需追赶
            return 0, []

        if last_date >= today:
            return 0, []

        # 从最后结算日次日开始查询交易日
        last_dt = datetime.strptime(last_date, "%Y-%m-%d")
        start_dt = last_dt + timedelta(days=1)
        start_str = start_dt.strftime("%Y-%m-%d")

        all_days = self._fetch_trading_days(start_str, today)

        # 排除 today 当天（尚未结束）
        missed = [d for d in all_days if d < today]
        return len(missed), missed

    def catch_up(
        self,
        portfolio_id: str,
        today: Optional[str] = None,
        runner_fn: Optional[Callable[[str, str], Dict]] = None,
    ) -> Dict:
        """逐日追赶结算。

        发现缺失交易日后，按时间顺序逐日调用 runner_fn 执行结算。
        每次调用传参 (portfolio_id, date_str)，期望返回 dict。

        Args:
            portfolio_id: 组合 ID。
            today: 基准日期 YYYY-MM-DD。默认使用系统当前日期。
            runner_fn: 单日结算回调函数，签名 ``fn(portfolio_id, date) -> dict``。
                       为 None 时使用默认的空结果。
                       典型实现为 ``SimulationRunner.run_daily_settlement``。

        Returns:
            {
                "missed": 6,
                "days": ["2026-06-01", ...],
                "results": [{...}, ...],
                "message": "追赶完成: 共 6 个交易日"
            }
        """
        missed_count, missed_days = self.get_missed_days(portfolio_id, today)
        if missed_count == 0:
            return {
                "missed": 0,
                "days": [],
                "results": [],
                "message": "无需追赶，已是最新",
            }

        results: List[Dict] = []
        for date_str in missed_days:
            if runner_fn is not None:
                result = runner_fn(portfolio_id, date_str)
            else:
                result = {"date": date_str, "status": "skipped"}
            results.append(result)

        return {
            "missed": missed_count,
            "days": missed_days,
            "results": results,
            "message": f"追赶完成: 共 {missed_count} 个交易日",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """通过 Tushare trade_cal 获取交易日列表。

        Args:
            start_date: 起始日期 YYYY-MM-DD。
            end_date: 结束日期 YYYY-MM-DD。

        Returns:
            交易日列表 YYYY-MM-DD，升序排列。失败返回空列表。
        """
        ts_start = start_date.replace("-", "")
        ts_end = end_date.replace("-", "")

        result = _call(
            "trade_cal",
            {"exchange": "SSE", "start_date": ts_start, "end_date": ts_end},
            fields="cal_date,is_open",
        )

        if not result or "items" not in result:
            return []

        fields = result.get("fields", [])
        days: List[str] = []
        for row in result["items"]:
            item = dict(zip(fields, row))
            is_open = item.get("is_open")
            if is_open is not None and int(is_open) == 1:
                cal_date = str(item.get("cal_date", ""))
                if len(cal_date) == 8:
                    days.append(
                        f"{cal_date[:4]}-{cal_date[4:6]}-{cal_date[6:8]}"
                    )
        return sorted(days)
