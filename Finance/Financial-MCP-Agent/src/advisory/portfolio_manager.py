"""投资组合持仓管理器 — CRUD + 交易执行 + 策略绑定。

提供持仓组合的创建、加载、保存、列表查询、买入/卖出交易执行、
市值与盈亏重算，以及策略绑定/解绑功能。

Usage:
    from src.advisory.portfolio_manager import PortfolioManager

    m = PortfolioManager()
    pf = m.create("我的组合")
    pf, trade = m.add_holding(pf, "600519.SH", "贵州茅台", 100, 1680.0)
    pf, trade = m.remove_holding(pf, "600519.SH", 100, 1690.0)
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.advisory.schemas import AdvisoryPortfolio, PortfolioHolding, TradeRecord


class PortfolioManager:
    """投资组合持仓管理器。

    职责范围:
    - 组合的 CRUD（创建/读取/更新/删除）。
    - 买入/卖出交易执行，含费率计算与资金校验。
    - 持仓市值、权重、盈亏的自动化重算。
    - 策略绑定与解绑。
    """

    # 交易费率
    COMMISSION_RATE = 0.0003   # 佣金 0.03%（买卖双向）
    STAMP_TAX_RATE = 0.001     # 印花税 0.1%（仅卖出）
    LOT_SIZE = 100             # A股 100 股整数倍

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, data_dir: Optional[str] = None):
        """初始化 PortfolioManager。

        Args:
            data_dir: 持仓数据文件目录。默认为 ``<项目根>/data/portfolios/``，
                      会自动创建该目录。
        """
        if data_dir is None:
            # 从当前文件位置向上推导项目根路径
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            data_dir = os.path.join(root, "data", "portfolios")
        self._data_dir = data_dir
        os.makedirs(self._data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _file_path(self, portfolio_id: str) -> str:
        """返回给定组合 ID 对应的 JSON 文件路径。"""
        return os.path.join(self._data_dir, f"{portfolio_id}.json")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        user_id: str = "default",
        initial_capital: float = 100000.0,
    ) -> AdvisoryPortfolio:
        """创建新投资组合。

        自动生成 portfolio_id (UUID4)，初始化现金为 ``initial_capital``，
        写入 JSON 文件后返回。

        Args:
            name: 组合名称。
            user_id: 用户标识，默认 ``"default"``。
            initial_capital: 初始本金，默认 100,000。

        Returns:
            已保存的 AdvisoryPortfolio 实例。
        """
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pf = AdvisoryPortfolio(
            portfolio_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            holdings={},
            total_cost=0.0,
            total_market_value=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            cash=initial_capital,
            initial_capital=initial_capital,
            created_at=now_iso,
            updated_at=now_iso,
            bound_strategy=None,
            strategy_params=None,
            status="active",
        )
        self.save(pf)
        return pf

    def load(self, portfolio_id: str) -> Optional[AdvisoryPortfolio]:
        """从 JSON 文件加载投资组合。

        Args:
            portfolio_id: 组合 ID。

        Returns:
            AdvisoryPortfolio 实例，文件不存在时返回 None。
        """
        file_path = self._file_path(portfolio_id)
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._from_dict(data)

    def save(self, pf: AdvisoryPortfolio) -> None:
        """序列化投资组合为 JSON 文件，自动更新 ``updated_at`` 时间戳。

        holdings 以 ``{stock_code: {field: value, ...}}`` 格式存储。

        Args:
            pf: 要保存的 AdvisoryPortfolio 实例。
        """
        pf.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = self._to_dict(pf)
        file_path = self._file_path(pf.portfolio_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_all(self, user_id: str = "default") -> List[AdvisoryPortfolio]:
        """列出指定用户的所有投资组合。

        Args:
            user_id: 用户标识，默认 ``"default"``。

        Returns:
            该用户的所有 AdvisoryPortfolio 列表。
        """
        result: List[AdvisoryPortfolio] = []
        if not os.path.isdir(self._data_dir):
            return result
        for filename in os.listdir(self._data_dir):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(self._data_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("user_id") == user_id:
                    result.append(self._from_dict(data))
            except (json.JSONDecodeError, OSError):
                continue  # 跳过损坏文件
        return result

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def add_holding(
        self,
        pf: AdvisoryPortfolio,
        stock_code: str,
        company_name: str,
        quantity: int,
        price: float,
        trade_date: Optional[str] = None,
    ) -> Tuple[AdvisoryPortfolio, TradeRecord]:
        """买入持仓 — 执行买入交易。

        流程:
        1. 数量向下取整至 100 股。
        2. 计算总成本 = 数量 * 价格 + 佣金 (0.03%)。
        3. 校验现金是否充足。
        4. 扣减现金。
        5. 更新或创建持仓记录（均价法更新成本价）。
        6. 重算市值/权重/盈亏。
        7. 自动保存。

        Args:
            pf: 当前投资组合（会被原地修改）。
            stock_code: 股票代码。
            company_name: 公司名称。
            quantity: 请求买入数量（自动向下取整至 100 的倍数）。
            price: 成交单价。
            trade_date: 交易日期（YYYY-MM-DD），为 None 时使用系统当前时间。

        Returns:
            (更新后的 AdvisoryPortfolio, TradeRecord) 元组。

        Raises:
            ValueError: 资金不足、数量不合法。
        """
        # 1. 100 股整数倍
        quantity = (quantity // self.LOT_SIZE) * self.LOT_SIZE
        if quantity <= 0:
            raise ValueError(f"买入数量必须至少为 {self.LOT_SIZE} 股")

        # 2. 费用计算
        cost = quantity * price
        commission = round(cost * self.COMMISSION_RATE, 2)
        total_required = cost + commission

        # 3. 资金校验
        if pf.cash < total_required:
            raise ValueError(
                f"可用资金不足: 需要 {total_required:.2f}, "
                f"可用 {pf.cash:.2f}"
            )

        # 4. 扣减现金
        pf.cash = round(pf.cash - total_required, 2)

        # 5. 更新/创建持仓
        if stock_code in pf.holdings:
            h = pf.holdings[stock_code]
            old_cost = h.quantity * h.cost_price
            new_qty = h.quantity + quantity
            h.cost_price = round((old_cost + cost) / new_qty, 2)
            h.quantity = new_qty
            h.current_price = price
        else:
            pf.holdings[stock_code] = PortfolioHolding(
                stock_code=stock_code,
                company_name=company_name,
                quantity=quantity,
                cost_price=price,
                current_price=price,
                market_value=0.0,
                weight=0.0,
            )

        # 6. 重算
        self._recalc_holdings(pf)

        # 7. 持久化
        self.save(pf)

        trade = TradeRecord(
            date=trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            action="buy",
            stock_code=stock_code,
            company_name=company_name,
            price=price,
            shares=quantity,
            commission=commission,
            reason=f"买入 {company_name} ({stock_code}) {quantity}股, "
                   f"成交价 {price:.2f}",
        )
        return pf, trade

    def remove_holding(
        self,
        pf: AdvisoryPortfolio,
        stock_code: str,
        quantity: int,
        price: float,
        trade_date: Optional[str] = None,
    ) -> Tuple[AdvisoryPortfolio, TradeRecord]:
        """卖出持仓 — 执行卖出交易。

        流程:
        1. 校验持仓存在且数量充足。
        2. 计算收入 = 数量 * 价格 - 佣金 (0.03%) - 印花税 (0.1%)。
        3. 增加现金。
        4. 减少或删除持仓记录。
        5. 重算市值/权重/盈亏。
        6. 自动保存。

        Args:
            pf: 当前投资组合（会被原地修改）。
            stock_code: 股票代码。
            quantity: 卖出数量。
            price: 成交单价。
            trade_date: 交易日期（YYYY-MM-DD），为 None 时使用系统当前时间。

        Returns:
            (更新后的 AdvisoryPortfolio, TradeRecord) 元组。

        Raises:
            ValueError: 持仓不存在、卖出数量超出持仓。
        """
        # 1. 校验
        if stock_code not in pf.holdings:
            raise ValueError(f"持仓中未找到 {stock_code}")
        holding = pf.holdings[stock_code]
        if quantity > holding.quantity:
            raise ValueError(
                f"卖出数量超出持仓: 请求 {quantity}股, "
                f"持有 {holding.quantity}股"
            )

        company_name = holding.company_name

        # 2. 费用计算
        proceeds = quantity * price
        commission = round(proceeds * self.COMMISSION_RATE, 2)
        stamp_tax = round(proceeds * self.STAMP_TAX_RATE, 2)
        net = round(proceeds - commission - stamp_tax, 2)

        # 3. 增加现金
        pf.cash = round(pf.cash + net, 2)

        # 4. 减少/删除持仓，同步更新剩余持仓的现价
        if quantity >= holding.quantity:
            del pf.holdings[stock_code]
        else:
            holding.quantity -= quantity
            holding.current_price = price  # 以卖出价作为最新市价
        self._recalc_holdings(pf)

        # 6. 持久化
        self.save(pf)

        trade = TradeRecord(
            date=trade_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            action="sell",
            stock_code=stock_code,
            company_name=company_name,
            price=price,
            shares=quantity,
            commission=commission,
            reason=f"卖出 {company_name} ({stock_code}) {quantity}股, "
                   f"成交价 {price:.2f}",
        )
        return pf, trade

    # ------------------------------------------------------------------
    # Holdings recalculation
    # ------------------------------------------------------------------

    def recalc(self, pf: AdvisoryPortfolio) -> AdvisoryPortfolio:
        """公开的重算接口 — 委托给 ``_recalc_holdings`` 并返回 pf。

        供 backtest_runner / simulation_runner 等外部调用者使用，
        避免直接访问私有方法 ``_recalc_holdings``。

        Args:
            pf: 投资组合（会被原地修改）。

        Returns:
            原地修改后的同一 AdvisoryPortfolio 引用（便于链式调用）。
        """
        self._recalc_holdings(pf)
        return pf

    def _recalc_holdings(self, pf: AdvisoryPortfolio) -> None:
        """重算持仓组合的市值、权重和总体盈亏。

        为每个持仓计算:
        - ``market_value = quantity * current_price``
        - ``weight = market_value / total_market_value * 100``

        更新组合级:
        - ``total_market_value``: 所有持仓市值之和
        - ``total_cost``: 所有持仓成本之和
        - ``total_pnl`` = total_market_value - total_cost
        - ``total_pnl_pct`` = total_pnl / total_cost * 100
        """
        total_mv = 0.0
        total_cost = 0.0

        for h in pf.holdings.values():
            h.market_value = round(h.quantity * h.current_price, 2)
            total_mv += h.market_value
            total_cost += h.quantity * h.cost_price

        pf.total_market_value = round(total_mv, 2)
        pf.total_cost = round(total_cost, 2)

        # 权重计算
        if total_mv > 0:
            for h in pf.holdings.values():
                h.weight = round(h.market_value / total_mv * 100, 2)
        else:
            for h in pf.holdings.values():
                h.weight = 0.0

        # 盈亏计算
        pf.total_pnl = round(pf.total_market_value - pf.total_cost, 2)
        if pf.total_cost > 0:
            pf.total_pnl_pct = round(
                pf.total_pnl / pf.total_cost * 100, 2
            )
        else:
            pf.total_pnl_pct = 0.0

    # ------------------------------------------------------------------
    # Strategy binding
    # ------------------------------------------------------------------

    def bind_strategy(
        self,
        pf: AdvisoryPortfolio,
        strategy_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """绑定一个交易策略到投资组合。

        Args:
            pf: 投资组合（会被原地修改并保存）。
            strategy_name: 策略注册名称。
            params: 策略参数字典，可选。
        """
        pf.bound_strategy = strategy_name
        pf.strategy_params = params or {}
        self.save(pf)

    def unbind_strategy(self, pf: AdvisoryPortfolio) -> None:
        """解绑当前绑定的策略。

        Args:
            pf: 投资组合（会被原地修改并保存）。
        """
        pf.bound_strategy = None
        pf.strategy_params = None
        self.save(pf)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _to_dict(self, pf: AdvisoryPortfolio) -> Dict[str, Any]:
        """将 AdvisoryPortfolio 转为 JSON 可序列化字典。

        holdings 以 ``{stock_code: {field: value, ...}}`` 嵌套字典存储。
        """
        return {
            "portfolio_id": pf.portfolio_id,
            "user_id": pf.user_id,
            "name": pf.name,
            "holdings": {
                code: {
                    "stock_code": h.stock_code,
                    "company_name": h.company_name,
                    "quantity": h.quantity,
                    "cost_price": h.cost_price,
                    "current_price": h.current_price,
                    "market_value": h.market_value,
                    "weight": h.weight,
                }
                for code, h in pf.holdings.items()
            },
            "total_cost": pf.total_cost,
            "total_market_value": pf.total_market_value,
            "total_pnl": pf.total_pnl,
            "total_pnl_pct": pf.total_pnl_pct,
            "cash": pf.cash,
            "initial_capital": pf.initial_capital,
            "created_at": pf.created_at,
            "updated_at": pf.updated_at,
            "bound_strategy": pf.bound_strategy,
            "strategy_params": pf.strategy_params,
            "status": pf.status,
        }

    def _from_dict(self, data: Dict[str, Any]) -> AdvisoryPortfolio:
        """从字典反序列化为 AdvisoryPortfolio。

        重建 ``holdings: Dict[str, PortfolioHolding]``。
        """
        raw_holdings = data.get("holdings", {})
        holdings: Dict[str, PortfolioHolding] = {}
        for code, h_data in raw_holdings.items():
            holdings[code] = PortfolioHolding(
                stock_code=h_data.get("stock_code", ""),
                company_name=h_data.get("company_name", ""),
                quantity=h_data.get("quantity", 0),
                cost_price=h_data.get("cost_price", 0.0),
                current_price=h_data.get("current_price", 0.0),
                market_value=h_data.get("market_value", 0.0),
                weight=h_data.get("weight", 0.0),
            )

        return AdvisoryPortfolio(
            portfolio_id=data.get("portfolio_id", ""),
            user_id=data.get("user_id", "default"),
            name=data.get("name", ""),
            holdings=holdings,
            total_cost=data.get("total_cost", 0.0),
            total_market_value=data.get("total_market_value", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            total_pnl_pct=data.get("total_pnl_pct", 0.0),
            cash=data.get("cash", 0.0),
            initial_capital=data.get("initial_capital", 100000.0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            bound_strategy=data.get("bound_strategy"),
            strategy_params=data.get("strategy_params"),
            status=data.get("status", "active"),
        )
