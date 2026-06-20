"""
市场仿真层 — 模拟盘与回测共用基础设施。
统一处理所有A股交易现实约束：佣金、印花税、滑点、成交量约束、涨跌停、停牌、T+1。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class OrderStatus(Enum):
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"


class RejectReason(Enum):
    SUSPENDED = "停牌"
    LIMIT_UP = "涨停无法买入"
    LIMIT_DOWN = "跌停无法卖出"
    LIQUIDITY = "流动性不足"
    T1_CONSTRAINT = "T+1约束"


@dataclass
class Order:
    """策略层生成的理想订单"""
    stock_code: str
    direction: str          # "buy" | "sell"
    quantity: int = 0       # 股数（100的整数倍）
    target_value: float = 0.0  # 目标成交金额
    order_type: str = "market"  # market | limit
    limit_price: float = 0.0    # 限价（仅limit订单）


@dataclass
class ExecutedOrder(Order):
    """市场仿真层执行后的实际成交"""
    actual_price: float = 0.0
    actual_value: float = 0.0
    commission: float = 0.0
    transfer_fee: float = 0.0
    stamp_tax: float = 0.0        # 仅卖出
    net_cost: float = 0.0          # 净成本（买入为正，卖出为负收入）
    status: str = "filled"         # filled/partial/rejected
    reject_reason: str = ""
    flags: List[str] = field(default_factory=list)  # 标记列表


@dataclass
class MarketData:
    """当日市场快照"""
    stock_code: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    pre_close: float = 0.0
    volume: float = 0.0            # 成交量（手）
    amount: float = 0.0            # 成交额（元）
    turnover_rate: float = 0.0     # 换手率
    is_suspended: bool = False
    is_limit_up: bool = False      # 涨停
    is_limit_down: bool = False    # 跌停
    is_hs300: bool = False         # 沪深300成分股
    market_cap: float = 0.0        # 总市值
    pe_ratio: float = 0.0          # PE ratio (from daily_basic)
    pb_ratio: float = 0.0          # PB ratio (from daily_basic)
    price_to_ma_ratio: float = 1.0 # Close / MA60, 1.0 = at MA
    risk_flags: list = None        # Risk flags from analysis (delist, audit, etc.)
    # ── 策略扩展字段（v2, 用于实现总纲 §5.3-5.5 完整规则）──
    industry: str = ""             # 申万一级行业名称
    industry_sw_code: str = ""     # 申万行业代码
    roe: float = 0.0              # ROE (from fina_indicator)
    revenue_growth: float = 0.0   # 营收增速 YoY
    dividend_yield: float = 0.0   # 股息率
    profit_growth_3y: float = 0.0 # 连续3年净利润增长指标 (>0表示满足)
    pe_percentile: float = -1.0   # PE历史分位数 (0-100, -1表示无数据)
    industry_pe_75th: float = 0.0 # 行业PE 75分位值
    industry_pe_90th: float = 0.0 # 行业PE 90分位值
    industry_pe_median: float = 0.0 # 行业PE中位数
    # 日内异常监控
    close_30min_spike: float = 0.0 # 收盘前30分钟异常波动幅度 (0=无异常, >0.03触发)


class MarketSimulator:
    """
    市场仿真层：接收订单，返回实际执行结果。

    处理流程:
        1. 停牌检查
        2. 涨跌停检查
        3. 成交量约束检查
        4. T+1约束检查
        5. 交易成本计算(佣金+过户费+印花税+滑点)
        6. 订单执行与结算
    """

    # ── 交易成本参数（可配置） ──
    COMMISSION_RATE = 0.00025      # 佣金 万2.5
    MIN_COMMISSION = 5.0           # 最低佣金 5元
    TRANSFER_FEE_RATE = 0.00002    # 过户费 万0.2
    STAMP_TAX_RATE = 0.001         # 印花税 千1（仅卖出）

    SLIPPAGE_BASE = 0.0005         # 基础滑点 万5
    SLIPPAGE_HS300 = 0.0002        # 大盘蓝筹 万2
    SLIPPAGE_SMALL = 0.001         # 小盘股 千1

    # ── 成交量约束 ──
    MAX_SINGLE_ORDER_RATIO = 0.05  # 单笔不超过日成交额5%
    MAX_DAILY_BUY_RATIO = 0.10     # 单日买入不超过日成交额10%

    # ── 涨跌停幅度 ──
    LIMIT_RATIO_MAIN = 0.10        # 主板 ±10%
    LIMIT_RATIO_GEM = 0.20         # 创业板/科创板 ±20%
    LIMIT_RATIO_BSE = 0.30         # 北交所 ±30%

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """可从config覆盖默认参数"""
        if config:
            self.COMMISSION_RATE = config.get("commission_rate", self.COMMISSION_RATE)
            self.MIN_COMMISSION = config.get("min_commission", self.MIN_COMMISSION)
            self.SLIPPAGE_BASE = config.get("slippage_base", self.SLIPPAGE_BASE)
            self.MAX_SINGLE_ORDER_RATIO = config.get("max_single_order_ratio", self.MAX_SINGLE_ORDER_RATIO)

        # 日内追踪（用于T+1和单日累计约束）
        self._daily_bought: Dict[str, float] = {}       # stock_code -> 当日买入金额
        self._t1_locked: Dict[str, int] = {}             # stock_code -> 剩余锁定天数
        self._suspension_tracker: Dict[str, int] = {}    # stock_code -> 连续停牌天数

    def reset_daily_state(self):
        """每日重置（开盘前调用）"""
        self._daily_bought.clear()
        # T+1: 将所有锁定减1，到期的移除
        expired = [code for code, days in self._t1_locked.items() if days <= 1]
        for code in expired:
            del self._t1_locked[code]
        for code in list(self._t1_locked.keys()):
            self._t1_locked[code] -= 1

    def _get_limit_ratio(self, stock_code: str) -> float:
        """根据股票代码判断涨跌停幅度"""
        code = stock_code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()
        if code.startswith(("30", "68")):
            return self.LIMIT_RATIO_GEM  # 创业板(30xxxx) / 科创板(68xxxx): ±20%
        if code.startswith("8"):
            return self.LIMIT_RATIO_BSE  # 北交所(8xxxxx): ±30%
        return self.LIMIT_RATIO_MAIN

    def _get_slippage(self, market_data: MarketData) -> float:
        """获取滑点"""
        if market_data.is_hs300:
            return self.SLIPPAGE_HS300
        if market_data.market_cap > 0 and market_data.market_cap < 5e9:  # 50亿以下=小盘
            return self.SLIPPAGE_SMALL
        return self.SLIPPAGE_BASE

    def _check_limit(self, order: Order, market_data: MarketData) -> Optional[str]:
        """检查涨跌停，返回拒绝原因或None"""
        if order.direction == "buy" and market_data.is_limit_up:
            # 盘中触及涨停但未封板 → 50%概率成交（简化处理：直接拒绝）
            return RejectReason.LIMIT_UP.value
        if order.direction == "sell" and market_data.is_limit_down:
            return RejectReason.LIMIT_DOWN.value
        return None

    def _check_suspension(self, market_data: MarketData) -> Optional[str]:
        """检查停牌"""
        if market_data.is_suspended:
            return RejectReason.SUSPENDED.value
        return None

    def _check_volume(self, order: Order, market_data: MarketData) -> Optional[str]:
        """检查成交量约束"""
        if market_data.amount <= 0:
            return None  # 数据缺失时放行
        max_single = market_data.amount * self.MAX_SINGLE_ORDER_RATIO
        daily_bought = self._daily_bought.get(order.stock_code, 0)
        remaining = market_data.amount * self.MAX_DAILY_BUY_RATIO - daily_bought
        if order.target_value > min(max_single, remaining):
            return RejectReason.LIQUIDITY.value
        return None

    def _check_t1(self, order: Order) -> Optional[str]:
        """检查T+1约束（当日买入的股票当日不能卖）"""
        if order.direction == "sell" and order.stock_code in self._t1_locked:
            return RejectReason.T1_CONSTRAINT.value
        return None

    def _calculate_trade_cost(self, order: Order, executed_price: float,
                               executed_value: float) -> tuple:
        """计算交易成本，返回(commission, transfer_fee, stamp_tax, net_cost)"""
        commission = max(executed_value * self.COMMISSION_RATE, self.MIN_COMMISSION)
        transfer_fee = executed_value * self.TRANSFER_FEE_RATE

        if order.direction == "buy":
            stamp_tax = 0.0
            net_cost = executed_value + commission + transfer_fee  # 实际支出
        else:
            stamp_tax = executed_value * self.STAMP_TAX_RATE
            net_cost = executed_value - commission - transfer_fee - stamp_tax  # 实际收入

        return commission, transfer_fee, stamp_tax, net_cost

    def execute_order(self, order: Order, market_data: MarketData) -> ExecutedOrder:
        """
        执行单个订单，返回实际成交结果。

        Args:
            order: 策略生成的理想订单
            market_data: 当日该股票的市场数据

        Returns:
            ExecutedOrder: 实际执行结果
        """
        result = ExecutedOrder(
            stock_code=order.stock_code,
            direction=order.direction,
            quantity=order.quantity,
            target_value=order.target_value,
            order_type=order.order_type,
            limit_price=order.limit_price,
            status=OrderStatus.FILLED.value,
        )

        # 1. 停牌检查
        reason = self._check_suspension(market_data)
        if reason:
            result.status = OrderStatus.REJECTED.value
            result.reject_reason = reason
            return result

        # 2. 涨跌停检查
        reason = self._check_limit(order, market_data)
        if reason:
            result.status = OrderStatus.REJECTED.value
            result.reject_reason = reason
            return result

        # 3. T+1约束（仅卖出）
        reason = self._check_t1(order)
        if reason:
            result.status = OrderStatus.REJECTED.value
            result.reject_reason = reason
            return result

        # 4. 确定执行价格（含滑点）
        slippage = self._get_slippage(market_data)
        base_price = market_data.close if market_data.close > 0 else market_data.open
        if order.direction == "buy":
            result.actual_price = base_price * (1 + slippage)
        else:
            result.actual_price = base_price * (1 - slippage)

        # 价格有效性检查
        if result.actual_price <= 0:
            result.status = OrderStatus.REJECTED.value
            result.reject_reason = "执行价格为0（数据异常）"
            return result

        # 5. 计算实际成交量和金额
        if order.quantity > 0:
            # 按股数计算（100的整数倍）
            result.quantity = (order.quantity // 100) * 100
            result.actual_value = result.quantity * result.actual_price
        elif order.target_value > 0:
            # 按金额计算
            result.quantity = int(order.target_value / result.actual_price / 100) * 100
            result.actual_value = result.quantity * result.actual_price
        else:
            result.status = OrderStatus.REJECTED.value
            result.reject_reason = "订单数量与金额均为零"
            return result

        if result.quantity <= 0:
            result.status = OrderStatus.REJECTED.value
            result.reject_reason = "计算后数量为零"
            return result

        # 6. 成交量约束检查
        reason = self._check_volume(order, market_data)
        if reason == RejectReason.LIQUIDITY.value:
            # 尝试缩减到最大允许量
            max_single = market_data.amount * self.MAX_SINGLE_ORDER_RATIO
            daily_bought = self._daily_bought.get(order.stock_code, 0)
            remaining = market_data.amount * self.MAX_DAILY_BUY_RATIO - daily_bought
            max_allowable = min(max_single, max(remaining, 0))
            if max_allowable <= 0:
                result.status = OrderStatus.REJECTED.value
                result.reject_reason = RejectReason.LIQUIDITY.value
                return result
            scale = max_allowable / result.actual_value
            result.quantity = int(result.quantity * scale / 100) * 100
            result.actual_value = result.quantity * result.actual_price
            result.status = OrderStatus.PARTIAL.value
            result.flags.append("流动性受限")

        # 7. 计算交易成本
        result.commission, result.transfer_fee, result.stamp_tax, result.net_cost = \
            self._calculate_trade_cost(order, result.actual_price, result.actual_value)

        # 8. 更新日内追踪
        if order.direction == "buy":
            self._daily_bought[order.stock_code] = \
                self._daily_bought.get(order.stock_code, 0) + result.actual_value
            self._t1_locked[order.stock_code] = 1  # T+1锁定1天

        return result

    def execute_orders(self, orders: List[Order],
                       market_data_map: Dict[str, MarketData]) -> List[ExecutedOrder]:
        """
        批量执行订单。

        Args:
            orders: 策略生成的理想订单列表
            market_data_map: {stock_code: MarketData} 当日市场数据

        Returns:
            List[ExecutedOrder]: 实际执行结果列表
        """
        results = []
        for order in orders:
            md = market_data_map.get(order.stock_code)
            if md is None:
                # 无市场数据 → 拒绝
                result = ExecutedOrder(
                    stock_code=order.stock_code,
                    direction=order.direction,
                    status=OrderStatus.REJECTED.value,
                    reject_reason="无市场数据",
                )
            else:
                result = self.execute_order(order, md)
            results.append(result)
        return results

    def get_holding_value(self, stock_code: str, shares: int,
                          market_data: MarketData) -> float:
        """计算持仓市值（用收盘价）"""
        if market_data.close > 0:
            return shares * market_data.close
        return shares * market_data.open

    def force_liquidate_suspended(self, stock_code: str, shares: int,
                                   last_price: float, days_suspended: int) -> tuple:
        """
        强制估值停牌股票。
        超过20个交易日 → 按停牌前价格×0.95估值
        返回 (估值价格, 估值金额, 是否强制)
        """
        if days_suspended > 20:
            forced_price = last_price * 0.95
            return forced_price, shares * forced_price, True
        return last_price, shares * last_price, False
