"""
智能投顾基础数据结构 — 定义所有后续任务共享的 dataclass 和 enum。

本模块提供：
- Enums: HealthColor, MatchLevel, TradeDirection
- Dataclasses: PortfolioHolding, AdvisoryPortfolio, StrategyConfig, TradeRecord,
  SettlementRecord, BacktestResult, SimulationStatus, HealthAssessment,
  PreferenceMatch, RecommendationItem
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ──


class HealthColor(Enum):
    """投资组合健康度颜色标识"""
    DEEP_GREEN = "deep_green"
    LIGHT_GREEN = "light_green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class MatchLevel(Enum):
    """投资偏好匹配等级"""
    HIGH = "high"
    BASIC = "basic"
    PARTIAL = "partial"
    DEVIATED = "deviated"


class TradeDirection(Enum):
    """交易方向"""
    BUY = "buy"
    SELL = "sell"


# ── Dataclasses ──


@dataclass
class PortfolioHolding:
    """投资组合中的个股持仓信息"""
    stock_code: str = ""
    company_name: str = ""
    quantity: int = 0
    cost_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0  # 占组合总市值权重 (0-100)


@dataclass
class AdvisoryPortfolio:
    """用户投资组合"""
    portfolio_id: str = ""
    user_id: str = ""
    name: str = ""
    holdings: Dict[str, PortfolioHolding] = field(default_factory=dict)
    total_cost: float = 0.0
    total_market_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    cash: float = 0.0
    initial_capital: float = 0.0
    created_at: str = ""          # ISO datetime
    updated_at: str = ""          # ISO datetime
    bound_strategy: Optional[str] = None  # 绑定的策略名称
    strategy_params: Optional[Dict[str, Any]] = None
    status: str = "active"        # active | closed


@dataclass
class StrategyConfig:
    """策略配置定义"""
    name: str = ""
    display_name: str = ""
    category: str = ""            # short_term | medium_term | long_term
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    is_custom: bool = False       # 是否为用户自定义策略
    source_file: str = ""         # 策略代码文件路径


@dataclass
class TradeRecord:
    """一条交易记录"""
    date: str = ""                # YYYY-MM-DD
    action: str = ""              # buy | sell
    stock_code: str = ""
    company_name: str = ""
    price: float = 0.0
    shares: int = 0
    commission: float = 0.0
    stamp_tax: float = 0.0
    reason: str = ""              # 交易原因说明


@dataclass
class SettlementRecord:
    """每日结算记录"""
    date: str = ""                # YYYY-MM-DD
    portfolio_id: str = ""
    holdings_snapshot: Dict[str, Any] = field(default_factory=dict)
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)


@dataclass
class BacktestResult:
    """一次回测的完整结果"""
    portfolio_id: str = ""
    strategy_name: str = ""
    start_date: str = ""          # YYYY-MM-DD
    end_date: str = ""            # YYYY-MM-DD
    initial_capital: float = 0.0
    final_value: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    settlement_records: List[SettlementRecord] = field(default_factory=list)


@dataclass
class SimulationStatus:
    """模拟盘运行状态"""
    portfolio_id: str = ""
    is_running: bool = False
    start_date: str = ""          # YYYY-MM-DD
    last_settlement_date: str = ""  # YYYY-MM-DD
    total_days: int = 0
    missed_days: int = 0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    pending_trades: List[TradeRecord] = field(default_factory=list)


@dataclass
class HealthAssessment:
    """投资组合健康度评估"""
    color: HealthColor = HealthColor.YELLOW
    color_label: str = ""
    summary: str = ""
    details: str = ""


@dataclass
class PreferenceMatch:
    """投资偏好匹配度"""
    level: MatchLevel = MatchLevel.BASIC
    level_label: str = ""
    reason: str = ""
    details: str = ""


@dataclass
class RecommendationItem:
    """推荐个股条目"""
    stock_code: str = ""
    company_name: str = ""
    reason: str = ""
    has_score_cache: bool = False
    score: Optional[float] = None
    term: Optional[str] = None      # short | medium | long
    basic_info: Optional[Dict[str, Any]] = None
