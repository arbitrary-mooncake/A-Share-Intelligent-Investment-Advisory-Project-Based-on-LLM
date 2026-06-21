"""
线路管理器 — 管理14条实盘模拟线的状态、持仓、收益追踪。
"""
from typing import Dict, Any, List, Optional
from datetime import datetime


# 线路定义（总纲 §3）
LINE_DEFINITIONS = {
    # 短线实盘（10条）— strategy值对应factory.py的strategy_type参数
    "S-L0": {"term": "short", "type": "ablation_base", "agents": "all", "strategy": "ablation"},
    "S-L1": {"term": "short", "type": "ablation", "agents": "-fundamental", "strategy": "ablation"},
    "S-L2": {"term": "short", "type": "ablation", "agents": "-technical", "strategy": "ablation"},
    "S-L3": {"term": "short", "type": "ablation", "agents": "-value", "strategy": "ablation"},
    "S-L4": {"term": "short", "type": "ablation", "agents": "-news", "strategy": "ablation"},
    "S-L5": {"term": "short", "type": "ablation", "agents": "-event", "strategy": "ablation"},
    "S-L6": {"term": "short", "type": "ablation", "agents": "-quality_risk", "strategy": "ablation"},
    "S-L7": {"term": "short", "type": "ablation", "agents": "-moneyflow", "strategy": "ablation"},
    "S-L8": {"term": "short", "type": "longhold", "agents": "all", "strategy": "longhold"},
    "S-L9": {"term": "short", "type": "llm_free", "agents": "none", "strategy": "llm_free"},
    # 中线实盘（2条）
    "M-L0": {"term": "medium", "type": "full_agent", "agents": "all", "strategy": "default"},
    "M-L1": {"term": "medium", "type": "llm_free", "agents": "none", "strategy": "llm_free"},
    # 长线实盘（2条）
    "L-L0": {"term": "long", "type": "full_agent", "agents": "all", "strategy": "default"},
    "L-L1": {"term": "long", "type": "llm_free", "agents": "none", "strategy": "llm_free"},
}

# 可消融的agent列表
ABLATION_AGENTS = ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"]


class LineState:
    """单条线路的运行时状态"""
    def __init__(self, line_id: str, initial_capital: float = 1000000.0):
        self.line_id = line_id
        self.definition = LINE_DEFINITIONS.get(line_id, {})
        self.term = self.definition.get("term", "short")
        self.cash = initial_capital
        self.holdings: Dict[str, int] = {}       # stock_code -> shares
        self.purchase_prices: Dict[str, float] = {}  # stock_code -> avg price
        self.hold_days: Dict[str, int] = {}      # stock_code -> days held
        self.total_value = initial_capital
        self.daily_returns: List[float] = []
        self.cumulative_return = 0.0
        self.max_drawdown = 0.0
        self.peak_value = initial_capital
        self.trade_count = 0
        self.last_rebalance_date = ""
        self.status = "active"

    def update_value(self, current_prices: Dict[str, float]):
        """根据当前价格更新持仓市值"""
        holdings_value = 0.0
        for code, shares in self.holdings.items():
            price = current_prices.get(code, 0)
            holdings_value += shares * price
        self.total_value = self.cash + holdings_value

        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
        dd = (self.peak_value - self.total_value) / self.peak_value if self.peak_value > 0 else 0
        self.max_drawdown = max(self.max_drawdown, dd)

    def record_daily_return(self, prev_value: float):
        """记录当日收益"""
        if prev_value > 0:
            daily_ret = (self.total_value - prev_value) / prev_value
            self.daily_returns.append(daily_ret)
            self.cumulative_return = (self.total_value - 1000000.0) / 1000000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "term": self.term,
            "type": self.definition.get("type", ""),
            "cash": self.cash,
            "holdings_count": len(self.holdings),
            "holdings": dict(self.holdings),
            "total_value": self.total_value,
            "cumulative_return_pct": round(self.cumulative_return * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "trade_count": self.trade_count,
            "status": self.status,
        }


class LineManager:
    """14条线路的集中管理器"""

    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.lines: Dict[str, LineState] = {}
        self._init_lines()

    def _init_lines(self):
        """初始化所有14条实盘线"""
        for line_id in LINE_DEFINITIONS:
            self.lines[line_id] = LineState(line_id, self.initial_capital)

    def get_line(self, line_id: str) -> Optional[LineState]:
        return self.lines.get(line_id)

    def get_lines_by_term(self, term: str) -> List[LineState]:
        return [l for l in self.lines.values() if l.term == term]

    def get_ablation_lines(self, term: str = "short") -> List[LineState]:
        return [l for l in self.lines.values()
                if l.term == term and l.definition.get("type") in ("ablation_base", "ablation")]

    def get_ablation_base(self, term: str = "short") -> Optional[LineState]:
        for l in self.lines.values():
            if l.term == term and l.definition.get("type") == "ablation_base":
                return l
        return None

    def sync_ablation_holdings(self, term: str = "short"):
        """交叉截面同步：所有消融线持仓同步为基准线持仓"""
        base = self.get_ablation_base(term)
        if not base:
            return
        for line in self.get_ablation_lines(term):
            if line.line_id == base.line_id:
                continue
            line.holdings = dict(base.holdings)
            line.purchase_prices = dict(base.purchase_prices)
            line.hold_days = dict(base.hold_days)
            line.cash = base.cash
            line.total_value = base.total_value

    def update_all_values(self, current_prices: Dict[str, float]):
        """更新所有线的持仓市值"""
        prev_values = {lid: l.total_value for lid, l in self.lines.items()}
        for line in self.lines.values():
            line.update_value(current_prices)
            line.record_daily_return(prev_values.get(line.line_id, self.initial_capital))

    def get_all_status(self) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in self.lines.values()]

    def get_holdings_summary(self) -> Dict[str, Any]:
        """所有线的持仓摘要"""
        summary = {}
        for line in self.lines.values():
            summary[line.line_id] = {
                "term": line.term,
                "holdings": {k: v for k, v in line.holdings.items() if v > 0},
                "cash": line.cash,
                "total_value": line.total_value,
                "cumulative_return_pct": round(line.cumulative_return * 100, 2),
            }
        return summary
