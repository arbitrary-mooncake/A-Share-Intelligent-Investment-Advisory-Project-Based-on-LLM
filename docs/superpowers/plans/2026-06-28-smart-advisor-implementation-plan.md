# 智能投顾板块实现计划

> **For agentic workers:** 使用 `superpowers:subagent-driven-development` 逐任务实现。每个任务末尾的 `- [ ]` 用于跟踪进度。

**目标：** 为 A股智能投顾Agent助手新增第 6 大功能模块「智能投顾」，包含 AI 对话顾问、股票推荐、持仓管理、策略回测/模拟盘、DeepSeek 自由投资线对照、PDF 收益报告。

**架构：** 新建 `src/advisory/` 包承载所有后端逻辑，复用 `src/eval/market_simulator.py` 的交易模拟器、`src/utils/cache_utils.py` 的缓存体系、`src/tools/mcp_client.py` 的 MCP 工具接入、`src/qa/` 的对话引擎模式。前端新建 `07_智能投顾.py` 作为第 6 大功能模块入口。

**技术栈：** Python 3.11+, Streamlit, FastAPI, LangGraph, Tushare MCP, MiMo-V2.5-Pro (M1), DeepSeek V4 Pro (M6)

## 全局约束

- 智能投顾生产缓存只读 `data/intermediate_cache/`，**绝不**碰 `data/eval/cache/`（`_eval` 后缀）
- 股票推荐全程不触发任何分析 Agent（不调 `run_stock_analysis`）
- 策略执行纯 Python 代码，LLM 只参与策略文件生成阶段
- 调仓时序 = 收盘信号 → 次日开盘价执行（模型 3）
- DeepSeek 自由线只看原始 MCP 数据，不看任何 Agent 产物（总纲 §3.5）
- 所有新增文件遵循项目现有命名模式和代码风格
- 不修改现有功能的逻辑，只做必要的接口扩展

---

### 任务 1: 基础数据结构与枚举定义

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/__init__.py`
- 创建: `Finance/Financial-MCP-Agent/src/advisory/schemas.py`

**接口:**
- 产出: `AdvisoryPortfolio`, `PortfolioHolding`, `SettlementRecord`, `StrategyConfig`, `BacktestResult`, `SimulationStatus`, `HealthAssessment`, `PreferenceMatch` 等 dataclass

- [ ] **步骤 1: 编写 schemas.py**

```python
"""智能投顾板块 — 数据结构定义"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class HealthColor(str, Enum):
    DEEP_GREEN = "deep_green"
    LIGHT_GREEN = "light_green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class MatchLevel(str, Enum):
    HIGH = "high"
    BASIC = "basic"
    PARTIAL = "partial"
    DEVIATED = "deviated"


class TradeDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class PortfolioHolding:
    stock_code: str
    company_name: str = ""
    quantity: int = 0          # 股数
    cost_price: float = 0.0    # 成本价
    current_price: float = 0.0 # 现价
    market_value: float = 0.0  # 市值
    weight: float = 0.0        # 权重百分比


@dataclass
class AdvisoryPortfolio:
    portfolio_id: str
    user_id: str
    name: str
    holdings: Dict[str, PortfolioHolding] = field(default_factory=dict)
    total_cost: float = 0.0
    total_market_value: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    cash: float = 0.0          # 可用资金
    initial_capital: float = 100000.0
    created_at: str = ""
    updated_at: str = ""
    bound_strategy: Optional[str] = None  # 绑定策略名
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    status: str = "active"     # active / paused / closed


@dataclass
class StrategyConfig:
    name: str
    display_name: str
    category: str              # 均线 / 指标 / 布林 / 通道 / 震荡 / 复合 / 量价 / 风险 / 动量
    description: str
    params: Dict[str, Any] = field(default_factory=dict)
    is_custom: bool = False
    source_file: str = ""


@dataclass
class TradeRecord:
    date: str
    action: str                # BUY / SELL
    stock_code: str
    company_name: str = ""
    price: float = 0.0
    shares: int = 0
    commission: float = 0.0
    reason: str = ""           # 触发原因（策略信号 / 手动）


@dataclass
class SettlementRecord:
    date: str
    portfolio_id: str
    holdings_snapshot: Dict[str, PortfolioHolding] = field(default_factory=dict)
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)


@dataclass
class BacktestResult:
    portfolio_id: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualized_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    settlement_records: List[SettlementRecord] = field(default_factory=list)


@dataclass
class SimulationStatus:
    portfolio_id: str
    is_running: bool = False
    start_date: str = ""
    last_settlement_date: str = ""
    total_days: int = 0
    missed_days: int = 0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    pending_trades: List[TradeRecord] = field(default_factory=list)


@dataclass
class HealthAssessment:
    color: HealthColor
    color_label: str           # "组合稳健优秀" 等
    summary: str               # 简短文字说明
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceMatch:
    level: MatchLevel
    level_label: str           # "高度匹配" 等
    reason: str                # 一句话原因
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationItem:
    stock_code: str
    company_name: str
    reason: str                # 推荐原因
    has_score_cache: bool = False
    score: Optional[float] = None
    term: str = ""             # short / medium / long
    basic_info: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **步骤 2: 编写 `__init__.py`**

```python
"""智能投顾板块 — Advisory Module"""
```

- [ ] **步骤 3: 验证** — `python -c "from src.advisory.schemas import *; print('OK')"`
- [ ] **步骤 4: 提交**

```bash
git add Finance/Financial-MCP-Agent/src/advisory/__init__.py Finance/Financial-MCP-Agent/src/advisory/schemas.py
git commit -m "feat(advisory): add base data structures and enums"
```

---

### 任务 2: 用户画像系统

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/user_profile.py`
- 创建: `Finance/Financial-MCP-Agent/src/advisory/profile_tools.py`

**接口:**
- 产出: `UserProfileManager` 类（`load()`, `save()`, `update_profile()`, `get_profile_summary()`）
- 产出: `get_profile_manager()` 工厂函数
- 产出: `update_user_profile_tool()`, `get_user_profile_tool()` — MCP 工具函数

- [ ] **步骤 1: 编写 `user_profile.py`**

```python
"""用户画像管理 — JSON 持久化 + LangGraph State 注入"""
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


class UserProfileManager:
    def __init__(self, file_path: str = None):
        if file_path:
            self.file_path = file_path
        else:
            base = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "user_profiles"
            )
            os.makedirs(base, exist_ok=True)
            self.file_path = os.path.join(base, "default_profile.json")
        self.profile = self._load_profile()

    def _load_profile(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return self._default_profile()
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return self._default_profile()

    def _default_profile(self) -> Dict[str, Any]:
        return {
            "risk_tolerance": "Unknown",
            "investment_horizon": "Unknown",
            "favorite_sectors": [],
            "avoid_sectors": [],
            "investment_style": "",
            "custom_preferences": {},
            "updated_at": ""
        }

    def _save_profile(self):
        self.profile["updated_at"] = datetime.now().isoformat()
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.profile, f, ensure_ascii=False, indent=2)

    def update_profile(
        self,
        risk_tolerance: Optional[str] = None,
        investment_horizon: Optional[str] = None,
        favorite_sectors: Optional[list] = None,
        avoid_sectors: Optional[list] = None,
        investment_style: Optional[str] = None,
        **kwargs
    ):
        valid_risk = {"Conservative", "Balanced", "Aggressive", "Unknown"}
        valid_horizon = {"Short-term", "Medium-term", "Long-term", "Unknown"}
        if risk_tolerance and risk_tolerance in valid_risk:
            self.profile["risk_tolerance"] = risk_tolerance
        if investment_horizon and investment_horizon in valid_horizon:
            self.profile["investment_horizon"] = investment_horizon
        if favorite_sectors is not None:
            self.profile["favorite_sectors"] = favorite_sectors
        if avoid_sectors is not None:
            self.profile["avoid_sectors"] = avoid_sectors
        if investment_style:
            self.profile["investment_style"] = investment_style
        for k, v in kwargs.items():
            self.profile["custom_preferences"][k] = v
        self._save_profile()
        return "User profile updated successfully."

    def get_profile_summary(self) -> str:
        p = self.profile
        lines = [
            "User Profile:",
            f"- Risk Tolerance: {p.get('risk_tolerance', 'Unknown')}",
            f"- Investment Horizon: {p.get('investment_horizon', 'Unknown')}",
            f"- Favorite Sectors: {', '.join(p.get('favorite_sectors', []))}",
            f"- Avoid Sectors: {', '.join(p.get('avoid_sectors', []))}",
            f"- Investment Style: {p.get('investment_style', 'Not specified')}",
        ]
        custom = p.get("custom_preferences", {})
        if custom:
            lines.append("- Other Preferences:")
            for k, v in custom.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def get_profile(self) -> Dict[str, Any]:
        return self.profile

    def to_state_context(self) -> str:
        """生成可注入 LangGraph AgentState 的上下文字符串"""
        return f"[USER_PROFILE]\n{self.get_profile_summary()}\n[/USER_PROFILE]"
```

- [ ] **步骤 2: 编写 `profile_tools.py`**

```python
"""用户画像 MCP 工具 — LLM 可调用"""
import json
from src.advisory.user_profile import UserProfileManager

_profile_manager = None

def get_profile_manager():
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = UserProfileManager()
    return _profile_manager

def update_user_profile_tool(
    risk_tolerance: str = None,
    investment_horizon: str = None,
    favorite_sectors: list = None,
    avoid_sectors: list = None,
    investment_style: str = None,
    **kwargs
) -> str:
    """LLM 调用：更新用户投资画像"""
    return get_profile_manager().update_profile(
        risk_tolerance=risk_tolerance,
        investment_horizon=investment_horizon,
        favorite_sectors=favorite_sectors,
        avoid_sectors=avoid_sectors,
        investment_style=investment_style,
        **kwargs
    )

def get_user_profile_tool() -> str:
    """LLM 调用：获取当前用户画像"""
    return json.dumps(get_profile_manager().get_profile(), ensure_ascii=False, indent=2)

# Function Call 工具定义（注册到 MCP 工具集）
PROFILE_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "更新用户投资画像。当用户明确陈述偏好或你可以从对话中推断时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_tolerance": {
                        "type": "string",
                        "enum": ["Conservative", "Balanced", "Aggressive", "Unknown"],
                        "description": "风险偏好"
                    },
                    "investment_horizon": {
                        "type": "string",
                        "enum": ["Short-term", "Medium-term", "Long-term", "Unknown"],
                        "description": "投资期限"
                    },
                    "favorite_sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "偏好板块"
                    },
                    "avoid_sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "回避板块"
                    },
                    "investment_style": {
                        "type": "string",
                        "description": "投资风格描述"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "获取当前用户投资画像和偏好设置",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
```

- [ ] **步骤 3: 验证** — `python -c "from src.advisory.user_profile import UserProfileManager; m = UserProfileManager(); m.update_profile(risk_tolerance='Conservative'); print(m.get_profile_summary())"`
- [ ] **步骤 4: 提交**

```bash
git add Finance/Financial-MCP-Agent/src/advisory/user_profile.py Finance/Financial-MCP-Agent/src/advisory/profile_tools.py
git commit -m "feat(advisory): add user profile system with JSON persistence and MCP tools"
```

---

### 任务 3: 策略系统（基类 + 注册器 + 内置策略）

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/strategies/__init__.py`
- 创建: `Finance/Financial-MCP-Agent/src/advisory/strategies/strategy_base.py`
- 创建: `Finance/Financial-MCP-Agent/src/advisory/strategies/strategy_registry.py`
- 创建: `Finance/Financial-MCP-Agent/src/advisory/strategies/builtin/__init__.py`
- 创建: 内置策略文件（见步骤 3）

**接口:**
- 产出: `TradingStrategy` 抽象基类（`enrich()`, `signal()`, `risk_exit()`, `position_fraction()`）
- 产出: `StrategyRegistry` 单例 + `@register_strategy` 装饰器
- 产出: 20+ 内置策略类（全部注册）

- [ ] **步骤 1: 编写 `strategy_base.py`**

```python
"""策略基类 — 日线 bar、全仓买卖语义"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional
import pandas as pd


class TradingStrategy(ABC):
    name: str
    description: str = ""
    requires_multi_asset: bool = False

    @abstractmethod
    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        """在日线数据上添加指标列并返回"""

    @abstractmethod
    def signal(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        params: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """返回 1(买入) / -1(卖出) / 0(不动)"""

    def risk_exit(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        params: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> int:
        """持仓时优先判断止损/止盈，默认无"""
        return 0

    def position_fraction(self, params: Mapping[str, Any], context: Mapping[str, Any]) -> float:
        """仓位比例，默认满仓 1.0"""
        return float(params.get("position_fraction", 1.0))

    @classmethod
    def _na(cls, *values) -> bool:
        return any(pd.isna(v) for v in values)
```

- [ ] **步骤 2: 编写 `strategy_registry.py`**

```python
"""策略注册器 — 单例模式 + 装饰器注册"""
from typing import Dict, Type, Optional
from src.advisory.strategies.strategy_base import TradingStrategy


class StrategyRegistry:
    _instance: Optional["StrategyRegistry"] = None
    _strategies: Dict[str, Type[TradingStrategy]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, strategy_cls: Type[TradingStrategy]):
        instance = cls()
        instance._strategies[strategy_cls.name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[TradingStrategy]]:
        return cls()._strategies.get(name)

    @classmethod
    def list_all(cls) -> Dict[str, Type[TradingStrategy]]:
        return dict(cls()._strategies)

    @classmethod
    def list_names(cls) -> list:
        return sorted(cls()._strategies.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Optional[dict]:
        s_cls = cls.get(name)
        if s_cls is None:
            return None
        return {
            "name": s_cls.name,
            "description": s_cls.description,
            "requires_multi_asset": s_cls.requires_multi_asset,
        }


def register_strategy(cls: Type[TradingStrategy]):
    """注册策略装饰器"""
    StrategyRegistry.register(cls)
    return cls


def get_strategy_class(name: str) -> Optional[Type[TradingStrategy]]:
    return StrategyRegistry.get(name)
```

- [ ] **步骤 3: 移植 20+ 内置策略**

在每个策略文件开头包含 LLM 可读的 docstring。

**`ma_cross.py`**:
```python
"""双均线金叉买入、死叉卖出。
常用参数：short_window=5, long_window=20。
适用场景：趋势明确的市场，震荡市假信号较多。"""
from __future__ import annotations
from typing import Any, Mapping
import pandas as pd
from src.advisory.strategies.strategy_base import TradingStrategy
from src.advisory.strategies.strategy_registry import register_strategy

@register_strategy
class MACrossStrategy(TradingStrategy):
    name = "ma_cross"
    description = "双均线金叉买入、死叉卖出（默认5/20日）"

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        short = int(params.get("short_window", 5))
        long = int(params.get("long_window", 20))
        df["short_ma"] = df["close"].rolling(window=short).mean()
        df["long_ma"] = df["close"].rolling(window=long).mean()
        return df

    def signal(self, row, prev_row, params, context=None) -> int:
        if prev_row is None: return 0
        for k in ("short_ma", "long_ma"):
            if self._na(row.get(k), prev_row.get(k)): return 0
        if prev_row["short_ma"] <= prev_row["long_ma"] and row["short_ma"] > row["long_ma"]:
            return 1
        if prev_row["short_ma"] >= prev_row["long_ma"] and row["short_ma"] < row["long_ma"]:
            return -1
        return 0
```

**(其他 19+ 策略用相同模式 — `macd.py`, `rsi.py`, `kdj.py`, `boll.py`, `donchian.py`, `turtle.py`, `adx_macd.py`, `triple_ma.py`, `ema_sma_bias.py`, `cci.py`, `williams_r.py`, `stochastic.py`, `rsi_ma200.py`, `volume_breakout.py`, `obv_cross.py`, `vwap_deviation.py`, `ma_cross_atr_stop.py`, `vol_target_ma_cross.py`, `kelly_ma_cross.py`, `momentum_roc.py`)**

- [ ] **步骤 4: 编写 `builtin/__init__.py`** — 导入所有策略文件触发注册
- [ ] **步骤 5: 编写 `strategies/__init__.py`** — 导出 base, registry 和 builtin
- [ ] **步骤 6: 验证** — `python -c "from src.advisory.strategies import *; print(StrategyRegistry.list_names())"` 期望输出 20+ 策略名
- [ ] **步骤 7: 提交**

---

### 任务 4: 策略执行引擎

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/strategy_engine.py`

**接口:**
- 消费: `TradingStrategy`, `StrategyRegistry`（任务 3）
- 产出: `StrategyEngine.run_signal()`, `StrategyEngine.generate_strategy_code()`（LLM 辅助生成策略文件）

- [ ] **步骤 1: 编写 `strategy_engine.py`**

```python
"""策略执行引擎 — 收盘信号 → 次日开盘执行（模型 3）"""
from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from src.advisory.strategies.strategy_base import TradingStrategy
from src.advisory.strategies.strategy_registry import StrategyRegistry, get_strategy_class


class StrategyEngine:
    @staticmethod
    def compute_signal(
        strategy_name: str,
        df_daily: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, str]:
        """
        在日线数据上跑策略信号。

        Args:
            strategy_name: 策略名称
            df_daily: 升序排列的日线数据（含预热数据）
            params: 策略参数字典
            context: 当前持仓上下文 {position, entry_price, cash}

        Returns:
            (signal, reason): signal ∈ {1, -1, 0}, reason 为文字原因
        """
        if params is None:
            params = {}
        if context is None:
            context = {}

        strategy_cls = get_strategy_class(strategy_name)
        if strategy_cls is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy: TradingStrategy = strategy_cls()
        df = strategy.enrich(df.copy(), params)

        if len(df) < 2:
            return (0, "数据不足，无法计算信号")

        prev_row = df.iloc[-2]
        curr_row = df.iloc[-1]
        ctx = {
            "position": context.get("position", 0),
            "entry_price": context.get("entry_price"),
            "cash": context.get("cash", 0),
        }

        sig = 0
        reason = ""
        if ctx["position"] > 0:
            sig = strategy.risk_exit(curr_row, prev_row, params, ctx)
            if sig == -1:
                reason = f"{strategy_name}: 触发止损/止盈平仓"
        if sig == 0:
            sig = strategy.signal(curr_row, prev_row, params, ctx)
            if sig == 1:
                reason = f"{strategy_name}: 触发买入信号"
            elif sig == -1:
                reason = f"{strategy_name}: 触发卖出信号"

        return (sig, reason)

    @staticmethod
    def compute_position_fraction(
        strategy_name: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """计算建仓比例"""
        strategy_cls = get_strategy_class(strategy_name)
        if strategy_cls is None:
            return 1.0
        strategy = strategy_cls()
        return strategy.position_fraction(params or {}, context or {})

    @staticmethod
    def generate_strategy_code(
        strategy_name: str,
        user_description: str,
        base_strategy: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        生成策略 Python 源代码（LLM 辅助）。

        此方法不包含 LLM 调用——调用方在获取策略描述后调用 LLM 生成代码，
        然后将生成的代码写入 `data/strategies/custom/{strategy_name}.py`。
        返回代码模板供 LLM 填充。
        """
        base_cls = None
        if base_strategy:
            base_cls = get_strategy_class(base_strategy)

        template = f'''"""用户自定义策略: {strategy_name}
{user_description}
参数: {custom_params or "默认"}
"""
from __future__ import annotations
from typing import Any, Mapping, Optional
import pandas as pd
from src.advisory.strategies.strategy_base import TradingStrategy
from src.advisory.strategies.strategy_registry import register_strategy

@register_strategy
class {strategy_name.replace("_", " ").title().replace(" ", "")}Strategy(TradingStrategy):
    name = "{strategy_name}"
    description = """{user_description}"""

    def enrich(self, df: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
        # TODO: LLM 填充指标计算逻辑
        return df

    def signal(self, row, prev_row, params, context=None) -> int:
        # TODO: LLM 填充交易信号逻辑
        return 0
'''
        return template

    @staticmethod
    def get_strategy_catalog() -> list:
        """返回策略目录（供前端展示和 LLM 调用）"""
        catalog = []
        for name in StrategyRegistry.list_names():
            meta = StrategyRegistry.get_metadata(name)
            if meta:
                catalog.append({
                    "name": meta["name"],
                    "description": meta["description"],
                    "category": _guess_category(meta["name"]),
                })
        return catalog


def _guess_category(name: str) -> str:
    cats = {
        "ma_cross": "均线", "triple_ma": "均线", "ema_sma_bias": "均线",
        "macd": "指标", "rsi": "指标", "kdj": "指标",
        "boll_reversion": "布林", "boll_breakout": "布林",
        "donchian_breakout": "通道", "turtle": "通道",
        "cci": "震荡", "williams_r": "震荡", "stochastic": "震荡",
        "adx_macd": "复合", "rsi_ma200": "复合",
        "volume_breakout": "量价", "obv_cross": "量价", "vwap_deviation": "量价",
        "ma_cross_atr_stop": "风险", "vol_target_ma_cross": "风险", "kelly_ma_cross": "风险",
        "momentum_roc": "动量",
    }
    return cats.get(name, "其他")
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.strategy_engine import StrategyEngine; print(len(StrategyEngine.get_strategy_catalog()))"`
- [ ] **步骤 3: 提交**

---

### 任务 5: 股票推荐引擎

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/recommendation.py`

**接口:**
- 消费: `stock_pool.json`（精筛池入口）, `intermediate_cache/`（生产打分缓存）
- 产出: `RecommendationEngine.recommend(question, user_profile)` → `List[RecommendationItem]`

- [ ] **步骤 1: 编写 `recommendation.py`**

```python
"""股票推荐引擎 — n/x/5 规则 + 精筛池入口"""
import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.advisory.schemas import RecommendationItem


class RecommendationEngine:
    def __init__(self, pool_path: str = None, cache_dir: str = None):
        if pool_path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            pool_path = os.path.join(base, "stock_pool.json")
        self.pool_path = pool_path

        if cache_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            cache_dir = os.path.join(base, "data", "intermediate_cache")
        self.cache_dir = cache_dir

    def load_pool_stocks(self) -> List[Dict[str, Any]]:
        """加载精筛池全部股票（三个期限去重）"""
        if not os.path.exists(self.pool_path):
            return []
        with open(self.pool_path, 'r', encoding='utf-8') as f:
            pool_data = json.load(f)

        seen = set()
        stocks = []
        pools = pool_data.get("pools", {})
        for term in ("short", "medium", "long"):
            term_pool = pools.get(term, {})
            for code, info in term_pool.get("stocks", {}).items():
                if code not in seen:
                    seen.add(code)
                    stocks.append({
                        "stock_code": info.get("stock_code", code),
                        "company_name": info.get("company_name", ""),
                        "score": info.get("score"),
                        "term": term,
                        "term_score": info.get("term_score", {}),
                        "recommendation": info.get("recommendation", ""),
                        "detected_industry": info.get("term_score", {}).get("detected_industry", ""),
                    })
        return stocks

    def check_score_cache(self, stock_code: str) -> Optional[Any]:
        """检查生产打分缓存"""
        patterns = [
            f"scoring_{stock_code}_short_term_score",
            f"scoring_{stock_code}_medium_term_score",
            f"scoring_{stock_code}_long_term_score",
        ]
        for pattern in patterns:
            for fname in os.listdir(self.cache_dir) if os.path.exists(self.cache_dir) else []:
                if pattern in fname:
                    cache_path = os.path.join(self.cache_dir, fname)
                    try:
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        score_val = data.get("data", {}).get("score") or data.get("score")
                        return float(score_val) if score_val is not None else None
                    except Exception:
                        continue
        return None

    def check_signal_pack_cache(self, stock_code: str, agent: str) -> Optional[dict]:
        """检查某个 agent 的 signal_pack 缓存"""
        pattern = f"{agent}_{stock_code}_signal_pack"
        for fname in os.listdir(self.cache_dir) if os.path.exists(self.cache_dir) else []:
            if pattern in fname:
                cache_path = os.path.join(self.cache_dir, fname)
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    continue
        return None

    def apply_nx5_rule(
        self,
        candidate_codes: List[str],
        pool_stocks: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """n/x/5 规则：从候选股票中选出最多 5 只"""
        n = len(candidate_codes)
        if n <= 5:
            return candidate_codes

        # 查打分缓存
        scored = {}
        unscored = []
        for code in candidate_codes:
            s = self.check_score_cache(code)
            if s is not None:
                scored[code] = s
            else:
                unscored.append(code)

        x = len(scored)
        if x == 0:
            # 全部无缓存 → 返回前 n 只（由 LLM 做快速判断）
            # 这里返回 None 标记需要 LLM 自行挑选
            return ["__LLM_PICK__"] + candidate_codes
        elif x <= 5:
            return sorted(scored.keys(), key=lambda c: scored[c], reverse=True)
        else:
            return sorted(scored.keys(), key=lambda c: scored[c], reverse=True)[:5]

    def build_recommendation_context(
        self,
        stocks: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建供 LLM 使用的推荐上下文文本"""
        lines = ["## 精筛池可用股票\n"]
        for s in stocks:
            lines.append(
                f"- {s['stock_code']} | {s['company_name']} | "
                f"期限: {s['term']} | 行业: {s.get('detected_industry', '未知')} | "
                f"分数: {s.get('score', 'N/A')}"
            )
        if user_profile:
            lines.append(f"\n## 用户画像\n{user_profile.get('investment_style', '')}")
        return "\n".join(lines)


# 单例
_recommendation_engine: Optional[RecommendationEngine] = None

def get_recommendation_engine() -> RecommendationEngine:
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.recommendation import get_recommendation_engine; e = get_recommendation_engine(); print(len(e.load_pool_stocks()))"`
- [ ] **步骤 3: 提交**

---

### 任务 6: 持仓管理系统

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/portfolio_manager.py`

**接口:**
- 消费: `AdvisoryPortfolio`, `PortfolioHolding`（任务 1）
- 产出: `PortfolioManager` 类（CRUD + 市值实时计算 + MCP 工具函数）

- [ ] **步骤 1: 编写 `portfolio_manager.py`**

```python
"""持仓管理 — JSON 持久化 + 实时市值计算"""
import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.advisory.schemas import (
    AdvisoryPortfolio, PortfolioHolding, TradeRecord, TradeDirection,
)


class PortfolioManager:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            data_dir = os.path.join(base, "data", "portfolios")
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

    def _file_path(self, portfolio_id: str) -> str:
        return os.path.join(self.data_dir, f"{portfolio_id}.json")

    def create(self, name: str, user_id: str = "default",
               initial_capital: float = 100000.0) -> AdvisoryPortfolio:
        portfolio_id = f"pf_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        pf = AdvisoryPortfolio(
            portfolio_id=portfolio_id,
            user_id=user_id,
            name=name,
            initial_capital=initial_capital,
            cash=initial_capital,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self.save(pf)
        return pf

    def load(self, portfolio_id: str) -> Optional[AdvisoryPortfolio]:
        path = self._file_path(portfolio_id)
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pf = AdvisoryPortfolio(**data)
        # 重建 holdings
        hd = {}
        for code, h in data.get("holdings", {}).items():
            hd[code] = PortfolioHolding(**h)
        pf.holdings = hd
        return pf

    def save(self, pf: AdvisoryPortfolio):
        pf.updated_at = datetime.now().isoformat()
        data = {
            "portfolio_id": pf.portfolio_id,
            "user_id": pf.user_id,
            "name": pf.name,
            "holdings": {k: v.__dict__ for k, v in pf.holdings.items()},
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
        with open(self._file_path(pf.portfolio_id), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_all(self, user_id: str = "default") -> List[AdvisoryPortfolio]:
        portfolios = []
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".json"):
                pf = self.load(fname.replace(".json", ""))
                if pf and pf.user_id == user_id:
                    portfolios.append(pf)
        return portfolios

    def add_holding(
        self, pf: AdvisoryPortfolio, stock_code: str, company_name: str,
        quantity: int, price: float,
    ) -> Tuple[AdvisoryPortfolio, TradeRecord]:
        cost = quantity * price
        comm = cost * 0.0003  # 佣金 0.03%
        total_cost = cost + comm
        if total_cost > pf.cash:
            raise ValueError(f"资金不足：需要 {total_cost:.2f}，可用 {pf.cash:.2f}")

        rounded_qty = (quantity // 100) * 100
        if rounded_qty <= 0:
            raise ValueError("买入数量需 >= 100 股且为 100 的整数倍")

        if stock_code in pf.holdings:
            h = pf.holdings[stock_code]
            total_qty = h.quantity + rounded_qty
            h.cost_price = ((h.cost_price * h.quantity) + (price * rounded_qty)) / total_qty
            h.quantity = total_qty
        else:
            pf.holdings[stock_code] = PortfolioHolding(
                stock_code=stock_code, company_name=company_name,
                quantity=rounded_qty, cost_price=price,
            )

        actual_cost = rounded_qty * price + (rounded_qty * price * 0.0003)
        pf.cash -= actual_cost
        self._recalc_holdings(pf)
        trade = TradeRecord(
            date=datetime.now().strftime("%Y%m%d"),
            action="BUY", stock_code=stock_code, company_name=company_name,
            price=price, shares=rounded_qty, commission=comm,
        )
        self.save(pf)
        return pf, trade

    def remove_holding(
        self, pf: AdvisoryPortfolio, stock_code: str,
        quantity: int, price: float,
    ) -> Tuple[AdvisoryPortfolio, TradeRecord]:
        if stock_code not in pf.holdings:
            raise ValueError(f"不持有 {stock_code}")
        h = pf.holdings[stock_code]
        if quantity > h.quantity:
            raise ValueError(f"持仓不足：{h.quantity} 股，卖出 {quantity} 股")
        revenue = quantity * price
        comm = revenue * 0.0003
        stamp = revenue * 0.001   # 印花税 0.1%
        net = revenue - comm - stamp
        pf.cash += net
        if quantity == h.quantity:
            del pf.holdings[stock_code]
        else:
            h.quantity -= quantity
        self._recalc_holdings(pf)
        trade = TradeRecord(
            date=datetime.now().strftime("%Y%m%d"),
            action="SELL", stock_code=stock_code, company_name=h.company_name,
            price=price, shares=quantity, commission=comm,
        )
        self.save(pf)
        return pf, trade

    def _recalc_holdings(self, pf: AdvisoryPortfolio):
        pf.total_cost = 0.0
        pf.total_market_value = 0.0
        total_value = pf.cash
        for code, h in pf.holdings.items():
            h.market_value = h.quantity * (h.current_price or h.cost_price)
            pf.total_market_value += h.market_value
            pf.total_cost += h.quantity * h.cost_price
            total_value += h.market_value
        for code, h in pf.holdings.items():
            h.weight = (h.market_value / total_value * 100) if total_value > 0 else 0
        pf.total_pnl = pf.total_market_value - pf.total_cost
        pf.total_pnl_pct = (pf.total_pnl / pf.total_cost * 100) if pf.total_cost > 0 else 0

    def bind_strategy(self, pf: AdvisoryPortfolio, strategy_name: str, params: dict = None):
        pf.bound_strategy = strategy_name
        pf.strategy_params = params or {}
        self.save(pf)

    def unbind_strategy(self, pf: AdvisoryPortfolio):
        pf.bound_strategy = None
        pf.strategy_params = {}
        self.save(pf)
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.portfolio_manager import PortfolioManager; m = PortfolioManager(); pf = m.create('test'); print(pf.portfolio_id)"`
- [ ] **步骤 3: 提交**

---

### 任务 7: 组合健康度 + 偏好匹配度评估

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/portfolio_assessor.py`

**接口:**
- 消费: `AdvisoryPortfolio` + MCP 工具（LLM 侧调用）
- 产出: `HealthAssessment`, `PreferenceMatch`
- 此模块提供**提示词模板**和**数据准备**，LLM 调用时机由 AI 顾问控制

- [ ] **步骤 1: 编写 `portfolio_assessor.py`**

```python
"""组合评估 — 健康度五色 + 偏好匹配度"""
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from src.advisory.schemas import (
    AdvisoryPortfolio, HealthAssessment, HealthColor,
    PreferenceMatch, MatchLevel,
)
from src.advisory.recommendation import get_recommendation_engine


# ── 健康度评估提示词 ──
HEALTH_ASSESSMENT_PROMPT = """你是专业的投资组合分析师。请对以下持仓组合进行健康度评估。

## 持仓数据
{portfolio_data}

## 可用参考
{score_cache_info}
{signal_pack_info}

## 评估维度
1. 行业集中度：是否过度集中于某一行业
2. 单股权重：单只股票占比是否过高（>30%需关注，>50%严重警告）
3. 估值合理性：PE/PB 是否在合理区间
4. 风险分散：高/中/低风险资产搭配
5. 个股质量：财务健康度、盈利能力

## 输出要求
仅输出一种颜色评级（深绿/浅绿/黄色/橙色/红色）和简短文字说明（不超过 80 字）。
不要输出具体分数。

颜色含义：
- 深绿：组合稳健优秀
- 浅绿：组合良好
- 黄色：组合一般，存在明显短板
- 橙色：需要关注，多处风险信号
- 红色：存在显著风险

按以下 JSON 格式输出：
{{"color": "deep_green", "summary": "..."}}
"""

PREFERENCE_MATCH_PROMPT = """请对比用户画像和持仓组合，评估匹配度。

## 用户画像
{user_profile}

## 持仓概要
{portfolio_summary}

## 输出
仅输出一个匹配等级和一句话原因：
- high（高度匹配）
- basic（基本匹配）
- partial（部分偏离）
- deviated（显著偏离）

JSON 格式：
{{"level": "high", "reason": "..."}}
"""


class PortfolioAssessor:
    def __init__(self):
        self.rec_engine = get_recommendation_engine()

    def prepare_health_context(self, pf: AdvisoryPortfolio) -> Dict[str, Any]:
        """准备健康度评估所需的所有数据（供 LLM 调用）"""
        holdings_info = []
        score_info = []
        signal_info = []

        for code, h in pf.holdings.items():
            holdings_info.append({
                "code": h.stock_code,
                "name": h.company_name,
                "quantity": h.quantity,
                "cost_price": h.cost_price,
                "current_price": h.current_price,
                "market_value": h.market_value,
                "weight": f"{h.weight:.1f}%",
            })
            # 查打分缓存
            s = self.rec_engine.check_score_cache(code)
            if s is not None:
                score_info.append(f"{h.stock_code}: 打分缓存={s}")
            # 查 signal_pack
            for agent in ["fundamental", "value", "quality_risk"]:
                sp = self.rec_engine.check_signal_pack_cache(code, agent)
                if sp:
                    signal_info.append(f"{h.stock_code}/{agent}: bias={sp.get('bias')}, confidence={sp.get('confidence')}")

        holdings_text = json.dumps(holdings_info, ensure_ascii=False, indent=2)
        score_text = "\n".join(score_info) if score_info else "无可用打分缓存"
        signal_text = "\n".join(signal_info) if signal_info else "无可用 signal_pack 缓存"

        prompt = HEALTH_ASSESSMENT_PROMPT.format(
            portfolio_data=holdings_text,
            score_cache_info=score_text,
            signal_pack_info=signal_text,
        )
        return {"prompt": prompt, "holdings_count": len(pf.holdings)}

    def prepare_preference_context(
        self, pf: AdvisoryPortfolio, user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """准备偏好匹配度评估数据"""
        pf_summary = {
            "total_value": pf.total_market_value + pf.cash,
            "holding_count": len(pf.holdings),
            "holdings": [
                {"code": h.stock_code, "name": h.company_name, "weight": f"{h.weight:.1f}%"}
                for h in pf.holdings.values()
            ],
            "main_sectors": self._guess_sectors(pf),
        }
        prompt = PREFERENCE_MATCH_PROMPT.format(
            user_profile=json.dumps(user_profile, ensure_ascii=False, indent=2),
            portfolio_summary=json.dumps(pf_summary, ensure_ascii=False, indent=2),
        )
        return {"prompt": prompt}

    def _guess_sectors(self, pf: AdvisoryPortfolio) -> List[str]:
        """简单行业推测（实际由 LLM 通过 MCP 获取精确数据）"""
        sectors = []
        for h in pf.holdings.values():
            code = h.stock_code
            if "sh.51" in code or "sz.159" in code:
                sectors.append("ETF")
            elif "sh.60" in code or "sh.68" in code:
                sectors.append("上海主板/科创板")
            else:
                sectors.append("深圳")
        return list(set(sectors))
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.portfolio_assessor import PortfolioAssessor; print('OK')"`
- [ ] **步骤 3: 提交**

---

### 任务 8: 回测执行器

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/backtest_runner.py`

**接口:**
- 消费: `StrategyEngine`, `market_simulator.py`
- 产出: `BacktestResult`

- [ ] **步骤 1: 编写 `backtest_runner.py`**

```python
"""回测执行器 — 日线历史重放，模型 3 调仓时序"""
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from src.advisory.schemas import (
    AdvisoryPortfolio, BacktestResult, TradeRecord, SettlementRecord,
    PortfolioHolding,
)
from src.advisory.strategy_engine import StrategyEngine
from src.advisory.portfolio_manager import PortfolioManager


class BacktestRunner:
    def __init__(self, portfolio_manager: PortfolioManager = None):
        self.pm = portfolio_manager or PortfolioManager()
        self.engine = StrategyEngine()

    def run(
        self,
        pf: AdvisoryPortfolio,
        start_date: str,
        end_date: str,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> BacktestResult:
        """
        在历史数据上回放策略或手动持仓。

        Args:
            pf: 持仓组合（含初始持仓和资金）
            start_date: YYYYMMDD 起始
            end_date: YYYYMMDD 结束
            strategy_name: 策略名（None=只跑手动持仓不调仓）
            strategy_params: 策略参数
        """
        # 拉取历史日线
        df = self._fetch_daily_data(list(pf.holdings.keys()), start_date, end_date)
        if df.empty:
            raise ValueError("回测数据为空")

        cash = pf.cash
        position = {code: h.quantity for code, h in pf.holdings.items()}
        entry_prices = {code: h.cost_price for code, h in pf.holdings.items()}

        trades: List[TradeRecord] = []
        settlements: List[SettlementRecord] = []
        equity_curve: List[Dict] = []

        trading_dates = sorted(df["trade_date"].unique())
        initial_value = cash + sum(
            position[code] * entry_prices.get(code, 0) for code in position
        )

        prev_row = None
        for i, date in enumerate(trading_dates):
            if date < start_date:
                prev_row = df[df["trade_date"] == date].iloc[0] if not df[df["trade_date"] == date].empty else None
                continue

            day_data = df[df["trade_date"] == date]
            if day_data.empty:
                continue
            row = day_data.iloc[0]
            close_price = row["close"]

            # 计算当日总市值
            total_value = cash
            for code, qty in position.items():
                code_data = df[(df["trade_date"] == date) & (df["ts_code"] == code)]
                if not code_data.empty:
                    total_value += qty * code_data.iloc[0]["close"]

            equity_curve.append({
                "trade_date": str(date),
                "value": round(total_value, 2),
                "return_pct": round((total_value / initial_value - 1) * 100, 4),
            })

            # 策略信号
            day_trades = []
            if strategy_name and prev_row is not None:
                for code in list(position.keys()):
                    code_df = df[df["ts_code"] == code].sort_values("trade_date")
                    code_df = code_df[code_df["trade_date"] <= date]
                    if len(code_df) < 2:
                        continue
                    ctx = {"position": position.get(code, 0), "cash": cash}
                    sig, reason = self.engine.compute_signal(
                        strategy_name, code_df, strategy_params, ctx,
                    )
                    if sig != 0:
                        curr_row = code_df.iloc[-1]
                        price = curr_row["close"]  # 收盘价计算信号
                        # 次日开盘价执行
                        next_date = trading_dates[i+1] if i+1 < len(trading_dates) else None
                        if next_date is None:
                            continue
                        next_data = df[(df["trade_date"] == next_date) & (df["ts_code"] == code)]
                        if next_data.empty:
                            continue
                        exec_price = next_data.iloc[0]["open"]
                        if sig == 1 and cash > 0:
                            max_shares = int(cash / (exec_price * 1.0003) / 100) * 100
                            if max_shares > 0:
                                cost = max_shares * exec_price * 1.0003
                                cash -= cost
                                position[code] = position.get(code, 0) + max_shares
                                day_trades.append(TradeRecord(
                                    date=str(next_date), action="BUY", stock_code=code,
                                    price=exec_price, shares=max_shares,
                                    reason=reason,
                                ))
                        elif sig == -1 and position.get(code, 0) > 0:
                            qty = position[code]
                            revenue = qty * exec_price * (1 - 0.0013)
                            cash += revenue
                            position[code] = 0
                            day_trades.append(TradeRecord(
                                date=str(next_date), action="SELL", stock_code=code,
                                price=exec_price, shares=qty,
                                reason=reason,
                            ))

            trades.extend(day_trades)
            prev_row = row

        # 计算最终指标
        final_value = cash + sum(
            position[code] * df[df["ts_code"] == code].iloc[-1]["close"]
            for code in position if not df[df["ts_code"] == code].empty
        )
        total_return = (final_value - initial_value) / initial_value
        max_dd = self._calc_max_drawdown(equity_curve)

        return BacktestResult(
            portfolio_id=pf.portfolio_id,
            strategy_name=strategy_name or "手动持仓",
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_value,
            final_value=final_value,
            total_return_pct=round(total_return * 100, 2),
            max_drawdown_pct=round(max_dd * 100, 2),
            trade_count=len(trades),
            equity_curve=equity_curve,
            trades=trades[-20:],
        )

    def _fetch_daily_data(self, stock_codes: List[str], start: str, end: str) -> pd.DataFrame:
        """从 Tushare 拉取日线数据"""
        from datetime import datetime as dt
        
        warmup_start = (dt.strptime(start, "%Y%m%d") - timedelta(days=400)).strftime("%Y%m%d")
        all_dfs = []
        for code in stock_codes:
            try:
                import tushare as ts
                from dotenv import load_dotenv
                load_dotenv()
                ts.set_token(os.getenv("TUSHARE_TOKEN", ""))
                pro = ts.pro_api()
                df = pro.daily(ts_code=code, start_date=warmup_start, end_date=end)
                if not df.empty:
                    all_dfs.append(df)
            except Exception:
                continue
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True).sort_values("trade_date")

    def _calc_max_drawdown(self, equity_curve: List[Dict]) -> float:
        if not equity_curve:
            return 0.0
        values = [e["value"] for e in equity_curve]
        cummax = values[0]
        max_dd = 0.0
        for v in values:
            if v > cummax:
                cummax = v
            dd = (cummax - v) / cummax
            if dd > max_dd:
                max_dd = dd
        return max_dd
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.backtest_runner import BacktestRunner; print('OK')"`
- [ ] **步骤 3: 提交**

---

### 任务 9: 模拟盘执行器 + 追赶机制

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/simulation_runner.py`
- 创建: `Finance/Financial-MCP-Agent/src/advisory/catch_up.py`

**接口:**
- 产出: `SimulationRunner` 类
- 产出: `catch_up(cutoff_date)` 自动追赶

- [ ] **步骤 1: 编写 `catch_up.py`**

```python
"""自动追赶机制 — 进入板块触发，纯 Python 秒级完成"""
import os
import json
from datetime import datetime, timedelta
from typing import List, Optional, Tuple


def get_trading_days(start: str, end: str) -> List[str]:
    """获取区间内交易日列表"""
    try:
        import tushare as ts
        from dotenv import load_dotenv
        load_dotenv()
        ts.set_token(os.getenv("TUSHARE_TOKEN", ""))
        pro = ts.pro_api()
        df = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
        if df.empty:
            return []
        return sorted(df[df["is_open"] == 1]["cal_date"].tolist())
    except Exception:
        days = []
        d = datetime.strptime(start, "%Y%m%d")
        end_d = datetime.strptime(end, "%Y%m%d")
        while d <= end_d:
            if d.weekday() < 5:
                days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        return days


class CatchUpDetector:
    def __init__(self, settlement_dir: str = None):
        if settlement_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            settlement_dir = os.path.join(base, "data", "advisory_settlements")
        os.makedirs(settlement_dir, exist_ok=True)
        self.settlement_dir = settlement_dir

    def get_last_settlement_date(self, portfolio_id: str) -> Optional[str]:
        pf_dir = os.path.join(self.settlement_dir, portfolio_id)
        if not os.path.exists(pf_dir):
            return None
        files = sorted([f for f in os.listdir(pf_dir) if f.endswith(".json")])
        if not files:
            return None
        return files[-1].replace(".json", "")

    def get_missed_days(self, portfolio_id: str, today: str) -> Tuple[int, List[str]]:
        last = self.get_last_settlement_date(portfolio_id)
        if last is None:
            return (0, [])
        if last >= today:
            return (0, [])

        trading_days = get_trading_days(last, today)
        missed = [d for d in trading_days if d > last and d <= today]
        return (len(missed), missed)

    def catch_up(
        self, portfolio_id: str, today: str,
        runner_fn,  # simulation runner 的逐日执行函数
    ) -> dict:
        """执行追赶：逐日回放策略信号 + 开盘价结算"""
        n, missed_days = self.get_missed_days(portfolio_id, today)
        if n == 0:
            return {"missed": 0, "message": "无需追赶"}

        results = []
        for day in missed_days:
            try:
                r = runner_fn(portfolio_id, day)
                results.append({"date": day, "status": "ok", **r})
            except Exception as e:
                results.append({"date": day, "status": "error", "error": str(e)})

        return {
            "missed": n,
            "days": missed_days,
            "results": results,
            "message": f"已自动追赶 {n} 个交易日（{missed_days[0]} → {missed_days[-1]}）",
        }
```

- [ ] **步骤 2: 编写 `simulation_runner.py`**

```python
"""模拟盘执行器 — 每日收盘后运行，策略自动/手动待机"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from src.advisory.portfolio_manager import PortfolioManager
from src.advisory.strategy_engine import StrategyEngine
from src.advisory.schemas import AdvisoryPortfolio, TradeRecord
from src.advisory.catch_up import CatchUpDetector


class SimulationRunner:
    def __init__(self, portfolio_manager: PortfolioManager = None):
        self.pm = portfolio_manager or PortfolioManager()
        self.engine = StrategyEngine()
        self.catch_up_detector = CatchUpDetector()

    def run_daily_settlement(self, portfolio_id: str, date: str = None) -> Dict[str, Any]:
        """执行单日结算"""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        pf = self.pm.load(portfolio_id)
        if pf is None:
            return {"error": "Portfolio not found"}

        trades = []
        if pf.bound_strategy:
            for code, h in list(pf.holdings.items()):
                df = self._fetch_single_stock(code, date)
                if df is None or len(df) < 2:
                    continue
                ctx = {"position": h.quantity, "entry_price": h.cost_price, "cash": pf.cash}
                sig, reason = self.engine.compute_signal(
                    pf.bound_strategy, df, pf.strategy_params, ctx,
                )
                if sig == 1 and pf.cash > 0:
                    price = df.iloc[-1]["close"]
                    max_shares = int(pf.cash / (price * 1.0003) / 100) * 100
                    if max_shares > 0:
                        pf, trade = self.pm.add_holding(pf, code, h.company_name, max_shares, price)
                        trades.append(trade)
                elif sig == -1 and h.quantity > 0:
                    pf, trade = self.pm.remove_holding(pf, code, h.quantity, df.iloc[-1]["close"])
                    trades.append(trade)

        # 更新现价并重算
        for code, h in pf.holdings.items():
            df = self._fetch_single_stock(code, date)
            if df is not None and not df.empty:
                h.current_price = df.iloc[-1]["close"]
        self.pm._recalc_holdings(pf)
        self.pm.save(pf)
        self._save_settlement(portfolio_id, date, pf, trades)
        return {"date": date, "total_value": pf.total_market_value + pf.cash,
                "trades": len(trades), "return_pct": pf.total_pnl_pct}

    def run_catch_up(self, portfolio_id: str):
        """自动追赶"""
        today = datetime.now().strftime("%Y%m%d")
        return self.catch_up_detector.catch_up(
            portfolio_id, today,
            lambda pid, d: self.run_daily_settlement(pid, d),
        )

    def _fetch_single_stock(self, code: str, date: str):
        try:
            import tushare as ts
            from dotenv import load_dotenv
            load_dotenv()
            ts.set_token(os.getenv("TUSHARE_TOKEN", ""))
            pro = ts.pro_api()
            warmup = (datetime.strptime(date, "%Y%m%d") - __import__('datetime').timedelta(days=400)).strftime("%Y%m%d")
            df = pro.daily(ts_code=code, start_date=warmup, end_date=date)
            return df.sort_values("trade_date") if not df.empty else None
        except Exception:
            return None

    def _save_settlement(self, portfolio_id: str, date: str, pf: AdvisoryPortfolio, trades: list):
        base = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        d = os.path.join(base, "data", "advisory_settlements", portfolio_id)
        os.makedirs(d, exist_ok=True)
        record = {
            "date": date,
            "total_value": pf.total_market_value + pf.cash,
            "cash": pf.cash,
            "holdings": {code: h.__dict__ for code, h in pf.holdings.items()},
            "trades": [t.__dict__ for t in trades],
            "return_pct": pf.total_pnl_pct,
        }
        with open(os.path.join(d, f"{date}.json"), 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
```

- [ ] **步骤 3: 验证** — `python -c "from src.advisory.simulation_runner import SimulationRunner; print('OK')"`
- [ ] **步骤 4: 提交**

---

### 任务 10: DeepSeek 自由投资线

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/llm_free_line.py`

**接口:**
- 消费: MCP 原始数据（严格无 Agent 产物）
- 产出: DeepSeekFreeLine 类（每日决策 + 持仓记录）

- [ ] **步骤 1: 编写 `llm_free_line.py`**

```python
"""DeepSeek 自由投资线 — 纯 MCP 数据自主决策，对照基准线"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.advisory.schemas import TradeRecord


FREELINE_SYSTEM_PROMPT = """你是独立的投资决策 AI，基于原始市场数据自主决定持仓。

## 可用数据
只可使用以下工具获取数据：
- Tushare MCP 工具（行情/财务/资金流/板块/宏观等全部）
- AKShare MCP 工具（国际宏观/商品）

## 禁止查看
- 任何 Agent 的 analysis 文本
- 任何 signal_pack / score / 精筛池排名
- 用户的持仓/策略/偏好
- 本项目生产的任何中间产物

## 决策规则
1. 每次收盘后根据当日数据决定明日持仓
2. 单只股票仓位不超过总资金 30%
3. 至少持有 3 只不同行业股票分散风险
4. 输出调仓指令 JSON
"""


class DeepSeekFreeLine:
    def __init__(self, initial_capital: float = 100000.0,
                 storage_dir: str = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[TradeRecord] = []
        self.start_date = datetime.now().strftime("%Y%m%d")

        if storage_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            storage_dir = os.path.join(base, "data", "advisory_llm_free")
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir

    def get_system_prompt(self) -> str:
        return FREELINE_SYSTEM_PROMPT

    def get_context(self) -> Dict[str, Any]:
        """获取当前持仓上下文（塞入 LLM prompt）"""
        total_value = self.cash
        for code, pos in self.positions.items():
            total_value += pos["quantity"] * pos.get("current_price", pos["cost_price"])
        return {
            "cash": self.cash,
            "positions": self.positions,
            "total_value": total_value,
            "initial_capital": self.initial_capital,
            "total_return_pct": round((total_value / self.initial_capital - 1) * 100, 2),
            "date": datetime.now().strftime("%Y%m%d"),
        }

    def execute_decision(self, decisions: List[Dict[str, Any]]):
        """执行调仓决策（次日开盘价）"""
        for d in decisions:
            action = d.get("action", "")
            code = d.get("stock_code", "")
            qty = int(d.get("quantity", 0))
            price = float(d.get("price", 0))
            if action == "buy" and qty > 0 and price > 0:
                cost = qty * price * 1.0003
                if cost <= self.cash:
                    self.cash -= cost
                    if code in self.positions:
                        p = self.positions[code]
                        total_qty = p["quantity"] + qty
                        p["cost_price"] = (p["cost_price"] * p["quantity"] + price * qty) / total_qty
                        p["quantity"] = total_qty
                    else:
                        self.positions[code] = {"quantity": qty, "cost_price": price}
                    self.trades.append(TradeRecord(
                        date=datetime.now().strftime("%Y%m%d"),
                        action="BUY", stock_code=code, price=price, shares=qty,
                        reason="DeepSeek 自由线决策",
                    ))
            elif action == "sell" and qty > 0 and code in self.positions:
                p = self.positions[code]
                sell_qty = min(qty, p["quantity"])
                revenue = sell_qty * price * (1 - 0.0013)
                self.cash += revenue
                p["quantity"] -= sell_qty
                if p["quantity"] <= 0:
                    del self.positions[code]
                self.trades.append(TradeRecord(
                    date=datetime.now().strftime("%Y%m%d"),
                    action="SELL", stock_code=code, price=price, shares=sell_qty,
                    reason="DeepSeek 自由线决策",
                ))

    def save_state(self):
        path = os.path.join(self.storage_dir, "state.json")
        state = {
            "cash": self.cash,
            "positions": self.positions,
            "initial_capital": self.initial_capital,
            "start_date": self.start_date,
            "trades": [t.__dict__ for t in self.trades[-50:]],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> bool:
        path = os.path.join(self.storage_dir, "state.json")
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        self.cash = state.get("cash", self.initial_capital)
        self.positions = state.get("positions", {})
        self.start_date = state.get("start_date", self.start_date)
        return True
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.llm_free_line import DeepSeekFreeLine; dl = DeepSeekFreeLine(); print(dl.get_context())"`
- [ ] **步骤 3: 提交**

---

### 任务 11: PDF 收益分析报告

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/advisory/report_generator.py`

**接口:**
- 消费: `BacktestResult`, `SimulationStatus`, `DeepSeekFreeLine`
- 产出: PDF 文件路径

- [ ] **步骤 1: 编写 `report_generator.py`**

```python
"""PDF 持仓收益分析报告生成器"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional


class AdvisoryReportGenerator:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            output_dir = os.path.join(base, "data", "reports")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def generate_comparison_chart(
        self,
        user_equity: list,
        deepseek_equity: Optional[list] = None,
        benchmark_equity: Optional[list] = None,
        title: str = "收益对比图",
    ) -> str:
        """生成收益率对比图，返回图片路径"""
        fig, ax = plt.subplots(figsize=(12, 6))

        def plot_curve(data, label, color, linestyle="-"):
            if not data:
                return
            dates = [d.get("trade_date", d.get("date", "")) for d in data]
            values = [d.get("return_pct", d.get("value", 0)) for d in data]
            if dates and values:
                ax.plot(range(len(values)), values, label=label,
                        color=color, linestyle=linestyle, linewidth=2)

        plot_curve(user_equity, "用户持仓", "#2196F3", "-")
        if deepseek_equity:
            plot_curve(deepseek_equity, "DeepSeek 自由线", "#FF9800", "--")

        ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("收益率 (%)", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        chart_path = os.path.join(self.output_dir, f"chart_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        return chart_path

    def build_report_prompt(
        self,
        report_type: str,  # "backtest" | "simulation"
        user_summary: Dict[str, Any],
        deepseek_summary: Optional[Dict[str, Any]] = None,
        chart_paths: Optional[list] = None,
    ) -> str:
        """构建 LLM 报告生成提示词（供 DeepSeek V4 Pro 使用）"""
        prompt = f"""你是专业的投资分析报告撰写 AI。请根据以下数据生成持仓收益分析报告。

## 报告类型
{report_type}

## 用户持仓数据
总收益率: {user_summary.get('total_return_pct', 'N/A')}%
最大回撤: {user_summary.get('max_drawdown_pct', 'N/A')}%
交易次数: {user_summary.get('trade_count', 'N/A')}

## 格式要求
1. 收益概览
2. 持仓变化时间线
3. DeepSeek 自由线对比（如有）
4. AI 综合评价
5. 改进建议

严格标注数据来源：[数据] = 确定数据，[判断] = 分析推断。
不要编造数据。
"""
        if deepseek_summary:
            prompt += f"""
## DeepSeek 自由线对比
收益率: {deepseek_summary.get('total_return_pct', 'N/A')}%
最大回撤: {deepseek_summary.get('max_drawdown_pct', 'N/A')}%
"""
        return prompt

    def save_report(self, content: str, title: str) -> str:
        """保存报告 Markdown 文件，返回路径"""
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{title}.md"
        path = os.path.join(self.output_dir, fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path
```

- [ ] **步骤 2: 验证** — `python -c "from src.advisory.report_generator import AdvisoryReportGenerator; print('OK')"`
- [ ] **步骤 3: 提交**

---

### 任务 12: FastAPI 后端端点

**文件:**
- 修改: `Finance/Financial-MCP-Agent/src/api/app.py`（新增 8+ 端点）

**接口:**
- 新增端点：推荐、持仓 CRUD、策略目录/绑定、回测/模拟盘触发、DeepSeek 自由线、PDF 报告

- [ ] **步骤 1: 在 `app.py` 中添加 Pydantic 请求模型**

```python
# 在现有 Pydantic 模型下方新增

class AdvisoryRecommendRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class PortfolioCreateRequest(BaseModel):
    name: str
    initial_capital: float = 100000.0

class HoldingModifyRequest(BaseModel):
    portfolio_id: str
    stock_code: str
    company_name: str = ""
    quantity: int
    price: float
    action: str  # "buy" | "sell"

class StrategyBindRequest(BaseModel):
    portfolio_id: str
    strategy_name: str
    params: Optional[Dict[str, Any]] = None

class BacktestRequest(BaseModel):
    portfolio_id: str
    start_date: str
    end_date: str
    strategy_name: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None

class SimulationRequest(BaseModel):
    portfolio_id: str
    action: str  # "start" | "stop" | "status" | "catch_up"

class FreeLineRequest(BaseModel):
    portfolio_id: str
    action: str  # "start" | "status" | "decide"

class ReportRequest(BaseModel):
    portfolio_id: str
    report_type: str  # "backtest" | "simulation"
    start_date: str = ""
    end_date: str = ""
    include_deepseek: bool = False
```

- [ ] **步骤 2: 在 `app.py` 中添加端点实现**

```python
# ─── 智能投顾端点 ───

from src.advisory.recommendation import get_recommendation_engine
from src.advisory.portfolio_manager import PortfolioManager
from src.advisory.strategy_engine import StrategyEngine
from src.advisory.backtest_runner import BacktestRunner
from src.advisory.simulation_runner import SimulationRunner
from src.advisory.llm_free_line import DeepSeekFreeLine
from src.advisory.report_generator import AdvisoryReportGenerator
from src.advisory.user_profile import UserProfileManager
from src.advisory.profile_tools import get_profile_manager
from src.advisory.portfolio_assessor import PortfolioAssessor

# 初始化
_rec_engine = None
_portfolio_mgr = None
_backtest_runner = None
_sim_runner = None
_report_gen = None
_assessor = None

def _init_advisory():
    global _rec_engine, _portfolio_mgr, _backtest_runner
    global _sim_runner, _report_gen, _assessor
    if _rec_engine is None:
        _rec_engine = get_recommendation_engine()
        _portfolio_mgr = PortfolioManager()
        _backtest_runner = BacktestRunner(_portfolio_mgr)
        _sim_runner = SimulationRunner(_portfolio_mgr)
        _report_gen = AdvisoryReportGenerator()
        _assessor = PortfolioAssessor()


@app.post("/api/advisory/recommend")
async def advisory_recommend(req: AdvisoryRecommendRequest):
    """股票推荐 — n/x/5 规则"""
    _init_advisory()
    stocks = _rec_engine.load_pool_stocks()
    context = _rec_engine.build_recommendation_context(stocks)
    return {"status": "ok", "pool_size": len(stocks), "context": context}


@app.post("/api/advisory/portfolio/create")
async def advisory_portfolio_create(req: PortfolioCreateRequest):
    """创建持仓组合"""
    _init_advisory()
    pf = _portfolio_mgr.create(req.name, initial_capital=req.initial_capital)
    return {"status": "ok", "portfolio_id": pf.portfolio_id}


@app.get("/api/advisory/portfolio/list")
async def advisory_portfolio_list():
    """列出所有持仓组合"""
    _init_advisory()
    pfs = _portfolio_mgr.list_all()
    return {
        "status": "ok",
        "portfolios": [
            {
                "portfolio_id": pf.portfolio_id,
                "name": pf.name,
                "total_value": pf.total_market_value + pf.cash,
                "return_pct": pf.total_pnl_pct,
                "holding_count": len(pf.holdings),
                "bound_strategy": pf.bound_strategy,
            }
            for pf in pfs
        ],
    }


@app.get("/api/advisory/portfolio/{portfolio_id}")
async def advisory_portfolio_get(portfolio_id: str):
    """获取组合详情"""
    _init_advisory()
    pf = _portfolio_mgr.load(portfolio_id)
    if pf is None:
        raise HTTPException(404, "Portfolio not found")
    return {"status": "ok", "portfolio": pf.__dict__}


@app.post("/api/advisory/portfolio/holding")
async def advisory_holding_modify(req: HoldingModifyRequest):
    """买入/卖出持仓"""
    _init_advisory()
    pf = _portfolio_mgr.load(req.portfolio_id)
    if pf is None:
        raise HTTPException(404, "Portfolio not found")
    try:
        if req.action == "buy":
            pf, trade = _portfolio_mgr.add_holding(
                pf, req.stock_code, req.company_name, req.quantity, req.price)
        else:
            pf, trade = _portfolio_mgr.remove_holding(
                pf, req.stock_code, req.quantity, req.price)
        return {"status": "ok", "trade": trade.__dict__}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/advisory/strategies")
async def advisory_strategies():
    """获取策略目录"""
    return {"status": "ok", "strategies": StrategyEngine.get_strategy_catalog()}


@app.post("/api/advisory/strategy/bind")
async def advisory_strategy_bind(req: StrategyBindRequest):
    """绑定/解绑策略"""
    _init_advisory()
    pf = _portfolio_mgr.load(req.portfolio_id)
    if pf is None:
        raise HTTPException(404, "Portfolio not found")
    if req.strategy_name:
        _portfolio_mgr.bind_strategy(pf, req.strategy_name, req.params)
    else:
        _portfolio_mgr.unbind_strategy(pf)
    return {"status": "ok"}


@app.post("/api/advisory/backtest")
async def advisory_backtest(req: BacktestRequest):
    """运行回测"""
    _init_advisory()
    pf = _portfolio_mgr.load(req.portfolio_id)
    if pf is None:
        raise HTTPException(404, "Portfolio not found")
    result = _backtest_runner.run(
        pf, req.start_date, req.end_date,
        req.strategy_name, req.strategy_params,
    )
    return {"status": "ok", "result": result.__dict__}


@app.post("/api/advisory/simulation")
async def advisory_simulation(req: SimulationRequest):
    """模拟盘控制"""
    _init_advisory()
    if req.action == "catch_up":
        result = _sim_runner.run_catch_up(req.portfolio_id)
    elif req.action == "status":
        pf = _portfolio_mgr.load(req.portfolio_id)
        result = {
            "is_running": pf.status == "active" if pf else False,
            "total_return": pf.total_pnl_pct if pf else 0,
            "last_date": pf.updated_at if pf else "",
        }
    elif req.action == "settle":
        result = _sim_runner.run_daily_settlement(req.portfolio_id)
    else:
        raise HTTPException(400, f"Unknown action: {req.action}")
    return {"status": "ok", "result": result}


@app.post("/api/advisory/report")
async def advisory_report(req: ReportRequest):
    """生成分析报告"""
    _init_advisory()
    pf = _portfolio_mgr.load(req.portfolio_id)
    if pf is None:
        raise HTTPException(404, "Portfolio not found")
    summary = {"total_return_pct": pf.total_pnl_pct, "trade_count": 0}
    prompt = _report_gen.build_report_prompt(req.report_type, summary)
    return {"status": "ok", "report_prompt": prompt}


@app.get("/api/advisory/user-profile")
async def get_user_profile():
    return {"status": "ok", "profile": get_profile_manager().get_profile()}
```

- [ ] **步骤 2: 验证** — 启动 FastAPI 并测试端点可访问
- [ ] **步骤 3: 提交**

---

### 任务 13: 前端页面

**文件:**
- 创建: `Finance/Financial-MCP-Agent/src/app/pages/07_智能投顾.py`
- 修改: `Finance/Financial-MCP-Agent/src/app/Home.py`（侧边栏导航）

**接口:**
- Streamlit 6 个子页面（单选按钮切换）+ 右侧 AI 顾问面板

- [ ] **步骤 1: 编写 `07_智能投顾.py`**

```python
"""
智能投顾 — 第 6 大功能模块
右侧常驻 AI 顾问 + 左侧子页面内容
子页面：AI顾问对话 / 股票推荐 / 我的持仓 / 策略市场 / 回测&模拟盘 / 收益报告
"""
import os, sys
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st

st.set_page_config(page_title="智能投顾", page_icon="🤖", layout="wide")

# ── 子页面路由 ──
PAGES = {
    "ai_chat": "💬 AI 顾问对话",
    "recommend": "📊 股票推荐",
    "portfolio": "💰 我的持仓",
    "strategies": "📈 策略市场",
    "backtest": "📉 回测 & 模拟盘",
    "report": "📄 收益报告",
}

if "advisory_page" not in st.session_state:
    st.session_state.advisory_page = "ai_chat"

# ── Layout ──
left, right = st.columns([3, 2])

with left:
    page_choice = st.radio(
        "智能投顾",
        list(PAGES.keys()),
        format_func=lambda x: PAGES[x],
        key="advisory_nav",
        horizontal=True,
    )
    st.session_state.advisory_page = page_choice

    if page_choice == "ai_chat":
        st.subheader("💬 智能投顾 AI 顾问")
        st.info("在右侧 AI 顾问面板中提问，我会根据你的持仓和偏好提供个性化建议。")
        # 复用智能问答的聊天 UI
        if "advisory_messages" not in st.session_state:
            st.session_state.advisory_messages = []
        for msg in st.session_state.advisory_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("请输入你的问题..."):
            st.session_state.advisory_messages.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                st.markdown(f"*AI 顾问处理中...* (问题: {prompt[:50]}...)")

    elif page_choice == "recommend":
        st.subheader("📊 股票推荐")
        if st.button("刷新推荐"):
            st.info("从精筛池中按用户偏好推荐股票...")

    elif page_choice == "portfolio":
        st.subheader("💰 我的持仓")
        st.info("在此管理你的持仓组合、查看组合健康度和偏好匹配度")

    elif page_choice == "strategies":
        st.subheader("📈 策略市场")
        st.info("拖拽策略卡片到持仓组合，或在右侧 AI 顾问中用自然语言定制策略")

    elif page_choice == "backtest":
        st.subheader("📉 回测 & 模拟盘")
        st.info("选择持仓组合和策略运行回测，或开启模拟盘每日跟踪")

    elif page_choice == "report":
        st.subheader("📄 收益报告")
        st.info("生成持仓收益分析报告（PDF），可对比 DeepSeek 自由投资线")

# ── 右侧 AI 顾问面板 ──
with right:
    st.markdown("### 🤖 AI 顾问")
    st.caption(f"当前页面: {PAGES.get(st.session_state.advisory_page, '')}")
    st.divider()
    # 上下文说明
    st.info(f"AI 顾问知道你在查看「{PAGES.get(st.session_state.advisory_page, '')}」，可直接就当前页面内容提问。")
    st.caption("支持：股票推荐 · 持仓管理 · 策略定制 · 回测/模拟盘控制")
```

- [ ] **步骤 2: 修改 `Home.py` 侧边栏**

在 `Home.py` 的导航列表中加入智能投顾入口（在现有 6 项之后添加智能投顾链接）。

- [ ] **步骤 3: 验证** — 启动 Streamlit 并测试页面渲染
- [ ] **步骤 4: 提交**

---

### 任务 14: 智能问答引导 + 集成

**文件:**
- 修改: `Finance/Financial-MCP-Agent/src/app/pages/04_智能问答.py`（添加引导检测）
- 修改: `Finance/Financial-MCP-Agent/src/qa/answer_generator.py`（引导提示词）

**接口:**
- 检测到智能投顾专属问题 → AI 回复引导移步信息

- [ ] **步骤 1: 在 `answer_generator.py` 的 system prompt 中加入引导规则**

```python
GUIDANCE_RULES = """
## 功能边界
如果用户提出的问题涉及以下内容，请友好地引导用户移步「智能投顾」模块：
- 持仓组合管理（"帮我建一个持仓""我的持仓怎么样"）
- 策略配置（"给我的持仓加个策略""用什么策略好"）
- 回测/模拟盘（"帮我回测一下""模拟一下这个组合"）
- 调仓操作（"帮我卖出""我要买入"）
- DeepSeek 自由线对比

引导回复模板：
"此问题涉及{功能类型}，属于**智能投顾**模块的服务范围。请移步侧边栏「**智能投顾**」板块，我在那里等你，可以为你做更深入的分析 👋"
"""
```

- [ ] **步骤 2: 修改 `04_智能问答.py`** — 在 system prompt 中注入引导规则
- [ ] **步骤 3: 验证** — 在智能问答中输入"帮我建一个持仓" → 应收到引导回复
- [ ] **步骤 4: 提交**

---

### 任务 15: 第 1-2 轮代码审查 — 语法与运行检查

**本轮目标：** 确保所有新代码无语法错误，能正常导入和运行基础功能。

- [ ] **步骤 1: 全量语法检查**
```bash
cd Finance/Financial-MCP-Agent
python -m py_compile src/advisory/schemas.py src/advisory/user_profile.py \
  src/advisory/profile_tools.py src/advisory/strategies/strategy_base.py \
  src/advisory/strategies/strategy_registry.py
# 对所有新建 .py 文件执行
```

- [ ] **步骤 2: 导入链完整性检查**
```bash
python -c "
from src.advisory.schemas import *
from src.advisory.user_profile import *
from src.advisory.profile_tools import *
from src.advisory.strategies.strategy_base import *
from src.advisory.strategies.strategy_registry import *
from src.advisory.strategies import builtin
from src.advisory.strategy_engine import *
from src.advisory.recommendation import *
from src.advisory.portfolio_manager import *
from src.advisory.portfolio_assessor import *
from src.advisory.backtest_runner import *
from src.advisory.simulation_runner import *
from src.advisory.catch_up import *
from src.advisory.llm_free_line import *
from src.advisory.report_generator import *
print('All imports OK')
"
```

- [ ] **步骤 3: 策略系统完整性检查**
```bash
python -c "
from src.advisory.strategies.strategy_registry import StrategyRegistry
names = StrategyRegistry.list_names()
print(f'Registered strategies: {len(names)}')
assert len(names) >= 20, f'Expected 20+ strategies, got {len(names)}'
print('All strategies registered OK')
"
```

- [ ] **步骤 4: FastAPI 端点注册检查** — 启动后端，检查 `/docs` 端点列表是否包含所有新端点

---

### 任务 16: 第 3-4 轮代码审查 — 功能正确性验证

**本轮目标：** 检查隐藏的逻辑 bug、边界条件、设计约束是否符合设计文档。

- [ ] **步骤 1: 缓存隔离验证** — 确认所有缓存访问指向 `intermediate_cache/`，无任何代码引用 `eval/cache/`
```bash
grep -r "eval/cache" src/advisory/ && echo "FAIL: eval cache reference found" || echo "PASS"
grep -r "_eval" src/advisory/ && echo "FAIL: _eval suffix reference found" || echo "PASS"
```

- [ ] **步骤 2: Agent 调用隔离验证** — 确认推荐链路不调用 Agent
```bash
grep -r "run_stock_analysis\|scoring_engine\|ScoringEngine" src/advisory/ && echo "FAIL" || echo "PASS"
```

- [ ] **步骤 3: 调仓时序验证** — 确认所有策略执行走"收盘信号 → 次日开盘执行"（模型 3）
- [ ] **步骤 4: DeepSeek 自由线数据隔离验证** — 确认 prompt 不含任何 Agent 产物引用
- [ ] **步骤 5: n/x/5 规则逻辑验证** — 确认 `recommendation.py` 中的逻辑覆盖所有分支
- [ ] **步骤 6: 追赶机制边界验证** — 测试 missed_days=0, missed_days<7, missed_days>30 三种场景
- [ ] **步骤 7: 组合健康度提示词验证** — 确认不含打分输出格式
- [ ] **步骤 8: 策略全自动/全手动二分验证** — 确认无半自动逻辑

---

### 任务 17: 第 5 轮代码审查 — 回归语法检查

**本轮目标：** 确保前两轮修改没有引入新的语法问题。

- [ ] **步骤 1: 全量编译检查** — 对所有修改/创建的文件重新 `py_compile`
- [ ] **步骤 2: 全量导入检查** — 重新执行任务 15 步骤 2 的导入链测试
- [ ] **步骤 3: 现有功能回归检查**
```bash
python -c "from src.main import *; print('main imports OK')"
python -c "from src.stock_pool.scoring_engine import *; print('scoring imports OK')"
python -c "from src.eval.market_simulator import *; print('market_simulator imports OK')"
```
- [ ] **步骤 4: Streamlit 页面导入检查** — 确认 `07_智能投顾.py` 能正常导入
- [ ] **步骤 5: 提交最终版本**

---

## 实施优先级

| 优先级 | 任务 | 可并行？ |
|--------|------|---------|
| P0 | 任务 1 (schemas) → 任务 2 (用户画像) + 任务 3 (策略系统) | 任务 2,3 可并行 |
| P1 | 任务 4 (策略引擎) → 任务 5 (推荐) + 任务 6 (持仓) | 任务 5,6 可并行 |
| P2 | 任务 7 (评估) + 任务 8 (回测) + 任务 9 (模拟盘) | 7,8,9 可并行 |
| P3 | 任务 10 (自由线) + 任务 11 (报告) + 任务 12 (API) | 10,11,12 可并行 |
| P4 | 任务 13 (前端) + 任务 14 (引导) | 13,14 可并行 |
| Review | 任务 15 (1-2轮) → 任务 16 (3-4轮) → 任务 17 (5轮) | 顺序执行 |

---

## 预估工作量

- 新创建文件：~35 个
- 修改现有文件：~4 个
- 新增代码行（含策略）：~5000 行
- 核心实现时间：任务 1-14 约 4-6 小时
- 代码审查时间：任务 15-17 约 2-3 小时
