"""
DeepSeek 自由投资线 — 总纲 §3.5 控制组。

DeepSeek V4 Pro 完全自主决策买卖，仅能访问原始 MCP 数据，
禁止访问任何 Agent 分析产物（analysis、signal_pack、score、精筛池）。

设计意图:
  LLM Free 对照线用于评测系统中衡量"AI 能否独立于 Agent 体系实现超额收益"。
  与依赖 Agent 评分的策略管线完全隔离，仅凭原始行情/财务/宏观数据自主判断。

架构位置:
  src/advisory/llm_free_line.py
  与 eval/strategies/llm_free.py 同属 LLM Free 体系，但职责不同:
    - eval/strategies/llm_free.py: 评测策略层，集成在 LineManager 中运行
    - src/advisory/llm_free_line.py: 独立投资线实例，可脱离评测系统单独运行
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FREELINE_SYSTEM_PROMPT — 核心约束定义
# ---------------------------------------------------------------------------

FREELINE_SYSTEM_PROMPT = """你是一位独立的 A 股投资决策者，管理一条自由投资线。

## 核心约束（总纲 §3.5）— 必须遵守

1. 【数据隔离】你只能调用原始 MCP 数据工具（行情、财务、宏观、资金流等）。
   严禁访问、引用或依赖以下 Agent 产出数据:
   - ❌ agent_analysis（任何分析文本）
   - ❌ agent_signal_pack（信号包 JSON）
   - ❌ agent_score（评分数值）
   - ❌ analysis_package（合并分析包）
   - ❌ 精筛池 / refined pool / 精选池
   - ❌ 用户持仓 / 用户策略 / 用户偏好

2. 【原始数据优先】仅基于以下类型的原始市场数据做决策:
   - 日线行情（open, close, high, low, volume, amount）
   - 估值数据（PE, PB, PS, 市值）
   - 财务指标（ROE, 毛利率, 增长率）
   - 资金流向（主力净流入, 北向资金）
   - 新闻/公告（原始标题+摘要）
   - 宏观指标（CPI, PMI, GDP, 利率）
   - 行业板块数据

3. 【独立判断】不得引用外部评分、评级或分析结论。
   所有判断必须来自你对原始数据的独立分析。

4. 【交易规范】
   - A 股买入以 100 股为整数倍
   - 佣金: 万分之三（0.03%）买卖双向
   - 印花税: 千分之一（0.1%）仅卖出时收取
   - 考虑流动性约束（涨停/跌停/停牌的股票不可交易）

## 输出格式

当你做出决策时，必须输出以下 JSON 格式。不要添加额外文字。

```json
{
  "reasoning": "决策逻辑简要说明（1-3句话）",
  "decisions": [
    {
      "action": "buy | sell",
      "stock_code": "股票代码",
      "quantity": 股数（买入须为100的整数倍）,
      "price": 预期成交价,
      "reason": "该笔交易的理由"
    }
  ]
}
```

如果不进行任何交易，输出:
```json
{
  "reasoning": "不交易的原因说明",
  "decisions": []
}
```"""

# ---------------------------------------------------------------------------
# 辅助函数: 提取 JSON
# ---------------------------------------------------------------------------

_JSON_PATTERNS = [
    # ```json ... ``` 代码块
    (re.compile(r'```json\s*(.*?)\s*```', re.DOTALL), 1),
    # ``` ... ``` 任意代码块
    (re.compile(r'```\s*(.*?)\s*```', re.DOTALL), 1),
    # 第一个 { ... } 对象（含嵌套）
    (re.compile(r'\{.*\}', re.DOTALL), 0),
]


def _extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON 对象。多重策略降级。"""
    if not text:
        return {}

    text = text.strip()

    # 策略 1-3: 正则提取
    for pattern, group_idx in _JSON_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return json.loads(match.group(group_idx))
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    # 策略 4: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error("_extract_json: 无法从响应中提取 JSON: %s", text[:300])
        return {}


# ---------------------------------------------------------------------------
# DeepSeekFreeLine
# ---------------------------------------------------------------------------

class DeepSeekFreeLine:
    """DeepSeek 自由投资线 — 总纲 §3.5 控制组。

    DeepSeek V4 Pro 独立决策，仅访问原始 MCP 数据。
    持有独立资金、仓位和交易记录，与主流 Agent 管线完全隔离。

    Usage:
        from src.advisory.llm_free_line import DeepSeekFreeLine

        line = DeepSeekFreeLine(initial_capital=100000.0)
        context = line.get_context()
        print(context["cash"])  # 100000.0
    """

    # 交易费率（与 PortfolioManager 一致）
    COMMISSION_RATE: float = 0.0003  # 佣金 0.03%（买卖双向）
    STAMP_TAX_RATE: float = 0.001  # 印花税 0.1%（仅卖出）
    LOT_SIZE: int = 100  # A 股 100 股整数倍

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(
        self,
        initial_capital: float = 100000.0,
        storage_dir: Optional[str] = None,
    ) -> None:
        """初始化 DeepSeek 自由投资线。

        Args:
            initial_capital: 初始资金，默认 100,000 元。
            storage_dir: 状态持久化目录。
                         默认 ``data/advisory_llm_free/``（相对于项目根目录）。
        """
        self.initial_capital: float = initial_capital
        self.cash: float = initial_capital
        self.positions: Dict[str, Dict[str, float]] = {}
        self.trades: List[TradeRecord] = []

        # storage_dir 默认相对于项目根（Finance/Financial-MCP-Agent/）
        if storage_dir is None:
            root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            storage_dir = os.path.join(root, "data", "advisory_llm_free")
        self._storage_dir: str = storage_dir
        os.makedirs(self._storage_dir, exist_ok=True)

        # LLM 客户端（DeepSeek V4 Pro）
        self._llm_client = self._init_llm_client()

    def _init_llm_client(self):
        """初始化 DeepSeek V4 Pro LLM 客户端。"""
        try:
            from src.utils.llm_clients import OpenAICompatibleClient
            from src.utils.model_config import get_eval_model_config, get_thinking_body

            model_config = get_eval_model_config("eval_llm_free")
            base_url = model_config["base_url"]
            thinking_body = get_thinking_body(base_url, enabled=True)

            client = OpenAICompatibleClient(
                api_key=model_config["api_key"],
                base_url=base_url,
                model=model_config["model_name"],
                env_prefix="",
                extra_body=thinking_body,
                http_timeout=600,
                http_connect_timeout=30,
            )
            logger.info(
                "DeepSeekFreeLine: LLM 客户端初始化成功 "
                "(model=%s)", model_config["model_name"]
            )
            return client
        except Exception as e:
            logger.warning("DeepSeekFreeLine: LLM 客户端初始化失败: %s", e)
            return None

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_context(self) -> Dict[str, Any]:
        """返回当前投资上下文。

        Returns:
            {
                "cash": float,
                "positions": {stock_code: {quantity, cost_price, current_price}, ...},
                "holdings_count": int,
                "total_market_value": float,
                "total_cost": float,
                "total_pnl": float,
                "total_value": float,  // cash + total_market_value
                "return_rate": float,
            }
        """
        total_market_value = 0.0
        total_cost = 0.0

        for code, pos in self.positions.items():
            qty = pos.get("quantity", 0)
            cost_price = pos.get("cost_price", 0.0)
            current_price = pos.get("current_price", 0.0)
            total_market_value += qty * current_price
            total_cost += qty * cost_price

        total_value = self.cash + total_market_value
        pnl = total_market_value - total_cost
        return_rate = (
            (total_value - self.initial_capital) / self.initial_capital * 100
            if self.initial_capital > 0
            else 0.0
        )

        return {
            "cash": round(self.cash, 2),
            "positions": dict(self.positions),
            "holdings_count": len(self.positions),
            "total_market_value": round(total_market_value, 2),
            "total_cost": round(total_cost, 2),
            "total_pnl": round(pnl, 2),
            "total_value": round(total_value, 2),
            "return_rate": round(return_rate, 2),
        }

    def get_system_prompt(self) -> str:
        """返回 FREELINE_SYSTEM_PROMPT。

        该 prompt 明确禁止 LLM 访问任何 Agent 产物数据，
        只允许使用原始 MCP 数据工具做自主决策。
        """
        return FREELINE_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # 交易执行
    # ------------------------------------------------------------------

    def execute_decision(
        self,
        decisions: List[Dict[str, Any]],
        trade_date: Optional[str] = None,
    ) -> List[TradeRecord]:
        """执行 LLM 返回的决策列表。

        对每个决策执行买入/卖出操作，含费率计算、资金校验、
        仓位更新和交易记录生成。

        Args:
            decisions: 决策列表，每项格式:
                {
                    "action": "buy" | "sell",
                    "stock_code": str,
                    "quantity": int,
                    "price": float,
                    "reason": str,
                }
            trade_date: 交易日期（YYYY-MM-DD）。为 None 时使用当前 UTC 日期。

        Returns:
            成功执行的 TradeRecord 列表。
        """
        if not decisions:
            return []

        if trade_date is None:
            trade_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        executed: List[TradeRecord] = []

        for decision in decisions:
            action = (decision.get("action") or "").strip().lower()
            stock_code = (decision.get("stock_code") or "").strip()
            raw_qty = int(decision.get("quantity", 0))
            price = float(decision.get("price", 0))
            reason = (decision.get("reason") or "").strip()

            if not stock_code or price <= 0 or raw_qty <= 0:
                logger.warning("execute_decision: 跳过无效决策: %s", decision)
                continue

            if action == "buy":
                record = self._execute_buy(stock_code, raw_qty, price, reason, trade_date)
                if record:
                    executed.append(record)
            elif action == "sell":
                record = self._execute_sell(stock_code, raw_qty, price, reason, trade_date)
                if record:
                    executed.append(record)
            else:
                logger.warning("execute_decision: 未知操作 '%s'，跳过", action)

        self.save_state()
        return executed

    # ------------------------------------------------------------------
    # 内部交易执行
    # ------------------------------------------------------------------

    def _execute_buy(
        self,
        stock_code: str,
        quantity: int,
        price: float,
        reason: str,
        trade_date: str,
    ) -> Optional[TradeRecord]:
        """执行买入交易。

        Args:
            stock_code: 股票代码。
            quantity: 请求买入数量（自动向下取整至 100 的倍数）。
            price: 成交单价。
            reason: 买入原因。
            trade_date: 交易日期。

        Returns:
            成功时返回 TradeRecord，失败时返回 None。
        """
        # 1. 100 股整数倍
        quantity = (quantity // self.LOT_SIZE) * self.LOT_SIZE
        if quantity <= 0:
            logger.warning("买入数量必须至少为 %d 股", self.LOT_SIZE)
            return None

        # 2. 费用计算
        cost = quantity * price
        commission = round(cost * self.COMMISSION_RATE, 2)
        total_required = cost + commission

        # 3. 资金校验
        if self.cash < total_required:
            logger.warning(
                "买入 %s 资金不足: 需要 %.2f, 可用 %.2f",
                stock_code, total_required, self.cash,
            )
            return None

        # 4. 扣减现金
        self.cash = round(self.cash - total_required, 2)

        # 5. 更新持仓（均价法）
        if stock_code in self.positions:
            pos = self.positions[stock_code]
            old_cost = pos["quantity"] * pos["cost_price"]
            new_qty = pos["quantity"] + quantity
            pos["cost_price"] = round((old_cost + cost) / new_qty, 4)
            pos["quantity"] = new_qty
            pos["current_price"] = price
        else:
            self.positions[stock_code] = {
                "quantity": quantity,
                "cost_price": price,
                "current_price": price,
            }

        # 6. 交易记录
        record = TradeRecord(
            date=trade_date,
            action="buy",
            stock_code=stock_code,
            price=price,
            shares=quantity,
            commission=commission,
            reason=reason or f"买入 {stock_code} {quantity}股，成交价 {price:.2f}",
        )
        self.trades.append(record)
        logger.info(
            "买入 %s: %d股 @ %.2f, 佣金 %.2f, 剩余现金 %.2f",
            stock_code, quantity, price, commission, self.cash,
        )
        return record

    def _execute_sell(
        self,
        stock_code: str,
        quantity: int,
        price: float,
        reason: str,
        trade_date: str,
    ) -> Optional[TradeRecord]:
        """执行卖出交易。

        Args:
            stock_code: 股票代码。
            quantity: 卖出数量。
            price: 成交单价。
            reason: 卖出原因。
            trade_date: 交易日期。

        Returns:
            成功时返回 TradeRecord，失败时返回 None。
        """
        # 1. 校验持仓
        if stock_code not in self.positions:
            logger.warning("卖出 %s 失败: 无此持仓", stock_code)
            return None

        pos = self.positions[stock_code]
        if quantity > pos["quantity"]:
            logger.warning(
                "卖出 %s 失败: 请求 %d股, 持有 %d股",
                stock_code, quantity, pos["quantity"],
            )
            return None

        # 2. 费用计算
        proceeds = quantity * price
        commission = round(proceeds * self.COMMISSION_RATE, 2)
        stamp_tax = round(proceeds * self.STAMP_TAX_RATE, 2)
        net = round(proceeds - commission - stamp_tax, 2)

        # 3. 增加现金
        self.cash = round(self.cash + net, 2)

        # 4. 更新/删除持仓
        if quantity >= pos["quantity"]:
            del self.positions[stock_code]
        else:
            pos["quantity"] -= quantity
            pos["current_price"] = price

        # 5. 交易记录
        record = TradeRecord(
            date=trade_date,
            action="sell",
            stock_code=stock_code,
            price=price,
            shares=quantity,
            commission=commission,
            reason=reason or f"卖出 {stock_code} {quantity}股，成交价 {price:.2f}",
        )
        self.trades.append(record)
        logger.info(
            "卖出 %s: %d股 @ %.2f, 佣金 %.2f, 印花税 %.2f, 净入金 %.2f",
            stock_code, quantity, price, commission, stamp_tax, net,
        )
        return record

    # ------------------------------------------------------------------
    # 位置更新（盘后结算用）
    # ------------------------------------------------------------------

    def update_price(self, stock_code: str, current_price: float) -> bool:
        """更新某只持仓股票的当前价格（用于每日结算）。

        Args:
            stock_code: 股票代码。
            current_price: 最新收盘价。

        Returns:
            更新成功返回 True，无此持仓返回 False。
        """
        if stock_code not in self.positions:
            return False
        self.positions[stock_code]["current_price"] = current_price
        return True

    # ------------------------------------------------------------------
    # 状态持久化
    # ------------------------------------------------------------------

    def _state_file_path(self) -> str:
        """返回 state.json 的完整路径。"""
        return os.path.join(self._storage_dir, "state.json")

    def save_state(self) -> None:
        """将当前状态持久化到 state.json。

        保存字段:
        - initial_capital
        - cash
        - positions: {stock_code: {quantity, cost_price, current_price}}
        - trades: TradeRecord 列表
        - updated_at: ISO 时间戳
        """
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": self.positions,
            "trades": [asdict(t) for t in self.trades],
            "updated_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        file_path = self._state_file_path()
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            logger.info("状态已保存至 %s", file_path)
        except OSError as e:
            logger.error("保存状态失败 %s: %s", file_path, e)

    def load_state(self) -> bool:
        """从 state.json 恢复状态。

        Returns:
            成功加载返回 True，文件不存在或损坏返回 False。
        """
        file_path = self._state_file_path()
        if not os.path.isfile(file_path):
            logger.info("未找到状态文件 %s，使用初始状态", file_path)
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("读取状态文件失败 %s: %s", file_path, e)
            return False

        self.initial_capital = state.get("initial_capital", self.initial_capital)
        self.cash = state.get("cash", self.initial_capital)
        self.positions = state.get("positions", {})

        raw_trades = state.get("trades", [])
        self.trades = []
        for t in raw_trades:
            if isinstance(t, dict):
                self.trades.append(
                    TradeRecord(
                        date=t.get("date", ""),
                        action=t.get("action", ""),
                        stock_code=t.get("stock_code", ""),
                        company_name=t.get("company_name", ""),
                        price=t.get("price", 0.0),
                        shares=t.get("shares", 0),
                        commission=t.get("commission", 0.0),
                        reason=t.get("reason", ""),
                    )
                )

        logger.info(
            "状态已恢复: cash=%.2f, positions=%d, trades=%d",
            self.cash, len(self.positions), len(self.trades),
        )
        return True

    # ------------------------------------------------------------------
    # LLM 决策辅助
    # ------------------------------------------------------------------

    def decide(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """调用 DeepSeek V4 Pro 并返回结构化决策结果。

        合并 system_prompt（总纲约束）和 user_prompt（当前市场数据、
        持仓上下文、交易目标），解析 LLM 返回的 JSON 决策。

        Args:
            user_prompt: 用户消息，包含当前市场数据和投资目标。
            system_prompt: 系统提示。为 None 时使用 `get_system_prompt()`。

        Returns:
            解析后的决策 JSON 字典。
        """
        if not self._llm_client:
            logger.error("decide: LLM 客户端不可用")
            return {"reasoning": "LLM 客户端不可用", "decisions": []}

        system = system_prompt if system_prompt is not None else FREELINE_SYSTEM_PROMPT

        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
            response = self._llm_client.get_completion(messages, max_retries=2)
            if not response:
                return {"reasoning": "LLM 返回空响应", "decisions": []}

            result = _extract_json(response)
            logger.info(
                "decide: reasoning=%s, decisions=%d",
                result.get("reasoning", "")[:80],
                len(result.get("decisions", [])),
            )
            return result

        except Exception as e:
            logger.error("decide: LLM 调用失败: %s", e)
            return {"reasoning": f"LLM 调用失败: {e}", "decisions": []}


# ---------------------------------------------------------------------------
# 如果通过 `from schemas import TradeRecord` 不可用，fallback 定义
# ---------------------------------------------------------------------------

try:
    from src.advisory.schemas import TradeRecord
except ImportError:
    # 本地 fallback dataclass（与 schemas.TradeRecord 保持字段一致）

    @dataclass
    class TradeRecord:
        """一条交易记录（fallback 定义，与 schemas.TradeRecord 兼容）。"""
        date: str = ""
        action: str = ""  # buy | sell
        stock_code: str = ""
        company_name: str = ""
        price: float = 0.0
        shares: int = 0
        commission: float = 0.0
        reason: str = ""
        st: float = 0.0  # stamp tax (保留字段，用于扩展)
