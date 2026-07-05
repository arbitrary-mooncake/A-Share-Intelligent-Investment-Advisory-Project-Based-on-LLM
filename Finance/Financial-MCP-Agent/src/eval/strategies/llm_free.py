"""LLM自主投资策略 — 总纲控制组。DeepSeek V4 Pro完全自主决策买卖。

总纲 §3 设计意图：
  S-L9 / M-L1 / L-L1 三条 llm_free 线构成整个评测系统的控制组。
  不依赖任何 Agent 评分管线，由 DeepSeek V4 Pro 直接接收精筛池股票
  和市场数据，自主判断并输出结构化买卖决策。这是测试"AI能否独立于
  Agent 体系实现超额收益"的核心实验。

与其他策略的本质区别：
  - 其他策略：纯代码规则，不调用 LLM（见 __init__.py 注释）
  - LLM-free：唯一调用 LLM 的策略，但 LLM 不依赖 Agent 评分
    （'free' 指不受 Agent pipeline 约束，而非'不使用 LLM'）
"""
import json
import logging
import re
from typing import Dict, Any, List

from src.eval.strategies.base import BaseStrategy, BuyOrder, SellOrder
from src.utils.llm_clients import OpenAICompatibleClient
from src.utils.model_config import get_eval_model_config, get_thinking_body

logger = logging.getLogger(__name__)


class LLMFreeStrategy(BaseStrategy):
    """LLM自主投资策略——总纲控制组。DeepSeek V4 Pro完全自主决策买卖。

    策略特点：
      1. 接收精筛池股票 + 完整市场数据快照
      2. 不依赖 Agent 评分（但可参考作为输入特征之一）
      3. LLM 一次性输出 buy_orders + sell_orders 结构化 JSON
      4. 按期限（短/中/长）有不同仓位上限和风控参数
    """

    def __init__(self, config=None, term="short"):
        super().__init__(config, term=term)
        # 获取 DeepSeek V4 Pro 模型配置（suffix=_6）
        self.model_config = get_eval_model_config("eval_llm_free")

        # 每期限最大持仓数
        self.max_positions_map = {"short": 12, "medium": 8, "long": 5}
        # 每期限单只股票仓位上限
        self.weight_limits = {"short": 0.12, "medium": 0.18, "long": 0.25}
        # 每期限最低现金比例
        self.min_cash_ratios = {"short": 0.10, "medium": 0.15, "long": 0.20}
        # 每期限买入评分阈值（仅当 scores 可用时作为基线参考）
        self.buy_thresholds = {"short": 60, "medium": 65, "long": 70}
        # 每期限卖出评分阈值
        self.sell_hard_thresholds = {"short": 35, "medium": 45, "long": 35}

        # 当前期限（由line definition注入，不从持仓推断）
        self._active_term = self._term

        # 构建 thinking 参数
        base_url = self.model_config["base_url"]
        thinking_body = get_thinking_body(base_url, enabled=True)

        # 初始化 LLM 客户端
        try:
            self.llm_client = OpenAICompatibleClient(
                api_key=self.model_config["api_key"],
                base_url=self.model_config["base_url"],
                model=self.model_config["model_name"],
                env_prefix="",  # 使用显式参数而非环境变量前缀
                extra_body=thinking_body,
                http_timeout=600,
                http_connect_timeout=30,
            )
            logger.info(
                "LLMFreeStrategy: DeepSeek V4 Pro client initialized "
                "(model=%s)", self.model_config["model_name"]
            )
        except Exception as e:
            logger.error("LLMFreeStrategy: Failed to init LLM client: %s", e)
            self.llm_client = None

    def _infer_term(self, holdings: Dict[str, int]) -> str:
        """根据持仓数量推断当前期限。

        规则：持仓数越多越可能是短线，越少越可能是长线。
        0~3持仓→长线，4~7持仓→中线，8+→短线。
        """
        n = sum(1 for v in holdings.values() if v > 0)
        if n <= 3:
            return "long"
        elif n <= 7:
            return "medium"
        return "short"

    # ── 公共接口 ────────────────────────────────────────────────────────

    def get_max_positions(self) -> int:
        return self.max_positions_map.get(self._active_term, 10)

    def get_single_weight_limit(self) -> float:
        return self.weight_limits.get(self._active_term, 0.10)

    def get_min_cash_ratio(self) -> float:
        return self.min_cash_ratios.get(self._active_term, 0.0)

    def select_stocks(self, pool, scores, holdings, cash, market_data_map=None):
        """LLM 自主选股：构建组合上下文，调用 DeepSeek V4 Pro 决策。

        Args:
            pool: 精筛池股票代码列表
            scores: {stock_code: score} Agent评分（作为输入特征，非硬约束）
            holdings: {stock_code: shares} 当前持仓
            cash: 可用现金
            market_data_map: {stock_code: MarketData} 市场快照

        Returns:
            List[BuyOrder]: LLM 决策的买入订单
        """
        self._active_term = self._term

        if not self.llm_client:
            logger.warning("LLMFreeStrategy: LLM client unavailable, returning empty orders")
            return []

        if not pool:
            return []

        max_pos = self.max_positions_map.get(self._active_term, 10)
        weight_limit = self.weight_limits.get(self._active_term, 0.10)

        # 构建上下文
        holdings_info = self._format_holdings_info(holdings, market_data_map)
        stocks_data = self._format_stocks_data(pool, scores, market_data_map)

        prompt = self._build_buy_prompt(
            stocks_data=stocks_data,
            holdings_info=holdings_info,
            cash=cash,
            term=self._active_term,
            max_positions=max_pos,
            single_weight_limit=weight_limit,
        )

        response = self._call_llm(prompt)
        if not response:
            logger.warning("LLMFreeStrategy: LLM returned no response for select_stocks")
            return []

        orders = self._parse_buy_orders(response)
        logger.info(
            "LLMFreeStrategy: select_stocks term=%s pool=%d → %d buy orders",
            self._active_term, len(pool), len(orders)
        )
        return orders

    def generate_sell_orders(self, holdings, scores, market_data_map=None,
                             purchase_prices=None, hold_days=None):
        """LLM 自主卖出判定：构建持仓上下文，调用 DeepSeek V4 Pro 决策。

        Args:
            holdings: {stock_code: shares}
            scores: {stock_code: score}
            market_data_map: {stock_code: MarketData}
            purchase_prices: {stock_code: avg_purchase_price}
            hold_days: {stock_code: days_held}

        Returns:
            List[SellOrder]: LLM 决策的卖出订单
        """
        self._active_term = self._term

        if not self.llm_client:
            logger.warning("LLMFreeStrategy: LLM client unavailable, returning empty orders")
            return []

        active_holdings = {k: v for k, v in holdings.items() if v > 0}
        if not active_holdings:
            return []

        # 构建持仓明细（含盈亏）
        holdings_detail = self._format_holdings_detail(
            active_holdings, scores, market_data_map,
            purchase_prices or {}, hold_days or {}
        )

        prompt = self._build_sell_prompt(
            holdings_detail=holdings_detail,
            cash=0,  # 卖出决策不依赖现金
            term=self._active_term,
            sell_hard=self.sell_hard_thresholds.get(self._active_term, 35),
        )

        response = self._call_llm(prompt)
        if not response:
            logger.warning("LLMFreeStrategy: LLM returned no response for generate_sell_orders")
            return []

        orders = self._parse_sell_orders(response)
        logger.info(
            "LLMFreeStrategy: generate_sell_orders term=%s holdings=%d → %d sell orders",
            self._active_term, len(active_holdings), len(orders)
        )
        return orders

    # ── LLM 调用 ─────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """调用 DeepSeek V4 Pro 并返回响应文本。"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.get_completion(messages, max_retries=2)
            return response or ""
        except Exception as e:
            logger.error("LLMFreeStrategy: LLM call failed: %s", e)
            return ""

    # ── 数据格式化 ──────────────────────────────────────────────────────

    def _format_stocks_data(self, pool: List[str],
                            scores: Dict[str, float],
                            market_data_map: Dict[str, Any]) -> str:
        """
        将精筛池股票格式化为 LLM 可读的文本表格。

        总纲 §3.5: LLM自由线不可使用任何Agent评分数值。
        scores参数保留用于API兼容但不再输出到prompt中。
        """
        lines = []
        for code in pool:
            md = market_data_map.get(code) if market_data_map else None

            parts = [f"| {code}"]
            if md:
                parts.append(f"close={md.close:.2f}")
                if hasattr(md, 'pre_close') and md.pre_close > 0:
                    chg_pct = (md.close - md.pre_close) / md.pre_close * 100
                    parts.append(f"chg={chg_pct:+.2f}%")
                if hasattr(md, 'turnover_rate') and md.turnover_rate:
                    parts.append(f"turnover={md.turnover_rate:.2%}")
                if hasattr(md, 'market_cap') and md.market_cap:
                    parts.append(f"cap={md.market_cap:.0f}亿")
                if hasattr(md, 'pe_ratio') and md.pe_ratio:
                    parts.append(f"PE={md.pe_ratio:.1f}")
                if hasattr(md, 'pb_ratio') and md.pb_ratio:
                    parts.append(f"PB={md.pb_ratio:.2f}")
                if hasattr(md, 'volume') and md.volume:
                    parts.append(f"vol={md.volume:.0f}手")
                if hasattr(md, 'is_limit_up') and md.is_limit_up:
                    parts.append("【涨停】")
                if hasattr(md, 'is_limit_down') and md.is_limit_down:
                    parts.append("【跌停】")
                if hasattr(md, 'is_suspended') and md.is_suspended:
                    parts.append("【停牌】")
                if hasattr(md, 'is_hs300') and md.is_hs300:
                    parts.append("沪深300")
            lines.append(" ".join(parts))

        return "\n".join(lines) if lines else "（无可用股票数据）"

    def _format_holdings_info(self, holdings: Dict[str, int],
                              market_data_map: Dict[str, Any]) -> str:
        """简洁的持仓摘要（用于选股时了解已有仓位）。"""
        active = {k: v for k, v in holdings.items() if v > 0}
        if not active:
            return "（空仓）"
        lines = []
        for code, shares in active.items():
            md = market_data_map.get(code) if market_data_map else None
            price = md.close if md and hasattr(md, 'close') and md.close else 0
            value = shares * price
            lines.append(f"  {code}: {shares}股, 市值≈{value:.0f}元")
        return "\n".join(lines)

    def _format_holdings_detail(self, holdings: Dict[str, int],
                                scores: Dict[str, float],
                                market_data_map: Dict[str, Any],
                                purchase_prices: Dict[str, float],
                                hold_days: Dict[str, int]) -> str:
        """完整的持仓明细（含盈亏，用于卖出决策）。"""
        lines = []
        for code, shares in holdings.items():
            if shares <= 0:
                continue
            score = scores.get(code, 0)
            md = market_data_map.get(code) if market_data_map else None
            current_price = md.close if md and hasattr(md, 'close') and md.close else 0
            cost_price = purchase_prices.get(code, 0)
            days = hold_days.get(code, 0)
            market_value = shares * current_price

            pnl_pct = ""
            if cost_price > 0 and current_price > 0:
                pnl_pct = f"盈亏={(current_price - cost_price) / cost_price * 100:+.1f}%"
            elif cost_price > 0 and current_price == 0:
                pnl_pct = "盈亏=无价格数据"

            parts = [
                f"| {code} | {shares}股 | cost={cost_price:.2f}",
                f"price={current_price:.2f} | 市值={market_value:.0f}",
            ]
            if pnl_pct:
                parts.append(f"| {pnl_pct}")
            parts.append(f"| 持有{days}天")

            # 额外信号
            signals = []
            if md:
                if hasattr(md, 'turnover_rate') and md.turnover_rate:
                    signals.append(f"换手={md.turnover_rate:.2%}")
                if hasattr(md, 'is_limit_up') and md.is_limit_up:
                    signals.append("涨停")
                if hasattr(md, 'is_limit_down') and md.is_limit_down:
                    signals.append("跌停")
                if hasattr(md, 'pe_ratio') and md.pe_ratio:
                    signals.append(f"PE={md.pe_ratio:.1f}")
            if signals:
                parts.append(f"| {' '.join(signals)}")

            lines.append(" ".join(parts))

        return "\n".join(lines) if lines else "（无持仓）"

    # ── Prompt 构建 ──────────────────────────────────────────────────────

    def _build_buy_prompt(self, stocks_data: str, holdings_info: str,
                          cash: float, term: str, max_positions: int,
                          single_weight_limit: float) -> str:
        """构建选股买入的组合管理 Prompt。"""
        term_labels = {"short": "短线（1-5个交易日）", "medium": "中线（1-4周）",
                       "long": "长线（1-6个月）"}

        return f"""你是一位专业的A股基金经理，管理一个{term_labels.get(term, term)}投资组合。

## 投资期限
{term_labels.get(term, term)}

## 账户状态
- 可用现金: {cash:,.0f} 元
- 最大持仓数: {max_positions} 只
- 单只股票仓位上限: {single_weight_limit:.0%}
- 当前持仓:
{holdings_info}

## 可选股票池（仅原始市场数据，不含任何Agent评分）
{stocks_data}

## 数据字段说明
- close: 当日收盘价（元）
- chg: 当日涨跌幅
- turnover: 换手率
- cap: 总市值（亿）
- PE: 市盈率
- PB: 市净率
- vol: 成交量（手）
- 沪深300: 是否为沪深300成分股

## 决策要求
1. 仅基于原始市场数据（价格/估值/流动性/市值等）独立判断，不得依赖任何外部评分
2. 考虑投资期限：短线注重动量和资金面，中线注重估值和基本面，长线注重质量和护城河
3. 每只入选股票必须给出买入金额（target_value），总和不超过可用现金
4. 不要买入【涨停】或【停牌】的股票，[跌停]的股票谨慎处理
5. 如果股票池质量不佳，可以少买或不买（保留现金）

## 输出格式
严格输出以下JSON格式，不要添加任何额外文字：
```json
{{
  "reasoning": "简要说明你的选股逻辑（1-3句话）",
  "buy_orders": [
    {{
      "stock_code": "股票代码，如000001",
      "target_value": 买入金额（元）,
      "reason": "买入理由（1句话）"
    }}
  ]
}}
```

请输出你的决策JSON："""

    def _build_sell_prompt(self, holdings_detail: str, cash: float,
                           term: str, sell_hard: float) -> str:
        """构建卖出判定的 Prompt。"""
        term_labels = {"short": "短线（1-5个交易日）", "medium": "中线（1-4周）",
                       "long": "长线（1-6个月）"}

        return f"""你是一位专业的A股基金经理，正在评估当前持仓的卖出决策。

## 投资期限
{term_labels.get(term, term)}

## 当前持仓明细
{holdings_detail}

## 数据字段说明
- cost: 买入均价（元）
- price: 当前价格（元）
- 盈亏: 相对买入价的盈亏百分比
- 持有X天: 已持有交易日数
- 换手: 当日换手率
- PE: 市盈率

## 卖出决策指引
- 仅基于原始市场数据判断（价格走势/估值变化/换手异常/涨跌停等）
- 短线考量：日涨幅过大、量价背离、换手异常
- 中线考量：估值过高、趋势破位、基本面数据恶化
- 长线考量：长期逻辑破坏、估值极端
- 可以部分卖出（sell_ratio: 0-1之间的小数），不需要全部清仓
- 如果持仓质量良好，可以不卖出（返回空数组）

## 输出格式
严格输出以下JSON格式，不要添加任何额外文字：
```json
{{
  "reasoning": "简要说明你的卖出逻辑（1-3句话）",
  "sell_orders": [
    {{
      "stock_code": "股票代码",
      "sell_ratio": 卖出比例（0-1之间的小数，1=全部卖出）,
      "reason": "卖出理由（1句话）"
    }}
  ]
}}
```

请输出你的决策JSON："""

    # ── 响应解析 ────────────────────────────────────────────────────────

    def _extract_json(self, response: str) -> dict:
        """从 LLM 响应中提取 JSON 对象。"""
        if not response:
            return {}

        text = response.strip()

        # 策略1: 提取 ```json ... ``` 代码块
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # 策略2: 提取 ``` ... ``` 任意代码块
        if "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        # 策略3: 提取第一个 { ... } 对象
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 策略4: 尝试直接解析整个文本
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        logger.error("LLMFreeStrategy: Failed to extract JSON from response: %s",
                     text[:200])
        return {}

    def _parse_buy_orders(self, response: str) -> List[BuyOrder]:
        """解析 LLM 返回的买入订单 JSON → List[BuyOrder]."""
        data = self._extract_json(response)
        if not data:
            return []

        orders = []
        raw_orders = data.get("buy_orders", [])
        if not isinstance(raw_orders, list):
            raw_orders = []

        for item in raw_orders:
            if not isinstance(item, dict):
                continue
            code = item.get("stock_code", "").strip()
            target_value = float(str(item.get("target_value", 0)).replace(",", ""))
            reason = item.get("reason", "")

            if not code or target_value <= 0:
                continue

            order = BuyOrder(code, target_value)
            order.reason = reason
            orders.append(order)

        logger.info("LLMFreeStrategy: reasoning=%s", data.get("reasoning", ""))
        return orders

    def _parse_sell_orders(self, response: str) -> List[SellOrder]:
        """解析 LLM 返回的卖出订单 JSON → List[SellOrder]."""
        data = self._extract_json(response)
        if not data:
            return []

        orders = []
        raw_orders = data.get("sell_orders", [])
        if not isinstance(raw_orders, list):
            raw_orders = []

        for item in raw_orders:
            if not isinstance(item, dict):
                continue
            code = item.get("stock_code", "").strip()
            sell_ratio = float(item.get("sell_ratio", 0))
            reason = item.get("reason", "")

            if not code or sell_ratio <= 0:
                continue

            # Clamp 到 [0, 1]
            sell_ratio = max(0.0, min(1.0, sell_ratio))
            orders.append(SellOrder(code, sell_ratio, reason))

        logger.info("LLMFreeStrategy: sell reasoning=%s", data.get("reasoning", ""))
        return orders

    # ── 仓位管理（基类提供默认等权分配，这里覆盖为 LLM 自主权重） ──

    def size_positions(self, buy_orders, total_capital, current_positions,
                       max_positions, single_weight_limit, min_cash_ratio):
        """仓位调整：LLM 已自主决定 target_value，这里做安全约束。

        约束规则：
          1. 单只股票不超过 single_weight_limit × total_capital
          2. 总买入额不超过 total_capital × (1 - min_cash_ratio)
          3. 买入数量不超过 max_positions - current_positions
        """
        available_cash = total_capital * (1 - min_cash_ratio)
        slots = max(0, max_positions - current_positions)

        if slots <= 0 or not buy_orders:
            return []

        # 如果 LLM 已经分配了 target_value，以 LLM 为准但施加约束
        sized = []
        total_allocated = 0.0
        for i, order in enumerate(buy_orders[:slots]):
            # 单只股票上限
            max_single = total_capital * single_weight_limit

            if order.target_value > 0:
                # LLM 已指定金额，施加约束
                target = min(order.target_value, max_single)
            else:
                # LLM 未指定金额，等权分配
                target = available_cash / max(slots, 1)
                target = min(target, max_single)

            # 确保总金额不超可用现金
            remaining = available_cash - total_allocated
            target = min(target, remaining / max(slots - i, 1))
            target = max(0, target)

            if target <= 0:
                continue

            order.target_value = target
            total_allocated += target
            sized.append(order)

        return sized
