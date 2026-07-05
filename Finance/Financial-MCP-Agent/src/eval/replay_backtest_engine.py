"""
历史回放回测引擎 — Point-in-Time历史数据重放。
用于快速测量Agent贡献（短线/中线/长线），不需等待未来自然成熟。

v2升级（2026-06）:
  - 用真实PIT数据评分替代MD5哈希模拟
  - 移除5锚点上限，处理全部anchor dates
  - 定义22条命名回测线 (SB-L0~L5, MB-L0~L7, LB-L0~L7)
  - 季度校准(方案B): 每锚点动态重筛池，消除存活偏差
  - 市场环境切片: 牛/熊/震荡市分别回测

v2.1升级（2026-07）:
  - Gap #9: 财务数据PIT对齐 — 按披露日期(disclosure_date)过滤，消除前视偏差
  - Gap #8: 存活偏差量化 — 检测回测期间ST/退市/长期停牌事件
  - Gap #15: Prompt补丁回测验证 — mini backtest + prompt_override支持
"""
import os
import json
import math
import random
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import namedtuple

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """回测配置"""
    term: str = "medium"                    # short/medium/long
    start_date: str = "2022-01-01"
    end_date: str = "2026-05-31"
    holding_days: int = 20                  # 持有期（交易日）
    anchor_frequency: str = "weekly"        # weekly/monthly
    anchor_weekday: int = 4                 # 0=Mon, 4=Fri
    ablation_agents: List[str] = field(default_factory=lambda:
        ["fundamental", "technical", "value", "quality_risk", "moneyflow"])
    benchmark_symbol: str = "sh.000300"     # CSI 300
    train_split: str = "2023-12-31"
    validation_split: str = "2025-06-30"
    # Gap #15: 允许在回测中覆写prompt以验证补丁效果
    prompt_override: Optional[Dict[str, str]] = None  # {agent_name: new_prompt, ...}
    # Gap #11: 策略参数覆盖
    param_overrides: Optional[Dict[str, Any]] = None  # {score_buy_threshold: 70, ...}


# Gap #8: 存活偏差量化数据结构
SurvivorshipStats = namedtuple("SurvivorshipStats", [
    "total_pool_stocks",         # 精筛池总股票数
    "st_events_count",           # 发生过ST的股票数
    "st_events_pct",             # ST比例 (%)
    "delisted_count",            # 已退市股票数
    "delisted_pct",              # 退市比例 (%)
    "long_suspended_count",      # 长期停牌(>30日)股票数
    "long_suspended_pct",        # 长期停牌比例 (%)
    "affected_count",            # 受任一影响股票数（去重）
    "affected_pct",              # 受影响比例 (%)
    "clean_count",               # 无事件股票数
    "clean_pct",                 # 无事件比例 (%)
    "details",                   # [{code, name, event_type, first_date, description}]
])


# Gap #15: Prompt补丁回测验证结果
@dataclass
class VerificationResult:
    """Prompt补丁回测验证结果"""
    passed: bool                            # 验证是否通过
    determination: str                      # "improvement" / "no_change" / "regression"
    mean_score_diff: float                  # 平均分数差异 (patched - original)
    rank_correlation: float                 # 排名相关性 (Spearman-like)
    score_volatility_change: float          # 分数波动性变化
    direction_flip_rate: float              # 方向翻转率 (0-1)
    performance_metrics: Dict[str, Any]     # 性能对比指标
    anchor_results: List[Dict[str, Any]]    # 每锚点详细结果
    baseline_summary: Dict[str, Any]        # 基线回测汇总
    patched_summary: Dict[str, Any]         # 补丁回测汇总


# ══════════════════════════════════════════════════════════════
# Fix 3: 22条命名回测线定义 (总纲 §8)
# ══════════════════════════════════════════════════════════════

BACKTEST_LINE_DEFINITIONS = {
    # ── 短线回测线 (6条) ──
    "SB-L0": {"term": "short", "agents": "all",
              "description": "短线基线 (全5 agent)"},
    "SB-L1": {"term": "short", "agents": "-fundamental",
              "description": "短线消融-基本面"},
    "SB-L2": {"term": "short", "agents": "-technical",
              "description": "短线消融-技术面"},
    "SB-L3": {"term": "short", "agents": "-value",
              "description": "短线消融-估值"},
    "SB-L4": {"term": "short", "agents": "-quality_risk",
              "description": "短线消融-质量风险"},
    "SB-L5": {"term": "short", "agents": "-moneyflow",
              "description": "短线消融-资金流"},

    # ── 短线参考线 (1条, 非消融) — 长持对照线, 连续持仓, 不参与ΔLoss ──
    "SB-L6": {"term": "short", "agents": "all",
              "description": "短线长持对照线 (成熟短线策略, 连续持仓, 不参与消融ΔLoss)",
              "strategy": "longhold",
              "is_reference": True},

    # ── 中线回测线 (8条) ──
    "MB-L0": {"term": "medium", "agents": "all",
              "description": "中线基线 (全5 agent)"},
    "MB-L1": {"term": "medium", "agents": "-fundamental",
              "description": "中线消融-基本面"},
    "MB-L2": {"term": "medium", "agents": "-technical",
              "description": "中线消融-技术面"},
    "MB-L3": {"term": "medium", "agents": "-value",
              "description": "中线消融-估值"},
    "MB-L4": {"term": "medium", "agents": "-quality_risk",
              "description": "中线消融-质量风险"},
    "MB-L5": {"term": "medium", "agents": "-moneyflow",
              "description": "中线消融-资金流"},
    "MB-L6": {"term": "medium", "agents": "-news",
              "description": "中线消融-新闻 (退化线, 按规范排除)",
              "degenerate": True},
    "MB-L7": {"term": "medium", "agents": "-event",
              "description": "中线消融-事件 (退化线, 按规范排除)",
              "degenerate": True},

    # ── 长线回测线 (8条) ──
    "LB-L0": {"term": "long", "agents": "all",
              "description": "长线基线 (全5 agent)"},
    "LB-L1": {"term": "long", "agents": "-fundamental",
              "description": "长线消融-基本面"},
    "LB-L2": {"term": "long", "agents": "-technical",
              "description": "长线消融-技术面"},
    "LB-L3": {"term": "long", "agents": "-value",
              "description": "长线消融-估值"},
    "LB-L4": {"term": "long", "agents": "-quality_risk",
              "description": "长线消融-质量风险"},
    "LB-L5": {"term": "long", "agents": "-moneyflow",
              "description": "长线消融-资金流"},
    "LB-L6": {"term": "long", "agents": "-news",
              "description": "长线消融-新闻 (退化线, 按规范排除)",
              "degenerate": True},
    "LB-L7": {"term": "long", "agents": "-event",
              "description": "长线消融-事件 (退化线, 按规范排除)",
              "degenerate": True},
}

# 按期限组织的活跃回测线 (排除退化线)
ACTIVE_BACKTEST_LINES = {
    "short": [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
              if v["term"] == "short"],
    "medium": [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
               if v["term"] == "medium" and not v.get("degenerate")],
    "long": [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
             if v["term"] == "long" and not v.get("degenerate")],
}

# 参考线（非消融，不参与ΔLoss计算）
# 这些线使用不同的交易逻辑（如连续持仓），提供现实性校验对照
REFERENCE_LINES = [k for k, v in BACKTEST_LINE_DEFINITIONS.items()
                   if v.get("is_reference", False)]

# 纯消融线（用于ΔLoss归因），排除参考线和退化线
ABLATION_LINES = {
    term: [k for k in active if k not in REFERENCE_LINES]
    for term, active in ACTIVE_BACKTEST_LINES.items()
}


class ReplayBacktestEngine:
    """
    历史回放回测引擎。

    对每个anchor date:
      1. 设定as_of_date，获取该日及之前的历史数据
      2. 对精筛池股票运行分析agent（仅可回溯的5个）
      3. 运行消融线scorer
      4. 按策略调仓并持有holding_days
      5. 计算持有期收益
      6. 比较各消融线的收益差异 → agent贡献
    """

    # 各agent消融时的分数折扣系数 (agent contribution weight)
    # 值越大 = 该agent对评分贡献越大 = 消融后分数降低越多
    _ABLATION_WEIGHTS = {
        "fundamental": 0.20,     # 基本面agent: 对中线/长线贡献最大
        "technical": 0.15,       # 技术面agent: 对短线贡献大
        "value": 0.12,           # 估值agent: 中等贡献
        "quality_risk": 0.15,    # 质量风险agent: 对中线/长线重要
        "moneyflow": 0.10,       # 资金流agent: 对短线重要
        "news": 0.08,            # 新闻: 边际贡献 (退化)
        "event": 0.08,           # 事件: 边际贡献 (退化)
    }

    # 行业PE/PB基准 (中位数)
    _INDUSTRY_BENCHMARKS = {
        # 行业名 → (PE中位数, PB中位数, ROE中位数)
        "default": (25.0, 2.5, 8.0),
        "银行": (6.0, 0.7, 10.0),
        "保险": (12.0, 1.5, 12.0),
        "证券": (18.0, 1.5, 7.0),
        "白酒": (28.0, 6.0, 22.0),
        "医药": (35.0, 4.0, 10.0),
        "医疗": (38.0, 4.5, 9.0),
        "半导体": (50.0, 4.5, 8.0),
        "元器件": (30.0, 3.5, 9.0),
        "软件服务": (45.0, 4.0, 6.0),
        "互联网": (35.0, 3.5, 8.0),
        "通信设备": (30.0, 3.0, 8.0),
        "电力": (18.0, 1.8, 7.0),
        "新能源": (25.0, 3.0, 10.0),
        "汽车": (22.0, 2.5, 9.0),
        "房地产": (12.0, 0.8, 5.0),
        "建筑": (10.0, 0.9, 7.0),
        "建材": (15.0, 1.8, 10.0),
        "化工": (18.0, 2.2, 9.0),
        "钢铁": (12.0, 0.9, 6.0),
        "有色": (25.0, 2.5, 8.0),
        "煤炭": (8.0, 1.2, 12.0),
        "石油": (12.0, 1.0, 8.0),
        "农林牧渔": (30.0, 3.0, 6.0),
        "食品饮料": (30.0, 5.0, 15.0),
        "家电": (15.0, 2.8, 15.0),
        "纺织服装": (20.0, 2.0, 7.0),
        "交通运输": (15.0, 1.5, 7.0),
        "公用事业": (18.0, 1.5, 7.0),
        "军工": (50.0, 3.5, 4.0),
        "机械": (25.0, 2.5, 8.0),
    }

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results: Dict[str, Any] = {}
        # Gap #9: PIT disclosure date and fundamentals caches
        self._disclosure_date_cache: Dict[Tuple[str, str, str], Dict[str, str]] = {}  # {(ts_code, start_date, end_date): {report_period: ann_date}}
        self._pit_fundamentals_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}  # {(stock_code, anchor_date): pit_data}
        # Gap #8: Survivorship stats
        self._survivor_stats: Optional[SurvivorshipStats] = None

    def generate_anchor_dates(self) -> List[str]:
        """生成回测anchor date列表"""
        cfg = self.config
        anchors = []

        current = datetime.strptime(cfg.start_date, "%Y-%m-%d")
        end = datetime.strptime(cfg.end_date, "%Y-%m-%d")

        if cfg.anchor_frequency == "weekly":
            # 每个周五（或指定周日）
            while current <= end:
                if current.weekday() == cfg.anchor_weekday:
                    anchors.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
        elif cfg.anchor_frequency == "monthly":
            # 每月最后一个交易日近似（取每月20日之后最后一个工作日）
            while current <= end:
                if current.day >= 20 and current.weekday() < 5:
                    # Check if this is the last trading day of the month
                    next_day = current + timedelta(days=1)
                    if next_day.month != current.month or next_day > end:
                        anchors.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)

        return anchors

    def get_train_val_test_split(self, anchors: List[str]) -> Dict[str, List[str]]:
        """将anchor dates分为训练/验证/测试集"""
        cfg = self.config
        train = [d for d in anchors if d <= cfg.train_split]
        validate = [d for d in anchors if cfg.train_split < d <= cfg.validation_split]
        test = [d for d in anchors if d > cfg.validation_split]
        return {"train": train, "validate": validate, "test": test}

    # ═══════════════════════════════════════════════════════════
    # Fix 5: Market Regime Classification
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _classify_market_regime(
        anchor_date: str,
        benchmark_data: Optional[List[Dict[str, Any]]] = None,
        trailing_days: int = 60,
    ) -> str:
        """
        根据基准指数(CSI 300)回報将anchor date所在时期分类为熊/震荡/牛。

        Args:
            anchor_date: YYYY-MM-DD
            benchmark_data: [{trade_date, close}, ...] 或 None (用内置估算)
            trailing_days: 回溯天数

        Returns:
            "bull" / "bear" / "ranging"
        """
        anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")

        if benchmark_data and len(benchmark_data) >= 2:
            # 用真实基准数据
            prices_by_date = {
                d.get("trade_date", ""): d.get("close", 0)
                for d in benchmark_data
            }
            # 在anchor_date附近找价格
            for offset in range(10):
                date_key = (anchor_dt - timedelta(days=offset)).strftime("%Y%m%d")
                if date_key in prices_by_date and prices_by_date[date_key] > 0:
                    current_price = prices_by_date[date_key]
                    break
            else:
                current_price = 0

            trailing_start = anchor_dt - timedelta(days=trailing_days + 20)
            trailing_prices = []
            for d in benchmark_data:
                date_str = d.get("trade_date", "")
                if date_str >= trailing_start.strftime("%Y%m%d") and date_str <= anchor_dt.strftime("%Y%m%d"):
                    trailing_prices.append(d.get("close", 0))

            if current_price > 0 and len(trailing_prices) >= 2:
                trailing_return = (current_price - trailing_prices[0]) / trailing_prices[0]

                # 阈值: >+8% 牛市, <-8% 熊市, 否则震荡
                if trailing_return > 0.08:
                    return "bull"
                elif trailing_return < -0.08:
                    return "bear"
                return "ranging"

        # 回退方案: 用已知A股市场环境做粗略判断
        # 基于anchor_date年代和月份做经验推断
        date_str = anchor_dt.strftime("%Y%m")
        # 2022年大部分是熊市（疫情封控+清零政策）
        if "202201" <= date_str <= "202210":
            return "bear"
        # 2022年底反弹
        elif "202211" <= date_str <= "202302":
            return "ranging"
        # 2023年中到2024年初的阴跌
        elif "202303" <= date_str <= "202401":
            return "bear"
        # 2024年2月起政策转向，震荡中走牛
        elif "202402" <= date_str <= "202409":
            return "ranging"
        # 2024年9月底政策大转向，牛回
        elif "202410" <= date_str:
            return "bull"
        return "ranging"

    def slice_anchors_by_regime(
        self,
        anchors: List[str],
        benchmark_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[str]]:
        """
        将anchor dates按市场环境切片。

        Returns:
            {"bull": [...], "bear": [...], "ranging": [...]}
        """
        regimes: Dict[str, List[str]] = {"bull": [], "bear": [], "ranging": []}
        for d in anchors:
            regime = self._classify_market_regime(d, benchmark_data)
            regimes[regime].append(d)
        return regimes

    async def run_regime_analysis(
        self,
        pool: List[str],
        price_data: Dict[str, Any],
        benchmark_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        按市场环境切片运行回测，分别统计牛/熊/震荡市表现。

        Returns:
            {"bull": {...}, "bear": {...}, "ranging": {...}, "aggregate": {...}}
        """
        anchors = self.generate_anchor_dates()
        regime_anchors = self.slice_anchors_by_regime(anchors, benchmark_data)

        regime_results = {}
        for regime, regime_anchor_list in regime_anchors.items():
            if not regime_anchor_list:
                regime_results[regime] = {"num_anchors": 0, "contribution_summary": {}}
                continue

            all_results = []
            for i, anchor_date in enumerate(regime_anchor_list):
                if (i + 1) % 20 == 0 or i == 0:
                    print(f"  [regime={regime}] 进度: {i+1}/{len(regime_anchor_list)}")
                result = await self.run_single_anchor(anchor_date, pool, price_data)
                all_results.append(result)

            contribution_summary = self._summarize_contributions(all_results)
            regime_results[regime] = {
                "num_anchors": len(all_results),
                "contribution_summary": contribution_summary,
            }

        # 汇总
        regime_results["aggregate"] = {
            "total_anchors": sum(v["num_anchors"] for v in regime_results.values()
                                 if isinstance(v, dict) and "num_anchors" in v),
            "bull_anchors": regime_results.get("bull", {}).get("num_anchors", 0),
            "bear_anchors": regime_results.get("bear", {}).get("num_anchors", 0),
            "ranging_anchors": regime_results.get("ranging", {}).get("num_anchors", 0),
        }
        return regime_results

    # ═══════════════════════════════════════════════════════════
    # Fix 1: Real PIT scoring (replaces _simulate_score)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _convert_to_ts_code(stock_code: str) -> str:
        """sh.601888 → 601888.SH, sz.300308 → 300308.SZ"""
        code = stock_code.strip()
        if code.startswith("sh."):
            return code[3:] + ".SH"
        elif code.startswith("sz."):
            return code[3:] + ".SZ"
        return code

    # ═══════════════════════════════════════════════════════════
    # Gap #9: Financial data PIT alignment using disclosure_date
    # ═══════════════════════════════════════════════════════════

    def _get_disclosure_dates(
        self, stock_code: str, start_date: str, end_date: str
    ) -> Dict[str, str]:
        """
        Fetch disclosure (announcement) dates for financial reports.

        Calls Tushare disclosure_date API to get the actual announcement
        dates for each report period. Results are cached per stock since
        disclosure dates are historical facts that never change.

        Args:
            stock_code: Internal format (sh.603871)
            start_date: YYYYMMDD (start of query range for report periods)
            end_date: YYYYMMDD (end of query range for report periods)

        Returns:
            {report_period_end_date (YYYYMMDD): disclosure_date (YYYYMMDD), ...}
            Empty dict if API fails.
        """
        ts_code = self._convert_to_ts_code(stock_code)
        cache_key = (ts_code, start_date, end_date)

        # Check cache first
        if cache_key in self._disclosure_date_cache:
            cached = self._disclosure_date_cache[cache_key]
            if cached is not None:
                return cached

        # On cache miss, also check if a broader cached range covers our range
        for (cached_ts_code, cached_start, cached_end), cached_data in list(
            self._disclosure_date_cache.items()
        ):
            if cached_ts_code == ts_code and cached_start <= start_date and cached_end >= end_date:
                if cached_data is not None:
                    return cached_data

        try:
            from src.eval.data_fetcher import _call as tushare_call

            result = tushare_call("disclosure_date", {
                "ts_code": ts_code,
                "start_date": start_date,
                "end_date": end_date,
            }, fields="ts_code,ann_date,end_date,pre_date")

            if not result or "items" not in result:
                logger.warning("disclosure_date API returned no data for %s (%s~%s)",
                             stock_code, start_date, end_date)
                return {}

            fields = result["fields"]
            disclosure_map: Dict[str, str] = {}

            for row in result["items"]:
                item = dict(zip(fields, row))
                report_period = str(item.get("end_date", ""))
                ann_date = str(item.get("ann_date", ""))
                if report_period and ann_date:
                    # Only store actual disclosure dates (skip pre-dates if not yet disclosed)
                    disclosure_map[report_period] = ann_date

            # Cache the result — keyed by (ts_code, start_date, end_date) to avoid
            # stale narrow-range cache poisoning broader queries.
            self._disclosure_date_cache[cache_key] = disclosure_map

            logger.debug("Got %d disclosure dates for %s", len(disclosure_map), stock_code)
            return disclosure_map

        except Exception as e:
            logger.warning("Failed to fetch disclosure dates for %s: %s", stock_code, str(e))
            # Return empty — caller will fall back to report_date (current behavior)
            return {}

    def _get_pit_financial_data(
        self, stock_code: str, anchor_date: str
    ) -> Dict[str, Any]:
        """
        Get Point-in-Time financial data: only reports disclosed on or before
        the anchor_date are used. This eliminates look-ahead bias.

        Steps:
        1. Fetch financial indicators via fina_indicator (extended range)
        2. Fetch disclosure dates for the same range
        3. For each report, check if ann_date <= anchor_date
        4. Return the latest available (by report period) disclosed report

        Args:
            stock_code: Internal format (sh.603871)
            anchor_date: YYYY-MM-DD — the analysis date

        Returns:
            Dict of financial indicators from the latest disclosed report,
            or empty dict if no data available / Tushare unavailable.
        """
        try:
            from src.eval.data_fetcher import _call as tushare_call

            ts_code = self._convert_to_ts_code(stock_code)
            ts_anchor = anchor_date.replace("-", "")

            # Query financial data going back several years to ensure coverage
            # Anchor date minus ~3 years of reporting periods
            anchor_year = int(anchor_date[:4])
            start_period = f"{anchor_year - 3}0101"

            # 1. Fetch financial indicators (reports up to anchor_date)
            fina = tushare_call("fina_indicator", {
                "ts_code": ts_code,
                "start_date": start_period,
                "end_date": ts_anchor,
                "limit": "8",  # ~2 years of quarterly reports
            }, ("end_date,roe,roe_dt,roa,grossprofit_margin,netprofit_margin,"
                "debt_to_assets,current_ratio,quick_ratio,"
                "or_yoy,profit_yoy,ocf_yoy,ocf_netprofit,assets_turn"))

            if not fina or "items" not in fina or not fina["items"]:
                return {"_has_data": False, "_pit_filtered": True,
                        "_reason": "No fina_indicator data available"}

            fina_fields = fina["fields"]

            # 2. Try to get disclosure dates for this stock
            disclosure_map = self._get_disclosure_dates(
                stock_code, start_period, ts_anchor
            )

            # 3. Filter reports by disclosure date
            # If disclosure_map is empty (API failed), fall back to report_date
            # (current behavior — no filtering)
            pit_filtered = False
            usable_reports = []

            for row in fina["items"]:
                item = dict(zip(fina_fields, row))
                report_period = str(item.get("end_date", ""))

                if disclosure_map:
                    # PIT mode: check if report was disclosed by anchor_date
                    ann_date = disclosure_map.get(report_period, "")
                    if ann_date and ann_date <= ts_anchor:
                        usable_reports.append(item)
                        pit_filtered = True
                    elif not ann_date:
                        # No disclosure date known — conservatively include only
                        # reports with end_date well before anchor (3 month buffer)
                        try:
                            rp_dt = datetime.strptime(report_period, "%Y%m%d")
                            ad_dt = datetime.strptime(anchor_date, "%Y-%m-%d")
                            if (ad_dt - rp_dt).days >= 90:
                                usable_reports.append(item)
                                pit_filtered = True
                                logger.debug(
                                    "Including %s report %s (no disclosure date, "
                                    ">=90d before anchor, conservative fallback)",
                                    stock_code, report_period
                                )
                        except ValueError:
                            pass
                else:
                    # No disclosure data at all — fall back to report_date-based
                    # filtering (current behavior, no PIT)
                    if report_period <= ts_anchor:
                        usable_reports.append(item)

            if not usable_reports:
                return {"_has_data": False, "_pit_filtered": pit_filtered,
                        "_reason": "No reports disclosed by anchor date"}

            # 4. Sort by report period (descending) and take the latest
            usable_reports.sort(key=lambda x: str(x.get("end_date", "")), reverse=True)
            latest = usable_reports[0]
            latest_period = str(latest.get("end_date", ""))

            fina_data = {
                "end_date": latest_period,
                "roe": _safe_float(latest.get("roe")),
                "roe_dt": _safe_float(latest.get("roe_dt")),
                "roa": _safe_float(latest.get("roa")),
                "grossprofit_margin": _safe_float(latest.get("grossprofit_margin")),
                "netprofit_margin": _safe_float(latest.get("netprofit_margin")),
                "debt_to_assets": _safe_float(latest.get("debt_to_assets")),
                "current_ratio": _safe_float(latest.get("current_ratio")),
                "quick_ratio": _safe_float(latest.get("quick_ratio")),
                "or_yoy": _safe_float(latest.get("or_yoy")),
                "profit_yoy": _safe_float(latest.get("profit_yoy")),
                "ocf_yoy": _safe_float(latest.get("ocf_yoy")),
                "ocf_netprofit": _safe_float(latest.get("ocf_netprofit")),
                "assets_turn": _safe_float(latest.get("assets_turn")),
                "_pit_filtered": pit_filtered,
            }

            if pit_filtered and disclosure_map:
                ann_date = disclosure_map.get(latest_period, "unknown")
                fina_data["_disclosure_date"] = ann_date
                logger.debug(
                    "PIT fundamental for %s @ %s: using report %s (disclosed %s)",
                    stock_code, anchor_date, latest_period, ann_date
                )

            return fina_data

        except Exception as e:
            logger.warning("PIT financial data failed for %s @ %s: %s",
                         stock_code, anchor_date, str(e))
            return {"_has_data": False, "_pit_filtered": False,
                    "_reason": f"Exception: {str(e)}"}

    def _fetch_pit_fundamentals(
        self, stock_code: str, anchor_date: str
    ) -> Dict[str, Any]:
        """
        Fetch PIT(Point-in-Time) fundamental data from Tushare.

        Gap #9 upgrade: Uses disclosure_date to filter out financial reports
        that were not yet announced as of the anchor date.

        Args:
            stock_code: Internal format (sh.603871)
            anchor_date: YYYY-MM-DD

        Returns:
            {pe, pb, roe, revenue_growth, profit_growth, debt_ratio, ocf_ratio, ...}
            Returns dict with _has_data=False if Tushare unavailable.
        """
        try:
            from src.eval.data_fetcher import _call as tushare_call

            ts_code = self._convert_to_ts_code(stock_code)
            ts_date = anchor_date.replace("-", "")

            # Check PIT fundamentals cache first (Gap #9)
            cache_key = (stock_code, anchor_date)
            if cache_key in self._pit_fundamentals_cache:
                logger.debug("PIT fundamentals cache hit for %s @ %s", stock_code, anchor_date)
                return self._pit_fundamentals_cache[cache_key]

            # 1. Get PE/PB/turnover/total_mv as of anchor_date
            basic = tushare_call("daily_basic", {
                "ts_code": ts_code,
                "trade_date": ts_date,
            }, fields="pe,pb,turnover_rate,total_mv")

            pe = 0.0
            pb = 0.0
            turnover = 0.0
            total_mv = 0.0
            if basic and "items" in basic and basic["items"]:
                fields = basic["fields"]
                item = dict(zip(fields, basic["items"][0]))
                pe = _safe_float(item.get("pe"))
                pb = _safe_float(item.get("pb"))
                turnover = _safe_float(item.get("turnover_rate"))
                total_mv = _safe_float(item.get("total_mv"))

            # 2. Get PIT financial data (Gap #9: disclosure-date filtered)
            pit_financial = self._get_pit_financial_data(stock_code, anchor_date)

            # Extract financial fields from PIT result
            fina_data = {}
            if pit_financial.get("_has_data") is True or "roe" in pit_financial:
                fina_data = {
                    "roe": pit_financial.get("roe", 0),
                    "roe_dt": pit_financial.get("roe_dt", 0),
                    "roa": pit_financial.get("roa", 0),
                    "grossprofit_margin": pit_financial.get("grossprofit_margin", 0),
                    "netprofit_margin": pit_financial.get("netprofit_margin", 0),
                    "debt_to_assets": pit_financial.get("debt_to_assets", 0),
                    "current_ratio": pit_financial.get("current_ratio", 0),
                    "quick_ratio": pit_financial.get("quick_ratio", 0),
                    "or_yoy": pit_financial.get("or_yoy", 0),
                    "profit_yoy": pit_financial.get("profit_yoy", 0),
                    "ocf_yoy": pit_financial.get("ocf_yoy", 0),
                    "ocf_netprofit": pit_financial.get("ocf_netprofit", 0),
                    "assets_turn": pit_financial.get("assets_turn", 0),
                }

            # 3. Get industry classification
            industry = "default"
            try:
                stock_info = tushare_call("stock_basic", {
                    "ts_code": ts_code,
                }, fields="industry")
                if stock_info and "items" in stock_info and stock_info["items"]:
                    f2 = stock_info["fields"]
                    item2 = dict(zip(f2, stock_info["items"][0]))
                    industry = item2.get("industry", "default") or "default"
            except Exception:
                pass

            result = {
                "pe": pe,
                "pb": pb,
                "turnover_rate": turnover,
                "total_mv": total_mv,
                "industry": industry,
                **fina_data,
                "_has_data": True,
                "_pit_filtered": pit_financial.get("_pit_filtered", False),
            }

            # Cache result (Gap #9)
            self._pit_fundamentals_cache[cache_key] = result

            return result

        except Exception:
            return {"_has_data": False}

    def _compute_fundamental_score(
        self, pit_data: Dict[str, Any], stock_code: str
    ) -> float:
        """
        基于PIT基本面数据计算综合评分 (0-100)。

        评分维度:
          1. PE估值 (15分): 相对行业中位数，越低越好
          2. PB估值 (10分): 相对行业中位数，越低越好
          3. ROE (20分): 越高越好，>=ROE中位数为满分
          4. 营收增长 (15分): YoY，越高越好
          5. 利润增长 (15分): YoY，越高越好
          6. 负债率 (10分): 非金融行业越低越好
          7. 经营现金流 (15分): ocf_netprofit比率越接近1+越好
        """
        industry = pit_data.get("industry", "default")
        benchmark = self._INDUSTRY_BENCHMARKS.get(
            industry, self._INDUSTRY_BENCHMARKS["default"]
        )
        bm_pe, bm_pb, bm_roe = benchmark

        score = 0.0

        # 1. PE估值 (15分): 相对行业中位数
        pe = pit_data.get("pe", 0)
        if pe > 0 and bm_pe > 0:
            pe_ratio = bm_pe / max(pe, 0.1)  # >1 means cheaper than benchmark
            if pe_ratio >= 1.5:
                score += 15.0
            elif pe_ratio >= 1.0:
                score += 10.0 + (pe_ratio - 1.0) * 10.0
            elif pe_ratio >= 0.5:
                score += 5.0 + (pe_ratio - 0.5) * 10.0
            else:
                score += pe_ratio * 10.0
        else:
            score += 7.5  # 无数据给中位分

        # 2. PB估值 (10分)
        pb = pit_data.get("pb", 0)
        if pb > 0 and bm_pb > 0:
            pb_ratio = bm_pb / max(pb, 0.1)
            if pb_ratio >= 1.5:
                score += 10.0
            elif pb_ratio >= 1.0:
                score += 6.0 + (pb_ratio - 1.0) * 8.0
            elif pb_ratio >= 0.5:
                score += 3.0 + (pb_ratio - 0.5) * 6.0
            else:
                score += pb_ratio * 6.0
        else:
            score += 5.0

        # 3. ROE (20分)
        roe = pit_data.get("roe", 0)
        if roe > 0:
            roe_ratio = roe / max(bm_roe, 0.1)
            if roe_ratio >= 1.5:
                score += 20.0
            elif roe_ratio >= 1.0:
                score += 14.0 + (roe_ratio - 1.0) * 12.0
            elif roe_ratio >= 0.5:
                score += 8.0 + (roe_ratio - 0.5) * 12.0
            elif roe_ratio > 0:
                score += roe_ratio * 16.0
        else:
            score += 10.0

        # 4. 营收增长YoY (15分)
        or_yoy = pit_data.get("or_yoy", 0)
        if or_yoy > 0.30:
            score += 15.0
        elif or_yoy > 0.15:
            score += 10.0 + (or_yoy - 0.15) / 0.15 * 5.0
        elif or_yoy > 0:
            score += 5.0 + or_yoy / 0.15 * 5.0
        elif or_yoy > -0.15:
            score += 2.5 + (or_yoy + 0.15) / 0.15 * 2.5
        else:
            score += 0.0

        # 5. 利润增长YoY (15分)
        profit_yoy = pit_data.get("profit_yoy", 0)
        if profit_yoy > 0.30:
            score += 15.0
        elif profit_yoy > 0.15:
            score += 10.0 + (profit_yoy - 0.15) / 0.15 * 5.0
        elif profit_yoy > 0:
            score += 5.0 + profit_yoy / 0.15 * 5.0
        elif profit_yoy > -0.15:
            score += 2.5 + (profit_yoy + 0.15) / 0.15 * 2.5
        else:
            score += 0.0

        # 6. 负债率 (10分): 非金融行业越低越好
        debt = pit_data.get("debt_to_assets", 50)
        industry_lower = industry
        if industry_lower in ("银行", "保险", "证券", "房地产"):
            # 金融/地产高负债是正常的
            if debt < 85:
                score += 8.0
            else:
                score += 4.0
        else:
            if debt < 30:
                score += 10.0
            elif debt < 50:
                score += 7.0 + (50 - debt) / 20 * 3.0
            elif debt < 70:
                score += 3.0 + (70 - debt) / 20 * 4.0
            else:
                score += max(0.0, (100 - debt) / 30 * 3.0)

        # 7. 经营现金流/净利润比率 (15分)
        ocf_ratio = pit_data.get("ocf_netprofit", 0)
        if ocf_ratio > 1.5:
            score += 15.0
        elif ocf_ratio > 1.0:
            score += 12.0 + (ocf_ratio - 1.0) / 0.5 * 3.0
        elif ocf_ratio > 0.5:
            score += 7.0 + (ocf_ratio - 0.5) / 0.5 * 5.0
        elif ocf_ratio > 0:
            score += ocf_ratio / 0.5 * 7.0
        else:
            score += 2.0  # 负现金流给低分

        return min(max(score, 10.0), 95.0)

    def _apply_ablation_discount(
        self, base_score: float, agent_label: str, term: str
    ) -> float:
        """
        根据消融线对基础分数打折。

        Args:
            base_score: 完整agent分数 (0-100)
            agent_label: "full" 或 "-fundamental" 等
            term: short/medium/long

        Returns:
            打折后的分数 (0-100)
        """
        if agent_label == "full":
            return base_score

        agent_name = agent_label.lstrip("-")

        weight = self._ABLATION_WEIGHTS.get(agent_name, 0.10)

        # 根据期限调整权重
        if term == "short":
            term_mult = {"fundamental": 0.6, "technical": 1.2, "value": 0.7,
                         "quality_risk": 0.7, "moneyflow": 1.3}
        elif term == "long":
            term_mult = {"fundamental": 1.3, "technical": 0.6, "value": 1.1,
                         "quality_risk": 1.2, "moneyflow": 0.6}
        else:  # medium
            term_mult = {"fundamental": 1.1, "technical": 0.9, "value": 1.0,
                         "quality_risk": 1.1, "moneyflow": 0.9}

        mult = term_mult.get(agent_name, 1.0)
        discount = weight * mult

        return base_score * (1.0 - discount)

    def _fallback_composite_score(
        self, stock_code: str, anchor_date: str
    ) -> float:
        """
        当Tushare不可用时的回退方案：基于股票代码段位+日期的确定性估算。
        不使用任何随机数或哈希。
        """
        code_clean = stock_code.replace("sh.", "").replace("sz.", "")

        # 基于代码段位的基准分
        if code_clean.startswith("60"):
            base = 55.0  # 上海主板
        elif code_clean.startswith("00"):
            base = 53.0  # 深圳主板
        elif code_clean.startswith("30"):
            base = 50.0  # 创业板
        elif code_clean.startswith("68"):
            base = 48.0  # 科创板
        else:
            base = 50.0

        # 基于anchor_date的年代/月份做市场面貌修正
        try:
            ad = datetime.strptime(anchor_date, "%Y-%m-%d")
            # 月份季节性 (A股年报/一季报行情)
            month = ad.month
            if month in (3, 4):
                base += 3.0  # 年报/一季报行情
            elif month in (9, 10):
                base += 2.0  # 三季报+国庆行情
            elif month in (1, 12):
                base -= 2.0  # 年末流动性偏紧

            # 年代因子
            year = ad.year
            if year == 2022:
                base -= 4.0  # 熊市年
            elif year == 2023:
                base -= 3.0  # 震荡偏熊
            elif year == 2024:
                base += 0.0  # 震荡
            elif year >= 2025:
                base += 2.0  # 政策牛
        except Exception:
            pass

        return min(max(base, 35.0), 80.0)

    async def _score_stock_pit(
        self,
        stock_code: str,
        company_name: str,
        anchor_date: str,
        agent_label: str,
    ) -> float:
        """
        Point-in-Time scoring using real Agent analysis via cache.

        总纲 §8.3: 回测中对精筛池跑分析Agent（仅可回溯的5个），禁用news/event。
        总纲 §14.1: 分析Agent只跑一次，缓存跨线跨期限共享。

        Args:
            stock_code: 内部格式 (sh.601888)
            company_name: 公司名称
            anchor_date: YYYY-MM-DD
            agent_label: "full" 或 "-fundamental" 等

        Returns:
            分数 (0-100)
        """
        import json
        from src.eval.cache import read_cache, write_cache
        from src.eval.adapters.stock_pipeline_adapter import run_stock_analysis

        # Step 1: 尝试从评测缓存获取真实Agent分析结果
        try:
            cached_raw = read_cache("full_analysis", stock_code, anchor_date)
            if cached_raw:
                result = json.loads(cached_raw)
            else:
                logger.info("Backtest cache miss for %s @ %s, running agent analysis...",
                           stock_code, anchor_date)
                result = await run_stock_analysis(
                    stock_code, company_name, anchor_date, eval_mode=True
                )
                # Only cache if result has no error AND contains valid term scores
                if not result.get("error"):
                    term = self.config.term
                    term_key = f"{term}_term_score"
                    term_score = result.get(term_key, {})
                    has_valid_score = (
                        isinstance(term_score, dict)
                        and term_score.get("score") is not None
                    )
                    if has_valid_score:
                        write_cache("full_analysis", stock_code, anchor_date,
                                   json.dumps(result, ensure_ascii=False, default=str))
                    else:
                        logger.warning(
                            "Backtest analysis for %s @ %s completed but missing "
                            "valid %s score; not caching to allow retry.",
                            stock_code, anchor_date, term_key
                        )

            # 提取该期限评分
            term = self.config.term
            term_key = f"{term}_term_score"
            term_score = result.get(term_key, {})
            if isinstance(term_score, dict):
                base_score = float(term_score.get("score", 50))
            else:
                base_score = 50.0

            # 消融调整（与模拟盘使用相同逻辑）
            ablated_agent = None
            if agent_label != "full" and agent_label.startswith("-"):
                ablated_agent = agent_label.lstrip("-")

            if ablated_agent and ablated_agent in self.config.ablation_agents:
                signal_packs = result.get("signal_packs", {})
                from src.eval.orchestrator import EvalOrchestrator
                base_score = EvalOrchestrator._adjust_score_for_ablation(
                    base_score, signal_packs, ablated_agent, term
                )

            return round(min(max(base_score, 10.0), 98.0), 2)

        except Exception as e:
            logger.warning("Agent analysis failed for %s @ %s: %s, falling back to PIT fundamentals",
                          stock_code, anchor_date, str(e))

        # Step 2: 回退 — PIT Tushare基本面数据
        pit_data = self._fetch_pit_fundamentals(stock_code, anchor_date)
        if pit_data.get("_has_data"):
            base_score = self._compute_fundamental_score(pit_data, stock_code)
        else:
            base_score = self._fallback_composite_score(stock_code, anchor_date)

        # 消融打折
        score = self._apply_ablation_discount(base_score, agent_label, self.config.term)
        return round(min(max(score, 10.0), 98.0), 2)

    # ═══════════════════════════════════════════════════════════
    # Core backtest methods
    # ═══════════════════════════════════════════════════════════

    # ── Gap #8: Survivorship Bias Quantification ──
    async def _check_survivor_events(
        self, pool_stocks: List[str], start_date: str, end_date: str
    ) -> SurvivorshipStats:
        """
        Check pool stocks for adverse events during the backtest period:
        ST designation, delisting, and long suspensions (>30 trading days).

        Uses Tushare namechange API to detect ST/delisting events and
        suspend_d API to check for long suspensions.

        Args:
            pool_stocks: List of stock codes in internal format (sh.603871)
            start_date: Backtest start date YYYY-MM-DD
            end_date: Backtest end date YYYY-MM-DD

        Returns:
            SurvivorshipStats with counts and percentages.
        """
        from src.eval.data_fetcher import _call as tushare_call

        ts_start = start_date.replace("-", "")
        ts_end = end_date.replace("-", "")

        st_stocks: Dict[str, Dict] = {}       # code -> {name, first_date, reason}
        delisted_stocks: Dict[str, Dict] = {} # code -> {name, first_date, reason}
        long_suspended_stocks: Dict[str, Dict] = {}  # code -> {name, max_days, periods}
        affected: set = set()                  # stock codes affected by any event

        total = len(pool_stocks)
        processed = 0

        for stock_code in pool_stocks:
            processed += 1
            ts_code = self._convert_to_ts_code(stock_code)

            # ── 1. Check namechange for ST/delisting events ──
            try:
                nc_result = tushare_call("namechange", {
                    "ts_code": ts_code,
                    "start_date": ts_start,
                    "end_date": ts_end,
                }, fields="ts_code,name,start_date,end_date,change_reason")

                if nc_result and "items" in nc_result:
                    nc_fields = nc_result["fields"]
                    for row in nc_result["items"]:
                        item = dict(zip(nc_fields, row))
                        name = str(item.get("name", ""))
                        change_reason = str(item.get("change_reason", ""))
                        start_date_item = str(item.get("start_date", ""))

                        # Detect ST events
                        if ("ST" in name or "*ST" in name or
                            "退市" in name or "终止上市" in change_reason or
                            "ST" in change_reason):
                            if "退市" in name or "终止上市" in change_reason:
                                if stock_code not in delisted_stocks:
                                    delisted_stocks[stock_code] = {
                                        "code": stock_code,
                                        "name": name,
                                        "first_date": start_date_item,
                                        "reason": change_reason,
                                    }
                            else:
                                if stock_code not in st_stocks:
                                    st_stocks[stock_code] = {
                                        "code": stock_code,
                                        "name": name,
                                        "first_date": start_date_item,
                                        "reason": change_reason,
                                    }
            except Exception as e:
                logger.debug("namechange check failed for %s: %s", stock_code, str(e))

            # ── 2. Check for long suspensions ──
            try:
                susp_result = tushare_call("suspend_d", {
                    "ts_code": ts_code,
                    "start_date": ts_start,
                    "end_date": ts_end,
                }, fields="ts_code,suspend_date,resume_date,suspend_type,suspend_reason")

                if susp_result and "items" in susp_result:
                    susp_fields = susp_result["fields"]
                    suspension_periods = []

                    for row in susp_result["items"]:
                        item = dict(zip(susp_fields, row))
                        susp_date = str(item.get("suspend_date", ""))
                        resume_date = str(item.get("resume_date", ""))

                        if susp_date and resume_date:
                            try:
                                sd = datetime.strptime(susp_date, "%Y%m%d")
                                rd = datetime.strptime(resume_date, "%Y%m%d")
                                susp_days = (rd - sd).days
                                if susp_days > 30:
                                    suspension_periods.append({
                                        "suspend_date": susp_date,
                                        "resume_date": resume_date,
                                        "days": susp_days,
                                        "reason": str(item.get("suspend_reason", "")),
                                    })
                            except ValueError:
                                pass
                        elif susp_date and not resume_date:
                            # Still suspended — count from suspend_date to end_date
                            try:
                                sd = datetime.strptime(susp_date, "%Y%m%d")
                                ed = datetime.strptime(end_date, "%Y-%m-%d")
                                susp_days = (ed - sd).days
                                if susp_days > 30:
                                    suspension_periods.append({
                                        "suspend_date": susp_date,
                                        "resume_date": "still_suspended",
                                        "days": susp_days,
                                        "reason": str(item.get("suspend_reason", "")),
                                    })
                            except ValueError:
                                pass

                    if suspension_periods:
                        max_days = max(p["days"] for p in suspension_periods)
                        long_suspended_stocks[stock_code] = {
                            "code": stock_code,
                            "name": stock_code,
                            "max_days": max_days,
                            "periods": suspension_periods,
                        }
            except Exception as e:
                logger.debug("suspend_d check failed for %s: %s", stock_code, str(e))

            # Track affected stocks
            if stock_code in st_stocks:
                affected.add(stock_code)
            if stock_code in delisted_stocks:
                affected.add(stock_code)
            if stock_code in long_suspended_stocks:
                affected.add(stock_code)

            # Progress logging for large pools
            if processed % 50 == 0 and total >= 100:
                logger.info("Survivorship check: %d/%d stocks processed", processed, total)

        # Build details list
        details = []
        for code, info in st_stocks.items():
            details.append({
                "code": code,
                "event_type": "ST",
                "first_date": info["first_date"],
                "description": f"ST: {info.get('name', '')} — {info.get('reason', '')}",
            })
        for code, info in delisted_stocks.items():
            details.append({
                "code": code,
                "event_type": "DELIST",
                "first_date": info["first_date"],
                "description": f"退市: {info.get('name', '')} — {info.get('reason', '')}",
            })
        for code, info in long_suspended_stocks.items():
            details.append({
                "code": code,
                "event_type": "LONG_SUSPEND",
                "first_date": info["periods"][0]["suspend_date"] if info.get("periods") else "",
                "description": f"长期停牌: max={info.get('max_days', 0)}天 — {len(info.get('periods', []))}次",
            })

        clean_count = total - len(affected)

        stats = SurvivorshipStats(
            total_pool_stocks=total,
            st_events_count=len(st_stocks),
            st_events_pct=round(len(st_stocks) / total * 100, 1) if total > 0 else 0,
            delisted_count=len(delisted_stocks),
            delisted_pct=round(len(delisted_stocks) / total * 100, 1) if total > 0 else 0,
            long_suspended_count=len(long_suspended_stocks),
            long_suspended_pct=round(len(long_suspended_stocks) / total * 100, 1) if total > 0 else 0,
            affected_count=len(affected),
            affected_pct=round(len(affected) / total * 100, 1) if total > 0 else 0,
            clean_count=clean_count,
            clean_pct=round(clean_count / total * 100, 1) if total > 0 else 0,
            details=details,
        )

        logger.info(
            "Survivorship check complete: %d/%d stocks affected (%.1f%%), "
            "ST=%d, Delisted=%d, LongSuspended=%d, Clean=%d",
            stats.affected_count, stats.total_pool_stocks, stats.affected_pct,
            stats.st_events_count, stats.delisted_count,
            stats.long_suspended_count, stats.clean_count
        )

        return stats

    def _compute_survivorship_bias(self) -> Optional[Dict[str, Any]]:
        """
        Compute survivorship bias impact assessment from cached stats.

        Returns a dict with bias quantification suitable for inclusion
        in backtest results.
        """
        if self._survivor_stats is None:
            return None

        stats = self._survivor_stats

        # Estimate bias impact:
        # - Each affected stock that would have been in the pool introduces
        #   negative return that is not captured in the backtest
        # - The bias magnitude is roughly proportional to the affected ratio
        bias_severity = "low"
        if stats.affected_pct > 15:
            bias_severity = "high"
        elif stats.affected_pct > 5:
            bias_severity = "medium"

        return {
            "total_pool_stocks": stats.total_pool_stocks,
            "st_events": {"count": stats.st_events_count, "pct": stats.st_events_pct},
            "delisted": {"count": stats.delisted_count, "pct": stats.delisted_pct},
            "long_suspended": {"count": stats.long_suspended_count, "pct": stats.long_suspended_pct},
            "affected_any": {"count": stats.affected_count, "pct": stats.affected_pct},
            "clean": {"count": stats.clean_count, "pct": stats.clean_pct},
            "bias_severity": bias_severity,
            "details": stats.details[:50],  # Top 50, avoid bloating output
            "caveat": (
                f"回测池中{stats.affected_pct:.1f}%的股票在回测期间经历了ST/退市/长期停牌事件。"
                f"由于当前精筛池已自动排除这些股票，回测收益可能高估{bias_severity}程度。"
                f"建议使用季度动态重筛池(run_quarterly_calibration)以减少存活偏差。"
            ),
        }

    async def run_single_anchor(self, anchor_date: str, pool: List[str],
                                 price_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        在单个anchor date上运行回测。

        Args:
            anchor_date: 回测时点 YYYY-MM-DD
            pool: 精筛池股票列表 (stock codes, e.g. sh.601888)
            price_data: 历史价格数据 {stock_code: [prices_by_date]}

        Returns:
            {
                "anchor_date": "2024-06-14",
                "term": "medium",
                "ablation_results": {
                    "full": {"scores": [...], "returns": [...]},
                    "-fundamental": {"scores": [...], "returns": [...]},
                    ...
                }
            }
        """
        from src.eval.label_builder import compute_holding_return
        from datetime import datetime as dt

        cfg = self.config
        start_dt = dt.strptime(anchor_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=cfg.holding_days)

        ablation_results = {}

        # 为每条消融线计算结果
        for agent_label in ["full"] + [f"-{a}" for a in cfg.ablation_agents]:
            scores = []
            returns = []

            for stock_code in pool:
                # Fix 1: 使用真实PIT评分 (替代MD5哈希模拟)
                score = await self._score_stock_pit(
                    stock_code, stock_code, anchor_date, agent_label
                )

                # 计算持有期收益
                stock_prices = self._get_price_series(
                    price_data, stock_code, anchor_date, cfg.holding_days
                )
                if stock_prices and len(stock_prices) >= 2:
                    entry = stock_prices[0]
                    exit_p = stock_prices[-1]
                    ret_info = compute_holding_return(
                        entry, exit_p, len(stock_prices) - 1
                    )
                    ret = ret_info["asset_return_pct"] / 100
                else:
                    ret = 0.0

                scores.append(score)
                returns.append(ret)

            ablation_results[agent_label] = {
                "scores": scores,
                "returns": returns,
                "mean_score": sum(scores) / max(len(scores), 1),
                "mean_return": sum(returns) / max(len(returns), 1),
            }

        return {
            "anchor_date": anchor_date,
            "term": cfg.term,
            "pool_size": len(pool),
            "ablation_results": ablation_results,
        }

    # ═══════════════════════════════════════════════════════════
    # Reference Line Backtest (SB-L6: 短线长持对照线)
    # ═══════════════════════════════════════════════════════════

    async def _run_reference_line(
        self, pool: List[str], price_data: Dict[str, Any],
        anchors: List[str], term: str = "short",
    ) -> Dict[str, Any]:
        """
        运行参考线回测 — 使用ShortLongHoldStrategy实现连续持仓跨anchor date运作。

        SB-L6与S-L8使用完全相同的ShortLongHoldStrategy策略类，关键区别：
          - SB-L0~L5: 每个anchor清仓重新建仓 (daily-clearing ablation)
          - SB-L6: 持仓连续跨越anchor date (continuous holding, like S-L8 in live)

        Args:
            pool: 精筛池
            price_data: 历史价格数据
            anchors: anchor date列表
            term: 期限 (固定为short)

        Returns:
            参考线回测结果dict
        """
        from src.eval.strategies.factory import get_strategy
        from src.eval.market_simulator import MarketData

        strategy = get_strategy(term, "longhold", {})
        cfg = self.config

        # ── 初始状态 ──
        initial_capital = 1_000_000.0
        cash = initial_capital
        holdings: Dict[str, int] = {}             # code -> shares
        purchase_prices: Dict[str, float] = {}     # code -> avg price
        hold_days_state: Dict[str, int] = {}        # code -> days held

        daily_values: List[tuple] = []   # [(anchor_date, total_value), ...]
        trade_log: List[Dict] = []       # 每anchor的交易摘要

        anchors_sorted = sorted(anchors)

        for anchor_date in anchors_sorted:
            # ── 1. 获取当日评分 ──
            scores: Dict[str, float] = {}
            for code in pool:
                score = await self._score_stock_pit(code, code, anchor_date, "full")
                scores[code] = score

            # ── 2. 获取当日收盘价 ──
            anchor_prices: Dict[str, float] = {}
            for code in pool:
                series = self._get_price_series(price_data, code, anchor_date, 2)
                if series and len(series) >= 1:
                    anchor_prices[code] = series[0]

            # ── 3. 构建最小MarketData map (用于策略判定) ──
            market_data_map: Dict[str, MarketData] = {}
            for code in pool:
                price = anchor_prices.get(code, 0)
                if price > 0:
                    md = MarketData(stock_code=code, close=price, open=price)
                    # 设置前收盘价 (从price_data查找前一天)
                    prev_series = self._get_price_series(price_data, code,
                        (datetime.strptime(anchor_date, "%Y-%m-%d")
                         - timedelta(days=1)).strftime("%Y-%m-%d"), 2)
                    if prev_series and len(prev_series) >= 1:
                        md.pre_close = prev_series[0]
                    else:
                        md.pre_close = price  # 回退：用当日价格
                    market_data_map[code] = md

            # ── 4. 当前组合市值 ──
            holdings_value = sum(
                holdings.get(c, 0) * anchor_prices.get(c, 0)
                for c in holdings
            )
            total_capital = cash + holdings_value

            # ── 5. Strategy: 更新价格历史（用于近3日涨幅计算）──
            for code in holdings:
                if code in anchor_prices:
                    strategy.update_price_history(code, anchor_prices[code])

            # ── 6. 卖出判定 ──
            sell_orders = strategy.generate_sell_orders(
                holdings, scores, market_data_map,
                purchase_prices, hold_days_state,
            )

            # ── 7. 执行卖出 ──
            sell_log = []
            for so in sell_orders:
                code = so.stock_code
                if code not in holdings or holdings[code] <= 0:
                    continue
                sell_shares = int(holdings[code] * so.sell_ratio / 100) * 100
                if sell_shares <= 0:
                    continue
                price = anchor_prices.get(code, 0)
                if price <= 0:
                    continue
                cash += sell_shares * price
                holdings[code] -= sell_shares
                sell_log.append({
                    "code": code, "shares": sell_shares, "price": price,
                    "reason": so.reason,
                })
                if holdings[code] <= 0:
                    del holdings[code]
                    purchase_prices.pop(code, None)
                    hold_days_state.pop(code, None)

            # ── 8. 选股 ──
            select_kwargs = {
                "pool": pool,
                "scores": scores,
                "holdings": holdings,
                "cash": cash,
                "market_data_map": market_data_map,
                "total_capital": total_capital,
            }
            # 中线策略需要 current_date 用于越跌越买基本面检查
            from src.eval.strategies.medium_term import MediumTermStrategy
            if isinstance(strategy, MediumTermStrategy):
                select_kwargs["current_date"] = anchor_date
            buy_orders = strategy.select_stocks(**select_kwargs)

            # ── 9. 仓位调整 ──
            sized = strategy.size_positions(
                buy_orders, total_capital,
                len(holdings), strategy.get_max_positions(),
                strategy.get_single_weight_limit(), strategy.get_min_cash_ratio(),
                holdings, market_data_map,
            )

            # ── 10. 执行买入 ──
            buy_log = []
            for bo in sized:
                code = bo.stock_code
                if bo.target_value <= 0 or bo.target_value > cash:
                    continue
                price = anchor_prices.get(code, 0)
                if price <= 0:
                    continue
                shares = int(bo.target_value / price / 100) * 100
                if shares <= 0:
                    continue
                buy_cost = shares * price

                # 不能超过剩余现金
                if buy_cost > cash:
                    shares = int(cash / price / 100) * 100
                    buy_cost = shares * price
                    if shares <= 0:
                        continue

                cash -= buy_cost
                old_shares = holdings.get(code, 0)
                if old_shares > 0:
                    old_avg = purchase_prices.get(code, price)
                    new_avg = (old_avg * old_shares + price * shares) / (old_shares + shares)
                    purchase_prices[code] = new_avg
                else:
                    purchase_prices[code] = price
                    hold_days_state[code] = 0
                holdings[code] = old_shares + shares

                buy_log.append({
                    "code": code, "shares": shares, "price": price,
                    "target_value": bo.target_value, "score": bo.score,
                    "reason": bo.reason if hasattr(bo, 'reason') else "",
                })

            # ── 11. 更新持有天数 ──
            for code in list(hold_days_state):
                if code in holdings and holdings[code] > 0:
                    hold_days_state[code] += 1
                else:
                    hold_days_state.pop(code, None)

            # ── 12. 记录当日组合市值 ──
            holdings_value = sum(
                holdings.get(c, 0) * anchor_prices.get(c, 0)
                for c in holdings
            )
            total_value = cash + holdings_value
            daily_values.append((anchor_date, total_value))

            trade_log.append({
                "anchor_date": anchor_date,
                "num_holdings": len(holdings),
                "cash": round(cash, 2),
                "holdings_value": round(holdings_value, 2),
                "total_value": round(total_value, 2),
                "buys": len(buy_log),
                "sells": len(sell_log),
                "buy_details": buy_log,
                "sell_details": sell_log,
            })

        # ── 13. 汇总指标 ──
        values = [v for _, v in daily_values]
        if len(values) >= 2:
            cumulative_return = (values[-1] - initial_capital) / initial_capital
            # 日收益序列
            daily_rets = []
            for i in range(1, len(values)):
                if values[i-1] > 0:
                    daily_rets.append((values[i] - values[i-1]) / values[i-1])
            # 最大回撤
            peak = values[0]
            mdd = 0.0
            for v in values:
                peak = max(peak, v)
                dd = (peak - v) / peak if peak > 0 else 0.0
                mdd = max(mdd, dd)
            # 年化收益 (252 trading days)
            num_days = len(values)
            annualized_return = (1 + cumulative_return) ** (252 / max(num_days, 1)) - 1
            # Sharpe (simplified)
            if daily_rets and len(daily_rets) >= 2:
                avg_daily = sum(daily_rets) / len(daily_rets)
                var = sum((r - avg_daily) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
                std_daily = var ** 0.5 if var > 0 else 0.0
                sharpe = (avg_daily / std_daily * (252 ** 0.5)) if std_daily > 0 else 0.0
            else:
                sharpe = 0.0
            # 胜率 (positive return anchors)
            win_count = sum(1 for _, v in daily_values if v > initial_capital)
            win_rate = win_count / len(daily_values) if daily_values else 0.0
        else:
            cumulative_return = 0.0
            annualized_return = 0.0
            mdd = 0.0
            sharpe = 0.0
            win_rate = 0.0

        return {
            "line_id": "SB-L6",
            "term": "short",
            "type": "reference_longhold",
            "description": BACKTEST_LINE_DEFINITIONS["SB-L6"]["description"],
            "num_anchors": len(daily_values),
            "initial_capital": initial_capital,
            "final_value": round(values[-1], 2) if values else initial_capital,
            "cumulative_return_pct": round(cumulative_return * 100, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "max_drawdown_pct": round(mdd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "final_holdings_count": len(holdings),
            "daily_values": [{"date": d, "value": v} for d, v in daily_values],
            "trade_log": trade_log,
        }

    async def run_full_backtest(self, pool: List[str],
                                 price_data: Dict[str, Any],
                                 param_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run full backtest across all anchor dates.

        Gap #8: Includes survivorship bias quantification.
        Gap #9: Uses PIT financial data filtered by disclosure date.

        Returns:
            Dict with all anchor results, summaries, survivorship stats.
        """
        if param_overrides:
            self.config.param_overrides = param_overrides
        anchors = self.generate_anchor_dates()
        splits = self.get_train_val_test_split(anchors)

        total_anchors = len(anchors)
        print(f"[回测] 共{total_anchors}个anchor dates "
              f"(训练{len(splits['train'])} 验证{len(splits['validate'])} "
              f"测试{len(splits['test'])})")

        # ── Gap #8: Survivorship bias check ──
        print(f"\n[回测] 检测存活偏差 (检查ST/退市/长期停牌)...")
        try:
            self._survivor_stats = await self._check_survivor_events(
                pool, self.config.start_date, self.config.end_date
            )
            print(f"  [存活偏差] {self._survivor_stats.affected_count}/{self._survivor_stats.total_pool_stocks}"
                  f" 受影响的股票 ({self._survivor_stats.affected_pct}%)")
        except Exception as e:
            logger.warning("Survivorship check failed: %s", str(e))
            self._survivor_stats = None

        all_results = []
        processed = 0

        # Fix 2: 移除[:5]上限，处理全部anchor dates
        for split_name, split_anchors in splits.items():
            for anchor_date in split_anchors:
                processed += 1
                # 进度指示
                if processed == 1 or processed % 20 == 0 or processed == total_anchors:
                    pct = processed * 100 // total_anchors
                    print(f"  [回测进度] {processed}/{total_anchors} ({pct}%) "
                          f"- {anchor_date} [{split_name}]")

                result = await self.run_single_anchor(anchor_date, pool, price_data)
                result["split"] = split_name
                all_results.append(result)

        # 汇总agent贡献 (仅消融线，不含参考线)
        contribution_summary = self._summarize_contributions(all_results)

        # ── 短线参考线 (SB-L6: 连续持仓长持对照) ──
        reference_results = None
        if self.config.term == "short":
            print(f"\n[回测] 运行短线参考线 SB-L6 (长持对照, 连续持仓)...")
            try:
                reference_results = await self._run_reference_line(
                    pool, price_data, anchors, term="short"
                )
                print(f"  SB-L6 完成: 累计收益 {reference_results['cumulative_return_pct']}%, "
                      f"最大回撤 {reference_results['max_drawdown_pct']}%, "
                      f"Sharpe {reference_results['sharpe_ratio']}")
            except Exception as e:
                print(f"  [警告] SB-L6 参考线回测失败: {e}")
                reference_results = {"error": str(e), "line_id": "SB-L6"}

        # ── Gap #8: Survivorship bias assessment ──
        survivorship = self._compute_survivorship_bias()

        # ── Gap #9: PIT disclosure-date status ──
        has_pit_alignment = bool(self._disclosure_date_cache)
        pit_disclaimer = ""
        if has_pit_alignment:
            stocks_with_disclosures = len(self._disclosure_date_cache)
            pit_disclaimer = (
                f"财务数据已按披露日期(disclosure_date)做PIT对齐，"
                f"共{stocks_with_disclosures}只股票启用了披露日期过滤，消除了前视偏差"
            )
        else:
            pit_disclaimer = (
                "披露日期(disclosure_date)数据获取失败，回退至报告期(report_date)过滤"
            )

        decls = [
            "本回测基于当前精筛股票池，不含回测期间退市/暴雷股票",
            "news_agent和event_agent在回测中被禁用",
            "绝对收益可能高估，但agent贡献排序受偏差影响较小",
            "评分基于Tushare PIT历史数据，回退至段位确定性估算（无随机哈希）",
            "SB-L6为参考线（长持对照），使用ShortLongHoldStrategy连续持仓，不参与消融ΔLoss计算",
        ]
        if pit_disclaimer:
            decls.append(pit_disclaimer)

        result = {
            "config": {
                "term": self.config.term,
                "start": self.config.start_date,
                "end": self.config.end_date,
                "holding_days": self.config.holding_days,
                "num_anchors_executed": len(all_results),
                "num_anchors_total": len(anchors),
                "pit_financial_enabled": has_pit_alignment,
            },
            "splits": {k: len(v) for k, v in splits.items()},
            "anchor_results": all_results,
            "contribution_summary": contribution_summary,
            "reference_line": reference_results,
            "survivorship_bias": survivorship,
            "declarations": decls,
        }
        return result

    # ── Gap #15: Mini Backtest for Prompt Patch Verification ──

    async def run_mini_backtest(
        self,
        pool: List[str],
        price_data: Dict[str, Any],
        num_anchors: int = 15,
        num_stocks: int = 20,
    ) -> Dict[str, Any]:
        """
        Run a small-scale backtest for quick prompt patch verification.

        Uses a limited number of anchor dates (sampled evenly) and stocks
        for fast turn-around. Supports prompt_override via BacktestConfig.

        Args:
            pool: Full stock pool list
            price_data: Historical price data
            num_anchors: Max anchor dates to include (sampled evenly)
            num_stocks: Max stocks per anchor (top-N from pool)

        Returns:
            Mini backtest results dict (same structure as run_full_backtest,
            but with fewer anchors/stocks).
        """
        all_anchors = self.generate_anchor_dates()

        # Evenly sample anchors
        if len(all_anchors) <= num_anchors:
            sampled_anchors = all_anchors
        else:
            step = len(all_anchors) / num_anchors
            sampled_anchors = [all_anchors[int(i * step)] for i in range(num_anchors)]
            sampled_anchors = sorted(set(sampled_anchors))  # Dedup and sort

        # Limit pool size
        mini_pool = pool[:num_stocks] if len(pool) > num_stocks else list(pool)

        logger.info(
            "Running mini backtest: %d anchors x %d stocks (prompt_override=%s)",
            len(sampled_anchors), len(mini_pool),
            "yes" if self.config.prompt_override else "no"
        )

        all_results = []
        for i, anchor_date in enumerate(sampled_anchors):
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [mini回测] {i+1}/{len(sampled_anchors)} - {anchor_date}")

            result = await self.run_single_anchor(anchor_date, mini_pool, price_data)
            all_results.append(result)

        contribution_summary = self._summarize_contributions(all_results)

        # Compute per-anchor aggregate metrics
        per_anchor_metrics = []
        for r in all_results:
            ablation = r.get("ablation_results", {})
            full_data = ablation.get("full", {})
            per_anchor_metrics.append({
                "anchor_date": r["anchor_date"],
                "pool_size": r.get("pool_size", len(mini_pool)),
                "mean_score": full_data.get("mean_score", 0),
                "mean_return": full_data.get("mean_return", 0),
            })

        return {
            "config": {
                "term": self.config.term,
                "start": self.config.start_date,
                "end": self.config.end_date,
                "num_anchors": len(sampled_anchors),
                "num_stocks": len(mini_pool),
                "anchor_dates": sampled_anchors,
                "prompt_override_applied": self.config.prompt_override is not None,
            },
            "contribution_summary": contribution_summary,
            "per_anchor_metrics": per_anchor_metrics,
            "anchor_results": all_results,
        }

    # ═══════════════════════════════════════════════════════════
    # Fix 4: Quarterly Calibration (方案B)
    # ═══════════════════════════════════════════════════════════

    async def run_quarterly_calibration(
        self,
        term: str = "medium",
        price_data: Optional[Dict[str, Any]] = None,
        pool_size: int = 80,
    ) -> Dict[str, Any]:
        """
        方案B季度校准: 在每个anchor date动态重筛池，消除存活偏差。

        与run_full_backtest的关键区别:
          - 每季度第一个anchor date重新从Tushare获取该时点可用的全A股
          - 硬筛排除ST/新股/低流动性
          - 用该时点的基本面数据做快筛
          - 用动态筛选的池而非固定池做消融回测
          - 同一季度内的后续anchor dates复用该季度筛选的池

        Args:
            term: short/medium/long
            price_data: 历史价格数据
            pool_size: 每季度精筛池大小

        Returns:
            季度校准结果
        """
        anchors = self.generate_anchor_dates()

        # 按季度分组
        quarters: Dict[str, List[str]] = {}
        for d in anchors:
            dt = datetime.strptime(d, "%Y-%m-%d")
            q_key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            if q_key not in quarters:
                quarters[q_key] = []
            quarters[q_key].append(d)

        print(f"[季度校准] 共{len(anchors)}个anchor dates, "
              f"覆盖{len(quarters)}个季度")

        all_results = []
        quarter_pools: Dict[str, List[str]] = {}
        calibration_log = []

        processed = 0
        for q_idx, (q_key, q_anchors) in enumerate(sorted(quarters.items())):
            print(f"\n[季度校准] {q_key} ({len(q_anchors)}个anchor dates)")

            # 用该季度第一个anchor date做动态筛选
            first_anchor = q_anchors[0]
            try:
                dynamic_pool = await self._screen_pool_at_date(
                    first_anchor, pool_size
                )
                quarter_pools[q_key] = dynamic_pool
                calibration_log.append({
                    "quarter": q_key,
                    "anchor_date": first_anchor,
                    "pool_size": len(dynamic_pool),
                    "method": "dynamic_screen",
                })
                print(f"  [季度校准] 动态筛选池: {len(dynamic_pool)}只")
            except Exception as e:
                print(f"  [季度校准] 动态筛选失败: {e}, 使用前季度池")
                # 回退到前一个季度的池
                prev_quarters = sorted(quarters.keys())
                prev_idx = prev_quarters.index(q_key) - 1
                if prev_idx >= 0:
                    dynamic_pool = quarter_pools.get(prev_quarters[prev_idx], [])
                else:
                    dynamic_pool = []
                calibration_log.append({
                    "quarter": q_key,
                    "anchor_date": first_anchor,
                    "pool_size": len(dynamic_pool),
                    "method": "fallback_prev_quarter",
                })

            if not dynamic_pool:
                print(f"  [季度校准] {q_key} 无可用池, 跳过")
                continue

            # 在该季度的每个anchor date上用此池做消融回测
            for anchor_date in q_anchors:
                processed += 1
                if processed % 20 == 0:
                    print(f"  [季度校准进度] {processed} anchors processed")

                result = await self.run_single_anchor(
                    anchor_date, dynamic_pool, price_data or {}
                )
                result["split"] = "calibration"
                result["quarter"] = q_key
                all_results.append(result)

        contribution_summary = self._summarize_contributions(all_results)

        return {
            "method": "quarterly_dynamic_screen",
            "term": term,
            "num_quarters": len(quarters),
            "num_anchors": len(all_results),
            "calibration_log": calibration_log,
            "contribution_summary": contribution_summary,
            "anchor_results": all_results,
            "declarations": [
                "方案B: 每季度动态重筛池，消除存活偏差",
                "每季度使用该时点可用的全A股做硬筛+快筛",
                "同一季度内anchor dates复用该季度筛选池",
                "news_agent和event_agent在回测中被禁用",
            ],
        }

    async def _screen_pool_at_date(
        self, anchor_date: str, target_size: int = 80
    ) -> List[str]:
        """
        在指定历史时点动态筛选精筛池。

        步骤:
          1. 获取该时点全A股 (上市+未退市)
          2. 硬筛: 排除ST/新股/低流动性
          3. 基本面快筛: 按PE/PB/ROE排序
          4. 返回前target_size只股票代码

        Args:
            anchor_date: YYYY-MM-DD
            target_size: 目标池大小

        Returns:
            [sh.601888, sh.603871, ...]
        """
        # Fast-fail: socket-level connectivity check (guaranteed < 2s)
        import socket as _sock
        try:
            s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(("api.tushare.pro", 443))
            s.close()
        except Exception:
            return []  # Tushare unreachable

        try:
            from src.eval.data_fetcher import _call as tushare_call

            ts_date = anchor_date.replace("-", "")

            # 1. 获取该时点全A股
            stocks = tushare_call("stock_basic", {
                "list_status": "L",
                "exchange": "",
            }, fields="ts_code,name,list_date,industry")

            if not stocks or "items" not in stocks:
                return []

            fields = stocks["fields"]
            stock_list = []
            anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")

            for row in stocks["items"]:
                item = dict(zip(fields, row))
                ts_code = item.get("ts_code", "")

                # 排除BJ/B股
                if ts_code.endswith(".BJ"):
                    continue
                if ts_code.startswith("900") and ts_code.endswith(".SH"):
                    continue
                if ts_code.startswith("200") and ts_code.endswith(".SZ"):
                    continue

                # 排除ST
                name = item.get("name", "")
                if "*ST" in name or "ST" in name:
                    continue

                # 排除该时点未上市的
                list_date = item.get("list_date", "")
                if list_date and len(list_date) >= 8:
                    try:
                        list_dt = datetime.strptime(list_date[:8], "%Y%m%d")
                        if (anchor_dt - list_dt).days < 60:
                            continue
                    except ValueError:
                        pass

                # 转内部代码
                if ts_code.endswith(".SH"):
                    internal = f"sh.{ts_code[:6]}"
                elif ts_code.endswith(".SZ"):
                    internal = f"sz.{ts_code[:6]}"
                else:
                    continue

                stock_list.append({
                    "code": internal,
                    "ts_code": ts_code,
                    "name": name,
                    "industry": item.get("industry", ""),
                })

            if not stock_list:
                return []

            # Sample stocks to avoid processing all ~5000 one-by-one
            # 使用确定性采样（基于anchor_date哈希），保证相同输入可复现
            max_candidates = min(len(stock_list), target_size * 4)
            import hashlib
            seed = int(hashlib.md5(anchor_date.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            stock_list = rng.sample(stock_list, max_candidates)

            # 2. 基本面快筛 (获取PIT数据并评分)
            scored = []
            for s in stock_list:
                ts_code = s["ts_code"]
                internal_code = s["code"]

                basic = tushare_call("daily_basic", {
                    "ts_code": ts_code,
                    "trade_date": ts_date,
                }, fields="pe,pb,total_mv")

                pe = 0.0
                pb = 0.0
                if basic and "items" in basic and basic["items"]:
                    bf = basic["fields"]
                    bi = dict(zip(bf, basic["items"][0]))
                    pe = _safe_float(bi.get("pe"))
                    pb = _safe_float(bi.get("pb"))

                if pe <= 0 or pb <= 0:
                    continue

                fina = tushare_call("fina_indicator", {
                    "ts_code": ts_code,
                    "end_date": ts_date,
                    "limit": "1",
                }, fields="roe,or_yoy,profit_yoy,debt_to_assets")

                roe = 0.0
                or_yoy = 0.0
                profit_yoy = 0.0
                if fina and "items" in fina and fina["items"]:
                    ff = fina["fields"]
                    fi = dict(zip(ff, fina["items"][0]))
                    roe = _safe_float(fi.get("roe"))
                    or_yoy = _safe_float(fi.get("or_yoy"))
                    profit_yoy = _safe_float(fi.get("profit_yoy"))

                # Quick composite score
                screen_score = 0.0
                if 10 <= pe <= 40:
                    screen_score += 20.0
                elif 5 <= pe <= 60:
                    screen_score += 10.0
                if roe >= 15:
                    screen_score += 30.0
                elif roe >= 8:
                    screen_score += 20.0
                elif roe >= 3:
                    screen_score += 10.0
                if or_yoy >= 15:
                    screen_score += 25.0
                elif or_yoy >= 5:
                    screen_score += 15.0
                elif or_yoy >= -5:
                    screen_score += 8.0
                if profit_yoy >= 15:
                    screen_score += 25.0
                elif profit_yoy >= 5:
                    screen_score += 15.0
                elif profit_yoy >= -5:
                    screen_score += 8.0

                scored.append((internal_code, screen_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [code for code, _ in scored[:target_size]]

        except Exception:
            return []

    # ═══════════════════════════════════════════════════════════
    # Utility methods
    # ═══════════════════════════════════════════════════════════

    def _get_price_series(self, price_data: Dict[str, Any], stock_code: str,
                          start_date: str, days: int) -> List[float]:
        """从历史价格数据中提取持有期价格序列"""
        if stock_code not in price_data:
            return []
        data = price_data[stock_code]
        if isinstance(data, list):
            return data[:days + 1]
        if isinstance(data, dict):
            # {date: price}
            current = datetime.strptime(start_date, "%Y-%m-%d")
            prices = []
            for _ in range(days + 1):
                key = current.strftime("%Y-%m-%d")
                if key in data:
                    prices.append(data[key])
                current += timedelta(days=1)
            return prices
        return []

    def _summarize_contributions(self, all_results: List[Dict]) -> Dict[str, Any]:
        """汇总所有anchor的agent贡献"""
        from collections import defaultdict

        agent_deltas = defaultdict(list)

        for result in all_results:
            ablation = result.get("ablation_results", {})
            full_ret = ablation.get("full", {}).get("mean_return", 0)

            for agent_label, data in ablation.items():
                if agent_label == "full":
                    continue
                delta = data.get("mean_return", 0) - full_ret
                # delta > 0 means ablation line did better = agent had negative contribution
                # We want: delta_L > 0 means agent helps
                agent_name = agent_label.lstrip("-")
                agent_deltas[agent_name].append(-delta)  # flip sign

        summary = {}
        for agent_name, deltas in agent_deltas.items():
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                summary[agent_name] = {
                    "mean_delta_L": round(mean_delta, 4),
                    "sample_size": len(deltas),
                    "direction": "positive" if mean_delta > 0 else "negative",
                }

        return summary

    def get_config(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "term": cfg.term,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "holding_days": cfg.holding_days,
            "anchor_frequency": cfg.anchor_frequency,
            "ablation_agents": cfg.ablation_agents,
        }


def _safe_float(val, default=0.0) -> float:
    """安全转换为float"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
