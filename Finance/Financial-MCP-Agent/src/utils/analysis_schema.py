"""
分析中间产物的结构化 Schema 定义。
使用轻量 dataclass，不引入重型依赖。
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


class SourceLevel:
    """证据可靠性层级"""
    OFFICIAL = "official_like"
    STRUCTURED = "structured"
    NEWS = "news"
    DERIVED = "derived"
    PROXY = "proxy"


SOURCE_PRIORITY = {
    SourceLevel.OFFICIAL: 5,
    SourceLevel.STRUCTURED: 4,
    SourceLevel.NEWS: 3,
    SourceLevel.DERIVED: 2,
    SourceLevel.PROXY: 1,
}

# ── 信号类目枚举（4.3 冲突检测归组用） ──
# 由分析 Agent 的 LLM 在产出 signal_pack 时按枚举自行声明，
# 代码侧只做枚举相等比较，不做词典/正则匹配。
SIGNAL_CATEGORIES = [
    "fundamentals_growth",          # 业绩成长性
    "fundamentals_profit_quality",  # 盈利质量
    "valuation",                    # 估值
    "balance_sheet",                # 资产负债结构
    "cashflow",                     # 现金流
    "governance",                   # 公司治理
    "capital_flow",                 # 资金流向
    "technical_trend",              # 技术趋势
    "sentiment",                    # 舆情情绪
    "catalyst_event",               # 事件催化
    "dividend",                     # 分红回报
    "ownership",                    # 股权结构
    "industry_policy",              # 行业与政策
    "liquidity",                    # 流动性/量价
    "risk_flag",                    # 风险事件
    "other",                        # 无法归类（兜底）
]

# signal_pack 缓存 schema 版本：
# v1 = 无 category 字段的旧格式；v2 = 信号含 category 枚举字段。
# read_signal_pack_cache 对版本低于当前版本的缓存按 miss 处理（4.9-1 兜底路径）。
SIGNAL_PACK_SCHEMA_VERSION = 2


@dataclass
class Signal:
    factor: str
    direction: int           # 1=看多, -1=看空, 0=中性
    strength: int            # 0-100
    confidence: float        # 0.0-1.0
    time_horizon: List[str]  # ["short","medium","long"]
    source_level: str
    freshness: str           # "intraday"|"daily"|"weekly"|"quarterly"|"unknown"
    risk_flags: List[str]
    note: str


@dataclass
class SignalPack:
    agent_name: str
    analysis_text: str
    bias: str                # "bullish"|"neutral"|"bearish"
    confidence: float
    data_quality_score: float
    key_points: List[str]
    signals: List[Signal]
    risk_flags: List[str]
    missing_data: List[str]
    source_summary: str
    as_of_date: str


@dataclass
class AnalysisPackage:
    as_of_date: str
    executed_agents: List[str]
    available_agents: List[str]
    missing_agents: List[str]
    global_risk_flags: List[str]
    global_missing_data: List[str]
    bullish_signals: List[dict]
    bearish_signals: List[dict]
    conflicting_signals: List[dict]
    source_priority_summary: Dict[str, Any]
    compact_prompt_context: str


@dataclass
class RiskGateResult:
    risk_level: str            # "low"|"medium"|"high"|"critical"
    risk_flags_found: List[str]
    score_cap: Optional[int]
    action_downgrade: Optional[str]
    abstain: bool
    abstain_reason: str
    data_quality_score: float
    warnings: List[str]


@dataclass
class DecisionPack:
    """最终决策结构化产物 — 评测系统的核心输入。

    signal_pack 描述"证据"，decision_pack 描述"最终决策"。
    """
    asset_type: str             # "stock" / "fund"
    symbol: str                 # 股票/基金代码
    name: str = ""              # 名称
    task_type: str = ""         # "single_stock" / "stock_pool" / "fund_analysis"
    term: str = ""              # "short" / "medium" / "long" / "fund"
    as_of_date: str = ""        # 评测时点 YYYY-MM-DD
    action: str = ""            # "strong_buy" / "buy" / "cautious_buy" / "hold" / "cautious_sell" / "sell" / "strong_sell"
    score: float = 0.0          # 0-100
    confidence: float = 0.0     # 0-1
    data_quality_score: float = 0.0  # 0-1
    risk_gate_applied: bool = False
    risk_gate_result: Optional[Dict[str, Any]] = None
    supporting_agents: Optional[List[str]] = None
    missing_agents: Optional[List[str]] = None
    key_positive_signals: Optional[List[str]] = None
    key_negative_signals: Optional[List[str]] = None
    conflicts: Optional[List[str]] = None
    model_profile: str = ""
    version_hash: str = ""
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def normalize(data: Dict[str, Any]) -> 'DecisionPack':
        """从dict构造DecisionPack，做类型标准化和fallback"""
        if not data or not isinstance(data, dict):
            return DecisionPack(asset_type="unknown", symbol="unknown")

        def _safe_float(v, default=0.0):
            try:
                return float(v)
            except (ValueError, TypeError):
                return default

        def _safe_bool(v, default=False):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes")
            return default

        return DecisionPack(
            asset_type=str(data.get("asset_type", "unknown")),
            symbol=str(data.get("symbol", "unknown")),
            name=str(data.get("name", "")),
            task_type=str(data.get("task_type", "")),
            term=str(data.get("term", "")),
            as_of_date=str(data.get("as_of_date", "")),
            action=str(data.get("action", "")),
            score=_safe_float(data.get("score"), 0.0),
            confidence=_safe_float(data.get("confidence"), 0.0),
            data_quality_score=_safe_float(data.get("data_quality_score"), 0.0),
            risk_gate_applied=_safe_bool(data.get("risk_gate_applied")),
            risk_gate_result=data.get("risk_gate_result") if isinstance(data.get("risk_gate_result"), dict) else None,
            supporting_agents=data.get("supporting_agents") if isinstance(data.get("supporting_agents"), list) else None,
            missing_agents=data.get("missing_agents") if isinstance(data.get("missing_agents"), list) else None,
            key_positive_signals=data.get("key_positive_signals") if isinstance(data.get("key_positive_signals"), list) else None,
            key_negative_signals=data.get("key_negative_signals") if isinstance(data.get("key_negative_signals"), list) else None,
            conflicts=data.get("conflicts") if isinstance(data.get("conflicts"), list) else None,
            model_profile=str(data.get("model_profile", "")),
            version_hash=str(data.get("version_hash", "")),
            meta=data.get("meta") if isinstance(data.get("meta"), dict) else None,
        )

    def to_json(self) -> Dict[str, Any]:
        """序列化为JSON兼容的dict"""
        import dataclasses
        return dataclasses.asdict(self)


FALLBACK_SIGNAL_PACK: Dict[str, Any] = {
    "agent_name": "unknown",
    "analysis_text": "",
    "bias": "neutral",
    "confidence": 0.3,
    "data_quality_score": 0.3,
    "key_points": ["结构化产物缺失，已使用文本fallback"],
    "signals": [],
    "risk_flags": ["structured_output_missing"],
    "missing_data": ["结构化产物缺失"],
    "source_summary": "fallback from text analysis",
    "as_of_date": "",
}
