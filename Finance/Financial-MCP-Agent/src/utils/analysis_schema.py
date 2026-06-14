"""
分析中间产物的结构化 Schema 定义。
使用轻量 dataclass，不引入重型依赖。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


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
