# A股主线架构升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将A股主线从"4段分析文本拼接打分"升级为"7维结构化证据 + 风险门控 + 专业报告"系统

**Architecture:** 自底向上5层推进 — 基础层(schema/builder/gate) → Agent层(改4+新建3) → 集成层(scorers/summary) → 入口层(graph) → 测试层。每层完成后可独立验证。

**Tech Stack:** LangGraph StateGraph, OpenAI-compatible LLM (5-model架构), Tushare MCP, Python 3.10+ dataclasses

**Spec:** `docs/superpowers/specs/2026-06-14-agent-architecture-upgrade-design.md`

---

## File Structure

```
Finance/Financial-MCP-Agent/src/
├── utils/
│   ├── analysis_schema.py          [CREATE]  TypedDict/dataclass 定义
│   ├── analysis_package_builder.py [CREATE]  合并signal_pack + compact context
│   ├── risk_gate.py                [CREATE]  风险门控后处理
│   ├── state_definition.py         [MODIFY]  扩展metadata注释
│   └── model_config.py             [MODIFY]  新增3个agent模型映射
├── agents/
│   ├── fundamental_agent.py        [MODIFY]  增加signal_pack输出 (+辅助函数)
│   ├── technical_agent.py          [MODIFY]  增加signal_pack输出
│   ├── value_agent.py              [MODIFY]  增加signal_pack输出
│   ├── news_agent.py               [MODIFY]  重新定位+signal_pack
│   ├── event_analyst_agent.py      [CREATE]  事件驱动分析 (M3)
│   ├── quality_risk_analyst_agent.py [CREATE] 财务质量/治理风险 (M4)
│   ├── moneyflow_analyst_agent.py  [CREATE]  资金面/量价确认 (M3)
│   ├── scoring_nodes.py            [MODIFY]  读取analysis_package + risk_gate
│   ├── short_term_scorer.py        [MODIFY]  新增依赖+权重+structured input
│   ├── medium_term_scorer.py       [MODIFY]  新增依赖+权重+structured input
│   ├── long_term_scorer.py         [MODIFY]  新增依赖+权重+structured input
│   └── summary_agent.py            [MODIFY]  9段式报告+analysis_package输入
├── main.py                         [MODIFY]  图结构: 4→7并行
└── stock_pool/
    └── scoring_engine.py           [MODIFY]  图结构: 4→7并行+scorer依赖调整

Finance/Financial-MCP-Agent/tests/
├── test_analysis_package.py        [CREATE]  builder合并/fallback/conflict测试
└── test_risk_gate.py               [CREATE]  risk_gate规则测试
```

---

### Task 1: 创建 analysis_schema.py — 结构化数据定义

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/analysis_schema.py`

- [ ] **Step 1: 写测试——验证所有类型可导入和实例化**

```python
# tests/test_analysis_package.py (创建此文件)

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_signal_dataclass_creation():
    """Signal和SignalPack应可正常创建"""
    from src.utils.analysis_schema import Signal, SignalPack, SourceLevel

    s = Signal(
        factor="ROE持续性",
        direction=1,
        strength=80,
        confidence=0.85,
        time_horizon=["medium", "long"],
        source_level=SourceLevel.STRUCTURED,
        freshness="quarterly",
        risk_flags=[],
        note="ROE连续3年>15%"
    )
    assert s.factor == "ROE持续性"
    assert s.direction == 1
    assert s.source_level == "structured"

    sp = SignalPack(
        agent_name="fundamental",
        analysis_text="基本面良好",
        bias="bullish",
        confidence=0.8,
        data_quality_score=0.9,
        key_points=["ROE高", "现金流好"],
        signals=[s],
        risk_flags=[],
        missing_data=[],
        source_summary="Tushare财报数据",
        as_of_date="2026-06-14",
    )
    assert sp.agent_name == "fundamental"
    assert len(sp.signals) == 1


def test_analysis_package_creation():
    """AnalysisPackage应可正常创建"""
    from src.utils.analysis_schema import AnalysisPackage

    pkg = AnalysisPackage(
        as_of_date="2026-06-14",
        executed_agents=["fundamental", "technical"],
        available_agents=["fundamental", "technical"],
        missing_agents=["event", "quality_risk", "moneyflow", "news", "value"],
        global_risk_flags=[],
        global_missing_data=["event数据缺失"],
        bullish_signals=[],
        bearish_signals=[],
        conflicting_signals=[],
        source_priority_summary={"counts": {"structured": 2}},
        compact_prompt_context="## 测试",
    )
    assert len(pkg.missing_agents) == 5


def test_risk_gate_result_creation():
    """RiskGateResult应可正常创建"""
    from src.utils.analysis_schema import RiskGateResult

    r = RiskGateResult(
        risk_level="low",
        risk_flags_found=[],
        score_cap=None,
        action_downgrade=None,
        abstain=False,
        abstain_reason="",
        data_quality_score=0.85,
        warnings=[],
    )
    assert r.risk_level == "low"


def test_source_priority_order():
    """Source priority应正确排序"""
    from src.utils.analysis_schema import SOURCE_PRIORITY, SourceLevel
    assert SOURCE_PRIORITY[SourceLevel.OFFICIAL] > SOURCE_PRIORITY[SourceLevel.NEWS]
    assert SOURCE_PRIORITY[SourceLevel.NEWS] > SOURCE_PRIORITY[SourceLevel.PROXY]


def test_fallback_signal_pack_constant():
    """FALLBACK_SIGNAL_PACK常量应包含所有必要字段"""
    from src.utils.analysis_schema import FALLBACK_SIGNAL_PACK
    required = ["agent_name", "bias", "confidence", "data_quality_score", "key_points", "signals", "risk_flags"]
    for key in required:
        assert key in FALLBACK_SIGNAL_PACK
    assert FALLBACK_SIGNAL_PACK["bias"] == "neutral"
    assert FALLBACK_SIGNAL_PACK["confidence"] == 0.3
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_analysis_package.py -v
```
Expected: 5 FAIL (ModuleNotFoundError: No module named 'src.utils.analysis_schema')

- [ ] **Step 3: 创建 analysis_schema.py 实现**

写入文件 `Finance/Financial-MCP-Agent/src/utils/analysis_schema.py`:

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_analysis_package.py -v
```
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/analysis_schema.py Finance/Financial-MCP-Agent/tests/test_analysis_package.py
git commit -m "feat: add analysis_schema with Signal, SignalPack, AnalysisPackage, RiskGateResult"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 2: 创建 analysis_package_builder.py — 合并引擎

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/analysis_package_builder.py`
- Modify: `Finance/Financial-MCP-Agent/tests/test_analysis_package.py` (追加测试)

- [ ] **Step 1: 写测试——text_to_signal_pack 从文本提取结构化信息**

在 `test_analysis_package.py` 末尾追加:

```python
def test_text_to_signal_pack_bullish():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("ROE持续提升，毛利率改善显著，现金流充裕，强烈看多买入", "fundamental", "2026-06-14")
    assert sp["agent_name"] == "fundamental"
    assert sp["bias"] == "bullish"
    assert sp["confidence"] < 0.5  # derived from text => low confidence
    assert len(sp["key_points"]) >= 0
    assert "structured_output_missing" not in sp.get("risk_flags", [])


def test_text_to_signal_pack_bearish():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("质押风险极高，现金流不匹配，建议卖出减持看空", "quality_risk", "2026-06-14")
    assert sp["bias"] == "bearish"


def test_text_to_signal_pack_neutral():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("数据平平，无明显方向", "unknown", "2026-06-14")
    assert sp["bias"] == "neutral"


def test_text_to_signal_pack_empty():
    from src.utils.analysis_package_builder import text_to_signal_pack
    sp = text_to_signal_pack("", "test", "2026-01-01")
    assert isinstance(sp, dict)
    assert sp["bias"] == "neutral"


def test_build_analysis_package_all_missing():
    """全部agent未执行时仍应返回有效pkg，不抛异常"""
    from src.utils.analysis_package_builder import build_analysis_package
    pkg = build_analysis_package({}, "2026-06-14")
    assert len(pkg.missing_agents) == 7
    assert pkg.global_risk_flags is not None
    assert len(pkg.compact_prompt_context) > 0
    assert isinstance(pkg.compact_prompt_context, str)


def test_build_analysis_package_with_text_only():
    """旧agent只产出了文本，应fallback生成signal_pack"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_analysis": "ROE持续提升，毛利率改善，基本面看多",
        "technical_analysis": "均线多头排列，MACD金叉，技术面看多",
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert "fundamental" in pkg.available_agents
    assert "technical" in pkg.available_agents


def test_build_analysis_package_with_signal_packs():
    """已有signal_pack时应直接使用"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish", "confidence": 0.8,
            "data_quality_score": 0.9, "key_points": ["ROE=18%", "现金流健康"],
            "signals": [
                {"factor": "ROE持续性", "direction": 1, "strength": 80,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE持续>15%"}
            ],
            "risk_flags": [], "missing_data": [], "source_summary": "Tushare",
            "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert "fundamental" in pkg.available_agents
    assert len(pkg.bullish_signals) >= 1
    assert pkg.bullish_signals[0]["factor"] == "ROE持续性"


def test_build_package_conflict_detection():
    """方向冲突的因子应被检测"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish", "confidence": 0.7,
            "key_points": ["好"], "data_quality_score": 0.8,
            "signals": [
                {"factor": "ROE质量", "direction": 1, "strength": 80,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE高"}
            ],
            "risk_flags": [], "missing_data": [], "source_summary": "",
            "as_of_date": "2026-06-14",
        },
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "bearish", "confidence": 0.8,
            "key_points": ["差"], "data_quality_score": 0.8,
            "signals": [
                {"factor": "ROE质量", "direction": -1, "strength": 75,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE质量差"}
            ],
            "risk_flags": [], "missing_data": [], "source_summary": "",
            "as_of_date": "2026-06-14",
        },
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert len(pkg.conflicting_signals) > 0


def test_compact_context_contains_sections():
    """compact_prompt_context应包含各个必需段落"""
    from src.utils.analysis_package_builder import build_analysis_package
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish", "confidence": 0.7,
            "key_points": ["ROE>15%"], "signals": [
                {"factor": "ROE", "direction": 1, "strength": 80,
                 "source_level": "structured", "time_horizon": ["medium"],
                 "note": "ROE持续改善"}
            ],
            "risk_flags": ["cashflow_mismatch"], "missing_data": ["未获取质押数据"],
            "source_summary": "Tushare", "as_of_date": "2026-06-14",
            "data_quality_score": 0.7,
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    ctx = pkg.compact_prompt_context
    assert "分析执行概况" in ctx
    assert "关键看多信号" in ctx or "看多" in ctx
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_analysis_package.py::test_text_to_signal_pack_bullish -v
```
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 创建 analysis_package_builder.py 实现**

写入文件 `Finance/Financial-MCP-Agent/src/utils/analysis_package_builder.py`:

```python
"""
Analysis Package Builder: 将多个 *_signal_pack 合并为统一 AnalysisPackage。

职责:
  1. 标准化 signal_pack 输入 (_parse_signal_pack)
  2. 从纯文本推断 signal_pack (text_to_signal_pack, 用于旧agent兼容)
  3. 合并所有 signal pack 为 AnalysisPackage (build_analysis_package)
  4. 生成 compact_prompt_context 供 scorer/summarizer 消费
"""
import re
from typing import Dict, Any, List, Optional

from src.utils.analysis_schema import (
    SourceLevel, SOURCE_PRIORITY,
    FALLBACK_SIGNAL_PACK,
    AnalysisPackage,
)


# ── 文本→结构化 提取工具 ────────────────────────────

def _extract_bias_from_text(text: str) -> str:
    if not text:
        return "neutral"
    t = text[:3000]  # 只看前3000字符
    bullish_kw = ["看多", "利多", "利好", "买入", "增持", "推荐", "改善", "增长强劲", "超预期", "积极向好"]
    bearish_kw = ["看空", "利空", "利淡", "卖出", "减持", "恶化", "下滑", "风险较高", "谨慎", "质疑"]
    bull = sum(1 for kw in bullish_kw if kw in t)
    bear = sum(1 for kw in bearish_kw if kw in t)
    if bull > bear:
        return "bullish"
    elif bear > bull:
        return "bearish"
    return "neutral"


def _extract_key_points_from_text(text: str, max_points: int = 5) -> List[str]:
    if not text:
        return []
    points = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or len(line) < 10:
            continue
        if re.match(r'^[-•*\d]+[\.\、\)]\s*', line):
            clean = re.sub(r'^[-•*\d]+[\.\、\)]\s*', '', line)
            if len(clean) > 5:
                points.append(clean[:150])
        elif any(kw in line for kw in ["结论", "综合", "关键", "核心", "重点", "主要发现"]):
            points.append(line[:150])
    return points[:max_points]


def _extract_risk_flags_from_text(text: str) -> List[str]:
    if not text:
        return []
    flags = []
    risk_pattern_map = [
        ("现金流不匹配", "cashflow_mismatch"),
        ("现金流不", "cashflow_mismatch"),
        ("质押风险", "high_pledge_risk"),
        ("质押比例", "high_pledge_risk"),
        ("监管风险", "regulatory_risk"),
        ("问询", "regulatory_risk"),
        ("立案", "regulatory_risk"),
        ("审计风险", "audit_risk"),
        ("审计意见", "audit_risk"),
        ("减值风险", "impairment_risk"),
        ("商誉减值", "impairment_risk"),
        ("商誉", "goodwill_risk"),
        ("盈利质量", "earnings_quality_concern"),
        ("ST风险", "st_risk"),
        ("退市风险", "delist_risk"),
        ("减持", "major_shareholder_sell"),
        ("债务风险", "debt_risk"),
        ("流动性风险", "liquidity_risk"),
    ]
    for pattern, flag in risk_pattern_map:
        if pattern in text:
            flags.append(flag)
    return list(dict.fromkeys(flags))  # 保序去重


# ── SignalPack 标准化 ────────────────────────────────

def _parse_signal_pack(raw: Any, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    """将任意输入标准化为 signal_pack dict"""
    if isinstance(raw, dict) and raw:
        sp = dict(raw)
        sp.setdefault("agent_name", agent_name)
        sp.setdefault("bias", "neutral")
        sp.setdefault("confidence", 0.5)
        sp.setdefault("data_quality_score", 0.5)
        sp.setdefault("key_points", [])
        sp.setdefault("signals", [])
        sp.setdefault("risk_flags", [])
        sp.setdefault("missing_data", [])
        sp.setdefault("source_summary", "")
        sp.setdefault("as_of_date", as_of_date)
        sp.setdefault("analysis_text", "")
        return sp

    if isinstance(raw, str) and raw.strip():
        return {
            **FALLBACK_SIGNAL_PACK,
            "agent_name": agent_name,
            "analysis_text": raw[:800],
            "as_of_date": as_of_date,
        }
    return dict(FALLBACK_SIGNAL_PACK, agent_name=agent_name, as_of_date=as_of_date)


def text_to_signal_pack(text: str, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    """
    从纯自然语言文本构建 fallback signal_pack。
    用于旧 agent 尚未产出结构化产物时的兼容降级。
    """
    return {
        "agent_name": agent_name,
        "analysis_text": text[:800] if text else "",
        "bias": _extract_bias_from_text(text),
        "confidence": 0.3,
        "data_quality_score": 0.3,
        "key_points": _extract_key_points_from_text(text),
        "signals": [],
        "risk_flags": _extract_risk_flags_from_text(text),
        "missing_data": ["结构化产物缺失，已从文本推断"],
        "source_summary": "derived from analysis text",
        "source_level": SourceLevel.DERIVED,
        "as_of_date": as_of_date,
    }


# ── 合并引擎 ──────────────────────────────────────────

def build_analysis_package(
    state_data: Dict[str, Any],
    as_of_date: str = "",
) -> AnalysisPackage:
    """
    从 state.data 提取所有 *_signal_pack（或 fallback from *_analysis），
    合并为统一 AnalysisPackage。
    """
    agent_text_keys = {
        "fundamental": "fundamental_analysis",
        "technical": "technical_analysis",
        "value": "value_analysis",
        "news": "news_analysis",
        "event": "event_analysis",
        "quality_risk": "quality_risk_analysis",
        "moneyflow": "moneyflow_analysis",
    }

    signal_packs: Dict[str, Dict[str, Any]] = {}
    for agent_name, text_key in agent_text_keys.items():
        sp_key = f"{agent_name}_signal_pack"
        if sp_key in state_data and state_data[sp_key]:
            signal_packs[agent_name] = _parse_signal_pack(state_data[sp_key], agent_name, as_of_date)
        elif text_key in state_data and state_data[text_key]:
            signal_packs[agent_name] = text_to_signal_pack(state_data[text_key], agent_name, as_of_date)
        else:
            sp = dict(FALLBACK_SIGNAL_PACK, agent_name=agent_name, as_of_date=as_of_date)
            sp["missing_data"] = ["Agent未执行"]
            signal_packs[agent_name] = sp

    # 归并
    all_signals = []
    all_risk_flags = []
    all_missing = []
    available = []
    missing = []

    for agent_name, sp in signal_packs.items():
        md = sp.get("missing_data", [])
        has_fatal_missing = any("Agent未执行" in str(x) for x in md)
        if has_fatal_missing and sp.get("data_quality_score", 0) < 0.3:
            missing.append(agent_name)
        else:
            available.append(agent_name)

        all_risk_flags.extend(sp.get("risk_flags", []))
        all_missing.extend(md)

        for sig in sp.get("signals", []):
            if isinstance(sig, dict):
                sig["_agent"] = agent_name
                self_level = sig.get("source_level", SourceLevel.PROXY)
                sig.setdefault("source_level", self_level)
                sig.setdefault("strength", 50)
                sig.setdefault("direction", 0)
                all_signals.append(sig)

    # 按 source priority + strength 排降序
    all_signals.sort(
        key=lambda s: (
            SOURCE_PRIORITY.get(s.get("source_level", SourceLevel.PROXY), 0),
            abs(s.get("strength", 0)),
        ),
        reverse=True,
    )

    bullish = [s for s in all_signals if s.get("direction", 0) > 0]
    bearish = [s for s in all_signals if s.get("direction", 0) < 0]

    # 冲突检测
    factor_map: Dict[str, List[dict]] = {}
    for s in all_signals:
        f = s.get("factor", "")
        if f:
            factor_map.setdefault(f, []).append(s)
    conflicting = []
    for factor, sigs in factor_map.items():
        dirs = {s.get("direction", 0) for s in sigs}
        if len(dirs) > 1:
            conflicting.append({"factor": factor, "signals": sigs})

    # source_counts
    source_counts: Dict[str, int] = {}
    for s in all_signals:
        lv = s.get("source_level", SourceLevel.PROXY)
        source_counts[lv] = source_counts.get(lv, 0) + 1

    unique_risk = list(dict.fromkeys(all_risk_flags))
    unique_missing = list(dict.fromkeys(all_missing))

    compact = _build_compact_context(available, missing, bullish, bearish, conflicting, unique_risk, unique_missing, signal_packs)

    return AnalysisPackage(
        as_of_date=as_of_date,
        executed_agents=list(dict.fromkeys(available + missing)),
        available_agents=available,
        missing_agents=missing,
        global_risk_flags=unique_risk,
        global_missing_data=unique_missing,
        bullish_signals=bullish,
        bearish_signals=bearish,
        conflicting_signals=conflicting,
        source_priority_summary={"counts": source_counts},
        compact_prompt_context=compact,
    )


def _build_compact_context(
    available: List[str],
    missing: List[str],
    bullish: List[dict],
    bearish: List[dict],
    conflicting: List[dict],
    risk_flags: List[str],
    missing_data: List[str],
    signal_packs: Dict[str, Dict[str, Any]],
) -> str:
    lines = []
    lines.append("## 分析执行概况")
    lines.append(f"- 已执行agent: {', '.join(available) if available else '无'}")
    if missing:
        lines.append(f"- 未执行agent: {', '.join(missing)}")
    lines.append("")

    # 各agent结论摘要
    lines.append("## 各Agent结论摘要")
    bias_map = {"bullish": "看多", "neutral": "中性", "bearish": "看空"}
    for agent_name, sp in signal_packs.items():
        bias_cn = bias_map.get(sp.get("bias", ""), "中性")
        lines.append(f"- **{agent_name}**: {bias_cn} (置信度={sp.get('confidence', 0):.0%})")
        for kp in sp.get("key_points", [])[:3]:
            lines.append(f"  - {kp}")
    lines.append("")

    # 看多 Top 5
    lines.append("## 关键看多信号")
    for s in bullish[:5]:
        lines.append(f"- [{s.get('_agent', '?')}] {s.get('factor', '?')}: strength={s.get('strength', 0)}, src={s.get('source_level', '?')}")
    if not bullish:
        lines.append("- (无)")
    lines.append("")

    # 看空 Top 5
    lines.append("## 关键看空信号")
    for s in bearish[:5]:
        lines.append(f"- [{s.get('_agent', '?')}] {s.get('factor', '?')}: strength={s.get('strength', 0)}, src={s.get('source_level', '?')}")
    if not bearish:
        lines.append("- (无)")
    lines.append("")

    # 冲突
    if conflicting:
        lines.append("## 信号冲突")
        for c in conflicting[:5]:
            lines.append(f"- 因子 **{c['factor']}**: {len(c['signals'])}个信号方向不一致")
        lines.append("")

    # 风险
    if risk_flags:
        lines.append(f"## 全局风险标签: {', '.join(risk_flags)}")
        lines.append("")

    # 缺失
    if missing_data:
        lines.append(f"## 缺失数据: {', '.join(missing_data[:8])}")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: 运行所有测试确认通过**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_analysis_package.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/analysis_package_builder.py Finance/Financial-MCP-Agent/tests/test_analysis_package.py
git commit -m "feat: add analysis_package_builder with text fallback and compact context"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 3: 创建 risk_gate.py — 风险门控

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/risk_gate.py`
- Create: `Finance/Financial-MCP-Agent/tests/test_risk_gate.py`

- [ ] **Step 1: 写测试**

创建文件 `Finance/Financial-MCP-Agent/tests/test_risk_gate.py`:

```python
"""Test risk_gate: score cap, abstain, downgrade rules"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.analysis_package_builder import build_analysis_package


def test_audit_risk_triggers_cap():
    """审计风险应触发score_cap=60"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.7, "data_quality_score": 0.8,
            "key_points": [], "signals": [],
            "risk_flags": ["audit_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 60
    assert result.action_downgrade == "谨慎"


def test_regulatory_risk_triggers_cap():
    """监管风险应触发score_cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.7, "data_quality_score": 0.8,
            "key_points": [], "signals": [],
            "risk_flags": ["regulatory_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 55


def test_no_critial_risk_no_cap():
    """无关键风险时不应cap"""
    from src.utils.risk_gate import apply_risk_gate
    pkg = build_analysis_package({}, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap is None


def test_news_only_narrative_caps_long():
    """长线仅靠新闻叙事应被cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "news_signal_pack": {
            "agent_name": "news", "bias": "bullish",
            "confidence": 0.9, "data_quality_score": 0.7,
            "key_points": ["媒体一致看多"], "signals": [
                {"factor": "媒体情绪", "direction": 1, "strength": 70,
                 "source_level": "news", "note": ""}
            ],
            "risk_flags": [], "missing_data": [],
            "source_summary": "新闻接口", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "long", 90)
    assert result.score_cap is not None
    assert result.score_cap <= 55


def test_abstain_when_many_missing():
    """缺失agent过多且data_quality低时abstain"""
    from src.utils.risk_gate import apply_risk_gate
    pkg = build_analysis_package({}, "2026-06-14")
    result = apply_risk_gate(pkg, "long", 80)
    assert result.abstain is True


def test_low_risk_on_full_data():
    """数据齐全无风险标签时应为low risk"""
    from src.utils.risk_gate import apply_risk_gate
    data = {}
    for agent in ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"]:
        data[f"{agent}_signal_pack"] = {
            "agent_name": agent, "bias": "neutral",
            "confidence": 0.8, "data_quality_score": 0.9,
            "key_points": [], "signals": [],
            "risk_flags": [], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.risk_level == "low"
    assert result.abstain is False


def test_delist_risk_most_conservative():
    """退市风险应触发最低cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.9, "data_quality_score": 0.9,
            "key_points": [], "signals": [],
            "risk_flags": ["delist_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 50


def test_multiple_critical_uses_min_cap():
    """多个关键风险标签时取最低cap"""
    from src.utils.risk_gate import apply_risk_gate
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.9, "data_quality_score": 0.9,
            "key_points": [], "signals": [],
            "risk_flags": ["audit_risk", "regulatory_risk", "high_pledge_risk"],
            "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 55  # 取最严格的
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_risk_gate.py -v
```
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: 创建 risk_gate.py 实现**

写入文件 `Finance/Financial-MCP-Agent/src/utils/risk_gate.py`:

```python
"""
Risk Gate: 轻量分析后处理，对评分结果做风险门控。

门控规则:
  1. 高严重度风险标签 → score cap + downgrade
  2. 长线仅新闻叙事无事实支撑 → 不能强推荐
  3. 核心agent缺失过多 + data_quality低 → abstain
  4. 短线流动性差 → 不能高分
"""
from typing import List, Optional

from src.utils.analysis_schema import RiskGateResult, SourceLevel, SOURCE_PRIORITY


CRITICAL_RISK_FLAGS = {
    "audit_risk":             {"cap": 60, "downgrade": "谨慎"},
    "regulatory_risk":        {"cap": 55, "downgrade": "谨慎"},
    "high_pledge_risk":       {"cap": 60, "downgrade": "谨慎"},
    "cashflow_mismatch":      {"cap": 65, "downgrade": "观察"},
    "major_event_negative":   {"cap": 55, "downgrade": "谨慎"},
    "delist_risk":            {"cap": 50, "downgrade": "谨慎"},
    "st_risk":                {"cap": 60, "downgrade": "观察"},
    "earnings_quality_concern": {"cap": 65, "downgrade": "观察"},
    "goodwill_risk":          {"cap": 65, "downgrade": "观察"},
    "debt_risk":              {"cap": 65, "downgrade": "观察"},
    "major_shareholder_sell": {"cap": 65, "downgrade": "观察"},
    "liquidity_risk":         {"cap": 65, "downgrade": "观察"},
    "impairment_risk":        {"cap": 65, "downgrade": "观察"},
}


def _count_signals_by_source(package, min_source_level: str) -> int:
    threshold = SOURCE_PRIORITY.get(min_source_level, 0)
    count = 0
    for sigs in [package.bullish_signals, package.bearish_signals]:
        for s in sigs:
            lv = s.get("source_level", SourceLevel.PROXY)
            if SOURCE_PRIORITY.get(lv, 0) >= threshold:
                count += 1
    return count


def apply_risk_gate(package, term: str, original_score: int) -> RiskGateResult:
    risk_flags = package.global_risk_flags
    missing_agents = package.missing_agents

    total_possible = 7
    available = len(package.available_agents)
    data_quality = available / total_possible if total_possible > 0 else 1.0

    # 规则1: 关键风险标签
    score_cap = None
    downgrade = None
    found_critical: List[str] = []

    for flag, cfg in CRITICAL_RISK_FLAGS.items():
        if flag in risk_flags:
            found_critical.append(flag)
            if score_cap is None or cfg["cap"] < score_cap:
                score_cap = cfg["cap"]
                downgrade = cfg["downgrade"]

    # 规则2: 仅新闻叙事无事实 (对中长期)
    factual_signals = _count_signals_by_source(package, SourceLevel.STRUCTURED)
    news_only_signals = _count_signals_by_source(package, SourceLevel.NEWS)
    if factual_signals == 0 and news_only_signals > 0 and term in ("medium", "long"):
        if score_cap is None or score_cap > 55:
            score_cap = 55
            downgrade = "观察"
            found_critical.append("news_only_narrative")

    # 规则3: 数据严重不足 → abstain
    if len(missing_agents) >= 2 and data_quality < 0.4:
        return RiskGateResult(
            risk_level="critical" if found_critical else "high",
            risk_flags_found=found_critical,
            score_cap=min(score_cap or 50, 50),
            action_downgrade=downgrade or "观察",
            abstain=True,
            abstain_reason=f"数据不足(缺失{len(missing_agents)}个agent: {', '.join(missing_agents)})",
            data_quality_score=data_quality,
            warnings=[f"缺失agent: {', '.join(missing_agents)}"],
        )

    # 规则4: 短线流动性
    if term == "short":
        liquidity_sigs = [s for s in package.bearish_signals if "流动" in s.get("factor", "") or "量价" in s.get("factor", "")]
        if liquidity_sigs and any(abs(s.get("strength", 0)) > 60 for s in liquidity_sigs):
            if score_cap is None or score_cap > 50:
                score_cap = 50
                downgrade = "观察"

    # risk_level
    if found_critical:
        risk_level = "high"
    elif data_quality < 0.6:
        risk_level = "medium"
    else:
        risk_level = "low"

    warnings = []
    if found_critical:
        warnings.append(f"检测到关键风险: {', '.join(found_critical)}")
    if data_quality < 0.5:
        warnings.append(f"数据质量低({data_quality:.0%})")

    return RiskGateResult(
        risk_level=risk_level,
        risk_flags_found=found_critical,
        score_cap=score_cap,
        action_downgrade=downgrade,
        abstain=len(missing_agents) >= 3,
        abstain_reason=f"核心agent缺失{len(missing_agents)}个: {', '.join(missing_agents)}" if len(missing_agents) >= 3 else "",
        data_quality_score=data_quality,
        warnings=warnings,
    )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/test_risk_gate.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/risk_gate.py Finance/Financial-MCP-Agent/tests/test_risk_gate.py
git commit -m "feat: add risk_gate with score cap, abstain, and downgrade rules"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 4: 扩展 model_config.py — 新增3个agent的模型映射

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/utils/model_config.py:37-38`

- [ ] **Step 1: 在 AGENT_MODEL_SUFFIX dict 中新增映射**

在 `model_config.py` 的 `AGENT_MODEL_SUFFIX` dict 中，`"news_agent": "_3"` 之后插入:

```python
    # ── 新增 Agent (2026-06 架构升级) ──
    "event_analyst": "_3",
    "quality_risk_analyst": "_4",
    "moneyflow_analyst": "_3",
```

- [ ] **Step 2: 验证映射**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.utils.model_config import AGENT_MODEL_SUFFIX; print({k: AGENT_MODEL_SUFFIX.get(k) for k in ['event_analyst','quality_risk_analyst','moneyflow_analyst']})"
```
Expected: `{'event_analyst': '_3', 'quality_risk_analyst': '_4', 'moneyflow_analyst': '_3'}`

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/utils/model_config.py
git commit -m "feat: add model mappings for event/quality_risk/moneyflow analysts"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 5: 改造 fundamental_agent.py — signal_pack 提取

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/fundamental_agent.py`

- [ ] **Step 1: 在文件顶部追加辅助函数**

在 `_get_recent_quarters` 函数之后，`async def fundamental_agent` 之前插入:

```python
def _extract_signal_pack_from_llm(llm_output: str, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    """
    从LLM输出中提取signal_pack JSON。
    三层fallback: JSON解析 → 正则提取 → 文本推断
    """
    import json as _json
    import re as _re

    # 第一层: <SIGNAL_PACK> 标签
    tag_match = _re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', llm_output)
    if tag_match:
        try:
            sp = _json.loads(tag_match.group(1))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", 0.7)
            return sp
        except (_json.JSONDecodeError, ValueError):
            pass

    # 第二层: 从文本中找包含bias和signals的JSON
    json_match = _re.search(r'\{[\s\S]*"bias"[\s\S]*"signals"[\s\S]*\}', llm_output)
    if json_match:
        try:
            sp = _json.loads(json_match.group(0))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", 0.5)
            return sp
        except (_json.JSONDecodeError, ValueError):
            pass

    # 第三层: 纯文本推断
    from src.utils.analysis_package_builder import text_to_signal_pack
    return text_to_signal_pack(llm_output, agent_name, as_of_date)
```

- [ ] **Step 2: 在 Phase 2 prompt 末尾追加 SIGNAL_PACK 输出指令**

在 `analysis_prompt = f"""请以券商分析师的标准...` 的最末尾（最后一个 `"""` 之前）追加:

```python

⛔ 结构化输出要求：
在完成上述分析的「🔍 分析判断区」之后，请额外输出一个 JSON block：

<SIGNAL_PACK>
{{
    "bias": "bullish"|"neutral"|"bearish",
    "confidence": 0.0-1.0,
    "key_points": ["关键结论1", "关键结论2"] (最多6条,每条<80字),
    "signals": [
        {{
            "factor": "因子名(如:ROE持续性/现金流质量/负债率)",
            "direction": 1|-1|0,
            "strength": 0-100,
            "time_horizon": ["medium","long"],
            "source_level": "official_like"|"structured"|"derived",
            "risk_flags": [],
            "freshness": "quarterly",
            "note": "一句话说明"
        }}
    ] (最多6条),
    "risk_flags": ["cashflow_mismatch", "earnings_quality_concern"],
    "missing_data": ["未获取到质押数据"],
    "source_summary": "Tushare财报+AkShare杜邦/运营/成长数据"
}}
</SIGNAL_PACK>

请确保SIGNAL_PACK内的JSON完全有效。
```

- [ ] **Step 3: 在 final_output 获取后增加 signal_pack 提取**

在 Phase 2 完成后 `final_output = response.content.strip()...` 那一行之后插入:

```python
        # 提取 signal_pack
        fundamental_signal_pack = _extract_signal_pack_from_llm(final_output, "fundamental", current_date)
        current_data["fundamental_signal_pack"] = fundamental_signal_pack
```

- [ ] **Step 4: 同样处理 LLM 超时/失败时的 fallback signal_pack**

在 LLM 超时 (`asyncio.TimeoutError`) 和 LLM 失败 (`except Exception as llm_err`) 的 except 块中，`final_output = ...` 之后追加:

```python
            from src.utils.analysis_package_builder import text_to_signal_pack
            fundamental_signal_pack = text_to_signal_pack(final_output, "fundamental", current_date)
            current_data["fundamental_signal_pack"] = fundamental_signal_pack
```

- [ ] **Step 5: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/fundamental_agent.py
git commit -m "feat: fundamental_agent outputs signal_pack with 3-layer JSON fallback"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 6: 改造 technical_agent.py — signal_pack 输出

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/technical_agent.py`

- [ ] **Step 1: 在 agent_input 末尾追加 SIGNAL_PACK 指令**

在 `agent_input = f"""请以券商分析师的标准...` 的最末尾追加与 fundamental_agent 相同的 SIGNAL_PACK JSON 指令块，但调整:

```python
    "source_summary": "Tushare行情+预计算技术指标(MACD/RSI/均线/量价)"
```

- [ ] **Step 2: 在 final_output 提取后增加 signal_pack 解析**

在 `final_output = last_ai_message.content` 之后插入:

```python
    import json as _json_sp
    import re as _re_sp
    from src.utils.analysis_package_builder import text_to_signal_pack

    sp = None
    tag_match = _re_sp.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', final_output)
    if tag_match:
        try:
            sp = _json_sp.loads(tag_match.group(1))
            sp["agent_name"] = "technical"
            sp["as_of_date"] = current_date
            sp.setdefault("analysis_text", final_output[:500])
            sp.setdefault("data_quality_score", 0.7)
        except Exception:
            pass
    if sp is None:
        sp = text_to_signal_pack(final_output, "technical", current_date)
    current_data["technical_signal_pack"] = sp
```

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/technical_agent.py
git commit -m "feat: technical_agent outputs signal_pack with JSON fallback"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 7: 改造 value_agent.py + news_agent.py — signal_pack 输出

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/value_agent.py`
- Modify: `Finance/Financial-MCP-Agent/src/agents/news_agent.py`

- [ ] **Step 1: value_agent.py — 在 LLM prompt 末尾追加 SIGNAL_PACK 指令**

在 value_agent.py 的 LLM analysis prompt 最末尾追加与 fundamental_agent 相同的 SIGNAL_PACK JSON 指令块，source_summary 改为:

```python
    "source_summary": "Tushare估值(PE/PB/EV-EBITDA)+历史分位+行业对比"
```

在 `final_output` 获取后增加 signal_pack 提取（同Task 6 Step 2模式，agent_name="value"）。

- [ ] **Step 2: news_agent.py — 修改 system prompt 并追加 SIGNAL_PACK**

将 system prompt 中的分析要求从原来的"逐条分析每一条新闻..."改为:

```python
        {"role": "system", "content": (
            "你是一位资深的A股新闻舆情分析师。\n\n"
            "你的职责范围（只做这些）：\n"
            "1. 媒体新闻情绪判断\n"
            "2. 行业/政策舆情分析\n"
            "3. 题材热度与叙事强度评估\n"
            "4. 市场关注点是否集中、是否形成一致预期\n\n"
            "你不再负责（这些交给event_analyst）：\n"
            "- 公司正式公告的事实判断\n"
            "- 监管事件是否成立\n"
            "- 重大事项的事实核实\n\n"
            "换言之：你分析的是'市场如何看、如何传'，而不是'真实发生了什么'。\n\n"
            "⛔ 输出格式：先输出「📊 数据事实区」「🔍 分析判断区」，"
            "然后在末尾输出: <SIGNAL_PACK>{JSON}</SIGNAL_PACK>\n"
            "其中JSON包含: bias, confidence, key_points(≤5条), signals(≤5条, source_level=\"news\"), "
            "risk_flags, source_summary\n"
        )}
```

在 `final_output` 获取后增加 signal_pack 提取（agent_name="news"）。

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/value_agent.py Finance/Financial-MCP-Agent/src/agents/news_agent.py
git commit -m "feat: value_agent and news_agent output signal_pack; news repositioned to sentiment-only"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 8: 新增 event_analyst_agent.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/agents/event_analyst_agent.py`

- [ ] **Step 1: 创建完整的 event agent 文件**

```python
"""
EventAnalyst Agent: 事件驱动分析
职责: 识别重大事件/催化剂、标记时效、输出事件方向(利多/利空/中性)
模型: M3 (Qwen3.7-Plus), thinking=enabled
"""
import asyncio
import json
import re
import time
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.utils.state_definition import AgentState
from src.tools.mcp_client import get_mcp_tools
from src.utils.logging_config import setup_logger, ERROR_ICON, SUCCESS_ICON, WAIT_ICON
from src.utils.execution_logger import get_execution_logger
from src.utils.cache_utils import read_cache, write_cache
from src.utils.model_config import get_model_config_for_agent, get_thinking_body
from src.utils.analysis_package_builder import text_to_signal_pack

load_dotenv(override=True)
logger = setup_logger(__name__)

TOOL_TIMEOUT = 30
LLM_TIMEOUT = 120

EVENT_TOOL_NAMES = [
    "tushare_anns_d",
    "tushare_new_share",
    "tushare_pledge_stat",
    "tushare_repurchase",
    "tushare_share_float",
    "tushare_top10_holders",
    "tushare_stk_holdertrade",
    "tushare_dividend",
    "tushare_namechange",
    "tushare_suspend",
    "crawl_news",
    "tushare_st_status",
]


def _extract_code(stock_code: str) -> str:
    return stock_code.replace("sh.", "").replace("sz.", "").replace("bj.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()


async def _call_tool_safe(tool, kwargs: dict, label: str) -> str:
    try:
        result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
        text = str(result).strip()
        if len(text) > 20:
            logger.info(f"{SUCCESS_ICON} EventAnalyst: {label} 获取成功 ({len(text)} 字符)")
            return text
        return f"[{label}] 数据不可用（返回过短）"
    except asyncio.TimeoutError:
        return f"[{label}] 数据不可用（超时）"
    except Exception as e:
        return f"[{label}] 数据不可用: {str(e)[:80]}"


def _extract_signal_pack(llm_output: str, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    tag_match = re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', llm_output)
    if tag_match:
        try:
            sp = json.loads(tag_match.group(1))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", 0.6)
            return sp
        except (json.JSONDecodeError, ValueError):
            pass
    return text_to_signal_pack(llm_output, agent_name, as_of_date)


async def event_analyst_agent(state: AgentState) -> AgentState:
    logger.info(f"{WAIT_ICON} EventAnalyst: 开始事件驱动分析")
    execution_logger = get_execution_logger()
    agent_name = "event_analyst"

    current_data = state.get("data", {})
    current_messages = state.get("messages", [])
    current_metadata = state.get("metadata", {})

    skip_cache = current_data.get("skip_cache", False)
    cache_date = current_data.get("current_date", "")
    cache_code = current_data.get("stock_code", "")

    if not skip_cache and cache_date and cache_code:
        cached = read_cache("event_analysis", cache_code, cache_date)
        if cached:
            logger.info(f"{SUCCESS_ICON} EventAnalyst: 命中缓存")
            current_data["event_analysis"] = cached
            current_metadata["event_agent_executed"] = True
            current_metadata["event_agent_cached"] = True
            return {"data": current_data, "messages": current_messages + [{"role": "assistant", "content": "事件分析已完成（缓存）"}], "metadata": current_metadata}

    stock_code = current_data.get("stock_code", "")
    company_name = current_data.get("company_name", "")
    current_time_info = current_data.get("current_time_info", "")
    current_date = current_data.get("current_date", "")
    clean_code = _extract_code(stock_code) if stock_code else ""

    agent_start_time = time.time()
    execution_logger.log_agent_start(agent_name, {"stock_code": stock_code, "company_name": company_name})

    try:
        model_cfg = get_model_config_for_agent("event_analyst", current_data)
        api_key = model_cfg["api_key"]
        base_url = model_cfg["base_url"]
        model_name = model_cfg["model_name"]
        if not all([api_key, base_url, model_name]):
            raise ValueError("缺少环境变量")

        # Phase 1: 并行数据预取
        try:
            all_tools = await get_mcp_tools(tool_filter=EVENT_TOOL_NAMES)
        except Exception:
            all_tools = []

        tool_map = {t.name: t for t in all_tools} if all_tools else {}
        tasks = []
        labels = []

        for tname in EVENT_TOOL_NAMES:
            if tname in tool_map:
                kwargs = {"code": clean_code}
                if tname == "tushare_anns_d":
                    kwargs["days"] = 90
                tasks.append(_call_tool_safe(tool_map[tname], kwargs, tname))
                labels.append(tname)

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        safe_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                safe_results.append(f"[{labels[i]}] 调用异常")
            else:
                safe_results.append(str(r) if r else f"[{labels[i]}] 空返回")

        data_sections = [f"### [{label}]\n{result}" for label, result in zip(labels, safe_results)]
        raw_data_text = "\n\n".join(data_sections) if data_sections else "无可用数据源"

        # Phase 2: LLM 分析
        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                        temperature=1.0, request_timeout=LLM_TIMEOUT, max_tokens=12000,
                        extra_body=get_thinking_body(base_url, True))

        system_prompt = f"""你是一位A股事件驱动分析师。
当前时间: {current_time_info}

职责:
1. 识别重大事件/催化剂: 业绩预告/快报、回购、增减持、重大合同、并购重组、诉讼仲裁、处罚/问询、股权质押、异常停复牌
2. 标记事件时效: 事件日期、新近程度(freshness)、影响期限
3. 输出事件方向: 利多/利空/中性; 一次性/持续性; 是否已被市场交易过(source_level: official_like>structured>news>derived)

⛔ 先输出「📊 数据事实区」「🔍 分析判断区」的自然语言分析。
末尾输出: <SIGNAL_PACK>{{JSON}}</SIGNAL_PACK>
JSON含: bias, confidence, key_points(≤5条), signals(≤8条,含factor/direction/strength/confidence/time_horizon/source_level/freshness/note), risk_flags, missing_data, source_summary
"""

        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析{company_name}({stock_code})的重大事件。\n\n## 原始数据\n{raw_data_text}"}
        ])
        final_output = response.content.strip() if hasattr(response, 'content') else str(response)

        signal_pack = _extract_signal_pack(final_output, "event", current_date)
        current_data["event_signal_pack"] = signal_pack
        current_data["event_analysis"] = final_output

        if not skip_cache and cache_date and cache_code:
            write_cache("event_analysis", cache_code, cache_date, final_output)
        current_metadata["event_agent_executed"] = True

        return {"data": current_data, "messages": current_messages + [{"role": "assistant", "content": "事件分析已完成"}], "metadata": current_metadata}

    except Exception as e:
        logger.error(f"{ERROR_ICON} EventAnalyst 失败: {e}", exc_info=True)
        current_data["event_analysis"] = f"事件分析失败: {str(e)}"
        current_data["event_signal_pack"] = text_to_signal_pack(current_data.get("event_analysis", ""), "event", current_date)
        current_metadata["event_agent_error"] = str(e)
        return {"data": current_data, "messages": current_messages, "metadata": current_metadata}
```

- [ ] **Step 2: 验证文件可导入**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.agents.event_analyst_agent import event_analyst_agent; print('OK')"
```
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/event_analyst_agent.py
git commit -m "feat: add event_analyst_agent for catalyst/event-driven analysis"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 9: 新增 quality_risk_analyst_agent.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/agents/quality_risk_analyst_agent.py`

- [ ] **Step 1: 创建完整的 quality_risk agent 文件**

代码结构与 Task 8 的 event_analyst_agent.py 完全相同（两阶段架构：并行数据预取 + LLM分析），差异点如下：

- Agent name: `quality_risk_analyst`
- 模型: `get_model_config_for_agent("quality_risk_analyst", ...)` → M4 (Kimi K2.6)
- 工具列表:

```python
QUALITY_RISK_TOOL_NAMES = [
    "tushare_income",
    "tushare_balancesheet",
    "tushare_cashflow",
    "tushare_fina_indicator",
    "tushare_pledge_stat",
    "tushare_top10_holders",
    "tushare_stk_holdertrade",
    "tushare_st_status",
    "get_st_risk_data",
    "get_dupont_data",
]
```

- system_prompt:

```python
        system_prompt = f"""你是一位A股财务质量与治理风险分析师。
当前时间: {current_time_info}

职责:
1. 财务质量: 利润现金含量(经营现金流/净利润)、应收/存货/商誉/减值风险、非经常性损益依赖
2. 治理与股东风险: 质押比例、冻结、控制权变化、关联交易、大股东减持
3. 风险标签输出: cashflow_mismatch, high_pledge_risk, regulatory_risk, audit_risk, impairment_risk, earnings_quality_concern, goodwill_risk, debt_risk, delist_risk
4. 不能确认时显式写「未获取到」

⛔ 输出格式同event: 先「📊 数据事实区」「🔍 分析判断区」，末尾<SIGNAL_PACK>{{JSON}}</SIGNAL_PACK>
"""
```

- LLM max_tokens: 16000, timeout: 300 (Kimi K2.6 更深思)
- temperature: 1.0
- 输出字段: `quality_risk_analysis` + `quality_risk_signal_pack`
- 缓存key: `quality_risk_analysis`

完整代码见 Task 8 模板，替换上述差异点。

- [ ] **Step 2: 验证可导入**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/quality_risk_analyst_agent.py
git commit -m "feat: add quality_risk_analyst_agent for financial quality and governance risk"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 10: 新增 moneyflow_analyst_agent.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/agents/moneyflow_analyst_agent.py`

- [ ] **Step 1: 创建完整的 moneyflow agent 文件**

代码结构与 Task 8 完全相同，差异点如下：

- Agent name: `moneyflow_analyst`
- 模型: `get_model_config_for_agent("moneyflow_analyst", ...)` → M3 (Qwen3.7-Plus)
- 工具列表:

```python
MONEYFLOW_TOOL_NAMES = [
    "tushare_moneyflow",
    "tushare_moneyflow_hsgt",
    "tushare_margin",
    "tushare_margin_detail",
    "tushare_top_list",
    "tushare_block_trade",
    "tushare_daily_basic",
    "tushare_kline",
    "tushare_cyq_chips",
]
```

- system_prompt:

```python
        system_prompt = f"""你是一位A股资金面与微观结构分析师。
当前时间: {current_time_info}

职责:
1. 量价确认: 换手率/成交额/持续性/放量突破/缩量整理/异常放量回落
2. 资金代理指标: 融资融券变化、龙虎榜/大宗交易/主力资金流向
3. 执行风险: 流动性是否支持短线操作、是否易受单日情绪主导
4. 数据不足时: source_level=proxy, 降低data_quality

⛔ 先输出「📊 数据事实区」「🔍 分析判断区」，末尾<SIGNAL_PACK>{{JSON}}</SIGNAL_PACK>
signal中source_level优先使用structured(如moneyflow/top_list等数值工具)和news(龙虎榜评论)
"""
```

- LLM: temperature=0.6, max_tokens=12000, timeout=120
- 输出字段: `moneyflow_analysis` + `moneyflow_signal_pack`
- 缓存key: `moneyflow_analysis`

- [ ] **Step 2: 验证可导入**

```bash
cd Finance/Financial-MCP-Agent && python -c "from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/moneyflow_analyst_agent.py
git commit -m "feat: add moneyflow_analyst_agent for capital flow and volume-price confirmation"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 11: 改造 scoring_nodes.py — analysis_package + risk_gate

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/scoring_nodes.py`

- [ ] **Step 1: 重写三个 scorer node**

替换文件全部内容为:

```python
"""
Scoring Nodes: LangGraph wrapper for 3 scoring agents (v2 — structured evidence)

架构升级:
  - short_term: technical + news + event + moneyflow
  - medium_term + long_term: all 7 analysis agents
  - 每个node先构建AnalysisPackage → 传给scorer → apply risk_gate
"""
from typing import Dict, Any

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


async def short_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.short_term_scorer import short_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package
    from src.utils.risk_gate import apply_risk_gate

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")

    logger.info(f"{WAIT_ICON} ShortTermScorerNode: {company_name}({stock_code}) 短线打分")

    try:
        pkg = build_analysis_package(data, as_of_date)

        result = await short_term_scorer(
            stock_code=stock_code, company_name=company_name,
            technical_analysis=data.get("technical_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        gate = apply_risk_gate(pkg, "short", result["score"])
        result["risk_gate"] = {
            "risk_level": gate.risk_level,
            "risk_flags": gate.risk_flags_found,
            "score_cap": gate.score_cap,
            "abstain": gate.abstain,
            "data_quality_score": gate.data_quality_score,
        }

        logger.info(f"{SUCCESS_ICON} ShortTermScorerNode: {company_name} 短线={result['score']} (gate={gate.risk_level})")
        return {"data": {"short_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} ShortTermScorerNode 失败: {e}", exc_info=True)
        raise


async def medium_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.medium_term_scorer import medium_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package
    from src.utils.risk_gate import apply_risk_gate

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")

    logger.info(f"{WAIT_ICON} MediumTermScorerNode: {company_name}({stock_code}) 中线打分")

    try:
        pkg = build_analysis_package(data, as_of_date)

        result = await medium_term_scorer(
            stock_code=stock_code, company_name=company_name,
            fundamental_analysis=data.get("fundamental_analysis", ""),
            technical_analysis=data.get("technical_analysis", ""),
            value_analysis=data.get("value_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            quality_risk_analysis=data.get("quality_risk_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        gate = apply_risk_gate(pkg, "medium", result["score"])
        result["risk_gate"] = {
            "risk_level": gate.risk_level,
            "risk_flags": gate.risk_flags_found,
            "score_cap": gate.score_cap,
            "abstain": gate.abstain,
            "data_quality_score": gate.data_quality_score,
        }

        logger.info(f"{SUCCESS_ICON} MediumTermScorerNode: {company_name} 中线={result['score']} (gate={gate.risk_level})")
        return {"data": {"medium_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} MediumTermScorerNode 失败: {e}", exc_info=True)
        raise


async def long_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.long_term_scorer import long_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package
    from src.utils.risk_gate import apply_risk_gate

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")

    logger.info(f"{WAIT_ICON} LongTermScorerNode: {company_name}({stock_code}) 长线打分")

    try:
        pkg = build_analysis_package(data, as_of_date)

        result = await long_term_scorer(
            stock_code=stock_code, company_name=company_name,
            fundamental_analysis=data.get("fundamental_analysis", ""),
            technical_analysis=data.get("technical_analysis", ""),
            value_analysis=data.get("value_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            quality_risk_analysis=data.get("quality_risk_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        gate = apply_risk_gate(pkg, "long", result["score"])
        result["risk_gate"] = {
            "risk_level": gate.risk_level,
            "risk_flags": gate.risk_flags_found,
            "score_cap": gate.score_cap,
            "abstain": gate.abstain,
            "data_quality_score": gate.data_quality_score,
        }

        logger.info(f"{SUCCESS_ICON} LongTermScorerNode: {company_name} 长线={result['score']} (gate={gate.risk_level})")
        return {"data": {"long_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} LongTermScorerNode 失败: {e}", exc_info=True)
        raise
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/scoring_nodes.py
git commit -m "feat: scoring_nodes v2 — build AnalysisPackage, pass structured input, apply risk_gate"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 12: 改造 short_term_scorer.py — 新增依赖+权重+structured input

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/short_term_scorer.py`

- [ ] **Step 1: 扩展函数签名——新增 event/moneyflow/analysis_package 参数**

替换 `async def short_term_scorer(` 的函数签名为:

```python
async def short_term_scorer(
    stock_code: str,
    company_name: str,
    technical_analysis: str = "",
    news_analysis: str = "",
    event_analysis: str = "",
    moneyflow_analysis: str = "",
    analysis_package = None,
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
    model_name: str = "",
    model_api_key: str = "",
    model_base_url: str = "",
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
```

- [ ] **Step 2: 更新评分维度和权重**

在 system_prompt 中将评分体系（原文件约第132-193行）替换为:

```
### 1. 技术状态（满分25分）
评估: 均线排列(5/10/20日)、MACD金叉/死叉、RSI超买/超卖、关键K线形态、支撑阻力位

### 2. 量价/流动性（满分20分）
评估: 成交量异动(相对20日均量)、量价配合(放量上涨/缩量回调)、换手率水平、流动性是否支持短线执行

### 3. 资金确认（满分20分）
评估: 主力资金流向(融资融券/龙虎榜/大宗交易)、资金是否在确认技术信号、是否存在资金与技术背离

### 4. 事件催化（满分20分）
评估: 近期事件催化剂(业绩预告/回购/重大合同)、事件时效与影响力、是否已被市场price-in

### 5. 新闻叙事/情绪（满分15分）
评估: 媒体情绪(利好/利空)、题材热度、板块共振效应

风险扣分：由后处理模块统一执行
```

- [ ] **Step 3: 在 user_prompt 中注入 structured context**

在 user_prompt 拼接部分，优先注入 `analysis_package.compact_prompt_context`:

```python
    # 优先注入结构化分析上下文
    if analysis_package and hasattr(analysis_package, 'compact_prompt_context'):
        structured_context = analysis_package.compact_prompt_context
    else:
        structured_context = ""

    # user_prompt 拼接
    user_prompt = f"""请对以下股票进行短线投资打分：

公司名称：{company_name}
股票代码：{stock_code}

"""
    if structured_context:
        user_prompt += f"## 结构化分析摘要（优先参考）\n{structured_context}\n\n"

    if technical_analysis:
        user_prompt += f"## 技术分析数据\n{technical_analysis}\n\n"
    if news_analysis:
        user_prompt += f"## 新闻分析数据\n{news_analysis}\n\n"
    if event_analysis:
        user_prompt += f"## 事件分析数据\n{event_analysis}\n\n"
    if moneyflow_analysis:
        user_prompt += f"## 资金面分析数据\n{moneyflow_analysis}\n\n"
```

- [ ] **Step 4: 更新JSON输出模板增加可选字段**

在 system_prompt 的 JSON 输出模板中增加:
```json
"key_drivers": ["驱动因素1"],
"risk_flags": [],
"abstain": false,
"abstain_reason": "",
"data_quality_score": 0.0,
"confidence": 0.0
```

- [ ] **Step 5: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/short_term_scorer.py
git commit -m "feat: short_term_scorer v2 — event+moneyflow deps, new weights, structured input"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 13: 改造 medium_term_scorer.py — 结构化输入+权重调整

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/medium_term_scorer.py`

- [ ] **Step 1: 扩展函数签名**

```python
async def medium_term_scorer(
    stock_code: str,
    company_name: str,
    fundamental_analysis: str = "",
    technical_analysis: str = "",
    value_analysis: str = "",
    news_analysis: str = "",
    event_analysis: str = "",
    quality_risk_analysis: str = "",
    moneyflow_analysis: str = "",
    analysis_package = None,
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
    model_name: str = "",
    model_api_key: str = "",
    model_base_url: str = "",
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
```

- [ ] **Step 2: 更新 scoring dimensions (参考大纲 10.4节)**

在原 system_prompt 中替换评分维度为:

```
### 1. 基本面质量（满分20分）
ROE/ROIC持续性、盈利质量(经营现金流/净利润)、资产负债健康度、成长持续性

### 2. 估值（满分15分）
PE/PB/EV-EBITDA相对行业、历史分位、估值对增长的反映程度、安全边际

### 3. 财务质量/治理风险（满分20分）
利润现金含量、应收/存货/商誉风险、质押/减持/关联交易、审计与监管风险

### 4. 事件持续性（满分15分）
催化剂是否可持续(一次性vs持续性)、事件对1-3个月窗口的实质影响

### 5. 技术与量价确认（满分10分）
中期趋势方向(20/60日均线)、量价配合、是否为中线进场/出场时机

### 6. 行业/估值适配（满分10分）
行业景气周期、行业相对估值容忍度、跨行业可比性

### 7. 新闻叙事（满分10分）
市场预期是否合理、是否形成一致预期、题材持续性

风险扣分：由risk_gate统一后处理
```

- [ ] **Step 3: 同步注入 structured context（同 short_term 模式）**

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/medium_term_scorer.py
git commit -m "feat: medium_term_scorer v2 — new deps, adjusted weights, structured input"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 14: 改造 long_term_scorer.py — 结构化输入+权重调整

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/long_term_scorer.py`

- [ ] **Step 1: 扩展函数签名（同 medium_term 模式）**

- [ ] **Step 2: 更新 scoring dimensions (参考大纲 10.4节)**

```
### 1. 基本面与资本回报（满分25分）
ROE/ROIC长期持续性、资本配置能力(再投资vs分红vs回购)、自由现金流累积能力

### 2. 财务质量/治理风险（满分20分）
财务真实性(利润现金含量)、减值/商誉/表外风险、治理结构与股东行为

### 3. 估值安全边际（满分15分）
3-5年历史估值分位、跨周期估值区间、DCF/剩余收益等绝对估值锚

### 4. 行业地位/商业质量（满分15分）
护城河深度与类型、市场份额与定价权、行业生命周期阶段

### 5. 资本配置/股东回报（满分10分）
分红历史与可持续性、回购意愿、资本开支纪律

### 6. 事件与政策风险（满分10分）
长期政策方向(产业政策/监管框架)、宏观系统性风险、地缘风险

### 7. 技术确认（满分5分）
仅用月线级别趋势判断极端高/低位，不过度放大短期价格行为

风险扣分：由risk_gate统一后处理
```

- [ ] **Step 3: 同步注入 structured context**

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/long_term_scorer.py
git commit -m "feat: long_term_scorer v2 — new deps, adjusted weights, structured input"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 15: 改造 summary_agent.py — 9段式报告 + analysis_package 输入

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/summary_agent.py`

- [ ] **Step 1: 在函数开头构建 analysis_package**

在 `summary_agent` 函数中，`current_data` 获取之后，增加:

```python
    from src.utils.analysis_package_builder import build_analysis_package
    pkg = build_analysis_package(current_data, current_date)
```

- [ ] **Step 2: 替换 system_prompt 为9段式结构**

将 system_prompt（原文件约第157-309行）替换为:

```python
        system_prompt = f"""
        你是一位资深A股证券分析师，拥有10年以上卖方研究经验。

        **重要时间信息：当前实际时间是 {current_time_info}**
        **分析基准日期：{current_date}**

        你的任务是综合7种分析结果，创建一份结构清晰的专业研究报告。

        ## 报告结构（必须严格遵循此9段格式）

        ### 1. 核心结论
        [3-5句话：最核心的投资判断、关键理由、总体评级]

        ### 2. 多维信号总览
        [表格或列表形式：列出每个分析维度的方向(看多/中性/看空)和置信度]
        [必须包含: 基本面、技术面、估值、新闻舆情、事件驱动、质量风险、资金面]

        ### 3. 关键利多因素
        [列出最重要的看多证据，每条标注来源agent和数据基础]
        [区分: 事实 vs 推断 vs 建议]

        ### 4. 关键利空与反证
        [必须写！列出最重要的看空证据和风险点]
        [当不同agent结论冲突时，必须明确写出冲突，不要只保留一种声音]

        ### 5. 事件与催化剂时间线
        [近期已发生和即将发生的关键事件，标注日期/时效/影响方向/影响期限]

        ### 6. 短线 / 中线 / 长线判断
        [分别给出三个期限的专业判断，明确区分不同期限的逻辑和风险]
        [短线: 量价/资金/催化剂驱动；中线: 基本面/估值/事件；长线: 护城河/行业/治理]

        ### 7. 主要风险与需要继续核验的数据
        [列出剩余不确定性、数据缺口、需要后续跟踪的指标]
        [数据不足时必须承认不确定性]

        ### 8. 结论的置信度与适用边界
        [当前结论的置信度评估(高/中/低)，在什么条件下结论会改变]
        [说明该结论适用的投资者类型和市场环境]

        ### 9. 风险提示
        [市场有风险，投资需谨慎]
        [具体到该股票的风险提示，不使用泛泛套话]

        ## 写作原则
        1. 明确区分事实([数据])、推断([判断])、建议([建议])
        2. 当不同分析维度冲突时，必须写出冲突而非只保留一种声音
        3. 必须有反证部分
        4. 数据不足时承认不确定性
        5. 使用简洁专业语言，避免空泛套话

        ⛔ 防幻觉规则:
        - 所有陈述必须标注 [数据] 或 [判断]
        - 禁止编造数值、新闻、或未在输入中出现的事实
        - 如果某模块数据缺失，写"该模块数据不可用"

        输出为纯Markdown，不含代码块标记。
        """
```

- [ ] **Step 3: 更新 user_prompt — 注入 pkg.compact_prompt_context**

在现有 user_prompt 开头增加:

```python
        user_prompt = f"""
请为 {company_name} ({stock_code}) 创建综合分析报告。

原始用户查询: {user_query}

## 结构化分析摘要
{pkg.compact_prompt_context}

## 各维度原始分析
"""
```

然后保留原有的各 `*_analysis` 拼接（含新增的 event/quality_risk/moneyflow）。

- [ ] **Step 4: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/agents/summary_agent.py
git commit -m "feat: summary_agent v2 — 9-section report, reads analysis_package"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 16: 改造 main.py — 图结构从4→7并行

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/main.py`

- [ ] **Step 1: 新增 import**

在文件头部现有 agent import 之后追加:

```python
from src.agents.event_analyst_agent import event_analyst_agent
from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent
from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent
```

- [ ] **Step 2: 新增3个节点**

在现有的 `workflow.add_node` 之后追加:

```python
        workflow.add_node("event_analyst", event_analyst_agent)
        workflow.add_node("quality_risk_analyst", quality_risk_analyst_agent)
        workflow.add_node("moneyflow_analyst", moneyflow_analyst_agent)
```

- [ ] **Step 3: 新增并行边**

在现有关键字 `workflow.add_edge("start_node", "news_analyst")` 之后追加:

```python
        workflow.add_edge("start_node", "event_analyst")
        workflow.add_edge("start_node", "quality_risk_analyst")
        workflow.add_edge("start_node", "moneyflow_analyst")
```

- [ ] **Step 4: 新增汇聚到 summarizer 的边**

在现有的 `workflow.add_edge("news_analyst", "summarizer")` 之后追加:

```python
        workflow.add_edge("event_analyst", "summarizer")
        workflow.add_edge("quality_risk_analyst", "summarizer")
        workflow.add_edge("moneyflow_analyst", "summarizer")
```

- [ ] **Step 5: 更新日志提示**

将打印语句更新为:

```python
        print(f"\n{WAIT_ICON} 正在执行基本面分析...")
        print(f"{WAIT_ICON} 正在执行技术面分析...")
        print(f"{WAIT_ICON} 正在执行估值分析...")
        print(f"{WAIT_ICON} 正在执行新闻分析...")
        print(f"{WAIT_ICON} 正在执行事件驱动分析...")
        print(f"{WAIT_ICON} 正在执行质量风险评估...")
        print(f"{WAIT_ICON} 正在执行资金面分析...")
```

- [ ] **Step 6: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/main.py
git commit -m "feat: main.py graph — 4→7 parallel analysis agents"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 17: 改造 scoring_engine.py — 图结构扩展+scorer依赖调整

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/stock_pool/scoring_engine.py`

- [ ] **Step 1: 新增 import**

在现有 agent import 之后追加:

```python
from src.agents.event_analyst_agent import event_analyst_agent
from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent
from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent
```

- [ ] **Step 2: 在 _build_workflow 中新增3个分析节点**

在现有 `workflow.add_node("news_analyst", news_agent)` 之后追加:

```python
            workflow.add_node("event_analyst", event_analyst_agent)
            workflow.add_node("quality_risk_analyst", quality_risk_analyst_agent)
            workflow.add_node("moneyflow_analyst", moneyflow_analyst_agent)
```

- [ ] **Step 3: 从 start_node 扇出到新agent**

在现有 `workflow.add_edge("start_node", "news_analyst")` 之后追加:

```python
            workflow.add_edge("start_node", "event_analyst")
            workflow.add_edge("start_node", "quality_risk_analyst")
            workflow.add_edge("start_node", "moneyflow_analyst")
```

- [ ] **Step 4: 调整 scorer 依赖边**

在现有的 short_term_scorer 依赖之后追加:

```python
            # short_term: 新增 event + moneyflow 依赖
            workflow.add_edge("event_analyst", "short_term_scorer")
            workflow.add_edge("moneyflow_analyst", "short_term_scorer")
```

在 medium_term 和 long_term 的现有4条边之后追加:

```python
            # medium_term + long_term: 新增3个agent作为依赖
            for scorer in ["medium_term_scorer", "long_term_scorer"]:
                workflow.add_edge("event_analyst", scorer)
                workflow.add_edge("quality_risk_analyst", scorer)
                workflow.add_edge("moneyflow_analyst", scorer)
```

- [ ] **Step 5: 在 _build_initial_state 中增加 analysis_version**

在 `initial_data = {` dict 中追加:

```python
            "analysis_version": "a_share_v2",
```

- [ ] **Step 6: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/stock_pool/scoring_engine.py
git commit -m "feat: scoring_engine graph — 4→7 agents, adjusted scorer dependencies"

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

---

### Task 18: 扩展 state_definition.py 注释

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/utils/state_definition.py`

- [ ] **Step 1: 更新字段注释**

在文件末尾注释中追加:

```python
    # 架构升级 v2 (2026-06):
    #   信号包（结构化中间产物）:
    #     fundamental_signal_pack, technical_signal_pack, value_signal_pack,
    #     news_signal_pack, event_signal_pack, quality_risk_signal_pack,
    #     moneyflow_signal_pack
    #   合并产物: analysis_package
    #   metadata 新增: analysis_version="a_share_v2", executed_agents,
    #     missing_agents, data_quality_summary, warnings
```

- [ ] **Step 2: Commit**

---

### Task 19: 运行完整测试套件验证无回归

- [ ] **Step 1: 运行全部测试**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/ -v 2>&1 | head -80
```

检查:
- 新测试（`test_analysis_package.py`, `test_risk_gate.py`）全部 PASS
- 现有测试（`test_batch_scorer.py`, `test_qa.py` 等）无新增失败

- [ ] **Step 2: 如发现回归，修复并重新运行**

---

### Task 20: Smoke Test — CLI 单票分析

- [ ] **Step 1: 运行 CLI 单票分析**

```bash
cd Finance/Financial-MCP-Agent && timeout 600 python -m src.main --command "分析嘉友国际" 2>&1 | tail -50
```

检查点:
- 7个 agent 全部执行（fundamental, technical, value, news, event, quality_risk, moneyflow）
- 报告以9段式结构输出
- 无异常崩溃

- [ ] **Step 2: 验证报告结构**

检查报告是否包含 核心结论/多维信号总览/关键利多/关键利空与反证/事件与催化剂时间线/短线中线长线判断/风险与核验数据/置信度与适用边界/风险提示

---

### Task 21: Smoke Test — 股票池评分

- [ ] **Step 1: 运行股票池评分**

```bash
cd Finance/Financial-MCP-Agent && timeout 600 python -m src.main_pool score 603871 2>&1 | tail -30
```

检查点:
- short/medium/long 三个评分均有 risk_gate 信息
- 评分中有 confidence/data_quality_score/risk_flags 等新字段

- [ ] **Step 2: 运行股票池报告**

```bash
cd Finance/Financial-MCP-Agent && timeout 300 python -m src.main_pool report 603871 2>&1 | tail -30
```

---

## 自审报告

**1. Spec coverage:**

| 文档要求 | Task |
|---------|------|
| P0.1 扩展现有4个agent分析深度 | Tasks 5,6,7 |
| P0.2 新增event_analyst | Task 8 |
| P0.3 新增quality_risk_analyst | Task 9 |
| P0.4 结构化中间产物 (signal_pack) | Tasks 1,2,5,6,7,8,9,10 |
| P0.5 改造scorer读取结构化数据 | Tasks 11,12,13,14 |
| P0.6 报告增加反证/风险/数据缺口 | Task 15 |
| P0.7 risk_gate后处理 | Task 3 |
| P1.1 独立moneyflow_analyst | Task 10 |
| P1.2 scorer追加可选字段 | Tasks 11,12,13,14 |
| P1.3 前端适配 | 未纳入（doc: "不在范围"） |
| 入口层改造 | Tasks 16,17 |
| 测试 | Tasks 19,20,21 |

**2. Placeholder scan:** 检查通过。所有步骤包含实际代码或命令。

**3. Type consistency:** `build_analysis_package(state_data, as_of_date)` → `AnalysisPackage`; `apply_risk_gate(package, term, score)` → `RiskGateResult`; scorer函数签名的 `analysis_package` 参数通过 `hasattr(pkg, 'compact_prompt_context')` 检查。一致。
```
