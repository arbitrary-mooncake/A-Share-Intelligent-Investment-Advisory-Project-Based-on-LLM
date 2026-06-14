# A股主线架构升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 将A股主线从"4段分析文本拼接打分"升级为"7维结构化证据 + 风险门控 + 专业报告"系统

**Architecture:** 自底向上（5层）：基础层(schema/builder/gate) → Agent层(4改+3新) → 集成层(scorers/summary) → 入口层(graph) → 测试层

**Tech Stack:** LangGraph, OpenAI-compatible LLM (5-model架构), Tushare MCP, Python 3.10+

---

### Task 1: 新增 analysis_schema.py — 信号与分析的TypedDict定义

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/analysis_schema.py`

- [ ] **Step 1: 创建 schema 文件**

```python
"""
分析中间产物的结构化 Schema 定义。
使用轻量 dataclass + TypedDict，不引入重型依赖。

字段参考: 子agent架构改进方案.md 第8.2节
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict


# ── Source Level 枚举（证据优先级） ──

class SourceLevel:
    """证据来源可靠性层级，用于 global_risk_flags 归并时的优先级判定"""
    OFFICIAL = "official_like"   # 正式结构化数据
    STRUCTURED = "structured"    # 可验证的数值工具输出
    NEWS = "news"                # 媒体/资讯
    DERIVED = "derived"          # 推断
    PROXY = "proxy"              # 临时代理

SOURCE_PRIORITY = {
    SourceLevel.OFFICIAL: 5,
    SourceLevel.STRUCTURED: 4,
    SourceLevel.NEWS: 3,
    SourceLevel.DERIVED: 2,
    SourceLevel.PROXY: 1,
}


# ── Signal Pack ──

@dataclass
class Signal:
    """单个信号"""
    factor: str              # 因子名，如"经营现金流/净利润匹配"
    direction: int           # 1=看多, -1=看空, 0=中性
    strength: int            # 信号强度 0-100
    confidence: float        # 置信度 0.0-1.0
    time_horizon: List[str]  # ["short", "medium", "long"]
    source_level: str        # SourceLevel 枚举值
    freshness: str           # "intraday"|"daily"|"weekly"|"quarterly"|"unknown"
    risk_flags: List[str]    # 关联风险标签
    note: str                # 一句话说明


@dataclass
class SignalPack:
    """单个 Agent 的结构化输出"""
    agent_name: str
    analysis_text: str                # 给人看的简洁总结（向后兼容）
    bias: str                         # "bullish"|"neutral"|"bearish"
    confidence: float                 # 结论置信度 0.0-1.0
    data_quality_score: float         # 数据完整度 0.0-1.0
    key_points: List[str]             # 关键结论（≤6条）
    signals: List[Signal]             # 结构化信号（≤8条）
    risk_flags: List[str]             # 该Agent输出的风险标签
    missing_data: List[str]           # 缺失数据项
    source_summary: str               # 数据来源简述
    as_of_date: str                   # "YYYY-MM-DD"


# 用于 json.loads 后构建 SignalPack 的 TypedDict（可选，解析校验用）
class SignalDict(TypedDict, total=False):
    factor: str
    direction: int
    strength: int
    confidence: float
    time_horizon: List[str]
    source_level: str
    freshness: str
    risk_flags: List[str]
    note: str


class SignalPackDict(TypedDict, total=False):
    agent_name: str
    analysis_text: str
    bias: str
    confidence: float
    data_quality_score: float
    key_points: List[str]
    signals: List[SignalDict]
    risk_flags: List[str]
    missing_data: List[str]
    source_summary: str
    as_of_date: str


# ── Analysis Package (多 Agent 合并产物) ──

@dataclass
class AnalysisPackage:
    """合并所有 Agent 的 signal pack 后的统一分析产物"""
    as_of_date: str
    executed_agents: List[str]
    available_agents: List[str]
    missing_agents: List[str]
    global_risk_flags: List[str]
    global_missing_data: List[str]
    bullish_signals: List[dict]    # 按 strength 降序
    bearish_signals: List[dict]    # 按 strength 降序
    conflicting_signals: List[dict]
    source_priority_summary: Dict[str, Any]
    compact_prompt_context: str    # 供 scorer/summary 直接使用的压缩文本


# ── Risk Gate Result ──

@dataclass
class RiskGateResult:
    """风险门控后处理结果"""
    risk_level: str              # "low"|"medium"|"high"|"critical"
    risk_flags_found: List[str]
    score_cap: Optional[int]     # 分数上限
    action_downgrade: Optional[str]  # 降级建议
    abstain: bool
    abstain_reason: str
    data_quality_score: float
    warnings: List[str]


# ── Scorer 输出扩展字段 ──

SCORER_OPTIONAL_FIELDS = {
    "confidence": 0.0,       # float
    "key_drivers": [],       # List[str]
    "risk_flags": [],        # List[str]
    "abstain": False,        # bool
    "abstain_reason": "",    # str
    "data_quality_score": 0.0,  # float
}

# ── Fallback 常量 ──

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

- [ ] **Step 2: Commit**

---

### Task 2: 新增 analysis_package_builder.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/analysis_package_builder.py`

- [ ] **Step 1: 创建 builder 模块**

```python
"""
Analysis Package Builder: 将多个 *_signal_pack 合并为统一 analysis_package
职责:
  1. 合并 signal packs
  2. 对重复因子去重归并
  3. source priority 排序
  4. 汇总 global_risk_flags
  5. 生成 compact_prompt_context
"""
import json
from typing import Dict, Any, List, Optional

from src.utils.analysis_schema import (
    SourceLevel, SOURCE_PRIORITY,
    FALLBACK_SIGNAL_PACK,
    AnalysisPackage, RiskGateResult,
)


def _parse_signal_pack(raw: Any, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    """将任意输入标准化为 signal_pack dict。失败时返回 fallback。"""
    if isinstance(raw, dict):
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
            "analysis_text": raw[:500],
            "as_of_date": as_of_date,
        }
    return dict(FALLBACK_SIGNAL_PACK, agent_name=agent_name, as_of_date=as_of_date)


def _extract_bias_from_text(text: str) -> str:
    """从纯文本中提取bias：看多/看空/中性"""
    if not text:
        return "neutral"
    t = text.lower()
    bullish_kw = ["看多", "利多", "利好", "买入", "增持", "推荐", "bullish", "积极", "改善", "增长强劲", "超预期"]
    bearish_kw = ["看空", "利空", "利淡", "卖出", "减持", "bearish", "恶化", "下滑", "风险较高", "谨慎"]
    bull = sum(1 for kw in bullish_kw if kw in t)
    bear = sum(1 for kw in bearish_kw if kw in t)
    if bull > bear:
        return "bullish"
    elif bear > bull:
        return "bearish"
    return "neutral"


def _extract_key_points_from_text(text: str, max_points: int = 5) -> List[str]:
    """从纯文本中用启发式规则提取关键要点"""
    import re
    points = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        # 取以 - 或数字开头或包含结论性关键词的行
        if re.match(r'^[-•\d]+[\.\、\)]', line):
            clean = re.sub(r'^[-•\d]+[\.\、\)]\s*', '', line)
            points.append(clean[:120])
        elif any(kw in line for kw in ["结论", "综合", "关键", "核心", "重点", "主要"]):
            points.append(line[:120])
    return points[:max_points]


def _extract_risk_flags_from_text(text: str) -> List[str]:
    """从纯文本中提取风险标签"""
    import re
    flags = []
    risk_patterns = [
        ("现金流不匹配", "cashflow_mismatch"),
        ("质押风险", "high_pledge_risk"),
        ("监管风险", "regulatory_risk"),
        ("审计风险", "audit_risk"),
        ("减值风险", "impairment_risk"),
        ("盈利质量", "earnings_quality_concern"),
        ("ST风险", "st_risk"),
        ("退市风险", "delist_risk"),
        ("减持", "major_shareholder_sell"),
        ("商誉", "goodwill_risk"),
        ("债务风险", "debt_risk"),
        ("流动性风险", "liquidity_risk"),
    ]
    for pattern, flag in risk_patterns:
        if pattern in text:
            flags.append(flag)
    return list(set(flags))  # 去重


def text_to_signal_pack(text: str, agent_name: str, as_of_date: str) -> Dict[str, Any]:
    """
    从纯自然语言文本构建 fallback signal_pack。
    用于旧 agent 尚未产出结构化产物时的兼容降级。
    """
    return {
        "agent_name": agent_name,
        "analysis_text": text[:800],
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


def build_analysis_package(
    state_data: Dict[str, Any],
    as_of_date: str = "",
) -> AnalysisPackage:
    """
    从 state.data 中提取所有 *_signal_pack（或 fallback from *_analysis），
    合并为统一 AnalysisPackage。

    Args:
        state_data: LangGraph state.data 字典
        as_of_date: 分析日期
    Returns:
        AnalysisPackage 实例
    """
    agent_fields = {
        "fundamental": "fundamental_analysis",
        "technical": "technical_analysis",
        "value": "value_analysis",
        "news": "news_analysis",
        "event": "event_analysis",
        "quality_risk": "quality_risk_analysis",
        "moneyflow": "moneyflow_analysis",
    }

    signal_packs: Dict[str, Dict[str, Any]] = {}
    for agent_name, text_key in agent_fields.items():
        sp_key = f"{agent_name}_signal_pack"
        if sp_key in state_data:
            signal_packs[agent_name] = _parse_signal_pack(
                state_data[sp_key], agent_name, as_of_date
            )
        elif text_key in state_data:
            signal_packs[agent_name] = text_to_signal_pack(
                state_data[text_key], agent_name, as_of_date
            )
        else:
            signal_packs[agent_name] = dict(FALLBACK_SIGNAL_PACK, agent_name=agent_name,
                                            missing_data=["Agent未执行"])

    # 归并
    all_signals = []
    global_risk_flags = []
    global_missing = []
    available = []
    missing = []

    for agent_name, sp in signal_packs.items():
        if sp.get("missing_data") and "Agent未执行" not in str(sp.get("missing_data")):
            available.append(agent_name)
        elif sp.get("data_quality_score", 0) < 0.3 and sp.get("missing_data") and "Agent未执行" in str(sp.get("missing_data")):
            missing.append(agent_name)
        else:
            available.append(agent_name)

        global_risk_flags.extend(sp.get("risk_flags", []))
        global_missing.extend(sp.get("missing_data", []))

        for sig in sp.get("signals", []):
            if isinstance(sig, dict):
                sig["_agent"] = agent_name
                all_signals.append(sig)

    # 按 source priority + strength 排序
    all_signals.sort(
        key=lambda s: (
            SOURCE_PRIORITY.get(s.get("source_level", "proxy"), 0),
            abs(s.get("strength", 0)),
        ),
        reverse=True,
    )

    bullish = [s for s in all_signals if s.get("direction", 0) > 0]
    bearish = [s for s in all_signals if s.get("direction", 0) < 0]

    # 冲突检测：同一因子方向相反的信号
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

    # source priority summary
    source_counts = {}
    for s in all_signals:
        lv = s.get("source_level", "proxy")
        source_counts[lv] = source_counts.get(lv, 0) + 1

    # 生成 compact_prompt_context
    compact = _build_compact_context(
        available, missing, bullish, bearish, conflicting,
        list(set(global_risk_flags)), list(set(global_missing)),
        signal_packs,
    )

    return AnalysisPackage(
        as_of_date=as_of_date,
        executed_agents=list(set(available + missing)),
        available_agents=available,
        missing_agents=missing,
        global_risk_flags=list(set(global_risk_flags)),
        global_missing_data=list(set(global_missing)),
        bullish_signals=bullish,
        bearish_signals=bearish,
        conflicting_signals=conflicting,
        source_priority_summary={"counts": source_counts, "top_levels": sorted(source_counts, key=lambda x: SOURCE_PRIORITY.get(x, 0), reverse=True)},
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
    """生成供 scorer/summarizer LLM 直接使用的压缩上下文文本"""
    lines = []
    lines.append(f"## 分析执行概况")
    lines.append(f"- 已执行agent: {', '.join(available) if available else '无'}")
    if missing:
        lines.append(f"- 未执行agent: {', '.join(missing)}")
    lines.append("")

    # 各agent结论摘要
    lines.append("## 各Agent结论摘要")
    for agent_name, sp in signal_packs.items():
        bias_icon = {"bullish": "🔴看多", "neutral": "⚪中性", "bearish": "🟢看空"}.get(sp.get("bias", ""), "⚪中性")
        lines.append(f"- **{agent_name}**: {bias_icon} (置信度={sp.get('confidence', 0):.0%})")
        for kp in sp.get("key_points", [])[:3]:
            lines.append(f"  - {kp}")
    lines.append("")

    # 看多信号 Top 5
    lines.append("## 关键看多信号")
    for s in bullish[:5]:
        lines.append(f"- [{s.get('_agent', '?')}] {s.get('factor', '?')}: strength={s.get('strength', 0)}, src={s.get('source_level', '?')}")
    lines.append("")

    # 看空信号 Top 5
    lines.append("## 关键看空信号")
    for s in bearish[:5]:
        lines.append(f"- [{s.get('_agent', '?')}] {s.get('factor', '?')}: strength={s.get('strength', 0)}, src={s.get('source_level', '?')}")
    lines.append("")

    # 冲突信号
    if conflicting:
        lines.append("## ⚠️ 信号冲突")
        for c in conflicting[:5]:
            lines.append(f"- 因子 **{c['factor']}**: {len(c['signals'])}个信号方向不一致")
        lines.append("")

    # 风险标签
    if risk_flags:
        lines.append(f"## 🚨 全局风险标签: {', '.join(risk_flags)}")
        lines.append("")

    # 缺失数据
    if missing_data:
        lines.append(f"## ❓ 缺失数据: {', '.join(missing_data)}")
        lines.append("")

    return "\n".join(lines)


def apply_risk_gate(
    analysis_package: AnalysisPackage,
    term: str,  # "short"|"medium"|"long"
    original_score: int,
) -> RiskGateResult:
    """
    对评分结果进行风险后处理。

    Args:
        analysis_package: 合并后的分析产物
        term: 期限
        original_score: LLM原始评分
    Returns:
        RiskGateResult
    """
    from src.utils.risk_gate import apply_risk_gate as _gate
    return _gate(analysis_package, term, original_score)
```

- [ ] **Step 2: Commit**

---

### Task 3: 新增 risk_gate.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/utils/risk_gate.py`

- [ ] **Step 1: 创建风险门控模块**

```python
"""
Risk Gate: 轻量分析后处理，对评分做风险门控。
不做完整客户适当性系统。
"""
from typing import Dict, Any, List, Optional

from src.utils.analysis_schema import RiskGateResult, SourceLevel


# 高严重度风险标签 → score cap 映射
CRITICAL_RISK_FLAGS = {
    "audit_risk": {"cap": 60, "downgrade": "谨慎"},
    "regulatory_risk": {"cap": 55, "downgrade": "谨慎"},
    "high_pledge_risk": {"cap": 60, "downgrade": "谨慎"},
    "cashflow_mismatch": {"cap": 65, "downgrade": "观察"},
    "major_event_negative": {"cap": 55, "downgrade": "谨慎"},
    "delist_risk": {"cap": 50, "downgrade": "谨慎"},
    "st_risk": {"cap": 60, "downgrade": "观察"},
    "earnings_quality_concern": {"cap": 65, "downgrade": "观察"},
    "goodwill_risk": {"cap": 65, "downgrade": "观察"},
    "debt_risk": {"cap": 65, "downgrade": "观察"},
    "major_shareholder_sell": {"cap": 65, "downgrade": "观察"},
    "liquidity_risk": {"cap": 65, "downgrade": "观察"},
    "impairment_risk": {"cap": 65, "downgrade": "观察"},
}


def _count_signals_by_source(package, min_source_level: str) -> int:
    """统计不低于指定 source level 的信号数量"""
    from src.utils.analysis_schema import SOURCE_PRIORITY
    threshold = SOURCE_PRIORITY.get(min_source_level, 0)
    count = 0
    for sigs in [package.bullish_signals, package.bearish_signals]:
        for s in sigs:
            lv = s.get("source_level", SourceLevel.PROXY)
            if SOURCE_PRIORITY.get(lv, 0) >= threshold:
                count += 1
    return count


def apply_risk_gate(
    analysis_package,
    term: str,
    original_score: int,
) -> RiskGateResult:
    """
    轻量风险门控核心逻辑。

    门控规则（按优先级）:
      1. 高严重度风险标签 → score cap + downgrade
      2. 长线仅新闻叙事无事实支撑 → 不能强推荐
      3. 核心agent缺失过多 + data_quality低 → abstain
      4. 短线流动性差 → 不能高分
    """
    risk_flags = analysis_package.global_risk_flags
    missing_agents = analysis_package.missing_agents
    bearish = analysis_package.bearish_signals

    # 计算 data_quality
    total_possible = 7
    available = len(analysis_package.available_agents)
    data_quality = available / total_possible if total_possible > 0 else 1.0

    # 规则1: 关键风险标签
    score_cap = None
    downgrade = None
    found_critical = []

    for flag in CRITICAL_RISK_FLAGS:
        if flag in risk_flags:
            found_critical.append(flag)
            cfg = CRITICAL_RISK_FLAGS[flag]
            if score_cap is None or cfg["cap"] < score_cap:
                score_cap = cfg["cap"]
                downgrade = cfg["downgrade"]

    # 规则2: 仅新闻叙事
    factual_signals = _count_signals_by_source(analysis_package, SourceLevel.STRUCTURED)
    news_only_signals = _count_signals_by_source(analysis_package, SourceLevel.NEWS)
    if factual_signals == 0 and news_only_signals > 0 and term in ("medium", "long"):
        if score_cap is None or score_cap > 55:
            score_cap = 55
            downgrade = "观察"
            found_critical.append("news_only_narrative")

    # 规则3: 数据严重不足
    if len(missing_agents) >= 2 and data_quality < 0.4:
        return RiskGateResult(
            risk_level="critical" if found_critical else "high",
            risk_flags_found=found_critical,
            score_cap=min(score_cap or 50, 50),
            action_downgrade=downgrade or "观察",
            abstain=True,
            abstain_reason=f"数据不足且核心agent缺失{len(missing_agents)}个: {', '.join(missing_agents)}",
            data_quality_score=data_quality,
            warnings=[f"缺失agent: {', '.join(missing_agents)}"],
        )

    # 规则4: 短线流动性风险
    if term == "short":
        liquidity_sigs = [s for s in bearish if "流动" in s.get("factor", "") or "量价" in s.get("factor", "")]
        if liquidity_sigs and any(abs(s.get("strength", 0)) > 60 for s in liquidity_sigs):
            if score_cap is None or score_cap > 50:
                score_cap = 50
                downgrade = "观察"

    # 风险等级
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
        warnings.append(f"数据质量低({data_quality:.0%})，结论置信度受影响")

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

- [ ] **Step 2: Commit**

---

### Task 4: 扩展 model_config.py — 新增3个agent的模型映射

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/utils/model_config.py`

- [ ] **Step 1: 在 AGENT_MODEL_SUFFIX dict 中新增3个agent映射**

在 `AGENT_MODEL_SUFFIX` dict 中，`"news_agent": "_3"` 之后新增:

```python
# ── 新增 Agent (2026-06 架构升级) ──
"event_analyst": "_3",            # Model 3: Qwen3.7-Plus — 事件/公告分析
"quality_risk_analyst": "_4",     # Model 4: Kimi K2.6 — 财务质量/治理风险深度分析
"moneyflow_analyst": "_3",        # Model 3: Qwen3.7-Plus — 资金面/量价确认
```

- [ ] **Step 2: 验证映射正确性**

Run: `python -c "from src.utils.model_config import AGENT_MODEL_SUFFIX; print(AGENT_MODEL_SUFFIX.get('event_analyst'))"`
Expected: `_3`

- [ ] **Step 3: Commit**

---

### Task 5: 扩展 state_definition.py — metadata字段

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/utils/state_definition.py`

- [ ] **Step 1: 更新注释，列出新增字段**

在现有文件末尾的注释中，追加新增字段说明:
```python
# 架构升级 v2 (2026-06) 新增字段：
#   信号包（结构化中间产物）:
#     fundamental_signal_pack, technical_signal_pack, value_signal_pack,
#     news_signal_pack, event_signal_pack, quality_risk_signal_pack,
#     moneyflow_signal_pack
#   合并产物:
#     analysis_package, risk_gate_result
#   metadata 新增:
#     analysis_version = "a_share_v2"
#     executed_agents, missing_agents, data_quality_summary, warnings
```

- [ ] **Step 2: Commit**

---

### Task 6: 改造 fundamental_agent.py — 增加 signal_pack 输出

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/fundamental_agent.py`

- [ ] **Step 1: 在 LLM analysis_prompt 末尾追加 JSON 输出指令**

在 Phase 2 的 `analysis_prompt` 末尾（在"⛔ 输出格式要求"之后），追加:

```python
        # Phase 2 prompt 追加结构化输出要求
        signal_pack_instruction = f"""
        
        ⛔ 结构化输出要求：
        在完成上述分析后，请额外输出一个 JSON block，包含以下结构化数据：
        
        <SIGNAL_PACK>
        {{
            "bias": "bullish"|"neutral"|"bearish",
            "confidence": 0.0-1.0,
            "key_points": ["关键结论1", "关键结论2", ...] (最多6条),
            "signals": [
                {{
                    "factor": "因子名",
                    "direction": 1|-1|0,
                    "strength": 0-100置信强度,
                    "time_horizon": ["medium", "long"],
                    "source_level": "official_like"|"structured"|"derived",
                    "note": "一句话说明"
                }}
            ] (最多6条),
            "risk_flags": ["cashflow_mismatch", "earnings_quality_concern", ...],
            "missing_data": ["未获取到质押数据", ...],
            "source_summary": "数据来源简述"
        }}
        </SIGNAL_PACK>
        
        请确保 SIGNAL_PACK 是有效的 JSON，放在 <SIGNAL_PACK> 和 </SIGNAL_PACK> 标签之间。
        """
        
        analysis_prompt = analysis_prompt + signal_pack_instruction
```

- [ ] **Step 2: 在 LLM response 解析后增加 signal_pack 提取**

在 Phase 2 完成后（`final_output` 获取后），追加:

```python
        # ── 提取 signal_pack ──────────────────────
        signal_pack = _extract_signal_pack(final_output, "fundamental", current_date, raw_data_text)
        current_data["fundamental_signal_pack"] = signal_pack
```

- [ ] **Step 3: 添加 signal_pack 提取工具函数（文件顶部）**

```python
def _extract_signal_pack(llm_output: str, agent_name: str, as_of_date: str, raw_data_text: str = "") -> Dict[str, Any]:
    """
    从 LLM 输出中提取 signal_pack JSON。
    三层fallback: JSON解析 → 正则提取 → 文本推断
    """
    import re as _re
    from src.utils.analysis_package_builder import text_to_signal_pack

    # 第一层: 从 <SIGNAL_PACK> 标签提取
    tag_match = _re.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', llm_output)
    if tag_match:
        try:
            sp = json.loads(tag_match.group(1))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", _estimate_data_quality(sp, raw_data_text))
            return sp
        except (json.JSONDecodeError, ValueError):
            pass

    # 第二层: 尝试从整体输出中找最外层JSON对象（回退）
    json_match = _re.search(r'\{[\s\S]*"bias"[\s\S]*"signals"[\s\S]*\}', llm_output)
    if json_match:
        try:
            sp = json.loads(json_match.group(0))
            sp["agent_name"] = agent_name
            sp["as_of_date"] = as_of_date
            sp.setdefault("analysis_text", llm_output[:500])
            sp.setdefault("data_quality_score", _estimate_data_quality(sp, raw_data_text))
            return sp
        except (json.JSONDecodeError, ValueError):
            pass

    # 第三层: 从纯文本推断
    return text_to_signal_pack(llm_output, agent_name, as_of_date)


def _estimate_data_quality(sp: Dict[str, Any], raw_data_text: str) -> float:
    """估算数据质量"""
    signals = sp.get("signals", [])
    missing = sp.get("missing_data", [])
    if not signals and missing:
        return 0.2
    if len(missing) > 5:
        return 0.3
    if raw_data_text and len(raw_data_text) < 200:
        return 0.3
    return 0.7
```

- [ ] **Step 4: 在文件头部 import json（如未导入）**

```python
import json
```

- [ ] **Step 5: Commit**

---

### Task 7: 改造 technical_agent.py — 增加 signal_pack 输出

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/technical_agent.py`

- [ ] **Step 1: 在 agent_input 末尾追加结构化输出要求**

在 agent_input 末尾（"⛔ 输出格式要求" 之后）追加 signal_pack JSON 指令（同 fundamental_agent 模式）。

- [ ] **Step 2: 在最终输出提取后增加 signal_pack 解析**

在 `final_output` 赋值后:

```python
            from src.utils.analysis_package_builder import text_to_signal_pack
            import re as _re_sp
            import json as _json_sp

            sp = None
            tag_match = _re_sp.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', final_output)
            if tag_match:
                try:
                    sp = _json_sp.loads(tag_match.group(1))
                    sp["agent_name"] = "technical"
                    sp["as_of_date"] = current_date
                    sp.setdefault("analysis_text", final_output[:500])
                except Exception:
                    pass
            if sp is None:
                sp = text_to_signal_pack(final_output, "technical", current_date)
            current_data["technical_signal_pack"] = sp
```

- [ ] **Step 3: Commit**

---

### Task 8: 改造 value_agent.py — 增加 signal_pack 输出

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/value_agent.py`

- [ ] **Step 1: 按 fundamental_agent 相同模式改造**

在 LLM prompt 末尾追加 SIGNAL_PACK JSON 指令；在响应解析后增加 `_extract_signal_pack` 调用（复用 fundamental_agent 中定义的函数，或直接 import）。

- [ ] **Step 2: Commit**

---

### Task 9: 改造 news_agent.py — 重新定位 + signal_pack 输出

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/news_agent.py`

- [ ] **Step 1: 修改 system prompt，明确 news 只负责舆情，不负责事件事实**

将 system prompt 中的分析要求改为:

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
                "⛔ 输出格式要求（防幻觉机制）：\n"
                "请将分析输出严格分为两个区域：\n\n"
                "## 📊 数据事实区\n"
                "列出上述新闻数据中的每一条真实新闻，逐条标注：序号、标题、来源、发布时间、核心内容摘要。\n"
                "如果新闻数据不足或无新闻数据，必须标注「新闻数据有限」而不是编造任何新闻。\n\n"
                "## 🔍 分析判断区\n"
                "基于上述新闻事实进行分析。每个判断必须引用数据事实区的具体新闻条目编号，"
                "使用「【基于新闻的推断】」标注推断性质。"
                "不得在任何地方编造数据事实区没有的数值、事件或新闻。\n\n"
                "⛔ 结构化输出要求：\n"
                "在分析完成后，额外输出一个 JSON block：\n"
                "<SIGNAL_PACK>\n"
                '{\n'
                '  "bias": "bullish"|"neutral"|"bearish",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "key_points": ["结论1", ...] (最多5条),\n'
                '  "signals": [{"factor":"...","direction":1|-1,"strength":0-100,"time_horizon":["short","medium"],"source_level":"news","note":"..."}],\n'
                '  "risk_flags": [],\n'
                '  "source_summary": "东方财富新闻接口"\n'
                '}\n'
                "</SIGNAL_PACK>\n"
            )}
```

- [ ] **Step 2: 在 final_output 后增加 signal_pack 提取**

```python
            from src.utils.analysis_package_builder import text_to_signal_pack
            import re as _re_sp
            import json as _json_sp

            sp = None
            tag_match = _re_sp.search(r'<SIGNAL_PACK>\s*(\{[\s\S]*?\})\s*</SIGNAL_PACK>', final_output)
            if tag_match:
                try:
                    sp = _json_sp.loads(tag_match.group(1))
                    sp["agent_name"] = "news"
                    sp["as_of_date"] = current_date
                    sp.setdefault("analysis_text", final_output[:500])
                except Exception:
                    pass
            if sp is None:
                sp = text_to_signal_pack(final_output, "news", current_date)
            current_data["news_signal_pack"] = sp
```

- [ ] **Step 3: Commit**

---

### Task 10: 新增 event_analyst_agent.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/agents/event_analyst_agent.py`

- [ ] **Step 1: 创建 event agent**

```python
"""
EventAnalyst Agent: 事件驱动分析
职责:
  1. 识别重大事件/催化剂（业绩预告、回购、增减持、并购重组、诉讼、处罚等）
  2. 标记事件时效与影响期限
  3. 输出明确的事件方向（利多/利空/中性，一次性/持续性）
"""
import asyncio
import json
import re
import time
from typing import Dict, Any, List

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
    "tushare_anns_d",          # 上市公司公告
    "tushare_new_share",       # 新股发行（IPO/增发）
    "tushare_pledge_stat",     # 股权质押统计
    "tushare_repurchase",      # 回购
    "tushare_share_float",     # 限售解禁
    "tushare_top10_holders",   # 十大股东变动
    "tushare_stk_holdertrade", # 股东增减持
    "tushare_dividend",        # 分红
    "tushare_namechange",      # 改名
    "tushare_suspend",         # 停复牌
    "crawl_news",              # 新闻（用于事件识别，非舆情）
    "tushare_st_status",       # ST状态
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
        logger.info(f"{WAIT_ICON} EventAnalyst: Phase 1 — 获取事件相关数据")
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
                if tname in ("tushare_anns_d",):
                    kwargs["days"] = 90
                tasks.append(_call_tool_safe(tool_map[tname], kwargs, tname))
                labels.append(tname)

        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        safe_results = [str(r) if not isinstance(r, Exception) else f"[{tname}] 调用异常" for r, tname in zip(results, labels)]

        data_sections = [f"### [{label}]\n{result}" for label, result in zip(labels, safe_results)]
        raw_data_text = "\n\n".join(data_sections) if data_sections else "无可用数据源"

        # Phase 2: LLM 分析
        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url,
                        temperature=1.0, request_timeout=LLM_TIMEOUT, max_tokens=12000,
                        extra_body=get_thinking_body(base_url, True))

        system_prompt = f"""你是一位A股事件驱动分析师。你的职责是识别和评估影响股价的重大事件。

分析维度：
1. **事件识别**：业绩预告/快报、回购、增减持、重大合同、并购重组、诉讼仲裁、处罚/问询、股权质押、异常停复牌
2. **时效标记**：事件日期、新近程度、影响期限（short/medium/long）
3. **方向判断**：利多/利空/中性；一次性/持续性；是否已被市场交易过
4. **来源置信度**：source_level标记
   - official_like: 正式公告、交易所披露
   - structured: 结构化数据工具输出
   - news: 新闻媒体
   - derived: 推断

当前时间: {current_time_info}

⛔ 输出格式：
先输出 "## 📊 数据事实区" → "## 🔍 分析判断区" 的自然语言分析。
然后在末尾输出信号包：

<SIGNAL_PACK>
{{
  "bias": "bullish"|"neutral"|"bearish",
  "confidence": 0.0-1.0,
  "key_points": ["..."],
  "signals": [
    {{
      "factor": "事件因子名",
      "direction": 1|-1|0,
      "strength": 0-100,
      "confidence": 0.0-1.0,
      "time_horizon": ["short","medium","long"],
      "source_level": "official_like"|"structured"|"news"|"derived",
      "freshness": "intraday"|"daily"|"weekly"|"quarterly",
      "note": "一句话"
    }}
  ],
  "risk_flags": ["major_event_negative", "regulatory_risk", ...],
  "missing_data": ["缺失项"],
  "source_summary": "数据来源"
}}
</SIGNAL_PACK>
"""

        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析{company_name}({stock_code})的重大事件。\n\n## 原始数据\n{raw_data_text}"}
        ])
        final_output = response.content.strip() if hasattr(response, 'content') else str(response)

        # 提取 signal_pack
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

- [ ] **Step 2: Commit**

---

### Task 11: 新增 quality_risk_analyst_agent.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/agents/quality_risk_analyst_agent.py`

- [ ] **Step 1: 创建 quality_risk agent**

同 event_analyst 的架构模式，但：

- 模型: M4 (Kimi K2.6)
- 工具: `tushare_income`, `tushare_balancesheet`, `tushare_cashflow`, `tushare_fina_indicator`, `tushare_pledge_stat`, `tushare_top10_holders`, `tushare_stk_holdertrade`, `tushare_st_status`, `get_st_risk_data`
- 分析维度（参考大纲 7.2节）:
  1. 财务质量：利润现金含量、应收/存货/商誉/减值风险
  2. 治理与股东风险：质押、冻结、减持、关联交易
  3. 风险标签：cashflow_mismatch, high_pledge_risk, regulatory_risk, audit_risk, impairment_risk, earnings_quality_concern
- 输出 `quality_risk_analysis` + `quality_risk_signal_pack`

代码结构与 event_analyst_agent.py 完全一致，替换工具列表、system_prompt、agent_name 即可。

- [ ] **Step 2: Commit**

---

### Task 12: 新增 moneyflow_analyst_agent.py

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/agents/moneyflow_analyst_agent.py`

- [ ] **Step 1: 创建 moneyflow agent**

同 event_analyst 的架构模式，但：

- 模型: M3 (Qwen3.7-Plus)
- 工具: `tushare_moneyflow`, `tushare_moneyflow_hsgt`, `tushare_margin`, `tushare_margin_detail`, `tushare_top_list`, `tushare_block_trade`, `tushare_daily_basic`, `tushare_kline`, `tushare_cyq_chips`
- 分析维度（参考大纲 7.3节）:
  1. 量价确认：换手率/成交额/持续性/放量突破/缩量整理
  2. 资金代理：融资融券/龙虎榜/大宗交易/主力资金
  3. 执行风险：流动性是否支持短线操作
- 输出 `moneyflow_analysis` + `moneyflow_signal_pack`

降级策略: 若核心 MCP 工具不可用，signal_pack source_level=proxy, data_quality_score降低。

- [ ] **Step 2: Commit**

---

### Task 13: 改造 scoring_nodes.py — 读取 analysis_package + risk_gate

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/scoring_nodes.py`

- [ ] **Step 1: 重写三个scorer node，统一使用分析产物**

三个 node 改为以下模式:

```python
async def short_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.short_term_scorer import short_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package, apply_risk_gate

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")

    logger.info(f"{WAIT_ICON} ShortTermScorerNode: 开始对 {company_name}({stock_code}) 进行短线打分")

    try:
        # 构建 analysis_package
        as_of_date = data.get("current_date", "")
        pkg = build_analysis_package(data, as_of_date)

        result = await short_term_scorer(
            stock_code=stock_code,
            company_name=company_name,
            # 旧接口兼容
            technical_analysis=data.get("technical_analysis", ""),
            news_analysis=data.get("news_analysis", ""),
            event_analysis=data.get("event_analysis", ""),
            moneyflow_analysis=data.get("moneyflow_analysis", ""),
            # 新增结构化输入
            analysis_package=pkg,
            current_time_info=data.get("current_time_info", ""),
            current_date=as_of_date,
            query=data.get("query", ""),
            model_name=data.get("model_name", ""),
            model_api_key=data.get("model_api_key", ""),
            model_base_url=data.get("model_base_url", ""),
            thinking_enabled=data.get("thinking_enabled", True),
        )

        # 风险门控
        gate_result = apply_risk_gate(pkg, "short", result["score"])
        result["risk_gate"] = {
            "risk_level": gate_result.risk_level,
            "risk_flags": gate_result.risk_flags_found,
            "score_cap": gate_result.score_cap,
            "abstain": gate_result.abstain,
            "data_quality_score": gate_result.data_quality_score,
        }

        logger.info(f"{SUCCESS_ICON} ShortTermScorerNode: {company_name} 短线评分={result['score']}")
        return {"data": {"short_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} ShortTermScorerNode 打分失败: {e}", exc_info=True)
        raise


async def medium_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.medium_term_scorer import medium_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package, apply_risk_gate

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")

    logger.info(f"{WAIT_ICON} MediumTermScorerNode: 开始对 {company_name}({stock_code}) 进行中线打分")

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

        gate_result = apply_risk_gate(pkg, "medium", result["score"])
        result["risk_gate"] = {
            "risk_level": gate_result.risk_level,
            "risk_flags": gate_result.risk_flags_found,
            "score_cap": gate_result.score_cap,
            "abstain": gate_result.abstain,
            "data_quality_score": gate_result.data_quality_score,
        }

        logger.info(f"{SUCCESS_ICON} MediumTermScorerNode: {company_name} 中线评分={result['score']}")
        return {"data": {"medium_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} MediumTermScorerNode 打分失败: {e}", exc_info=True)
        raise


async def long_term_scorer_node(state: AgentState) -> Dict[str, Any]:
    from src.agents.long_term_scorer import long_term_scorer
    from src.utils.analysis_package_builder import build_analysis_package, apply_risk_gate

    data = state.get("data", {})
    stock_code = data.get("stock_code", "")
    company_name = data.get("company_name", "")
    as_of_date = data.get("current_date", "")

    logger.info(f"{WAIT_ICON} LongTermScorerNode: 开始对 {company_name}({stock_code}) 进行长线打分")

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

        gate_result = apply_risk_gate(pkg, "long", result["score"])
        result["risk_gate"] = {
            "risk_level": gate_result.risk_level,
            "risk_flags": gate_result.risk_flags_found,
            "score_cap": gate_result.score_cap,
            "abstain": gate_result.abstain,
            "data_quality_score": gate_result.data_quality_score,
        }

        logger.info(f"{SUCCESS_ICON} LongTermScorerNode: {company_name} 长线评分={result['score']}")
        return {"data": {"long_term_score": result}}

    except Exception as e:
        logger.error(f"{ERROR_ICON} LongTermScorerNode 打分失败: {e}", exc_info=True)
        raise
```

- [ ] **Step 2: Commit**

---

### Task 14: 改造 short_term_scorer.py — 新增依赖 + 权重调整

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/short_term_scorer.py`

- [ ] **Step 1: 函数签名新增参数**

```python
async def short_term_scorer(
    stock_code: str,
    company_name: str,
    technical_analysis: str = "",
    news_analysis: str = "",
    event_analysis: str = "",           # 新增
    moneyflow_analysis: str = "",       # 新增
    analysis_package = None,             # 新增：AnalysisPackage 对象
    current_time_info: str = "",
    current_date: str = "",
    query: str = "",
    model_name: str = "",
    model_api_key: str = "",
    model_base_url: str = "",
    thinking_enabled: bool = True,
) -> Dict[str, Any]:
```

- [ ] **Step 2: 权重调整（参考大纲 10.4节）**

将评分体系改为：
- 技术状态 25 (原25)
- 量价/流动性 20 (原30)
- 资金确认 20 (新增，从原量价中拆分)
- 事件催化 20 (新增)
- 新闻叙事/情绪 15 (原25)
- 风险扣分: 后处理

system prompt 中评分维度对应更新，新增 event_context 和 moneyflow_context 的数据拼接。

- [ ] **Step 3: 优先读取 analysis_package.compact_prompt_context**

```python
    if analysis_package and hasattr(analysis_package, 'compact_prompt_context'):
        structured_context = analysis_package.compact_prompt_context
    else:
        structured_context = ""
```

在 user_prompt 中优先注入 `structured_context`。

- [ ] **Step 4: 在输出JSON中增加可选字段**

在 JSON 模板中增加 `"key_drivers": [], "risk_flags": [], "abstain": false, "abstain_reason": "", "data_quality_score": 0.0, "confidence": 0.0`

- [ ] **Step 5: Commit**

---

### Task 15: 改造 medium_term_scorer.py — 结构化输入 + 权重调整

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/medium_term_scorer.py`

- [ ] **Step 1: 函数签名新增参数**

新增 `event_analysis`, `quality_risk_analysis`, `moneyflow_analysis`, `analysis_package` 参数（同 short_term 模式）。

- [ ] **Step 2: 权重调整（参考大纲 10.4节）**

- 基本面质量 20 (原25)
- 估值 15 (原20)
- 财务质量/治理风险 20 (新增)
- 事件持续性 15 (新增)
- 技术与量价确认 10 (原15)
- 行业/估值适配 10 (不变)
- 新闻叙事 10 (原10)
- 风险扣分: 后处理

- [ ] **Step 3: 优先读取 analysis_package.compact_prompt_context**

同 short_term 模式。

- [ ] **Step 4: Commit**

---

### Task 16: 改造 long_term_scorer.py — 结构化输入 + 权重调整

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/long_term_scorer.py`

- [ ] **Step 1: 函数签名新增参数**

新增 `event_analysis`, `quality_risk_analysis`, `moneyflow_analysis`, `analysis_package` 参数。

- [ ] **Step 2: 权重调整（参考大纲 10.4节）**

- 基本面与资本回报 25 (原30中拆分)
- 财务质量/治理风险 20 (新增)
- 估值安全边际 15 (原20)
- 行业地位/商业质量 15 (原20中拆分)
- 资本配置/股东回报 10 (新增)
- 事件与政策风险 10 (新增)
- 技术确认 5 (不变)
- 风险扣分: 后处理

- [ ] **Step 3: Commit**

---

### Task 17: 改造 summary_agent.py — 9段式报告 + analysis_package 输入

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/agents/summary_agent.py`

- [ ] **Step 1: 在 summary_agent 函数开头构建 analysis_package**

```python
    from src.utils.analysis_package_builder import build_analysis_package
    pkg = build_analysis_package(current_data, current_date)
```

- [ ] **Step 2: 替换 system_prompt 为9段式报告结构**

按大纲 12.2 节要求:
1. `## 核心结论`
2. `## 多维信号总览`
3. `## 关键利多因素`
4. `## 关键利空与反证`
5. `## 事件与催化剂时间线`
6. `## 短线 / 中线 / 长线判断`
7. `## 主要风险与需要继续核验的数据`
8. `## 结论的置信度与适用边界`
9. `## 风险提示`

在 user_prompt 中注入 `pkg.compact_prompt_context` 和各 `*_analysis` 文本。

- [ ] **Step 3: 保持旧文本字段兼容**

继续从 state.data 读取 `*_analysis` 字符串作为 fallback 输入。

- [ ] **Step 4: Commit**

---

### Task 18: 改造 src/main.py — 图结构扩展

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/main.py`

- [ ] **Step 1: 新增 import**

```python
from src.agents.event_analyst_agent import event_analyst_agent
from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent
from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent
from src.utils.analysis_package_builder import build_analysis_package
```

- [ ] **Step 2: 图结构扩展（7并行 → summarizer）**

在 workflow 定义中新增3个节点和边:

```python
# 新增节点
workflow.add_node("event_analyst", event_analyst_agent)
workflow.add_node("quality_risk_analyst", quality_risk_analyst_agent)
workflow.add_node("moneyflow_analyst", moneyflow_analyst_agent)

# 并行边
workflow.add_edge("start_node", "event_analyst")
workflow.add_edge("start_node", "quality_risk_analyst")
workflow.add_edge("start_node", "moneyflow_analyst")

# 汇聚到 summarizer
workflow.add_edge("event_analyst", "summarizer")
workflow.add_edge("quality_risk_analyst", "summarizer")
workflow.add_edge("moneyflow_analyst", "summarizer")
```

- [ ] **Step 3: 更新日志提示**

更新 `print(f"\n{WAIT_ICON} ...")` 部分，增加新 Agent 的提示。

- [ ] **Step 4: Commit**

---

### Task 19: 改造 scoring_engine.py — 图结构扩展

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/stock_pool/scoring_engine.py`

- [ ] **Step 1: 新增 import**

```python
from src.agents.event_analyst_agent import event_analyst_agent
from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent
from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent
```

- [ ] **Step 2: 图结构扩展**

```python
# 新增分析节点
workflow.add_node("event_analyst", event_analyst_agent)
workflow.add_node("quality_risk_analyst", quality_risk_analyst_agent)
workflow.add_node("moneyflow_analyst", moneyflow_analyst_agent)

# 从 start_node 并行扇出
workflow.add_edge("start_node", "event_analyst")
workflow.add_edge("start_node", "quality_risk_analyst")
workflow.add_edge("start_node", "moneyflow_analyst")

# short_term: 新增 event + moneyflow 依赖
workflow.add_edge("event_analyst", "short_term_scorer")
workflow.add_edge("moneyflow_analyst", "short_term_scorer")

# medium_term + long_term: 等待全部7个
for scorer in ["medium_term_scorer", "long_term_scorer"]:
    workflow.add_edge("event_analyst", scorer)
    workflow.add_edge("quality_risk_analyst", scorer)
    workflow.add_edge("moneyflow_analyst", scorer)
```

- [ ] **Step 3: 在 _build_initial_state 中增加 metadata version**

```python
initial_data["analysis_version"] = "a_share_v2"
```

- [ ] **Step 4: Commit**

---

### Task 20: 测试 — analysis_package_builder 与 risk_gate 单元测试

**Files:**
- Create: `Finance/Financial-MCP-Agent/tests/test_analysis_package.py`
- Create: `Finance/Financial-MCP-Agent/tests/test_risk_gate.py`

- [ ] **Step 1: 创建 analysis_package 测试**

```python
"""Test analysis_package_builder: merge, dedup, source priority, compact context"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.analysis_package_builder import (
    build_analysis_package, text_to_signal_pack,
    _parse_signal_pack, _extract_bias_from_text,
    _extract_key_points_from_text, _extract_risk_flags_from_text,
)
from src.utils.analysis_schema import FALLBACK_SIGNAL_PACK


def test_text_to_signal_pack_basic():
    sp = text_to_signal_pack("公司基本面持续改善，ROE稳步提升", "fundamental", "2026-06-14")
    assert sp["agent_name"] == "fundamental"
    assert sp["bias"] in ("bullish", "neutral", "bearish")
    assert sp["confidence"] < 0.5  # derived from text => low confidence


def test_build_package_all_missing():
    """全部agent未执行时仍应返回有效pkg"""
    pkg = build_analysis_package({}, "2026-06-14")
    assert len(pkg.missing_agents) == 7
    assert pkg.global_risk_flags is not None
    assert len(pkg.compact_prompt_context) > 0


def test_build_package_with_existing_analysis():
    """旧agent只产出了文本，应能fallback"""
    data = {
        "fundamental_analysis": "ROE持续提升，毛利率改善，基本面看多",
        "technical_analysis": "均线多头排列，MACD金叉，技术面看多",
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert "fundamental" in pkg.available_agents
    assert "technical" in pkg.available_agents
    assert len(pkg.bullish_signals) >= 0


def test_build_package_with_signal_packs():
    """已有signal_pack时应直接使用"""
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental",
            "bias": "bullish",
            "confidence": 0.8,
            "data_quality_score": 0.9,
            "key_points": ["ROE=18%", "现金流健康"],
            "signals": [
                {"factor": "ROE", "direction": 1, "strength": 80, "source_level": "structured", "time_horizon": ["medium"], "note": "ROE持续>15%"}
            ],
            "risk_flags": [],
            "missing_data": [],
            "source_summary": "Tushare财报",
            "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert "fundamental" in pkg.available_agents
    assert len(pkg.bullish_signals) >= 1
    assert pkg.bullish_signals[0]["factor"] == "ROE"


def test_text_extract_bias():
    assert _extract_bias_from_text("强烈看多，买入增持推荐") == "bullish"
    assert _extract_bias_from_text("利空消息，建议卖出减持") == "bearish"
    assert _extract_bias_from_text("数据平平无明显方向") == "neutral"


def test_text_extract_key_points():
    text = "- 营收增长20%\n- 净利率提升至15%\n结论：基本面稳健"
    points = _extract_key_points_from_text(text)
    assert len(points) > 0


def test_text_extract_risk_flags():
    text = "公司存在质押风险，现金流不匹配问题严重"
    flags = _extract_risk_flags_from_text(text)
    assert "high_pledge_risk" in flags
    assert "cashflow_mismatch" in flags


def test_fallback_never_returns_none():
    """Fallback必须在任何输入下返回有效dict"""
    sp = _parse_signal_pack(None, "test", "2026-01-01")
    assert isinstance(sp, dict)
    assert sp["bias"] in ("bullish", "neutral", "bearish")
    assert "structured_output_missing" in sp.get("risk_flags", [])


def test_conflict_detection():
    """因子方向冲突应被检测"""
    data = {
        "fundamental_signal_pack": {
            "bias": "bullish", "confidence": 0.7, "key_points": ["好"],
            "signals": [
                {"factor": "ROE", "direction": 1, "strength": 80, "source_level": "structured", "note": "ROE高"}
            ],
            "risk_flags": [], "missing_data": [],
        },
        "quality_risk_signal_pack": {
            "bias": "bearish", "confidence": 0.8, "key_points": ["差"],
            "signals": [
                {"factor": "ROE", "direction": -1, "strength": 75, "source_level": "structured", "note": "ROE质量差"}
            ],
            "risk_flags": [], "missing_data": [],
        },
    }
    pkg = build_analysis_package(data, "2026-06-14")
    assert len(pkg.conflicting_signals) > 0
```

- [ ] **Step 2: 创建 risk_gate 测试**

```python
"""Test risk_gate: 风险标签→score cap, abstain, downgrade rules"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.risk_gate import apply_risk_gate, CRITICAL_RISK_FLAGS
from src.utils.analysis_package_builder import build_analysis_package


def test_audit_risk_triggers_cap():
    data = {
        "quality_risk_signal_pack": {
            "agent_name": "quality_risk", "bias": "neutral",
            "confidence": 0.7, "key_points": [], "signals": [],
            "risk_flags": ["audit_risk"], "missing_data": [],
            "source_summary": "", "as_of_date": "2026-06-14",
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    assert result.score_cap == 60
    assert result.action_downgrade == "谨慎"


def test_high_score_without_critical_risk():
    pkg = build_analysis_package({}, "2026-06-14")
    result = apply_risk_gate(pkg, "medium", 85)
    # 没有关键风险时不应cap
    assert result.score_cap is None
    assert result.risk_level == "high"  # data_quality low due to all missing


def test_abstain_when_many_agents_missing():
    """缺失agent过多时应abstain"""
    # 模拟只有1个agent有数据
    data = {
        "fundamental_signal_pack": {
            "agent_name": "fundamental", "bias": "bullish",
            "confidence": 0.7, "key_points": ["数据有限"], "signals": [],
            "risk_flags": [], "missing_data": [],
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "long", 80)
    assert result.abstain is True


def test_news_only_narrative_not_strong_recommend_long():
    """长线仅靠新闻叙事不能强推荐"""
    data = {
        "news_signal_pack": {
            "agent_name": "news", "bias": "bullish",
            "confidence": 0.6, "key_points": ["媒体看多"], "signals": [
                {"factor": "媒体情绪", "direction": 1, "strength": 70, "source_level": "news", "note": ""}
            ],
            "risk_flags": [], "missing_data": [],
        }
    }
    pkg = build_analysis_package(data, "2026-06-14")
    result = apply_risk_gate(pkg, "long", 90)
    assert result.score_cap is not None
    assert result.score_cap <= 55
```

- [ ] **Step 3: 运行测试并确认通过**

```bash
python -m pytest tests/test_analysis_package.py tests/test_risk_gate.py -v
```

- [ ] **Step 4: Commit**

---

### Task 21: 运行现有pytest套件验证无回归

**Files:**
- Verify: `Finance/Financial-MCP-Agent/tests/`

- [ ] **Step 1: 运行全部现有测试**

```bash
cd Finance/Financial-MCP-Agent && python -m pytest tests/ -v
```

检查是否所有测试通过（或之前已存在的失败未被修复的是否仍一致）。

- [ ] **Step 2: 如有回归，修复后重新运行**

---

### Task 22: Smoke Test — 验证CLI和API正常

**Files:**
- Test: CLI ``python -m src.main --command "分析嘉友国际"``

- [ ] **Step 1: 确认单票分析流程跑通**

启动分析，确认7个agent全部执行，报告以9段式输出。

- [ ] **Step 2: 确认股票池评分流程跑通**

```bash
python -m src.main_pool score 603871
```

确认 short/medium/long 三个评分均有 risk_gate 信息。

- [ ] **Step 3: 确认API接口正常**

确认 `/api/report` 和 `/api/score-all/{stock_code}` 至少能完成一次真实任务。
```
