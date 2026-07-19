"""
Analysis Package Builder: 将多个 *_signal_pack 合并为统一 AnalysisPackage。

职责:
  1. 标准化 signal_pack 输入 (_parse_signal_pack)
  2. 从纯文本推断 signal_pack (text_to_signal_pack, 用于旧agent兼容)
  3. 合并所有 signal pack 为 AnalysisPackage (build_analysis_package)
  4. 生成 compact_prompt_context 供 scorer/summarizer 消费
"""
import re
from typing import Dict, Any, List

from src.utils.analysis_schema import (
    SourceLevel, SOURCE_PRIORITY,
    FALLBACK_SIGNAL_PACK,
    AnalysisPackage,
)


# ── 文本→结构化 提取工具 ────────────────────────────

def _extract_bias_from_text(text: str) -> str:
    if not text:
        return "neutral"
    t = text[:3000]
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
    return list(dict.fromkeys(flags))


# ── 信号字段标准化映射 ──────────────────────────────

_DIRECTION_MAP = {
    "positive": 1, "bullish": 1, "long": 1, "buy": 1,
    "看多": 1, "利多": 1, "正向": 1, "积极": 1, "偏多": 1,
    "negative": -1, "bearish": -1, "short": -1, "sell": -1,
    "看空": -1, "利空": -1, "负向": -1, "消极": -1, "偏空": -1,
    "neutral": 0, "中性": 0, "持平": 0, "观望": 0,
}

_STRENGTH_MAP = {
    "strong": 8, "强": 8, "very_strong": 9, "极强": 9,
    "medium": 5, "中": 5, "moderate": 5, "一般": 5,
    "weak": 3, "弱": 3, "mild": 3,
    "none": 0, "无": 0,
}


def normalize_signal_pack(sp: Dict[str, Any]) -> Dict[str, Any]:
    """标准化 signal_pack 中的字段类型和值映射。

    LLM 输出不规范时，direction 可能是中文/英文字符串（如"利多"/"short"），
    strength 可能是"强"/"medium"等。本函数统一转换为数值类型。
    """
    bias_raw = sp.get("bias", "")
    if isinstance(bias_raw, str) and bias_raw not in ("bullish", "bearish", "neutral"):
        mapped = None
        if bias_raw in ("看多", "利多", "偏多", "积极向好", "positive"):
            mapped = "bullish"
        elif bias_raw in ("看空", "利空", "偏空", "消极", "negative"):
            mapped = "bearish"
        if mapped:
            sp["bias"] = mapped

    for sig in sp.get("signals", []):
        if not isinstance(sig, dict):
            continue

        d = sig.get("direction", 0)
        if isinstance(d, str):
            sig["direction"] = _DIRECTION_MAP.get(d.strip(), 0)
        try:
            sig["direction"] = int(sig.get("direction", 0))
        except (ValueError, TypeError):
            sig["direction"] = 0

        s = sig.get("strength", 5)
        if isinstance(s, str):
            s_stripped = s.strip()
            if s_stripped in _STRENGTH_MAP:
                sig["strength"] = _STRENGTH_MAP[s_stripped]
            else:
                try:
                    sig["strength"] = int(s_stripped)
                except (ValueError, TypeError):
                    sig["strength"] = 5
        try:
            sig["strength"] = int(sig.get("strength", 5))
        except (ValueError, TypeError):
            sig["strength"] = 5

        try:
            sig["confidence"] = float(sig.get("confidence", 0.5))
        except (ValueError, TypeError):
            sig["confidence"] = 0.5

    return sp


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
        # Normalize numeric fields — LLM may output strings
        try:
            sp["confidence"] = float(sp.get("confidence", 0.5))
        except (ValueError, TypeError):
            sp["confidence"] = 0.5
        try:
            sp["data_quality_score"] = float(sp.get("data_quality_score", 0.5))
        except (ValueError, TypeError):
            sp["data_quality_score"] = 0.5
        return normalize_signal_pack(sp)

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


def _is_explicit_agent_failure(value: Any) -> bool:
    """Recognize explicit execution failures before compatibility fallback.

    Failure messages are not analysis evidence.  Treating them as low-confidence
    neutral text makes a broken agent look available to the risk gate.
    """
    if isinstance(value, dict):
        status = str(value.get("status", "")).strip().lower()
        validity = str(value.get("validity", "")).strip().lower()
        if value.get("error") or value.get("_error"):
            return True
        if status in {"error", "failed", "timeout", "invalid"}:
            return True
        if validity in {"invalid", "error", "failed"}:
            return True
        if value.get("_scorer_failed") is True:
            return True
        value = value.get("analysis_text", "")

    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    failure_markers = (
        "agent执行失败",
        "agent 执行失败",
        "分析失败:",
        "分析失败：",
        "数据获取失败",
        "调用失败",
        "请求超时",
        "pipeline failed",
        "agent failed",
        "agent execution failure",
        "agent execution failed",
    )
    return any(marker in normalized for marker in failure_markers)


# ── 合并引擎 ──────────────────────────────────────────

def build_analysis_package(
    state_data: Dict[str, Any],
    as_of_date: str = "",
) -> AnalysisPackage:
    """
    从 state.data 提取所有 *_signal_pack（或 fallback from *_analysis），
    合并为统一 AnalysisPackage。
    """
    try:
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
            raw_signal_pack = state_data.get(sp_key)
            raw_text = state_data.get(text_key)
            if _is_explicit_agent_failure(raw_signal_pack) or _is_explicit_agent_failure(raw_text):
                sp = dict(FALLBACK_SIGNAL_PACK, agent_name=agent_name, as_of_date=as_of_date)
                sp["validity"] = "invalid"
                sp["status"] = "failed"
                sp["confidence"] = 0.0
                sp["data_quality_score"] = 0.0
                sp["missing_data"] = ["Agent执行失败"]
                sp["source_summary"] = "agent execution failure"
                sp["error_type"] = "agent_execution_failure"
                signal_packs[agent_name] = sp
            elif raw_signal_pack:
                signal_packs[agent_name] = _parse_signal_pack(state_data[sp_key], agent_name, as_of_date)
            elif raw_text:
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
            has_fatal_missing = any(
                marker in str(x) for x in md
                for marker in ("Agent未执行", "Agent执行失败")
            )
            dqs = sp.get("data_quality_score", 0)
            try:
                dqs = float(dqs)
            except (ValueError, TypeError):
                dqs = 0.3
            if has_fatal_missing and dqs <= 0.3:
                missing.append(agent_name)
            else:
                available.append(agent_name)

            all_risk_flags.extend(sp.get("risk_flags", []))
            all_missing.extend(md)

            for sig in sp.get("signals", []):
                if isinstance(sig, dict):
                    sig["_agent"] = agent_name
                    sig.setdefault("source_level", SourceLevel.PROXY)
                    # Normalize numeric fields — LLM may output strings for int/float values
                    try:
                        sig["strength"] = int(sig.get("strength", 50))
                    except (ValueError, TypeError):
                        sig["strength"] = 50
                    try:
                        sig["direction"] = int(sig.get("direction", 0))
                    except (ValueError, TypeError):
                        sig["direction"] = 0
                    try:
                        sig["confidence"] = float(sig.get("confidence", 0.5))
                    except (ValueError, TypeError):
                        sig["confidence"] = 0.5
                    all_signals.append(sig)

        # 按 source priority + strength 排降序
        all_signals.sort(
            key=lambda s: (
                SOURCE_PRIORITY.get(s.get("source_level", SourceLevel.PROXY), 0),
                abs(int(s.get("strength", 0)) if not isinstance(s.get("strength"), int) else s.get("strength", 0)),
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
    except Exception:
        return AnalysisPackage(
            as_of_date=as_of_date,
            executed_agents=[],
            available_agents=[],
            missing_agents=["fundamental","technical","value","news","event","quality_risk","moneyflow"],
            global_risk_flags=["builder_error"],
            global_missing_data=["AnalysisPackage builder encountered an error"],
            bullish_signals=[],
            bearish_signals=[],
            conflicting_signals=[],
            source_priority_summary={"error": "builder failed"},
            compact_prompt_context="## 分析产物构建失败\n分析数据合并过程中出现错误，请检查原始分析输出。",
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

    lines.append("## 各Agent结论摘要")
    bias_map = {
        "bullish": "看多", "neutral": "中性", "bearish": "看空",
        "看多": "看多", "看空": "看空", "中性": "中性",
        "偏多": "看多", "偏空": "看空", "利多": "看多", "利空": "看空",
    }
    for agent_name, sp in signal_packs.items():
        bias_cn = bias_map.get(sp.get("bias", ""), "中性")
        try:
            conf = float(sp.get('confidence', 0))
        except (ValueError, TypeError):
            conf = 0.0
        try:
            dqs = float(sp.get('data_quality_score', 0.5))
        except (ValueError, TypeError):
            dqs = 0.5
        sig_count = len([s for s in sp.get("signals", []) if isinstance(s, dict)])
        lines.append(f"- **{agent_name}**: {bias_cn} (置信度={conf:.0%}, 数据质量={dqs:.0%}, {sig_count}条信号)")
        for kp in sp.get("key_points", [])[:3]:
            lines.append(f"  - {kp}")
    lines.append("")

    lines.append("## 关键看多信号")
    for s in bullish[:5]:
        lines.append(f"- [{s.get('_agent', '?')}] {s.get('factor', '?')}: strength={s.get('strength', 0)}, src={s.get('source_level', '?')}")
    if not bullish:
        lines.append("- (无)")
    lines.append("")

    lines.append("## 关键看空信号")
    for s in bearish[:5]:
        lines.append(f"- [{s.get('_agent', '?')}] {s.get('factor', '?')}: strength={s.get('strength', 0)}, src={s.get('source_level', '?')}")
    if not bearish:
        lines.append("- (无)")
    lines.append("")

    if conflicting:
        lines.append("## 信号冲突")
        for c in conflicting[:5]:
            lines.append(f"- 因子 **{c['factor']}**: {len(c['signals'])}个信号方向不一致")
        lines.append("")

    if risk_flags:
        lines.append(f"## 全局风险标签: {', '.join(risk_flags)}")
        lines.append("")

    if missing_data:
        lines.append(f"## 缺失数据: {', '.join(missing_data[:8])}")
        lines.append("")

    return "\n".join(lines)
