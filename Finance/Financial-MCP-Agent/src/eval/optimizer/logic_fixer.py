"""
逻辑修复器 — 检测并修复代码逻辑问题（总纲 §11.5）。

异常检测规则（基于 loss + contribution 数据的统计分析）：
  1. **单Agent极端负贡献**：某个 agent 的 delta_L_total 显著正偏离（>2σ 或 >0.05）
     → 可能该 agent 的评分逻辑与真实表现反向
  2. **相关Agent贡献背离**：依赖链上的 agent 贡献方向不一致
     （如 technical 正贡献但 short_term_scorer 负贡献）
     → 可能 scorer 的融合权重/逻辑有误
  3. **Loss维度异常集中**：某个 loss 子维度异常高（>0.7 且占比 >50%）
     → 可能该维度的计算或阈值有误
  4. **不稳定性检测**：同一 agent 的贡献在不同批次间剧烈波动（CV > 2.0）
     → 可能 agent 逻辑对输入质量过度敏感

规则引擎（RuleEngine）：在调用 LLM 之前先进行纯规则检测。
如果问题能 100% 被已知规则覆盖，直接应用确定性修复，跳过 LLM。
否则回退到 LLM-based/generate_fix() 模板生成。

所有检测均为纯 Python 实现，不依赖 LLM。
generate_fix() 和 generate_test() 生成代码模板（用户需手动验证）。
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


# ── 规则引擎 ──────────────────────────────────────────────────────

class RuleEngine:
    """纯规则检测引擎 — 在涉及 LLM 之前先判断问题是否可通过已知规则修复。

    提供4类检测器：
      1. 评分计算Bug检测 (check_score_calculation_bug)
      2. 权重归一化检查 (check_weight_normalization)
      3. 阈值一致性检查 (check_threshold_consistency)
      4. 数据字段名匹配检查 (check_data_field_mismatch)
    """

    # 已知的评分计算Bug模式
    SCORE_BUG_PATTERNS = [
        {
            "name": "score_range_inversion",
            "pattern": r"(?:score|scoring)\s*[=:]\s*(?:100\s*-\s*|1\.0\s*-\s*)",
            "description": "评分可能使用了反向映射（100-score 或 1.0-score），导致高分对应差表现",
            "fix_hint": "检查评分赋值的正负方向，确保高分=好表现、低分=差表现",
        },
        {
            "name": "weight_out_of_bounds",
            "pattern": r"(?:weight|权重)\s*[=:]\s*[-]?\d+\.\d+\s*[*/]",
            "description": "权重计算中可能出现负值或 >10 的极端值",
            "fix_hint": "检查权重计算的取值范围，确保在 [0, 1] 区间",
        },
        {
            "name": "missing_normalization",
            "pattern": r"(?:weight|w_)\w*\s*=\s*\w+\s*/\s*\w+",
            "description": "权重有除法但缺少显式归一化步骤",
            "fix_hint": "添加归一化步骤：weight_i = weight_i / sum(all_weights)",
        },
        {
            "name": "integer_division",
            "pattern": r"(?:score|count|total)\s*/\s*(?:2|3|5|10)\b",
            "description": "整数除法可能导致精度丢失",
            "fix_hint": "改用浮点除法：value / 2.0 或 from __future__ import division",
        },
    ]

    @staticmethod
    def check_score_calculation_bug(error_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """检测评分计算中的常见 Bug 模式。

        Args:
            error_pattern: 包含 code_snippet / error_message / agent_name 的字典

        Returns:
            {"matched": bool, "pattern_name": str, "fix_hint": str, "confidence": float}
        """
        code = error_pattern.get("code_snippet", "")
        error_msg = error_pattern.get("error_message", "")

        # 合并为搜索文本
        search_text = f"{code}\n{error_msg}"

        for pattern_info in RuleEngine.SCORE_BUG_PATTERNS:
            if re.search(pattern_info["pattern"], search_text, re.IGNORECASE):
                return {
                    "matched": True,
                    "pattern_name": pattern_info["name"],
                    "description": pattern_info["description"],
                    "fix_hint": pattern_info["fix_hint"],
                    "confidence": 0.90 if code else 0.60,
                }

        # 检查是否 score 输出范围异常
        if re.search(r"(?:score|Score)\s*[><=]+\s*(?:-?\d{3,}|1\d{3,})", search_text):
            return {
                "matched": True,
                "pattern_name": "score_range_anomaly",
                "description": "评分值出现超出预期范围（如 >999 或负数），可能是溢出或未裁剪",
                "fix_hint": "添加 score = max(0, min(100, score)) 裁剪逻辑",
                "confidence": 0.85,
            }

        return {"matched": False, "pattern_name": None, "fix_hint": None, "confidence": 0.0}

    @staticmethod
    def check_weight_normalization(params: Dict[str, Any]) -> Dict[str, Any]:
        """检查权重是否归一化（Σ=1.0）。

        Args:
            params: 包含多个参数及其值的字典，其中 weight 相关参数以 'w_' 或 '_weight_' 结尾

        Returns:
            {"all_normalized": bool, "violations": [...], "fix_suggestion": str}
        """
        violations = []

        # 按组检测：查找以权重前缀归组的参数
        # 例如：short_weight_tech, short_weight_volume → "short_weight" 组
        weight_groups: Dict[str, Dict[str, float]] = {}
        for key, value in params.items():
            if not isinstance(value, (int, float)):
                continue
            # 匹配 _w_xxx 或 _weight_xxx 模式的分组
            match = re.match(r"^(.+?)(?:_w_|_weight_)(.+)$", key)
            if match:
                group_key = match.group(1) + "_w"  # 如 "loss_w", "effect_short_w"
                weight_name = match.group(2)
                if group_key not in weight_groups:
                    weight_groups[group_key] = {}
                weight_groups[group_key][weight_name] = float(value)

        for group_key, weights in weight_groups.items():
            if len(weights) < 2:
                continue
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                violations.append({
                    "group": group_key,
                    "total": total,
                    "drift": total - 1.0,
                    "weights": dict(weights),
                })

        if violations:
            return {
                "all_normalized": False,
                "violations": violations,
                "fix_suggestion": (
                    f"以下权重组的 Σ != 1.0: "
                    + ", ".join(f"{v['group']}(Σ={v['total']:.3f})" for v in violations)
                    + "。请对每组执行 weights_i = weights_i / sum(weights) 归一化。"
                ),
            }

        return {"all_normalized": True, "violations": [], "fix_suggestion": None}

    @staticmethod
    def check_threshold_consistency(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查阈值列表中是否存在逻辑间隙或重叠。

        常见问题：
          - buy_threshold > sell_threshold（买卖阈值颠倒）
          - 相邻阈值之间有未覆盖的区间（gap）
          - 阈值范围超出 [0, 100] 或 [0.0, 1.0]

        Args:
            rules: 规则列表，每项 {"name": str, "threshold": float/int, "type": "buy"/"sell"/"cap"}

        Returns:
            {"consistent": bool, "gaps": [...], "overlaps": [...], "fix_suggestions": [...]}
        """
        gaps = []
        overlaps = []
        fix_suggestions = []

        # 分离购买和卖出阈值
        buy_rules = [r for r in rules if r.get("type") == "buy"]
        sell_rules = [r for r in rules if r.get("type") == "sell"]
        cap_rules = [r for r in rules if r.get("type") == "cap"]

        # 检查: buy_threshold 不应高于 sell_threshold（否则永远不买）
        for buy in buy_rules:
            for sell in sell_rules:
                if buy.get("threshold", 0) > sell.get("threshold", 100):
                    overlaps.append({
                        "type": "buy_above_sell",
                        "buy_rule": buy.get("name"),
                        "buy_val": buy.get("threshold"),
                        "sell_rule": sell.get("name"),
                        "sell_val": sell.get("threshold"),
                        "message": (
                            f"购买阈值 ({buy.get('threshold')}) > 卖出阈值 ({sell.get('threshold')})，"
                            f"可能导致永远不买入"
                        ),
                    })
                    fix_suggestions.append(
                        f"确保 buy_threshold <= sell_threshold: "
                        f"将 {buy.get('name')} 从 {buy.get('threshold')} 降低到 ≤ {sell.get('threshold')}，"
                        f"或将 {sell.get('name')} 提高到 ≥ {buy.get('threshold')}"
                    )

        # 检查：cap 值是否合理
        for cap in cap_rules:
            threshold = cap.get("threshold", 50)
            cap_name = cap.get("name", "unknown")
            if threshold > 100 or threshold < 0:
                gaps.append({
                    "type": "cap_out_of_bounds",
                    "rule": cap_name,
                    "value": threshold,
                    "message": f"上限值 {threshold} 超出合理范围 [0, 100]",
                })
                fix_suggestions.append(f"将 {cap_name} 裁剪到 [0, 100] 范围")

        # 检查：相邻 cap 阈值是否分层合理（不允许相邻 cap 差 < 3）
        sorted_caps = sorted(cap_rules, key=lambda r: r.get("threshold", 50))
        for i in range(len(sorted_caps) - 1):
            curr = sorted_caps[i].get("threshold", 50)
            nxt = sorted_caps[i + 1].get("threshold", 50)
            if abs(curr - nxt) < 3 and abs(curr - nxt) > 0:
                overlaps.append({
                    "type": "caps_too_close",
                    "rule_a": sorted_caps[i].get("name"),
                    "rule_b": sorted_caps[i + 1].get("name"),
                    "value_a": curr,
                    "value_b": nxt,
                    "message": f"上限值 {curr} 和 {nxt} 过于接近（差={abs(curr - nxt)}）",
                })

        consistent = len(gaps) == 0 and len(overlaps) == 0

        return {
            "consistent": consistent,
            "gaps": gaps,
            "overlaps": overlaps,
            "fix_suggestions": fix_suggestions,
        }

    @staticmethod
    def check_data_field_mismatch(
        code_refs: List[str], schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """检查代码引用的字段名是否与数据schema一致。

        检测模式：
          - 字段名拼写错误（如 "clsoe" vs "close"）
          - 废弃字段名（如 "pct_chg" vs "pct_change"）
          - 已迁移/重命名的字段

        Args:
            code_refs: 代码中引用的字段名列表
            schema: 已知的数据schema键集合（可选）

        Returns:
            {"mismatches": [...], "suggestions": [...]}
        """
        mismatches = []
        suggestions = []

        # 常见拼写错误映射
        common_typos = {
            "clsoe": "close",
            "volme": "volume",
            "amout": "amount",
            "pct_chg": "pct_change",
            "turnover_r": "turnover_rate",
            "pe_ttm": "pe_ttm",
            "pre_close": "prev_close",
            "change_pct": "pct_chg",
            "high_price": "high",
            "low_price": "low",
            "open_price": "open",
            "close_price": "close",
            "daily_amout": "amount",
            "avg_price": "vwap",
            "total_shares": "total_share",
            "circ_market_cap": "circ_mv",
            "total_market_cap": "total_mv",
        }

        # 已知废弃/重命名字段
        deprecated_fields = {
            "vol": "改用 volume",
            "pct_chg": "改用 pct_change（更明确）",
            "pe": "改用 pe_ttm 或 pe_lyr",
            "pb": "改用 pb_lf",
            "total_mv": "total_market_cap",
            "circ_mv": "circ_market_cap",
        }

        for ref in code_refs:
            ref_lower = ref.lower().strip()
            # 检查拼写错误
            if ref_lower in common_typos:
                correct = common_typos[ref_lower]
                mismatches.append({
                    "field": ref,
                    "issue": "likely_typo",
                    "suggested_fix": correct,
                    "message": f"字段 '{ref}' 可能是拼写错误，建议改为 '{correct}'",
                })
                suggestions.append(f"将 '{ref}' 改为 '{correct}'")

            # 检查废弃字段
            if ref_lower in deprecated_fields:
                if not any(m["field"] == ref for m in mismatches):
                    mismatches.append({
                        "field": ref,
                        "issue": "deprecated",
                        "suggestion": deprecated_fields[ref_lower],
                        "message": f"字段 '{ref}' 已废弃，{deprecated_fields[ref_lower]}",
                    })
                    suggestions.append(f"更新 '{ref}' 引用为 {deprecated_fields[ref_lower]}")

            # 如果提供了 schema，检查字段是否存在
            if schema and ref_lower not in schema and ref not in schema:
                # 尝试找最接近的字段名
                closest = RuleEngine._find_closest_field(ref_lower, schema)
                if closest:
                    mismatches.append({
                        "field": ref,
                        "issue": "not_in_schema",
                        "closest_match": closest,
                        "message": f"字段 '{ref}' 不在 schema 中，最接近的是 '{closest}'",
                    })
                    suggestions.append(f"将 '{ref}' 改为 '{closest}'")

        return {
            "mismatches": mismatches,
            "suggestions": suggestions,
            "has_issues": len(mismatches) > 0,
        }

    @staticmethod
    def _find_closest_field(field: str, schema: Dict[str, Any]) -> Optional[str]:
        """在 schema 中找最接近的字段名（基于 Levenshtein 距离）。"""
        candidates = []
        field_lower = field.lower()
        for key in schema.keys():
            key_lower = key.lower()
            if field_lower == key_lower:
                return key
            # 简单相似度检查：开头相同 或 包含关系
            if key_lower.startswith(field_lower[:3]) or field_lower.startswith(key_lower[:3]):
                candidates.append(key)
        return candidates[0] if len(candidates) == 1 else (candidates[0] if candidates else None)


class LogicFixer:
    """逻辑修复器 — 检测并修复代码逻辑问题。

    输入：loss 数据 + agent 贡献数据
    输出：异常列表 + 修复建议 + 测试代码
    """

    # 阈值常量
    DELTA_SIGNIFICANCE_THRESHOLD = 0.03       # delta_L_total 显著性阈值
    DELTA_SIGMA_MULTIPLIER = 2.0              # 标准差倍数阈值
    LOSS_DIM_HIGH_THRESHOLD = 0.7             # 单维度 loss 过高阈值
    LOSS_DIM_RATIO_THRESHOLD = 0.50           # 单维度占总 loss 比例过高阈值
    CV_INSTABILITY_THRESHOLD = 2.0            # 变异系数不稳定性阈值

    def __init__(self):
        pass

    # ── Public API ────────────────────────────────────────────────

    def analyze_anomalies(
        self,
        loss_data: Dict[str, Any],
        contribution_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """分析 loss 和贡献数据，检测异常模式。

        Args:
            loss_data: 来自 LossEngine 的 loss 分解结果，格式：
                {
                    "L_total": float,
                    "L_effect": float,
                    "return_detail": {"L_return": ..., "L_rank_ic": ..., ...},
                    "risk_detail": {"L_risk": ..., ...},
                    "structure_detail": {"L_structure": ..., ...},
                    "stability_detail": {"L_stability": ..., ...},
                    "efficiency_detail": {"L_efficiency": ..., ...},
                }
            contribution_data: agent 贡献数据，格式：
                {
                    "contributions": [
                        {"agent_name": str, "delta_L_total": float, ...},
                        ...
                    ],
                    "batch_id": str (可选)
                }

        Returns:
            异常列表，每项:
            {
                "anomaly_type": "extreme_negative_contribution" | "contribution_divergence"
                              | "loss_dimension_concentration" | "instability",
                "severity": "high" | "medium" | "low",
                "description": str,
                "affected_agent": str 或 null,
                "evidence": {...},  # 具体数据证据
                "confidence": float,  # 0-1
            }
        """
        anomalies: List[Dict[str, Any]] = []

        contributions = contribution_data.get("contributions", [])
        if not contributions:
            return anomalies

        # 1. 极端负贡献检测
        anomalies.extend(self._detect_extreme_negative(contributions))

        # 2. 贡献背离检测
        anomalies.extend(self._detect_divergence(contributions))

        # 3. Loss 维度异常集中检测
        anomalies.extend(self._detect_loss_concentration(loss_data))

        # 4. 不稳定性检测（如果有历史数据）
        if "history" in contribution_data:
            anomalies.extend(
                self._detect_instability(contribution_data["history"])
            )

        return anomalies

    # ── 异常检测子方法 ─────────────────────────────────────────────

    def _detect_extreme_negative(
        self, contributions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """检测极端负贡献的 agent。

        规则：delta_L_total > 0 表示 agent 增加了 loss（负贡献）。
        如果某个 agent 的 delta 超过 2σ 或超过 0.05 绝对值 → 标记。
        """
        anomalies = []
        deltas = [c.get("delta_L_total", 0.0) for c in contributions]
        n = len(deltas)
        if n < 2:
            return anomalies

        mean_delta = sum(deltas) / n
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
        sigma = math.sqrt(max(variance, 1e-10))

        for c in contributions:
            d = c.get("delta_L_total", 0.0)
            # 只关注负贡献（delta > 0 = agent 增加 loss）
            if d <= self.DELTA_SIGNIFICANCE_THRESHOLD:
                continue

            agent = c.get("agent_name", "unknown")
            z_score = (d - mean_delta) / max(sigma, 1e-10)

            if z_score > self.DELTA_SIGMA_MULTIPLIER or d > 0.05:
                severity = "high" if d > 0.08 else "medium"
                anomalies.append({
                    "anomaly_type": "extreme_negative_contribution",
                    "severity": severity,
                    "description": (
                        f"Agent '{agent}' 显著增加总 Loss "
                        f"(delta={d:.4f}, z-score={z_score:.2f})，"
                        f"可能评分逻辑与真实表现反向"
                    ),
                    "affected_agent": agent,
                    "evidence": {
                        "delta_L_total": d,
                        "z_score": z_score,
                        "mean_delta": mean_delta,
                        "sigma": sigma,
                    },
                    "confidence": min(1.0, max(0.3, z_score / 4.0)),
                })

        return anomalies

    def _detect_divergence(
        self, contributions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """检测相关 agent 之间贡献方向不一致。

        检查 scorer agent 与其依赖的分析 agent 之间的贡献方向是否背离。
        例如：如果 technical_agent 正贡献但 short_term_scorer 负贡献，这可能表明
        scorer 的融合逻辑有问题。
        """
        anomalies = []

        # 建立贡献映射
        contrib_map = {c.get("agent_name", ""): c.get("delta_L_total", 0.0)
                       for c in contributions}

        # 已知的 scorer → 依赖 agent 映射
        scorer_dependency_map = {
            "short_term_scorer": ["technical_analysis", "news_analysis",
                                  "event_analysis", "moneyflow_analysis"],
            "medium_term_scorer": ["fundamental_analysis", "value_analysis",
                                   "quality_risk_analysis", "event_analysis",
                                   "technical_analysis", "news_analysis",
                                   "moneyflow_analysis"],
            "long_term_scorer": ["fundamental_analysis", "value_analysis",
                                 "quality_risk_analysis", "event_analysis",
                                 "technical_analysis", "news_analysis",
                                 "moneyflow_analysis"],
        }

        for scorer, deps in scorer_dependency_map.items():
            scorer_delta = contrib_map.get(scorer, 0.0)
            if abs(scorer_delta) < 0.005:
                continue

            # 检查 scorer 的贡献方向是否与多数依赖 agent 相反
            dep_deltas = [
                contrib_map.get(d, 0.0) for d in deps
                if d in contrib_map
            ]
            if len(dep_deltas) < 2:
                continue

            # 依赖 agent 的平均方向
            dep_mean = sum(dep_deltas) / len(dep_deltas)
            # dep_delta < 0 意味着依赖 agent 正贡献
            if dep_mean < -0.01 and scorer_delta > 0.01:
                # 依赖 agents 正贡献但 scorer 负贡献 → 背离
                anomalies.append({
                    "anomaly_type": "contribution_divergence",
                    "severity": "high",
                    "description": (
                        f"依赖 agent 平均正贡献 (mean_delta={dep_mean:.4f}) "
                        f"但 scorer '{scorer}' 负贡献 (delta={scorer_delta:.4f})，"
                        f"可能 scorer 融合权重或逻辑有误"
                    ),
                    "affected_agent": scorer,
                    "evidence": {
                        "scorer_delta": scorer_delta,
                        "dep_mean_delta": dep_mean,
                        "dep_deltas": {d: contrib_map.get(d) for d in deps if d in contrib_map},
                    },
                    "confidence": 0.75,
                })

            # Also check: dep agents optimistic (positive delta), scorer pessimistic (negative delta)
            if dep_mean > 0.01 and scorer_delta < -0.01:
                anomalies.append({
                    "anomaly_type": "contribution_divergence",
                    "severity": "high",
                    "description": (
                        f"依赖 agent 平均负贡献 (mean_delta={dep_mean:.4f}) "
                        f"但 scorer '{scorer}' 正贡献 (delta={scorer_delta:.4f})，"
                        f"可能 scorer 融合权重或逻辑有误"
                    ),
                    "affected_agent": scorer,
                    "evidence": {
                        "scorer_delta": scorer_delta,
                        "dep_mean_delta": dep_mean,
                        "dep_deltas": {d: contrib_map.get(d) for d in deps if d in contrib_map},
                    },
                    "confidence": 0.75,
                })

        return anomalies

    def _detect_loss_concentration(
        self, loss_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """检测 Loss 是否过度集中在某个子维度。

        如果某个子维度 loss > 0.7 且占总 loss 比例 > 50% → 标记。
        """
        anomalies = []
        L_total = loss_data.get("L_total", 0.0)
        if L_total < 0.05:
            return anomalies  # total loss 太小，不分析

        # 需要检查的子维度及其（路径，标签）
        dimension_paths = [
            ("return_detail", "L_return", "收益端"),
            ("return_detail", "L_rank_ic", "排序IC"),
            ("return_detail", "L_direction", "方向准确率"),
            ("return_detail", "L_calibration", "校准误差"),
            ("return_detail", "L_extreme", "极端判断"),
            ("return_detail", "L_excess", "超额收益"),
            ("risk_detail", "L_risk", "风险端"),
            ("risk_detail", "dd_penalty", "回撤惩罚"),
            ("risk_detail", "vol_penalty", "波动率惩罚"),
            ("risk_detail", "downside_penalty", "下行偏差"),
            ("risk_detail", "consec_penalty", "连续亏损"),
            ("structure_detail", "L_structure", "结构端"),
            ("structure_detail", "concentration_penalty", "集中度"),
            ("structure_detail", "turnover_penalty", "换手率"),
            ("structure_detail", "cash_penalty", "现金拖累"),
            ("structure_detail", "sector_penalty", "行业集中"),
            ("stability_detail", "L_stability", "稳定性端"),
            ("efficiency_detail", "L_efficiency", "效率端"),
        ]

        for parent_key, child_key, label in dimension_paths:
            parent = loss_data.get(parent_key, {})
            if not isinstance(parent, dict):
                continue
            val = parent.get(child_key, 0.0)
            if val > self.LOSS_DIM_HIGH_THRESHOLD and val > L_total * self.LOSS_DIM_RATIO_THRESHOLD:
                severity = "high" if val > 0.85 else "medium"
                anomalies.append({
                    "anomaly_type": "loss_dimension_concentration",
                    "severity": severity,
                    "description": (
                        f"维度 '{label}' 的 Loss ({val:.3f}) 异常高，"
                        f"占总 Loss ({L_total:.3f}) 的 {val/L_total*100:.0f}%"
                    ),
                    "affected_agent": None,
                    "evidence": {
                        "dimension": label,
                        "dimension_loss": val,
                        "total_loss": L_total,
                        "ratio": val / L_total,
                    },
                    "confidence": min(1.0, (val / L_total) / 1.5),
                })

        return anomalies

    def _detect_instability(
        self, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """检测 agent 贡献在不同批次间的剧烈波动。

        history 格式：[{"batch_id": ..., "contributions": [...]}, ...]
        """
        anomalies = []
        if len(history) < 3:
            return anomalies

        # 收集每个 agent 在各批次中的 delta_L_total
        agent_history: Dict[str, List[float]] = {}
        for batch in history:
            contributions = batch.get("contributions", [])
            for c in contributions:
                agent = c.get("agent_name", "unknown")
                d = c.get("delta_L_total", 0.0)
                if agent not in agent_history:
                    agent_history[agent] = []
                agent_history[agent].append(d)

        for agent, deltas in agent_history.items():
            if len(deltas) < 3:
                continue
            mean_d = sum(deltas) / len(deltas)
            # 使用总体均值（包括接近0的值）来计算标准差，避免除以接近0的均值
            variance = sum((d - mean_d) ** 2 for d in deltas) / max(len(deltas) - 1, 1)
            sigma = math.sqrt(max(variance, 1e-10))

            # 使用 sigma/|mean| 作为 CV 近似（mean 可能接近 0）
            # 当 mean 接近 0 时，使用 sigma 与阈值比较
            if abs(mean_d) > 0.005:
                cv = sigma / abs(mean_d)
            else:
                cv = sigma / 0.01  # 归一化到 1% level

            if cv > self.CV_INSTABILITY_THRESHOLD:
                anomalies.append({
                    "anomaly_type": "instability",
                    "severity": "medium",
                    "description": (
                        f"Agent '{agent}' 的贡献在不同批次间剧烈波动 "
                        f"(CV={cv:.2f}, mean={mean_d:.4f}, sigma={sigma:.4f})，"
                        f"可能对输入数据质量过度敏感"
                    ),
                    "affected_agent": agent,
                    "evidence": {
                        "agent": agent,
                        "cv": cv,
                        "mean": mean_d,
                        "sigma": sigma,
                        "n_batches": len(deltas),
                        "deltas": deltas,
                    },
                    "confidence": min(1.0, cv / 4.0),
                })

        return anomalies

    # ── 规则驱动修复 ──────────────────────────────────────────────

    def can_fix_by_rules(self, diagnosis: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """检查问题是否可以 100% 通过纯规则修复，无需 LLM。

        根据 diagnosis 中的信息尝试匹配已知规则模式。
        如果匹配成功，返回 (True, fix_payload, rule_name)。
        如果无法被规则覆盖，返回 (False, None, None)。

        Args:
            diagnosis: 诊断信息，可能包含:
                - anomaly_type (str): 异常类型
                - evidence (dict): 证据数据
                - code_snippet (str): 相关代码片段
                - params (dict): 参数值
                - affected_files (list): 受影响文件
                - error_message (str): 错误消息

        Returns:
            (can_fix, fix_payload, rule_name) 三元组
        """
        anomaly_type = diagnosis.get("anomaly_type", "")
        evidence = diagnosis.get("evidence", {})
        code_snippet = diagnosis.get("code_snippet", "")
        error_message = diagnosis.get("error_message", "")

        # Rule 1: 评分计算Bug
        if anomaly_type == "extreme_negative_contribution" or code_snippet or error_message:
            result = RuleEngine.check_score_calculation_bug({
                "code_snippet": code_snippet,
                "error_message": error_message,
                "agent_name": diagnosis.get("affected_agent", ""),
            })
            if result.get("matched"):
                fix_payload = {
                    "method": "rule_direct_fix",
                    "rule_name": result["pattern_name"],
                    "description": result["description"],
                    "changes": [{
                        "type": "modify",
                        "location": diagnosis.get("affected_agent", "unknown"),
                        "before": "当前评分逻辑（存在计算方向或范围 Bug）",
                        "after": result["fix_hint"],
                        "reason": f"规则引擎检测到已知 Bug 模式: {result['pattern_name']}",
                    }],
                    "patched_code": (
                        f"# [LogicFixer/RuleEngine] 自动修复: {result['pattern_name']}\n"
                        f"# {result['description']}\n"
                        f"# 修复指引: {result['fix_hint']}\n"
                    ),
                    "test_needed": True,
                    "confidence": result["confidence"],
                }
                return (True, fix_payload, result["pattern_name"])

        # Rule 2: 权重归一化
        params = diagnosis.get("params", {})
        if params:
            norm_check = RuleEngine.check_weight_normalization(params)
            if not norm_check["all_normalized"]:
                fix_payload = {
                    "method": "rule_direct_fix",
                    "rule_name": "weight_normalization",
                    "description": "权重组 Σ != 1.0，自动归一化",
                    "changes": [
                        {
                            "type": "modify",
                            "location": v["group"],
                            "before": f"Σ={v['total']:.4f}（未归一化）",
                            "after": "归一化到 Σ=1.0",
                            "reason": f"权重组 {v['group']} 和为 {v['total']:.4f}，需归一化",
                        }
                        for v in norm_check.get("violations", [])
                    ],
                    "patched_code": (
                        "# [LogicFixer/RuleEngine] 权重自动归一化\n"
                        + norm_check.get("fix_suggestion", "")
                    ),
                    "test_needed": True,
                    "confidence": 0.95,
                    "normalized_params": {
                        v["group"]: {
                            k: w / v["total"] for k, w in v["weights"].items()
                        }
                        for v in norm_check.get("violations", [])
                    },
                }
                return (True, fix_payload, "weight_normalization")

        # Rule 3: 阈值一致性
        rules = diagnosis.get("rules", diagnosis.get("thresholds", []))
        if rules:
            threshold_check = RuleEngine.check_threshold_consistency(rules)
            if not threshold_check["consistent"]:
                fix_payload = {
                    "method": "rule_direct_fix",
                    "rule_name": "threshold_consistency",
                    "description": "检测到阈值逻辑不一致",
                    "changes": [
                        {"type": "modify", "location": "threshold_config",
                         "before": str(gap),
                         "after": "修正后的阈值",
                         "reason": gap.get("message", "")}
                        for gap in threshold_check.get("gaps", []) + threshold_check.get("overlaps", [])
                    ],
                    "patched_code": (
                        "# [LogicFixer/RuleEngine] 阈值一致性修复\n"
                        + "\n".join(threshold_check.get("fix_suggestions", []))
                    ),
                    "test_needed": True,
                    "confidence": 0.88,
                }
                return (True, fix_payload, "threshold_consistency")

        # Rule 4: 数据字段不匹配
        code_refs = diagnosis.get("code_refs", diagnosis.get("field_refs", []))
        schema = diagnosis.get("schema", diagnosis.get("data_schema"))
        if code_refs:
            mismatch_check = RuleEngine.check_data_field_mismatch(code_refs, schema)
            if mismatch_check.get("has_issues"):
                fix_payload = {
                    "method": "rule_direct_fix",
                    "rule_name": "data_field_mismatch",
                    "description": "检测到代码引用的字段名可能有问题",
                    "changes": [
                        {"type": "modify",
                         "location": m.get("field", "unknown"),
                         "before": m.get("field", ""),
                         "after": m.get("suggested_fix", m.get("closest_match", "")),
                         "reason": m.get("message", "")}
                        for m in mismatch_check.get("mismatches", [])
                    ],
                    "patched_code": (
                        "# [LogicFixer/RuleEngine] 字段名修复\n"
                        + "\n".join(mismatch_check.get("suggestions", []))
                    ),
                    "test_needed": True,
                    "confidence": 0.82,
                }
                return (True, fix_payload, "data_field_mismatch")

        # 无法被已知规则覆盖
        return (False, None, None)

    # ── 修复生成 ───────────────────────────────────────────────────

    def fix(self, diagnosis: Dict[str, Any], source_code: str = "") -> Dict[str, Any]:
        """主修复入口 — 先尝试规则驱动修复，失败才回退到模板生成。

        流程：
          1. 调用 can_fix_by_rules() 检查是否可被规则覆盖
          2. 如果可以 → 直接返回规则修复结果（确定性、快速）
          3. 如果不行 → 调用 generate_fix() 生成 LLM 模板

        Args:
            diagnosis: 诊断信息字典，格式与 can_fix_by_rules 相同
            source_code: 受影响模块的源代码

        Returns:
            同 generate_fix() 的返回格式
        """
        can_fix, fix_payload, rule_name = self.can_fix_by_rules(diagnosis)
        if can_fix and fix_payload is not None:
            fix_payload["fix_source"] = "rule_engine"
            fix_payload["rule_name"] = rule_name
            return fix_payload

        # 规则不覆盖，回退到 LLM 模板生成
        anomaly = diagnosis.get("anomaly", diagnosis)
        result = self.generate_fix(anomaly, source_code)
        result["fix_source"] = "template_based"
        return result

    def generate_fix(
        self,
        anomaly: Dict[str, Any],
        source_code: str = "",
    ) -> Dict[str, Any]:
        """为检测到的异常生成修复建议。

        Args:
            anomaly: 来自 analyze_anomalies() 的异常条目
            source_code: 受影响模块的源代码（可选，用于定位修复位置）

        Returns:
            {
                "patched_code": str,       # 修复后的代码（如果有源码则包含实际修改，否则为模板）
                "changes": [{type, location, before, after, reason}, ...],
                "test_needed": bool,
            }
        """
        # 先检查规则引擎是否能处理
        diagnosis = {"anomaly": anomaly, "anomaly_type": anomaly.get("anomaly_type", ""),
                     "evidence": anomaly.get("evidence", {}),
                     "affected_agent": anomaly.get("affected_agent", "")}
        can_fix, fix_payload, rule_name = self.can_fix_by_rules(diagnosis)
        if can_fix and fix_payload is not None:
            fix_payload["fix_source"] = "rule_engine"
            fix_payload["rule_name"] = rule_name
            return fix_payload

        # 规则不覆盖，使用模板生成
        anomaly_type = anomaly.get("anomaly_type", "")
        affected_agent = anomaly.get("affected_agent", "")

        if anomaly_type == "extreme_negative_contribution":
            return self._fix_extreme_negative(anomaly, source_code, affected_agent)
        elif anomaly_type == "contribution_divergence":
            return self._fix_divergence(anomaly, source_code, affected_agent)
        elif anomaly_type == "loss_dimension_concentration":
            return self._fix_loss_concentration(anomaly, source_code)
        elif anomaly_type == "instability":
            return self._fix_instability(anomaly, source_code, affected_agent)
        else:
            return {
                "patched_code": source_code,
                "changes": [],
                "test_needed": False,
            }

    def _fix_extreme_negative(
        self, anomaly: Dict[str, Any], source_code: str, agent: str
    ) -> Dict[str, Any]:
        """为极端负贡献 agent 生成修复建议。"""
        evidence = anomaly.get("evidence", {})
        delta = evidence.get("delta_L_total", 0)

        changes = [{
            "type": "modify",
            "location": f"{agent} 评分/权重逻辑",
            "before": "当前评分逻辑（待检查的代码）",
            "after": "建议：检查评分方向是否正确（正相关 vs 负相关），"
                     f"当前 delta={delta:.4f} > 0，agent 增加了 Loss",
            "reason": f"Agent 的评分方向可能与真实收益方向相反，需验证评分-收益的秩相关性",
        }]
        return {
            "patched_code": source_code or (
                f"# [LogicFixer] 极端负贡献修复建议 for {agent}\n"
                f"# delta_L_total = {delta:.4f}\n"
                f"# 1. 检查 agent 的评分和权重是否正确（方向检查）\n"
                f"# 2. 验证评分区间映射（0-100 vs 实际预期收益）\n"
                f"# 3. 检查是否误用了反转逻辑（如将坏消息评高分）\n"
            ),
            "changes": changes,
            "test_needed": True,
        }

    def _fix_divergence(
        self, anomaly: Dict[str, Any], source_code: str, scorer: str
    ) -> Dict[str, Any]:
        """为 scorer 融合背离生成修复建议。"""
        evidence = anomaly.get("evidence", {})

        changes = [{
            "type": "modify",
            "location": f"{scorer} 的融合权重逻辑",
            "before": "当前融合逻辑",
            "after": (
                "建议：检查依赖 agent 的输出如何在 scorer 中融合。\n"
                f"  依赖 agents 平均贡献: {evidence.get('dep_mean_delta', 'N/A')}\n"
                f"  当前 scorer 贡献: {evidence.get('scorer_delta', 'N/A')}\n"
                "  可能原因：(a) 权重分配错误 (b) 某个依赖 agent 输出被反向解读 "
                "(c) scorer prompt 中的评分指引有误导"
            ),
            "reason": "Scorer 与其依赖 agent 的贡献方向不一致，融合逻辑可能有问题",
        }]
        return {
            "patched_code": source_code or (
                f"# [LogicFixer] 融合背离修复建议 for {scorer}\n"
                f"# 确认各依赖 agent 的权重是否合理\n"
                f"# 检查是否有 agent 输出被错误解读（如 signal bias 方向弄反）\n"
            ),
            "changes": changes,
            "test_needed": True,
        }

    def _fix_loss_concentration(
        self, anomaly: Dict[str, Any], source_code: str
    ) -> Dict[str, Any]:
        """为 Loss 维度异常集中生成修复建议。"""
        evidence = anomaly.get("evidence", {})
        dim = evidence.get("dimension", "Unknown")

        changes = [{
            "type": "modify",
            "location": f"LossEngine 中 {dim} 的计算逻辑",
            "before": "当前计算阈值/映射",
            "after": (
                f"建议：检查 {dim} 的阈值映射是否过于严格。"
                f"当前维度 loss={evidence.get('dimension_loss', 'N/A')}，"
                f"占总 loss {evidence.get('ratio', 0)*100:.0f}%。"
                "考虑：(a) 调宽阈值 (b) 检查输入数据质量 (c) 降低该维度在加权中的权重"
            ),
            "reason": f"{dim} 维度 loss 异常高，可能阈值设置不当或数据质量问题",
        }]
        return {
            "patched_code": source_code or (
                f"# [LogicFixer] Loss 维度集中修复建议\n"
                f"# 维度: {dim}\n"
                f"# 请检查 LossEngine 中相关阈值和映射逻辑\n"
            ),
            "changes": changes,
            "test_needed": True,
        }

    def _fix_instability(
        self, anomaly: Dict[str, Any], source_code: str, agent: str
    ) -> Dict[str, Any]:
        """为不稳定性生成修复建议。"""
        evidence = anomaly.get("evidence", {})

        changes = [{
            "type": "modify",
            "location": f"{agent} 的输入处理逻辑",
            "before": "当前处理逻辑",
            "after": (
                f"建议：增强 {agent} 对输入数据质量的鲁棒性。"
                f"当前 CV={evidence.get('cv', 'N/A'):.2f}。\n"
                "  可能原因：(a) agent 过于依赖单一数据源 "
                "(b) 数据缺失时无合理的默认值 (c) 评分对极端值过于敏感\n"
                "  修复方向：添加输入数据validation、使用稳健统计量、添加数据缺失fallback"
            ),
            "reason": f"Agent 贡献在不同批次间波动较大 (CV={evidence.get('cv', 0):.2f})",
        }]
        return {
            "patched_code": source_code or (
                f"# [LogicFixer] 不稳定性修复建议 for {agent}\n"
                f"# 添加输入数据 validation 和缺失值 fallback 逻辑\n"
                f"# 使用中位数替代均值、添加数据质量评分门槛\n"
            ),
            "changes": changes,
            "test_needed": True,
        }

    # ── 测试生成 ───────────────────────────────────────────────────

    def generate_test(self, anomaly: Dict[str, Any], fix: Dict[str, Any]) -> str:
        """为修复生成一个单元测试骨架。

        Args:
            anomaly: 异常条目
            fix: 来自 generate_fix() 的修复结果

        Returns:
            Python 单元测试代码字符串
        """
        anomaly_type = anomaly.get("anomaly_type", "unknown")
        affected = anomaly.get("affected_agent", "unknown")
        severity = anomaly.get("severity", "low")

        test_name = f"test_fix_{anomaly_type}_{affected}".replace("-", "_")

        # 公共 preamble
        preamble = (
            '"""Auto-generated test by LogicFixer for anomaly: '
            f'{anomaly.get("description", "")}'
            '"""\n'
            'import pytest\n\n\n'
        )

        if anomaly_type == "extreme_negative_contribution":
            test_code = (
                preamble +
                f"def {test_name}():\n"
                f'    """验证 {affected} 的评分方向是否正确。"""\n'
                f"    # TODO: 替换为实际的 agent 调用\n"
                f"    # 1. 使用已知方向的数据（好坏各半）测试 agent 输出\n"
                f"    # 2. 验证 score 与真实收益的方向一致性\n"
                f"    # 3. 断言方向准确率 > 0.5（基线水平）\n"
                f"    pass\n"
            )
        elif anomaly_type == "contribution_divergence":
            test_code = (
                preamble +
                f"def {test_name}():\n"
                f'    """验证 {affected} 的融合方向与依赖 agent 一致。"""\n'
                f"    # TODO: 替换为实际的 scorer 调用\n"
                f"    # 1. Mock 所有依赖 agent 的 signal_pack（全 bullish）\n"
                f"    # 2. 验证 scorer 输出 bias 方向与输入一致\n"
                f"    # 3. Mock 全 bearish，验证 scorer 方向再次一致\n"
                f"    pass\n"
            )
        elif anomaly_type == "loss_dimension_concentration":
            test_code = (
                preamble +
                f"def {test_name}():\n"
                f'    """验证 loss 维度阈值在合理范围内。"""\n'
                f"    from src.eval.loss_engine import LossEngine\n"
                f"    # 1. 使用边缘场景数据测试 LossEngine\n"
                f"    # 2. 验证单维度 loss 不超过总 loss 的阈值比例\n"
                f"    # 3. 验证合理数据下 loss < 0.5\n"
                f"    engine = LossEngine()\n"
                f"    # TODO: 填入具体测试数据\n"
                f"    result = engine.compute_total_loss(\n"
                f'        "medium",\n'
                f"        scores=[50, 50, 50, 50, 50],\n"
                f"        returns=[0.01, 0.02, -0.01, 0.0, 0.015],\n"
                f"    )\n"
                f"    assert result['L_total'] >= 0.0\n"
                f"    assert result['L_total'] <= 1.0\n"
            )
        elif anomaly_type == "instability":
            test_code = (
                preamble +
                f"def {test_name}():\n"
                f'    """验证 {affected} 对数据扰动的鲁棒性。"""\n'
                f"    # TODO: 替换为实际的 agent 调用\n"
                f"    # 1. 用相同数据多次调用 agent（理论上应得到一致输出）\n"
                f"    # 2. 对小量噪声扰动，验证输出变化在合理范围（<10%）\n"
                f"    # 3. 验证缺失数据场景有合理的 fallback 输出\n"
                f"    pass\n"
            )
        else:
            test_code = (
                preamble +
                f"def {test_name}():\n"
                f'    """验证修复的正确性 (anomaly_type={anomaly_type})。"""\n'
                f"    # TODO: 根据 anomaly 描述编写具体测试\n"
                f"    pass\n"
            )

        return test_code
