"""
反幻觉验证层 — 五层防护体系，用于DeepSeek V4 Pro (eval_orchestrator) 的输出安全。

按照 评分智能体开发总纲 §12 "反幻觉机制（DeepSeek V4 Pro专用）" 实现：

  Layer 1: 输入结构化   — 将原始分析数据转为结构化JSON包，剥离LLM生成的叙事文本
  Layer 2: 输出验证     — 验证LLM输出符合期望JSON Schema，拒绝幻觉字段
  Layer 3: 代码验证层   — 数值溯源、实体验证、逻辑一致性、比较有效性
  Layer 4: 自洽性校验   — 同prompt不同seed多次运行，对比结论一致性
  Layer 5: 置信度标注   — 综合五层结果输出 HIGH/MEDIUM/LOW 置信度标签

设计原则:
  - 每层独立可调用，也支持 pipeline 串联
  - 绝不崩溃：所有异常均捕获，返回安全结果
  - LOW置信度结论不触发任何自动修改行为（总纲 §12.5）
"""
import json
import re
import copy
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

from src.utils.model_config import get_eval_model_config


# ──────────────────────────── Exceptions ────────────────────────────

class ValidationError(ValueError):
    """输出验证失败，包含具体失败的字段和原因。"""

    def __init__(self, message: str, failed_fields: List[str] = None,
                 hallucinated_fields: List[str] = None):
        super().__init__(message)
        self.failed_fields = failed_fields or []
        self.hallucinated_fields = hallucinated_fields or []


class VerificationFailure(ValueError):
    """代码验证层发现无法通过的安全问题。"""

    def __init__(self, message: str, layer: str = "", details: Dict = None):
        super().__init__(message)
        self.layer = layer
        self.details = details or {}


# ──────────────────────────── Data Classes ────────────────────────────

@dataclass
class VerificationResult:
    """单层验证结果。"""
    layer: str
    passed: bool
    score: float  # 0.0 ~ 1.0
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False


@dataclass
class ConfidenceLabel:
    """置信度标签（总纲 §12.5）。"""
    level: str  # HIGH / MEDIUM / LOW
    reasons: List[str] = field(default_factory=list)
    layer_results: List[VerificationResult] = field(default_factory=list)
    can_auto_act: bool = False  # LOW置信度不触发自动修改

    def __post_init__(self):
        self.can_auto_act = self.level != "LOW"


# ──────────────────────────── Helpers ────────────────────────────

# 用于识别数值的模式（整数、小数、百分数、科学计数）
# 使用 re.ASCII 避免 \w 匹配中文字符（Python 3 默认 \w 含 Unicode）
_NUMERIC_PATTERN = re.compile(
    r'(?<![\w.])[-+]?'
    r'(?:\d{1,3}(?:,\d{3})*|\d+)'
    r'(?:\.\d+)?'
    r'(?:[eE][-+]?\d+)?'
    r'(?![\w])',
    flags=re.ASCII,
)

# A股股票代码模式
_STOCK_CODE_PATTERN = re.compile(
    r'(?:sh|sz|bj)\.\d{6}|'        # tushare格式: sh.600000
    r'(?:SH|SZ|BJ)\.\d{6}|'
    r'\b(?:60[0123]\d{3}|'          # 沪市主板
    r'688\d{3}|'                     # 科创板
    r'00[0123]\d{3}|'                # 深市主板
    r'30[0123]\d{3}|'                # 创业板
    r'8[3-9]\d{3})'          # 北交所 (6-digit codes, match by length)
)

# Agent名称白名单（来自总纲和项目实际Agent）
_VALID_AGENT_NAMES = {
    "fundamental", "technical", "value", "news", "event",
    "quality_risk", "moneyflow",
    "fundamental_agent", "technical_agent", "value_agent",
    "news_agent", "event_agent", "event_analyst",
    "quality_risk_agent", "quality_risk_analyst",
    "moneyflow_agent", "moneyflow_analyst",
    "short_term_scorer", "medium_term_scorer", "long_term_scorer",
    "summary_agent",
}

# 合法ticket类型
_VALID_TICKET_TYPES = {"PARAM_TUNE", "PROMPT_PATCH", "LOGIC_FIX",
                       "ARCH_CHANGE", "RESEARCH"}

# 合法置信度值
_VALID_CONFIDENCE_VALUES = {"HIGH", "MEDIUM", "LOW"}


def _extract_numerics(text: str) -> List[float]:
    """从文本中提取所有数值。"""
    values = []
    for m in _NUMERIC_PATTERN.finditer(text):
        try:
            v = m.group().replace(",", "")
            values.append(float(v))
        except ValueError:
            continue
    return values


def _extract_stock_codes(text: str) -> List[str]:
    """从文本中提取所有A股股票代码。"""
    return [m.group() for m in _STOCK_CODE_PATTERN.finditer(text)]


def _deep_search_value(obj: Any, target: float, tolerance: float = 1e-8) -> bool:
    """在嵌套结构中递归搜索某个数值，返回是否找到近似匹配。"""
    if isinstance(obj, (int, float)):
        if isinstance(obj, bool):
            return False
        return abs(float(obj) - target) < tolerance
    if isinstance(obj, str):
        return False
    if isinstance(obj, (list, tuple)):
        return any(_deep_search_value(v, target, tolerance) for v in obj)
    if isinstance(obj, dict):
        return any(_deep_search_value(v, target, tolerance) for v in obj.values())
    return False


def _collect_all_numerics(obj: Any) -> List[float]:
    """收集嵌套结构中的所有数值。"""
    result = []
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        result.append(float(obj))
    elif isinstance(obj, str):
        pass
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            result.extend(_collect_all_numerics(v))
    elif isinstance(obj, dict):
        for v in obj.values():
            result.extend(_collect_all_numerics(v))
    return result


def _collect_all_strings(obj: Any) -> List[str]:
    """收集嵌套结构中的所有字符串。"""
    result = []
    if isinstance(obj, str):
        result.append(obj)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            result.extend(_collect_all_strings(v))
    elif isinstance(obj, dict):
        for v in obj.values():
            result.extend(_collect_all_strings(v))
    return result


# ──────────────────────────── Layer 1: 输入结构化 ────────────────────────────

def structure_input(data: dict) -> dict:
    """
    总纲 §12.1 — 输入结构化。

    将原始分析数据转为结构化JSON包，确保:
      1. 只保留预计算指标，剥离LLM生成的叙事文本
      2. 所有字段有显式类型标记
      3. 数值字段从字符串转为原生 float/int
      4. 移除可能被LLM"自行领悟"的模糊自然语言描述

    Args:
        data: 原始分析数据字典，可能包含 analysis 文本、signal_pack 等混合内容

    Returns:
        清洗后的结构化输入字典，每个叶子节点带有 _type 标记
    """
    try:
        cleaned = _structure_recursive(data, depth=0)
        # 顶层添加元信息
        cleaned["_meta"] = {
            "structured_at": datetime.now().isoformat(),
            "total_fields": _count_leaf_fields(cleaned),
            "primary_keys": list(cleaned.keys()),
        }
        return cleaned
    except Exception:
        return {"_meta": {"structured_at": datetime.now().isoformat(),
                          "error": "input structuring failed, returning safe minimal"},
                "raw_keys": list(data.keys()) if isinstance(data, dict) else []}


def _structure_recursive(obj: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """递归清洗结构化数据。"""
    if depth > max_depth:
        return str(obj)

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            # 跳过明显的自由叙事文本字段（由LLM生成的）
            if _is_narrative_field(k):
                continue
            # 跳过过长文本（>500字视为叙事）
            if isinstance(v, str) and len(v) > 500:
                continue
            result[k] = _structure_recursive(v, depth + 1, max_depth)
        return result

    if isinstance(obj, list):
        return [_structure_recursive(v, depth + 1, max_depth) for v in obj]

    if isinstance(obj, str):
        # 尝试将字符串转为数值
        return _try_parse_numeric(obj)

    return obj


def _is_narrative_field(key: str) -> bool:
    """判断字段名是否指向叙事文本（总纲 §12.1 要求排除）。"""
    narrative_keywords = [
        "analysis", "narrative", "summary", "description",
        "opinion", "comment", "interpretation", "评估", "分析",
        "总结", "描述", "观点", "评论", "推断",
    ]
    key_lower = key.lower()
    return any(kw in key_lower for kw in narrative_keywords)


def _try_parse_numeric(s: str) -> Union[str, int, float]:
    """将字符串尝试解析为数值类型。"""
    s = s.strip()
    if not s:
        return s
    # 尝试int
    try:
        i = int(s)
        # 检查不是浮点伪装后丢失精度
        if '.' not in s:
            return i
    except (ValueError, TypeError):
        pass
    # 尝试float
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    return s


def _count_leaf_fields(obj: Any) -> int:
    """统计叶子字段数。"""
    if isinstance(obj, dict):
        return sum(_count_leaf_fields(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_leaf_fields(v) for v in obj)
    return 1


# ──────────────────────────── Layer 2: 输出验证 ────────────────────────────

_VALIDATION_SCHEMA = {
    "type": "object",
    "required": ["diagnosis", "optimization_suggestions"],
    "properties": {
        "diagnosis": {
            "type": "object",
            "required": ["top_findings"],
            "properties": {
                "top_findings": {"type": "array"},
                "agent_ranking": {"type": "array"},
                "market_regime_insights": {"type": "object"},
            },
        },
        "optimization_suggestions": {"type": "array"},
        "narrative_summary": {"type": "string"},
    },
}


def validate_output(response: str, schema: dict = None,
                    source_data: dict = None) -> dict:
    """
    总纲 §12.2 — 输出验证。

    验证LLM输出:
      1. 符合期望JSON Schema（top_findings, optimization_suggestions等）
      2. 所有引用数据ID可在source_data中找到
      3. 股票代码合法且存在于输入数据中
      4. 无幻觉字段（字段名不在schema白名单中）

    Args:
        response: LLM输出的原始字符串（预期为JSON）
        schema: 期望的JSON Schema（默认使用_VALIDATION_SCHEMA）
        source_data: 输入数据，用于数据ID回溯验证

    Returns:
        解析并验证通过后的字典

    Raises:
        ValidationError: 验证失败时抛出，包含失败字段和幻觉字段明细
    """
    schema = schema or _VALIDATION_SCHEMA
    source_data = source_data or {}
    failed = []
    hallucinated = []

    # Step 1: 提取JSON
    parsed = _extract_json_from_response(response)
    if parsed is None:
        raise ValidationError(
            "无法从LLM输出中提取有效JSON",
            failed_fields=["_entire_response"]
        )

    # Step 2: 验证顶层required字段
    if not isinstance(parsed, dict):
        raise ValidationError(
            "LLM输出顶层必须是JSON对象",
            failed_fields=["_entire_response"]
        )

    required = schema.get("required", [])
    for field in required:
        if field not in parsed:
            failed.append(field)

    # Step 3: 验证 diagnosis 子结构
    diag = parsed.get("diagnosis", {})
    diag_schema = schema.get("properties", {}).get("diagnosis", {})
    diag_required = diag_schema.get("required", [])
    for field in diag_required:
        if field not in diag:
            failed.append(f"diagnosis.{field}")

    # Step 4: 验证 top_findings 中的 confidence 值
    top_findings = diag.get("top_findings", [])
    for i, finding in enumerate(top_findings):
        if isinstance(finding, dict):
            conf = finding.get("confidence", "")
            if conf and conf not in _VALID_CONFIDENCE_VALUES:
                failed.append(f"diagnosis.top_findings[{i}].confidence")
                hallucinated.append(f"diagnosis.top_findings[{i}].confidence='{conf}'")

    # Step 5: 验证 optimization_suggestions 中 ticket type
    suggestions = parsed.get("optimization_suggestions", [])
    for i, sug in enumerate(suggestions):
        if isinstance(sug, dict):
            ttype = sug.get("type", "")
            if ttype and ttype not in _VALID_TICKET_TYPES:
                failed.append(f"optimization_suggestions[{i}].type")
                hallucinated.append(
                    f"optimization_suggestions[{i}].type='{ttype}'")

    # Step 6: 数据ID引用验证（如果有source_data）
    if source_data:
        data_id_issues = _validate_data_id_references(parsed, source_data)
        hallucinated.extend(data_id_issues)

    # Step 7: 股票代码验证
    stock_issues = _validate_stock_codes_in_output(response, source_data)
    if stock_issues:
        hallucinated.extend(stock_issues)

    if failed or hallucinated:
        raise ValidationError(
            f"输出验证失败: {len(failed)}个缺失字段, "
            f"{len(hallucinated)}个疑似幻觉",
            failed_fields=failed,
            hallucinated_fields=hallucinated,
        )

    return parsed


def _extract_json_from_response(response: str) -> Optional[dict]:
    """从LLM输出中提取JSON对象。支持纯JSON、markdown code block、嵌入式JSON。"""
    response = response.strip()

    # 尝试直接解析
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # 尝试 ```json ... ``` code block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找最外层 { ... }
    brace_start = response.find('{')
    brace_end = response.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(response[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _validate_data_id_references(parsed: dict, source_data: dict) -> List[str]:
    """验证LLM输出中引用的数据ID是否在source_data中存在。"""
    issues = []
    # 收集所有evidence_ids
    top_findings = parsed.get("diagnosis", {}).get("top_findings", [])
    for i, finding in enumerate(top_findings):
        if isinstance(finding, dict):
            evidence_ids = finding.get("supporting_evidence_ids", [])
            for eid in evidence_ids:
                if isinstance(eid, str) and not _find_in_nested_dict(source_data, eid):
                    issues.append(
                        f"diagnosis.top_findings[{i}].supporting_evidence_ids: "
                        f"'{eid}' 在输入数据中未找到"
                    )

    # 验证rationale中的引用
    suggestions = parsed.get("optimization_suggestions", [])
    for i, sug in enumerate(suggestions):
        if isinstance(sug, dict):
            rationale = sug.get("rationale", "")
            if rationale and "证据ID" in rationale:
                # 提取 rationale 中提到的数据ID
                for m in re.finditer(r'[a-z]+_\w+_\w+_\w+', rationale):
                    eid = m.group()
                    if not _find_in_nested_dict(source_data, eid):
                        issues.append(
                            f"optimization_suggestions[{i}].rationale: "
                            f"引用的ID '{eid}' 在输入数据中未找到"
                        )

    return issues


def _find_in_nested_dict(data: dict, target_key: str) -> bool:
    """递归查找某个key是否存在于嵌套字典中。"""
    if target_key in data:
        return True
    for v in data.values():
        if isinstance(v, dict):
            if _find_in_nested_dict(v, target_key):
                return True
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _find_in_nested_dict(item, target_key):
                    return True
    return False


def _validate_stock_codes_in_output(text: str, source_data: dict) -> List[str]:
    """验证输出中的股票代码是否在source_data中存在。"""
    issues = []
    codes_in_output = _extract_stock_codes(text)

    # 收集source_data中所有出现的股票代码
    source_codes = set()
    all_strings = _collect_all_strings(source_data)
    for s in all_strings:
        source_codes.update(_extract_stock_codes(s))

    for code in codes_in_output:
        if code not in source_codes:
            issues.append(f"股票代码 '{code}' 不在输入数据中，可能是幻觉")
    return issues


# ──────────────────────────── Layer 3: 代码验证层 ────────────────────────────

def verify_numerical_traceability(changes: dict, source_data: dict) -> Tuple[bool, List[str]]:
    """
    总纲 §12.3.1 — 数值溯源验证。

    扫描LLM输出中的所有数字，确认每个数字都能在输入指标JSON中找到对应项。
    找不到的数字标记为"疑似幻觉"。

    Args:
        changes: LLM输出的修改建议/诊断（已解析为dict）
        source_data: 输入指标JSON包

    Returns:
        (passed: bool, flagged_items: list of human-readable issue descriptions)
    """
    source_nums = set(round(n, 6) for n in _collect_all_numerics(source_data))
    flagged = []

    # 先收集changes中的所有字符串，从中提取数值并检查
    change_strings = _collect_all_strings(changes)
    full_text = "\n".join(change_strings)
    output_nums = _extract_numerics(full_text)

    for num in output_nums:
        rounded = round(num, 6)
        found = False
        for sn in source_nums:
            if abs(rounded - sn) < 1e-4:
                found = True
                break
        # 也检查嵌套结构
        if not found:
            if not _deep_search_value(source_data, num, tolerance=1e-4):
                flagged.append(f"数值 {num} 在输入数据中找不到对应项")

    return len(flagged) == 0, flagged


def verify_entity_existence(entities: List[Dict[str, str]],
                            reference_data: dict) -> Tuple[bool, List[str]]:
    """
    总纲 §12.3.2 — 实体验证。

    扫描LLM输出中的Agent名称、文件名、路径，确认每个实体在系统中真实存在。

    Args:
        entities: 实体列表，每项为 {"type": "agent"/"file"/"path", "name": "..."}
        reference_data: 包含known_agents和known_paths的参考数据

    Returns:
        (passed: bool, issues: list of human-readable issue descriptions)
    """
    issues = []
    known_agents = reference_data.get("known_agents", _VALID_AGENT_NAMES)
    known_paths = reference_data.get("known_paths", set())

    for entity in entities:
        etype = entity.get("type", "")
        name = entity.get("name", "")

        if etype == "agent":
            if name not in known_agents:
                issues.append(f"Agent '{name}' 未在系统中注册")

        elif etype in ("file", "path"):
            if name not in known_paths and known_paths:
                import os
                # 如果提供了完整路径，检查文件是否存在
                if os.path.isabs(name):
                    if not os.path.exists(name):
                        issues.append(f"文件/路径 '{name}' 不存在")
                else:
                    # 非绝对路径，检查是否在已知路径列表中
                    matched = False
                    for kp in known_paths:
                        if name in kp or kp.endswith(name):
                            matched = True
                            break
                    if not matched:
                        issues.append(f"文件/路径 '{name}' 未在已知路径集中找到")

    return len(issues) == 0, issues


def verify_logical_consistency(before: dict, after: dict) -> List[str]:
    """
    总纲 §12.3.3 — 逻辑一致性验证。

    检测矛盾:
      - claim说"正贡献"但delta_L为负
      - suggestion说"修改文件X"但X不在白名单
      - agent排名与delta值方向不一致
      - ticket类型与建议操作不匹配

    Args:
        before: 修改前的状态数据
        after: LLM输出的修改建议/诊断数据

    Returns:
        矛盾描述列表（空列表表示一致）
    """
    contradictions = []

    # 检测 top_findings 中的方向矛盾
    top_findings = after.get("diagnosis", {}).get("top_findings", [])
    agents_data = before.get("agents", {})

    for i, finding in enumerate(top_findings):
        if not isinstance(finding, dict):
            continue
        claim = finding.get("claim", "").lower()
        evidence_ids = finding.get("supporting_evidence_ids", [])

        # 检查"正贡献" vs delta_L_total < 0
        if "正贡献" in claim or "正向" in claim or "positive" in claim:
            for eid in evidence_ids:
                for agent_name, agent_data in agents_data.items():
                    if isinstance(agent_data, dict) and agent_name in eid:
                        delta = agent_data.get("delta_L_total",
                                               agent_data.get("delta_L_return", 0))
                        if isinstance(delta, (int, float)) and delta < -0.001:
                            contradictions.append(
                                f"top_findings[{i}]: claim='{claim}' 但 "
                                f"{agent_name}.delta_L_total={delta} 为负值"
                            )

        # 检查"负贡献" vs delta_L_total > 0
        if "负贡献" in claim or "拖累" in claim or "negative" in claim:
            for eid in evidence_ids:
                for agent_name, agent_data in agents_data.items():
                    if isinstance(agent_data, dict) and agent_name in eid:
                        delta = agent_data.get("delta_L_total",
                                               agent_data.get("delta_L_return", 0))
                        if isinstance(delta, (int, float)) and delta > 0.001:
                            contradictions.append(
                                f"top_findings[{i}]: claim='{claim}' 但 "
                                f"{agent_name}.delta_L_total={delta} 为正值"
                            )

    # 检测 optimization_suggestions 中的 ticket type 矛盾
    suggestions = after.get("optimization_suggestions", [])
    for i, sug in enumerate(suggestions):
        if not isinstance(sug, dict):
            continue
        ttype = sug.get("type", "")
        target_file = sug.get("target_file", "")

        # ARCH_CHANGE应该是manual路由但suggestion可能错标为auto
        if ttype == "PARAM_TUNE":
            target_param = sug.get("target_param", "")
            # 参数调优应该是具体参数路径
            if target_param and not any(kw in target_param.lower()
                                        for kw in ["weight", "param", "threshold",
                                                    "score", "tolerance"]):
                contradictions.append(
                    f"optimization_suggestions[{i}]: type=PARAM_TUNE 但 "
                    f"target_param='{target_param}' 不像一个参数路径"
                )

    # 检测agent排名方向一致性
    agent_ranking = after.get("diagnosis", {}).get("agent_ranking", [])
    if agent_ranking and agents_data:
        for i, ranked in enumerate(agent_ranking):
            if isinstance(ranked, dict):
                agent_name = ranked.get("agent", ranked.get("name", ""))
                rank_delta = ranked.get("delta", ranked.get("contribution", 0))
                actual_data = agents_data.get(agent_name, {})
                if isinstance(actual_data, dict):
                    actual_delta = actual_data.get("delta_L_total", None)
                    if actual_delta is not None and isinstance(rank_delta, (int, float)):
                        # 符号必须一致
                        if (rank_delta > 0 and actual_delta < -0.001) or \
                           (rank_delta < 0 and actual_delta > 0.001):
                            contradictions.append(
                                f"agent_ranking[{i}]: {agent_name} 排名delta={rank_delta} "
                                f"与输入数据delta_L_total={actual_delta} 方向相反"
                            )

    return contradictions


def verify_comparison_validity(comparisons: List[Dict[str, Any]],
                                baseline: dict) -> Tuple[bool, List[str]]:
    """
    总纲 §12.3.4 — 比较验证。

    如果LLM输出说 "A > B" 或 "X比Y好"，代码验证是否真的 A > B。

    Args:
        comparisons: 比较声明列表，每项:
            {"left": "agent_name.field", "operator": "gt"/"lt"/"eq",
             "right": float, "claim_text": "原始声明文本"}
        baseline: 包含实际数值的基准数据

    Returns:
        (passed: bool, issues: list of description strings)
    """
    issues = []

    for i, comp in enumerate(comparisons):
        left_path = comp.get("left", "")
        operator = comp.get("operator", "gt")
        right_val = comp.get("right")
        right_path = comp.get("right_path")
        claim_text = comp.get("claim_text", f"comparison[{i}]")

        # Resolve right value: use direct numeric if available, else resolve path
        if right_val is None and right_path:
            right_val = _resolve_path(baseline, right_path)
            if right_val is None:
                issues.append(f"{claim_text}: 无法在baseline中找到右侧路径 '{right_path}'")
                continue
        if right_val is None:
            right_val = 0

        # 在baseline中查找left_path对应的实际值
        left_val = _resolve_path(baseline, left_path)
        if left_val is None:
            issues.append(f"{claim_text}: 无法在baseline中找到 '{left_path}'")
            continue

        if not isinstance(left_val, (int, float)):
            issues.append(f"{claim_text}: '{left_path}' 的值 ({left_val}) 不是数值")
            continue

        if not isinstance(right_val, (int, float)):
            issues.append(f"{claim_text}: 右侧值 ({right_val}) 不是数值，跳过比较")
            continue

        valid = False
        if operator == "gt":
            valid = left_val > right_val
        elif operator == "lt":
            valid = left_val < right_val
        elif operator == "eq":
            valid = abs(left_val - right_val) < 1e-6

        if not valid:
            issues.append(
                f"{claim_text}: 声明 {left_path}(={left_val}) {operator} {right_val} "
                f"但实际不成立"
            )

    return len(issues) == 0, issues


def _resolve_path(data: dict, path: str) -> Any:
    """解析点分隔路径，从嵌套字典中获取值。例如 'agents.fundamental.delta_L_total'。"""
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


# ──────────────────────────── Layer 4: 自洽性校验 ────────────────────────────

def check_self_consistency(results: List[dict],
                           key_fields: List[str] = None) -> dict:
    """
    总纲 §12.4 — 自洽性校验 (Self-Consistency)。

    比较多次LLM运行输出的关键结论是否一致。
    通常用于: agent贡献排序、优化建议类型判断、top findings等关键分析。

    Note: 实际运行LLM多次由调用方负责（需要模型配置和API调用）。
          此函数接收已有的多次运行结果进行比较。

    Args:
        results: 多次LLM运行输出的结果列表 (已解析为dict)
        key_fields: 需要比较的关键字段路径列表，如
            ["diagnosis.top_findings", "optimization_suggestions"]
            默认比较 diagnosis.top_findings 和 optimization_suggestions

    Returns:
        {
            "consistency_score": float,     # 0.0 ~ 1.0
            "total_checks": int,
            "consistent_checks": int,
            "discrepancies": List[dict],    # 不一致的详情
            "is_consistent": bool,
        }
    """
    if len(results) < 2:
        return {
            "consistency_score": 1.0,
            "total_checks": 0,
            "consistent_checks": 0,
            "discrepancies": [],
            "is_consistent": True,
            "note": "少于2个结果，跳过自洽性校验",
        }

    key_fields = key_fields or [
        "diagnosis.top_findings",
        "optimization_suggestions",
    ]

    discrepancies = []
    total_checks = 0
    consistent_checks = 0

    base = results[0]
    for field_path in key_fields:
        base_val = _resolve_path(base, field_path)
        if base_val is None:
            continue

        for run_idx, other in enumerate(results[1:], start=2):
            other_val = _resolve_path(other, field_path)
            if other_val is None:
                continue

            total_checks += 1

            # 根据字段类型进行不同的比较
            if field_path == "diagnosis.top_findings":
                is_consistent, detail = _compare_top_findings(
                    base_val, other_val, run_idx)
            elif field_path == "optimization_suggestions":
                is_consistent, detail = _compare_suggestions(
                    base_val, other_val, run_idx)
            else:
                is_consistent, detail = _compare_generic(
                    base_val, other_val, field_path, run_idx)

            if is_consistent:
                consistent_checks += 1
            else:
                discrepancies.append(detail)

    consistency_score = consistent_checks / max(total_checks, 1)

    return {
        "consistency_score": round(consistency_score, 3),
        "total_checks": total_checks,
        "consistent_checks": consistent_checks,
        "discrepancies": discrepancies,
        "is_consistent": consistency_score >= 0.7,
    }


def _compare_top_findings(a: list, b: list, run_idx: int) -> Tuple[bool, dict]:
    """比较两次运行的top_findings是否一致。"""
    # 提取findings的核心结论
    a_claims = set()
    b_claims = set()
    for f in (a or []):
        if isinstance(f, dict):
            a_claims.add(f.get("claim", "")[:80])
    for f in (b or []):
        if isinstance(f, dict):
            b_claims.add(f.get("claim", "")[:80])

    overlap = a_claims & b_claims
    total = len(a_claims | b_claims)
    jaccard = len(overlap) / max(total, 1)

    consistent = jaccard >= 0.5
    return consistent, {
        "field": "diagnosis.top_findings",
        "run_compared": run_idx,
        "jaccard_similarity": round(jaccard, 3),
        "claims_only_in_run1": list(a_claims - b_claims),
        "claims_only_in_other_run": list(b_claims - a_claims),
        "consistent": consistent,
    }


def _compare_suggestions(a: list, b: list, run_idx: int) -> Tuple[bool, dict]:
    """比较两次运行的优化建议类型是否一致。"""
    a_types = set()
    b_types = set()
    for s in (a or []):
        if isinstance(s, dict) and "type" in s:
            a_types.add(s["type"])
    for s in (b or []):
        if isinstance(s, dict) and "type" in s:
            b_types.add(s["type"])

    overlap = a_types & b_types
    total = len(a_types | b_types)
    jaccard = len(overlap) / max(total, 1)

    consistent = jaccard >= 0.5
    return consistent, {
        "field": "optimization_suggestions",
        "run_compared": run_idx,
        "jaccard_similarity": round(jaccard, 3),
        "types_only_in_run1": list(a_types - b_types),
        "types_only_in_other_run": list(b_types - a_types),
        "consistent": consistent,
    }


def _compare_generic(a: Any, b: Any, field: str, run_idx: int) -> Tuple[bool, dict]:
    """通用比较。"""
    a_hash = _stable_hash(a)
    b_hash = _stable_hash(b)
    consistent = a_hash == b_hash
    return consistent, {
        "field": field,
        "run_compared": run_idx,
        "run1_hash": a_hash,
        "other_hash": b_hash,
        "consistent": consistent,
    }


def _stable_hash(obj: Any) -> str:
    """生成稳定的哈希值用于比较。"""
    try:
        serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]
    except Exception:
        return "hash_error"


# ──────────────────────────── Layer 5: 置信度标注 ────────────────────────────

def label_confidence(verification_results: List[VerificationResult]) -> ConfidenceLabel:
    """
    总纲 §12.5 — 置信度标注。

    综合五层验证结果，对LLM输出标注置信度:
      - HIGH: 所有4层验证全部通过，无任何问题
      - MEDIUM: 有1-2层有minor问题（非critical），不影响核心结论
      - LOW: 有任何一层出现critical问题，或≥3层有失败

    Args:
        verification_results: 每层验证返回的VerificationResult列表

    Returns:
        ConfidenceLabel with level, reasons, and can_auto_act flag
    """
    reasons = []
    failed_layers = []
    critical_failures = []
    all_passed = True

    for vr in verification_results:
        if not vr.passed:
            all_passed = False
            failed_layers.append(vr.layer)
            if vr.is_critical:
                critical_failures.append(vr.layer)
        if vr.issues:
            for issue in vr.issues:
                reasons.append(f"[{vr.layer}] {issue}")

    # 决策逻辑
    if all_passed:
        level = "HIGH"
        reasons.insert(0, "所有验证层通过，无任何问题")
    elif critical_failures or len(failed_layers) >= 3:
        level = "LOW"
        if critical_failures:
            reasons.insert(0, f"关键验证失败: {', '.join(critical_failures)}")
        else:
            reasons.insert(0, f"多个验证层失败 ({len(failed_layers)}): {', '.join(failed_layers)}")
    else:
        level = "MEDIUM"
        reasons.insert(0, f"部分验证层有minor问题 ({len(failed_layers)}): {', '.join(failed_layers)}")

    label = ConfidenceLabel(
        level=level,
        reasons=reasons,
        layer_results=verification_results,
    )

    return label


# ──────────────────────────── Pipeline ────────────────────────────

class AntiHallucinationPipeline:
    """
    五层反幻觉验证流水线。

    按总纲 §12 的完整流程串联五层防护:
      1. 输入结构化 (structure_input)
      2. 输出验证 (validate_output)
      3. 代码验证层 (4项检查)
      4. 自洽性校验 (check_self_consistency)
      5. 置信度标注 (label_confidence)

    Usage:
        pipeline = AntiHallucinationPipeline()
        result = pipeline.run(llm_output, source_data, known_entities={})
        if result["confidence"]["level"] == "LOW":
            # 不触发任何自动修改
            pass
    """

    def __init__(self, model_profile: str = "eval_orchestrator"):
        self.model_profile = model_profile
        self.model_config = get_eval_model_config(model_profile)

    def run(self, llm_output: str,
            source_data: dict,
            known_entities: dict = None,
            multi_run_results: List[dict] = None) -> dict:
        """
        执行完整的五层反幻觉验证流水线。

        Args:
            llm_output: LLM原始输出字符串
            source_data: 输入DeepSeek V4 Pro的原始指标数据
            known_entities: 已知实体（agents, paths等），用于Layer 3实体检查
            multi_run_results: 多次运行的结果（用于Layer 4自洽性），≥2时启用

        Returns:
            {
                "structured_input": dict,       # Layer 1结果
                "validated_output": dict,       # Layer 2结果
                "verifications": list,          # Layer 3结果列表
                "self_consistency": dict,       # Layer 4结果
                "confidence": dict,             # Layer 5结果
                "overall_pass": bool,
                "summary": str,
            }
        """
        known_entities = known_entities or {}
        verifications: List[VerificationResult] = []
        overall_pass = True

        # ── Layer 1: 输入结构化 ──
        structured = structure_input(source_data)

        # ── Layer 2: 输出验证 ──
        try:
            validated = validate_output(llm_output, source_data=structured)
            verifications.append(VerificationResult(
                layer="Layer2_OutputValidation",
                passed=True,
                score=1.0,
                details={"fields_checked": len(validated.keys())},
            ))
        except ValidationError as e:
            validated = {}
            is_critical = len(e.failed_fields) > 2 or len(e.hallucinated_fields) > 0
            verifications.append(VerificationResult(
                layer="Layer2_OutputValidation",
                passed=False,
                score=max(0.0, 1.0 - 0.2 * len(e.failed_fields)),
                issues=[str(e)] + e.hallucinated_fields,
                details={
                    "failed_fields": e.failed_fields,
                    "hallucinated_fields": e.hallucinated_fields,
                },
                is_critical=is_critical,
            ))
            overall_pass = False
        except Exception as e:
            validated = {}
            verifications.append(VerificationResult(
                layer="Layer2_OutputValidation",
                passed=False,
                score=0.0,
                issues=[f"验证异常: {e}"],
                is_critical=True,
            ))
            overall_pass = False

        # ── Layer 3: 代码验证层 ──
        l3_verifications = self._run_code_verification(validated, structured,
                                                       known_entities, llm_output)
        verifications.extend(l3_verifications)
        for v in l3_verifications:
            if not v.passed and v.is_critical:
                overall_pass = False

        # ── Layer 4: 自洽性校验 ──
        if multi_run_results and len(multi_run_results) >= 2:
            sc = check_self_consistency(multi_run_results)
            sc_passed = sc.get("is_consistent", False)
            sc_is_critical = not sc_passed and sc.get("consistency_score", 1.0) < 0.3
            verifications.append(VerificationResult(
                layer="Layer4_SelfConsistency",
                passed=sc_passed,
                score=sc.get("consistency_score", 0.0),
                issues=[f"discrepancy: {d}" for d in sc.get("discrepancies", [])],
                details=sc,
                is_critical=sc_is_critical,
            ))
            if not sc_passed and sc_is_critical:
                overall_pass = False
        else:
            verifications.append(VerificationResult(
                layer="Layer4_SelfConsistency",
                passed=True,
                score=1.0,
                details={"note": "单次运行，跳过自洽性校验"},
            ))

        # ── Layer 5: 置信度标注 ──
        confidence = label_confidence(verifications)

        # 汇总
        passed_layers = sum(1 for v in verifications if v.passed)
        total_layers = len(verifications)

        summary_parts = [f"置信度: {confidence.level}"]
        if not overall_pass:
            summary_parts.append(
                f"验证失败层: {[v.layer for v in verifications if not v.passed]}")
        summary_parts.append(f"({passed_layers}/{total_layers}层通过)")

        return {
            "structured_input": structured,
            "validated_output": validated,
            "verifications": [
                {
                    "layer": v.layer,
                    "passed": v.passed,
                    "score": v.score,
                    "issues": v.issues,
                    "is_critical": v.is_critical,
                }
                for v in verifications
            ],
            "self_consistency": sc if multi_run_results and len(multi_run_results) >= 2 else None,
            "confidence": {
                "level": confidence.level,
                "reasons": confidence.reasons,
                "can_auto_act": confidence.can_auto_act,
            },
            "overall_pass": overall_pass,
            "summary": " | ".join(summary_parts),
        }

    def _run_code_verification(self, validated: dict, structured: dict,
                               known_entities: dict, raw_output: str) -> List[VerificationResult]:
        """执行Layer 3的四项代码验证。"""
        results = []
        known_entities = known_entities or {}

        # 3.1 数值溯源
        trace_pass, trace_issues = verify_numerical_traceability(validated, structured)
        results.append(VerificationResult(
            layer="Layer3_1_NumericalTraceability",
            passed=trace_pass,
            score=0.0 if not trace_pass and len(trace_issues) > 3 else
                  1.0 if trace_pass else 0.6,
            issues=trace_issues,
            is_critical=not trace_pass and len(trace_issues) > 3,
        ))

        # 3.2 实体验证 — 从LLM输出提取实体
        entities = _extract_entities_from_output(raw_output, known_entities, validated)
        ent_pass, ent_issues = verify_entity_existence(entities, known_entities)
        results.append(VerificationResult(
            layer="Layer3_2_EntityExistence",
            passed=ent_pass,
            score=0.0 if not ent_pass else 1.0,
            issues=ent_issues,
            is_critical=not ent_pass and any(
                "agent" in e.get("type", "") for e in entities),
        ))

        # 3.3 逻辑一致性
        contradictions = verify_logical_consistency(structured, validated)
        logic_pass = len(contradictions) == 0
        results.append(VerificationResult(
            layer="Layer3_3_LogicalConsistency",
            passed=logic_pass,
            score=0.0 if len(contradictions) > 2 else
                  1.0 if logic_pass else 0.5,
            issues=contradictions,
            is_critical=not logic_pass,
        ))

        # 3.4 比较验证
        comparisons = _extract_comparisons_from_output(raw_output)
        if comparisons:
            comp_pass, comp_issues = verify_comparison_validity(comparisons, structured)
            results.append(VerificationResult(
                layer="Layer3_4_ComparisonValidity",
                passed=comp_pass,
                score=1.0 if comp_pass else 0.5,
                issues=comp_issues,
                is_critical=False,
            ))

        return results


def _extract_entities_from_output(text: str, known_entities: dict,
                                   validated: dict) -> List[Dict[str, str]]:
    """从LLM输出中提取Agent名、文件名、路径等实体。"""
    entities = []

    # Agent名称 — sort by length descending to avoid substring duplicates
    known_agents = known_entities.get("known_agents", _VALID_AGENT_NAMES)
    for agent_name in sorted(known_agents, key=len, reverse=True):
        if agent_name in text:
            entities.append({"type": "agent", "name": agent_name})

    # 文件路径（匹配 src/... 或 config/... 等模式）
    path_pattern = re.compile(
        r'(?:src|config|data|tests)/[\w/.-]+\.(?:py|json|yaml|yml|toml|md)')
    for m in path_pattern.finditer(text):
        entities.append({"type": "path", "name": m.group()})

    # 从validated中提取target_file
    suggestions = validated.get("optimization_suggestions", [])
    for sug in suggestions:
        if isinstance(sug, dict) and "target_file" in sug:
            entities.append({"type": "file", "name": sug["target_file"]})

    return entities


def _extract_comparisons_from_output(text: str) -> List[Dict[str, Any]]:
    """从LLM输出文本中提取比较声明。"""
    comparisons = []
    # 匹配形如 "A(0.5) > B(0.3)" 或 "X比Y高"
    comp_pattern = re.compile(
        r'(\w+(?:\.\w+)*)\s*([><])\s*(\w+(?:\.\w+)*)')
    for m in comp_pattern.finditer(text):
        left = m.group(1)
        op = m.group(2)
        right = m.group(3)
        # Try to resolve right as a numeric value
        try:
            right_val = float(right)
        except ValueError:
            # Try to parse as path and resolve from baseline later;
            # mark as raw so verify_comparison_validity can resolve it
            right_val = None
        comparisons.append({
            "left": left,
            "operator": "gt" if op == ">" else "lt",
            "right": right_val,
            "right_path": right if right_val is None else None,
            "claim_text": m.group(0),
            "raw": right_val is None,  # True if right needs path resolution
        })
    return comparisons


# ──────────────────────────── Convenience API ────────────────────────────

def quick_verify(llm_output: str, source_data: dict) -> dict:
    """
    快速单次验证 — 不需要多次运行结果，执行 Layer 1-3 + Layer 5。

    适合在不需要自洽性校验的轻量场景使用。

    Args:
        llm_output: LLM原始输出字符串
        source_data: 原始输入数据

    Returns:
        与 Pipeline.run() 同格式的结果字典
    """
    pipeline = AntiHallucinationPipeline()
    return pipeline.run(llm_output, source_data, multi_run_results=None)


def verify_with_consistency(llm_outputs: List[str],
                            source_data: dict) -> dict:
    """
    带自洽性校验的完整验证 — 执行全部五层防护。

    Args:
        llm_outputs: 至少2个LLM输出（同一prompt不同seed的运行结果）
        source_data: 原始输入数据

    Returns:
        与 Pipeline.run() 同格式的结果字典
    """
    if not llm_outputs:
        return {"error": "llm_outputs is empty", "consistency_score": 0.0, "is_consistent": False}

    pipeline = AntiHallucinationPipeline()

    # 分别验证每个输出，获取parsed结果
    parsed_results = []
    for output in llm_outputs:
        try:
            parsed = _extract_json_from_response(output)
            if parsed:
                parsed_results.append(parsed)
        except Exception:
            pass

    return pipeline.run(
        llm_outputs[0],
        source_data,
        multi_run_results=parsed_results,
    )
