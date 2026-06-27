"""
QA反幻觉验证层 — 代码级事后验证，补充提示词层面的两区输出防幻觉机制。

移植自 src/eval/optimizer/anti_hallucination.py Layer 3 (代码验证层)，
适配QA的自然语言输出场景。

核心功能:
  1. 数值溯源 — 回答中的每个数字必须在证据数据中找到来源
  2. 实体验证 — 回答中的股票代码/名称必须真实存在
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# 数值提取: 匹配整数、小数（含千分位逗号）、百分比数字
# (?!\d) 仅禁止数字紧跟，允许单字母后缀如 32.5x（PE倍数常见写法）
_NUMERIC_PATTERN = re.compile(
    r'(?<![a-zA-Z_])(?:\d+(?:,\d{3})*)(?:\.\d+)?(?:\s*[%％])?(?!\d)'
)

# A股股票代码: 沪/深/北交所 6位数字
_STOCK_CODE_PATTERN = re.compile(
    r'(?<!\d)(?:sh\.|sz\.|bj\.)?'
    r'(?:60[0123]\d{3}|688\d{3}|00[0123]\d{3}|30[0123]\d{3}|'
    r'4[3-9]\d{4}|8[3-9]\d{4}|9[2-9]\d{4})(?!\d)'
)

# 忽略的数值模式（非金融数据的数字）
_IGNORE_NUMERIC_PATTERNS = [
    re.compile(r'^\d{1,2}[-/]\d{1,2}([-/]\d{2,4})?$'),   # 日期
    re.compile(r'^\d{1,2}:\d{2}(:\d{2})?$'),               # 时间
    re.compile(r'^\d{4}年\d{1,2}月\d{1,2}日$'),           # 中文日期
    re.compile(r'^\d+[\.-]\d+[\.-]\d+$'),                   # 版本号
    re.compile(r'^\d{6,}$'),                                 # 股票代码/长ID
    re.compile(r'^20\d{2}$'),                                 # 年份 (2000-2099)
]


@dataclass
class VerificationResult:
    """数值溯源验证结果"""
    passed: bool = True
    total_numbers: int = 0
    traceable_count: int = 0
    untraceable: List[dict] = field(default_factory=list)
    traceability_score: float = 1.0
    issues: List[str] = field(default_factory=list)


def _extract_numerics(text: str) -> List[float]:
    """从文本中提取所有数值（跳过日期、时间、代码等）"""
    values = []
    for m in _NUMERIC_PATTERN.finditer(text):
        raw = m.group().strip()
        # 跳过忽略模式
        if any(pat.match(raw) for pat in _IGNORE_NUMERIC_PATTERNS):
            continue
        # 清理千分位逗号和百分号
        cleaned = raw.replace(",", "").replace("%", "").replace("％", "").strip()
        try:
            val = float(cleaned)
            # 跳过过大的值（很可能是股票代码）和过小的值
            if 0.001 < val < 1_000_000_000:
                values.append(val)
        except ValueError:
            continue
    return values


def _deep_contains_value(text: str, target: float, tolerance: float = 0.01) -> bool:
    """在文本中搜索目标数值（相对容差）"""
    evidence_nums = _extract_numerics(text)
    for en in evidence_nums:
        if en == 0 and target == 0:
            return True
        if en != 0 and abs(target - en) / max(abs(en), 1) < tolerance:
            return True
    return False


def verify_answer_traceability(answer: str, evidence_text: str) -> VerificationResult:
    """
    验证QA回答中的数值是否能在证据数据中找到来源。

    Args:
        answer: LLM生成的回答文本
        evidence_text: 证据数据原始文本

    Returns:
        VerificationResult 包含详细验证结果
    """
    result = VerificationResult()

    if not answer or not evidence_text:
        result.passed = True  # 无数据时不标记为失败
        result.issues.append("回答或证据数据为空，跳过验证")
        return result

    # 提取回答中的数值
    answer_nums = _extract_numerics(answer)
    result.total_numbers = len(answer_nums)

    if result.total_numbers == 0:
        result.passed = True
        return result

    # 逐个数验证
    for num in answer_nums:
        if _deep_contains_value(evidence_text, num, tolerance=0.01):
            result.traceable_count += 1
        else:
            # 尝试更宽松的容差（5%）
            if _deep_contains_value(evidence_text, num, tolerance=0.05):
                result.traceable_count += 1
            else:
                # 截取回答中包含该数值的上下文
                context = _get_number_context(answer, num)
                result.untraceable.append({
                    "value": num,
                    "context": context,
                })

    # 计算可追溯性评分
    result.traceability_score = (
        result.traceable_count / result.total_numbers
        if result.total_numbers > 0 else 1.0
    )
    result.passed = result.traceability_score >= 0.7  # 70%阈值

    if result.untraceable:
        result.issues.append(
            f"{len(result.untraceable)}/{result.total_numbers} 个数值"
            f"在证据数据中找不到来源"
        )

    return result


def _get_number_context(text: str, target: float, window: int = 60) -> str:
    """获取文本中包含目标数值的上下文片段"""
    # 简单实现：搜索数值在文本中的位置
    target_strs = [str(int(target)) if target == int(target) else f"{target:.2f}"]
    # 也搜千分位格式
    if target >= 1000:
        target_strs.append(f"{target:,.0f}")
        target_strs.append(f"{target:,.2f}")

    for ts in target_strs:
        idx = text.find(ts)
        if idx >= 0:
            start = max(0, idx - window // 2)
            end = min(len(text), idx + len(ts) + window // 2)
            return text[start:end].replace("\n", " ")

    return f"（数值 {target} 上下文未定位）"


# ── 实体存在性验证 ────────────────────────────

def verify_stock_entities(answer: str) -> List[str]:
    """
    验证回答中提到的A股股票代码是否格式合法。

    注意：此函数仅在进程内做格式验证，不调用 Tushare API。
    完整的实体验证（调用 stock_basic）留待 P3 实现。

    Returns:
        格式不合法的代码列表（空=全部合法）
    """
    codes = _STOCK_CODE_PATTERN.findall(answer)
    invalid = []
    for code in codes:
        clean = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        if len(clean) != 6:
            invalid.append(code)
        elif not clean.isdigit():
            invalid.append(code)
    return invalid


async def verify_stock_entities_api(answer: str) -> dict:
    """
    P3.2: API-backed stock entity verification.

    Extracts A-share stock codes from the answer, queries Tushare stock_basic
    to verify they are real listed stocks. Runs in executor to avoid blocking.

    Returns:
        {"valid": [...], "invalid": [...], "not_found": [...], "error": str|None}
    """
    codes = _STOCK_CODE_PATTERN.findall(answer)
    if not codes:
        return {"valid": [], "invalid": [], "not_found": [], "error": None}

    # Separate valid-format from invalid-format
    valid_format = []
    invalid_format = []
    for code in codes:
        clean = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        if len(clean) == 6 and clean.isdigit():
            valid_format.append(clean)
        else:
            invalid_format.append(code)

    if not valid_format:
        return {"valid": [], "invalid": invalid_format, "not_found": [], "error": None}

    # Convert to Tushare format
    ts_codes = []
    for raw in valid_format:
        if raw.startswith(("6", "688", "5")):
            ts_codes.append(f"{raw}.SH")
        elif raw.startswith(("4", "8")):
            ts_codes.append(f"{raw}.BJ")
        else:
            ts_codes.append(f"{raw}.SZ")

    # Query Tushare in thread executor
    try:
        import asyncio
        from src.utils.tushare_client import get_stock_info_batch
        loop = asyncio.get_running_loop()
        found = await loop.run_in_executor(None, get_stock_info_batch, ts_codes)
    except Exception as e:
        logger.warning(f"Entity verification API call failed: {e}")
        return {"valid": [], "invalid": invalid_format,
                "not_found": valid_format, "error": str(e)[:200]}

    if not found:
        return {"valid": [], "invalid": invalid_format,
                "not_found": valid_format, "error": None}

    not_found = []
    valid = []
    for i, ts_code in enumerate(ts_codes):
        if ts_code in found:
            valid.append({
                "code": valid_format[i],
                "name": found[ts_code].get("name", ""),
                "industry": found[ts_code].get("industry", ""),
            })
        else:
            not_found.append(valid_format[i])

    return {
        "valid": valid,
        "invalid": invalid_format,
        "not_found": not_found,
        "error": None,
    }


# ── P3.3: 跨轮一致性检查 ────────────────────────────

@dataclass
class ConsistencyResult:
    """跨轮一致性检查结果"""
    passed: bool = True
    consistency_score: float = 1.0
    contradictions: List[dict] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def _extract_claims(text: str) -> List[dict]:
    """从文本中提取关键声明（因子+方向+数值）"""
    claims = []
    # 方向词
    bullish_words = ["上涨", "增长", "上升", "改善", "提升", "看好", "买入", "增持",
                     "流入", "净买入", "扩大", "加速", "提高", "走强"]
    bearish_words = ["下跌", "下降", "下滑", "恶化", "降低", "看空", "卖出", "减持",
                     "流出", "净卖出", "缩小", "减速", "走弱", "承压"]

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        nums = _extract_numerics(line)
        if not nums:
            continue

        direction = None
        if any(w in line for w in bullish_words):
            direction = "bullish"
        elif any(w in line for w in bearish_words):
            direction = "bearish"

        if direction:
            claims.append({
                "text": line[:200],
                "numbers": nums,
                "direction": direction,
            })

    return claims


def check_cross_turn_consistency(
    assistant_answers: List[str],
    question_texts: List[str] = None,
) -> ConsistencyResult:
    """
    P3.3: 跨轮一致性检查。

    Compares consecutive assistant answers in a session to detect
    self-contradictions: opposite directions on the same factor,
    or numerical claims that differ by >20% across turns.

    Args:
        assistant_answers: list of assistant answer texts in order
        question_texts: optional corresponding user questions

    Returns:
        ConsistencyResult with contradictions and score
    """
    result = ConsistencyResult()

    if len(assistant_answers) < 2:
        result.passed = True
        result.issues.append("少于2轮对话，跳过一致性检查")
        return result

    for i in range(len(assistant_answers) - 1):
        prev_claims = _extract_claims(assistant_answers[i])
        curr_claims = _extract_claims(assistant_answers[i + 1])

        for pc in prev_claims:
            for cc in curr_claims:
                # 检查方向矛盾
                if pc["direction"] and cc["direction"] and pc["direction"] != cc["direction"]:
                    # 查找共同的数值因子
                    common_nums = set(round(n, 1) for n in pc["numbers"]) & set(round(n, 1) for n in cc["numbers"])
                    if common_nums:
                        continue  # 相同数值不同方向可能是正常讨论

                    # 检查数值差距是否>20%
                    for pn in pc["numbers"]:
                        for cn in cc["numbers"]:
                            if pn > 0 and cn > 0:
                                gap = abs(pn - cn) / max(pn, cn)
                                if gap < 0.2 and pc["direction"] != cc["direction"]:
                                    result.contradictions.append({
                                        "turn_pair": (i + 1, i + 2),
                                        "prev_claim": pc["text"][:100],
                                        "curr_claim": cc["text"][:100],
                                        "prev_direction": pc["direction"],
                                        "curr_direction": cc["direction"],
                                        "value_gap": round(gap, 2),
                                    })

    if result.contradictions:
        result.passed = False
        total_turns = max(len(assistant_answers) - 1, 1)
        result.consistency_score = round(
            1.0 - min(len(result.contradictions) / total_turns, 1.0), 3
        )
        result.issues.append(
            f"发现{len(result.contradictions)}处可能的跨轮矛盾"
        )

    return result
