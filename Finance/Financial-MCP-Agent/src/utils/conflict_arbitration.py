"""
实质冲突仲裁层（4.3 定稿）：代码检测 + 触发式快速 LLM 仲裁。

职责边界：
- 代码做检测（deterministic_scorer.detect_material_conflicts：枚举归组 + 数值比较）；
- LLM 只回答"这两份证据哪份更可信"，输出有界折扣系数 {"dominant", "discount"}
  或 {"no_conflict": true}——LLM 产出的是权重参数，分数仍全部由代码计算；
- 仲裁失败/解析失败/未触发仲裁 → 一律退回 source_level 默认折扣规则，
  任何环节失败都向下跌落到纯代码，不产生 invalid 分数；
- 仲裁结果按冲突内容签名缓存，同样的冲突不重复仲裁。
"""
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

from src.utils.analysis_schema import SOURCE_PRIORITY
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# 默认折扣：未仲裁/仲裁失败时，低 source 优先级一方 × 0.5
DEFAULT_DISCOUNT = 0.5

# 仲裁结果文件缓存目录（按冲突签名缓存，长期有效——同样证据的冲突结论不变）
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "arbitration_cache",
)


def _conflict_signature(conflict: Dict[str, Any]) -> str:
    """冲突内容签名：双方信号的关键字段决定缓存键。"""
    def _brief(sig: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "factor": sig.get("factor", ""),
            "direction": sig.get("direction", 0),
            "strength": sig.get("strength", 0),
            "confidence": sig.get("confidence", 0),
            "source_level": sig.get("source_level", ""),
            "note": str(sig.get("note", ""))[:200],
        }
    raw = json.dumps(
        {"category": conflict.get("category", ""),
         "bullish": _brief(conflict["bullish"]),
         "bearish": _brief(conflict["bearish"])},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _read_cached_arbitration(signature: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(_CACHE_DIR, f"{signature}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cached_arbitration(signature: str, result: Dict[str, Any]) -> None:
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = os.path.join(_CACHE_DIR, f"{signature}.json")
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        pass  # 缓存失败不影响主流程


def _default_rule_discounts(conflict: Dict[str, Any]) -> Dict[int, float]:
    """source_level 默认折扣：低优先级一方 × DEFAULT_DISCOUNT（纯代码兜底）。"""
    bull = conflict["bullish"]
    bear = conflict["bearish"]
    bull_pri = SOURCE_PRIORITY.get(bull.get("source_level", "proxy"), 1)
    bear_pri = SOURCE_PRIORITY.get(bear.get("source_level", "proxy"), 1)
    if bull_pri == bear_pri:
        return {id(bull): DEFAULT_DISCOUNT, id(bear): DEFAULT_DISCOUNT}
    loser = bull if bull_pri < bear_pri else bear
    return {id(loser): DEFAULT_DISCOUNT}


def _parse_arbitration_output(text: str) -> Optional[Dict[str, Any]]:
    """严格解析仲裁输出：{"dominant": ..., "discount": ...} 或 {"no_conflict": true}。"""
    match = re.search(r'\{[\s\S]*\}', text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except Exception:
        return None
    if data.get("no_conflict") is True:
        return {"no_conflict": True}
    dominant = data.get("dominant")
    if dominant not in ("bullish", "bearish", "neither"):
        return None
    try:
        discount = float(data.get("discount", 0.5))
    except (ValueError, TypeError):
        return None
    return {"dominant": dominant, "discount": max(0.0, min(1.0, discount))}


async def _llm_arbitrate(conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """调用快速模型（M3 档）仲裁单组冲突。失败返回 None（调用方落兜底规则）。"""
    from langchain_openai import ChatOpenAI
    from src.utils.model_config import get_model_config_for_agent, get_thinking_body

    model_cfg = get_model_config_for_agent("conflict_arbiter")
    if not all([model_cfg.get("api_key"), model_cfg.get("base_url"), model_cfg.get("model_name")]):
        return None

    bull = conflict["bullish"]
    bear = conflict["bearish"]

    def _desc(sig: Dict[str, Any]) -> str:
        return (
            f"- 因子: {sig.get('factor', '?')}｜方向: {'看多' if sig.get('direction', 0) > 0 else '看空'}"
            f"｜强度: {sig.get('strength', '?')}｜置信度: {sig.get('confidence', '?')}"
            f"｜来源级别: {sig.get('source_level', '?')}｜来源Agent: {sig.get('_agent', '?')}\n"
            f"  依据: {str(sig.get('note', ''))[:300]}"
        )

    prompt = f"""你是金融证据仲裁员。以下两条关于同一股票同一类目（{conflict.get('category', '?')}）的信号对同一事实给出了矛盾判断。请判断哪份证据更可信。

信号A（看多）:
{_desc(bull)}

信号B（看空）:
{_desc(bear)}

仲裁规则：
1. 正式公告/交易所数据 > 数值工具 > 媒体报道 > 推断 > 间接代理；
2. 若两条信号其实不矛盾（不同视角/不同时间维度），返回 no_conflict；
3. 若一方明显更可信，返回 dominant 为胜方，并给出败方的折扣系数 discount（0=完全采信胜方，1=不折扣）；
4. 若无法判断，dominant 返回 neither。

只输出严格 JSON，不要输出其他内容：
{{"dominant": "bullish"|"bearish"|"neither", "discount": 0.0-1.0}}
或
{{"no_conflict": true}}"""

    try:
        llm = ChatOpenAI(
            model=model_cfg["model_name"],
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            temperature=0.2,
            request_timeout=60,
            max_tokens=1000,
            extra_body=get_thinking_body(model_cfg["base_url"], enabled=False),
        )
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        text = response.content.strip() if hasattr(response, "content") else str(response)
        return _parse_arbitration_output(text)
    except Exception as e:
        logger.warning(f"冲突仲裁 LLM 调用失败（落兜底规则）: {e}")
        return None


def _result_to_discounts(conflict: Dict[str, Any], result: Dict[str, Any]) -> Dict[int, float]:
    """仲裁结果 → 信号折扣映射。"""
    if result.get("no_conflict"):
        return {}
    bull, bear = conflict["bullish"], conflict["bearish"]
    dominant = result.get("dominant")
    discount = result.get("discount", DEFAULT_DISCOUNT)
    if dominant == "bullish":
        return {id(bear): discount}
    if dominant == "bearish":
        return {id(bull): discount}
    # neither：双方都打折
    return {id(bull): DEFAULT_DISCOUNT, id(bear): DEFAULT_DISCOUNT}


async def arbitrate_conflicts(
    conflicts: List[Dict[str, Any]],
    enabled: bool = True,
) -> Dict[int, float]:
    """对实质冲突逐组仲裁，返回 {id(signal): discount}。

    - 每组冲突先查签名缓存；
    - 缓存未命中且 enabled → LLM 仲裁（结果写缓存）；
    - LLM 失败/未启用 → source_level 默认折扣规则。
    """
    discounts: Dict[int, float] = {}
    for conflict in conflicts:
        signature = _conflict_signature(conflict)
        result = _read_cached_arbitration(signature)
        if result is None and enabled:
            result = await _llm_arbitrate(conflict)
            if result is not None:
                _write_cached_arbitration(signature, result)
        if result is None:
            discounts.update(_default_rule_discounts(conflict))
        else:
            discounts.update(_result_to_discounts(conflict, result))
    return discounts
