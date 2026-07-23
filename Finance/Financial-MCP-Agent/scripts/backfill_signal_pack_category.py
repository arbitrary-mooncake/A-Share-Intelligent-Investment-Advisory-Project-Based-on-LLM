"""
signal_pack category 一次性回填脚本（4.9-1 已确认方案）。

背景：signal_pack Schema 新增必填 category 字段后，旧缓存没有该字段。
本脚本用 DeepSeek V4 Flash 批量读取旧 signal_pack，为每条 signal 从固定枚举
判定 category 并写回原文件。

安全措施：
- 默认 dry-run，只统计不写盘；--apply 才执行写入；
- 枚举校验 + 信号数量对齐校验，失败的文件原样保留（后续按 schema 版本 miss 重跑）；
- 回填只新增 category 与溯源标记，不修改 created_at/其他任何字段（不延长缓存寿命）;
- 写盘用临时文件 + 原子替换，中断不产生半文件。

用法:
    python scripts/backfill_signal_pack_category.py            # dry-run 统计
    python scripts/backfill_signal_pack_category.py --apply    # 实际回填
    python scripts/backfill_signal_pack_category.py --apply --batch-size 10 --limit 100
"""
import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.analysis_schema import SIGNAL_CATEGORIES, SIGNAL_PACK_SCHEMA_VERSION
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CACHE_DIR = os.path.join(_BASE_DIR, "data", "intermediate_cache")

_PACK_FILE_PATTERN = re.compile(
    r"^(?:fundamental|technical|value|news|event|quality_risk|moneyflow)"
    r"_analysis_signal_pack_(?:sh|sz|bj)_\d+_\d{4}-\d{2}-\d{2}(?:_eval)?\.json$"
)


def _needs_backfill(pack: Dict[str, Any]) -> bool:
    signals = pack.get("signals")
    if not isinstance(signals, list) or not signals:
        return False
    return any(
        isinstance(s, dict) and "category" not in s for s in signals
    )


def scan_targets(cache_dir: str) -> List[str]:
    targets = []
    for fname in sorted(os.listdir(cache_dir)):
        if not _PACK_FILE_PATTERN.match(fname):
            continue
        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                pack = json.load(f)
        except Exception:
            continue
        if _needs_backfill(pack):
            targets.append(path)
    return targets


def _build_prompt(batch: List[Dict[str, Any]]) -> str:
    enum_text = ", ".join(SIGNAL_CATEGORIES)
    items = []
    for i, item in enumerate(batch):
        sig_lines = []
        for j, sig in enumerate(item["pack"]["signals"]):
            sig_lines.append(
                f"  signal[{j}]: factor={sig.get('factor', '?')} "
                f"| direction={sig.get('direction', 0)} "
                f"| note={str(sig.get('note', ''))[:150]}"
            )
        items.append(
            f"[pack {i}] agent={item['pack'].get('agent_name', '?')}\n" + "\n".join(sig_lines)
        )
    packs_text = "\n\n".join(items)
    return f"""你是金融信号分类员。下面每个 pack 是一组投资信号，请为每条 signal 判定 category（信号类目）。

可选枚举（必须严格从中选择，不得自创）：{enum_text}

类目含义速查：fundamentals_growth=业绩成长, fundamentals_profit_quality=盈利质量, valuation=估值,
balance_sheet=资产负债, cashflow=现金流, governance=公司治理, capital_flow=资金流向,
technical_trend=技术趋势, sentiment=舆情情绪, catalyst_event=事件催化, dividend=分红回报,
ownership=股权结构, industry_policy=行业政策, liquidity=流动性/量价, risk_flag=风险事件, other=无法归类

{packs_text}

只输出严格 JSON（categories 数组长度必须与该 pack 的 signal 数量一致）：
{{"results": [{{"index": 0, "categories": ["...", "..."]}}, ...]}}"""


def _parse_response(text: str, batch: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """解析并校验 LLM 输出。返回 {batch_index: categories}，校验失败的 pack 不包含在内。"""
    match = re.search(r'\{[\s\S]*\}', text or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {}
    results = data.get("results")
    if not isinstance(results, list):
        return {}
    valid: Dict[int, List[str]] = {}
    for entry in results:
        try:
            idx = int(entry.get("index", -1))
            categories = entry["categories"]
        except (AttributeError, KeyError, ValueError, TypeError):
            continue
        if not (0 <= idx < len(batch)):
            continue
        expected = len(batch[idx]["pack"]["signals"])
        if not isinstance(categories, list) or len(categories) != expected:
            continue
        if any(c not in SIGNAL_CATEGORIES for c in categories):
            continue
        valid[idx] = categories
    return valid


async def _call_flash(prompt: str) -> Optional[str]:
    from langchain_openai import ChatOpenAI
    from src.utils.model_config import get_eval_model_config, get_thinking_body

    model_cfg = get_eval_model_config("eval_analysis")  # DeepSeek V4 Flash
    if not all([model_cfg.get("api_key"), model_cfg.get("base_url"), model_cfg.get("model_name")]):
        raise RuntimeError("DeepSeek V4 Flash 模型配置缺失（OPENAI_COMPATIBLE_*_5）")
    llm = ChatOpenAI(
        model=model_cfg["model_name"],
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        temperature=0.0,
        request_timeout=120,
        max_tokens=2000,
        extra_body=get_thinking_body(model_cfg["base_url"], enabled=False),
    )
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
    return response.content.strip() if hasattr(response, "content") else str(response)


def _write_back(path: str, categories: List[str]) -> None:
    with open(path, "r", encoding="utf-8") as f:
        pack = json.load(f)
    for sig, cat in zip(pack["signals"], categories):
        sig["category"] = cat
        sig["category_source"] = "v4flash_backfill"
    pack["_schema_version"] = SIGNAL_PACK_SCHEMA_VERSION
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


async def run(cache_dir: str, apply: bool, batch_size: int, limit: int) -> None:
    targets = scan_targets(cache_dir)
    if limit > 0:
        targets = targets[:limit]
    total_signals = 0
    for path in targets:
        try:
            with open(path, "r", encoding="utf-8") as f:
                pack = json.load(f)
            total_signals += sum(
                1 for s in pack.get("signals", []) if isinstance(s, dict) and "category" not in s
            )
        except Exception:
            continue
    print(f"待回填文件: {len(targets)} 个，待归类信号: {total_signals} 条")
    if not apply:
        print("dry-run 模式（未写盘）。加 --apply 执行实际回填。")
        return

    written, failed = 0, 0
    for start in range(0, len(targets), batch_size):
        batch_paths = targets[start:start + batch_size]
        batch = []
        for path in batch_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    batch.append({"path": path, "pack": json.load(f)})
            except Exception:
                failed += 1
        if not batch:
            continue
        try:
            text = await _call_flash(_build_prompt(batch))
        except Exception as e:
            logger.error(f"批次 {start // batch_size + 1} LLM 调用失败: {e}")
            failed += len(batch)
            continue
        valid = _parse_response(text or "", batch)
        for idx, item in enumerate(batch):
            categories = valid.get(idx)
            if categories is None:
                failed += 1
                continue
            try:
                _write_back(item["path"], categories)
                written += 1
            except Exception as e:
                logger.error(f"写回失败 {item['path']}: {e}")
                failed += 1
        print(f"进度 {min(start + batch_size, len(targets))}/{len(targets)} "
              f"(已写回 {written}, 失败 {failed})")
    print(f"完成：写回 {written} 个文件，失败 {failed} 个（失败文件保持原样，将按版本 miss 重跑）")


def main() -> None:
    parser = argparse.ArgumentParser(description="signal_pack category 回填（4.9-1）")
    parser.add_argument("--cache-dir", default=_CACHE_DIR)
    parser.add_argument("--apply", action="store_true", help="实际写盘（默认 dry-run）")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（0=全部）")
    args = parser.parse_args()
    asyncio.run(run(args.cache_dir, args.apply, args.batch_size, args.limit))


if __name__ == "__main__":
    main()
