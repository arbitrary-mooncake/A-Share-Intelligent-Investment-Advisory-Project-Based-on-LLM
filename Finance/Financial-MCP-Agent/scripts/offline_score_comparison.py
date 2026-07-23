"""
离线比对脚本（4.3 定稿：无影子期的质量保障）。

直接消费 data/intermediate_cache/ 中的历史 signal_pack，用确定性 scorer 重算三期限
分数，与同股同日的历史 LLM 分数比对——不重新跑任何 Agent、零 LLM 成本。

产出:
1. 逐股对比表 + 汇总指标（Spearman / MAE / 均值差 / Top-N 重叠 / 风险闸门一致率）
2. strength 字段分布报告（4.9-11：0 值占比、按 Agent 分组、量纲分布）
3. 历史报告矛盾率审计（4.9-2：reports/ 中 33 份报告的 S/M/L 方向 vs 确定性评级）

用法:
    python scripts/offline_score_comparison.py [--cache-dir PATH] [--top-n 20]
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.analysis_package_builder import build_analysis_package
from src.utils.deterministic_scorer import (
    collect_signals, compute_score, detect_material_conflicts, map_score_to_rating,
)
from src.utils.conflict_arbitration import arbitrate_conflicts
from src.utils.risk_gate import apply_risk_gate

AGENTS = ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"]
TERMS = ["short", "medium", "long"]

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_CACHE_DIR = os.path.join(_BASE_DIR, "data", "intermediate_cache")
_REPORTS_DIR = os.path.join(_BASE_DIR, "reports")
_OUTPUT_DIR = os.path.join(_BASE_DIR, "data", "analysis_reports")


# ── 缓存扫描 ────────────────────────────────────────────

def scan_signal_packs(cache_dir: str) -> Dict[Tuple[str, str], Dict[str, dict]]:
    """{(code, date): {agent_key: pack}}"""
    packs: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)
    pattern = re.compile(
        r"^(fundamental|technical|value|news|event|quality_risk|moneyflow)"
        r"_analysis_signal_pack_((?:sh|sz|bj)_\d+)_(\d{4}-\d{2}-\d{2})(?:_eval)?\.json$"
    )
    for fname in os.listdir(cache_dir):
        m = pattern.match(fname)
        if not m:
            continue
        agent, code, date = m.group(1), m.group(2), m.group(3)
        try:
            with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
                packs[(code, date)][agent] = json.load(f)
        except Exception:
            continue
    return packs


def scan_llm_scores(cache_dir: str) -> Dict[Tuple[str, str, str], float]:
    """{(term, code, date): llm_score}（scorer 缓存 content 为 JSON 字符串）"""
    scores: Dict[Tuple[str, str, str], float] = {}
    pattern = re.compile(
        r"^(short|medium|long)_term_scorer_((?:sh|sz|bj)_\d+)_(\d{4}-\d{2}-\d{2})(?:_eval)?\.json$"
    )
    for fname in os.listdir(cache_dir):
        m = pattern.match(fname)
        if not m:
            continue
        term, code, date = m.group(1), m.group(2), m.group(3)
        try:
            with open(os.path.join(cache_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            content = data.get("content", "")
            payload = json.loads(content) if isinstance(content, str) else content
            score = payload.get("score")
            if isinstance(score, (int, float)):
                scores[(term, code, date)] = float(score)
        except Exception:
            continue
    return scores


# ── 统计工具 ────────────────────────────────────────────

def _ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


# ── 确定性重算 ──────────────────────────────────────────

def _is_etf_code(code: str) -> bool:
    c = code.replace("sh_", "").replace("sz_", "").replace("bj_", "")
    return c.startswith(("51", "58", "15", "16", "18"))


async def recompute_deterministic(
    agent_packs: Dict[str, dict], term: str, code: str,
) -> Dict[str, Any]:
    """同包重算：AnalysisPackage 重建 → 纯代码冲突折扣 → 确定性分数 → risk_gate。"""
    state_data = {f"{agent}_signal_pack": pack for agent, pack in agent_packs.items()}
    pkg = build_analysis_package(state_data, "")
    signals = collect_signals(pkg)
    conflicts = detect_material_conflicts(signals)
    # 离线比对不调 LLM：enabled=False → 全部走 source_level 默认折扣（纯代码）
    discounts = await arbitrate_conflicts(conflicts, enabled=False)
    result = compute_score(term, signals, pkg, is_etf=_is_etf_code(code), signal_discounts=discounts)
    gate = apply_risk_gate(pkg, term, result["score"])
    if gate.score_cap is not None:
        result["score"] = round(min(result["score"], gate.score_cap), 1)
    result["risk_gate_abstain"] = gate.abstain
    result["risk_gate_flags"] = gate.risk_flags_found
    return result


# ── strength 分布（4.9-11） ─────────────────────────────

def strength_distribution(packs: Dict[Tuple[str, str], Dict[str, dict]]) -> Dict[str, Any]:
    per_agent: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "zero_with_direction": 0, "bucket_1_10": 0, "bucket_11_50": 0, "bucket_51_100": 0}
    )
    for agent_packs in packs.values():
        for agent, pack in agent_packs.items():
            for sig in (pack or {}).get("signals", []):
                if not isinstance(sig, dict):
                    continue
                st = per_agent[agent]
                st["total"] += 1
                try:
                    strength = int(sig.get("strength", 0))
                except (ValueError, TypeError):
                    strength = 0
                try:
                    direction = int(sig.get("direction", 0))
                except (ValueError, TypeError):
                    direction = 0
                if strength == 0 and direction != 0:
                    st["zero_with_direction"] += 1
                if 1 <= strength <= 10:
                    st["bucket_1_10"] += 1
                elif 11 <= strength <= 50:
                    st["bucket_11_50"] += 1
                elif 51 <= strength <= 100:
                    st["bucket_51_100"] += 1
    return dict(per_agent)


# ── 历史报告矛盾率审计（4.9-2） ─────────────────────────

_DIRECTION_BULLISH = ("看多", "看好", "买入", "积极", "超配")
_DIRECTION_BEARISH = ("看空", "不看好", "回避", "卖出", "减持", "谨慎")


def _parse_report_direction(section_text: str) -> str:
    for kw in _DIRECTION_BEARISH:
        if kw in section_text:
            return "bearish"
    for kw in _DIRECTION_BULLISH:
        if kw in section_text:
            return "bullish"
    return "neutral"


def _rating_to_direction(rating: str) -> str:
    if rating in ("强烈买入", "买入"):
        return "bullish"
    if rating in ("减持", "回避"):
        return "bearish"
    return "neutral"


def parse_report_judgments(report_path: str) -> Dict[str, str]:
    """抽取报告第 6 节三期限方向结论。返回 {term: direction}。"""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return {}
    m = re.search(r"###\s*6\.\s*短线\s*/\s*中线\s*/\s*长线判断([\s\S]*?)(?=###\s*7\.|$)", text)
    if not m:
        return {}
    section = m.group(1)
    judgments: Dict[str, str] = {}
    term_map = [("short", r"短线[（(][^)）]*[)）]\*?\*?[:：]\*?\*?([^\n*]+)"),
                ("medium", r"中线[（(][^)）]*[)）]\*?\*?[:：]\*?\*?([^\n*]+)"),
                ("long", r"长线[（(][^)）]*[)）]\*?\*?[:：]\*?\*?([^\n*]+)")]
    for term, pat in term_map:
        tm = re.search(pat, section)
        if tm:
            judgments[term] = _parse_report_direction(tm.group(1))
    return judgments


async def audit_reports(
    packs: Dict[Tuple[str, str], Dict[str, dict]],
) -> Dict[str, Any]:
    """历史报告方向 vs 确定性评级方向（需同股同日 signal_pack 才能比对）。"""
    results: List[Dict[str, Any]] = []
    fname_pattern = re.compile(r"report_.+_(\d{6})_(\d{8})_\d+\.md$")
    if not os.path.isdir(_REPORTS_DIR):
        return {"total_reports": 0, "comparable": 0, "items": []}
    for fname in sorted(os.listdir(_REPORTS_DIR)):
        m = fname_pattern.match(fname)
        if not m or not fname.endswith(".md"):
            continue
        code6, date8 = m.group(1), m.group(2)
        date = f"{date8[:4]}-{date8[4:6]}-{date8[6:8]}"
        judgments = parse_report_judgments(os.path.join(_REPORTS_DIR, fname))
        if not judgments:
            continue
        # 尝试 sh/sz 两种前缀匹配缓存
        agent_packs = None
        for prefix in (f"sh_{code6}", f"sz_{code6}"):
            if (prefix, date) in packs:
                agent_packs = packs[(prefix, date)]
                code = prefix
                break
        item: Dict[str, Any] = {"report": fname, "date": date, "judgments": judgments}
        if agent_packs is None:
            item["comparable"] = False
            item["note"] = "无同股同日 signal_pack 缓存"
        else:
            item["comparable"] = True
            item["terms"] = {}
            for term, report_dir in judgments.items():
                det = await recompute_deterministic(agent_packs, term, code)
                det_dir = _rating_to_direction(map_score_to_rating(det["score"]))
                contradictory = (
                    {report_dir, det_dir} == {"bullish", "bearish"}
                )
                item["terms"][term] = {
                    "report_direction": report_dir,
                    "det_score": det["score"],
                    "det_rating": map_score_to_rating(det["score"]),
                    "det_direction": det_dir,
                    "contradictory": contradictory,
                }
        results.append(item)
    comparable_items = [i for i in results if i.get("comparable")]
    n_contra = sum(
        1 for i in comparable_items for t in i["terms"].values() if t["contradictory"]
    )
    n_compared = sum(len(i["terms"]) for i in comparable_items)
    return {
        "total_reports": len(results),
        "comparable": len(comparable_items),
        "compared_terms": n_compared,
        "contradictory_terms": n_contra,
        "contradiction_rate": round(n_contra / n_compared, 3) if n_compared else None,
        "items": results,
    }


# ── 主流程 ──────────────────────────────────────────────

async def run(cache_dir: str, top_n: int) -> Dict[str, Any]:
    packs = scan_signal_packs(cache_dir)
    llm_scores = scan_llm_scores(cache_dir)

    # category 覆盖率：旧 Schema 缓存无 category，确定性 scorer 会跳过这些信号，
    # 分数退化为缺失惩罚分布——必须在报告中显式标注，防止误读为公式质量问题。
    total_signals = 0
    categorized_signals = 0
    for agent_packs in packs.values():
        for pack in agent_packs.values():
            for sig in (pack or {}).get("signals", []):
                if isinstance(sig, dict):
                    total_signals += 1
                    if sig.get("category"):
                        categorized_signals += 1
    category_coverage = round(categorized_signals / total_signals, 4) if total_signals else 0.0

    comparisons: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TERMS}
    for (term, code, date), llm_score in sorted(llm_scores.items()):
        agent_packs = packs.get((code, date))
        if not agent_packs:
            continue
        det = await recompute_deterministic(agent_packs, term, code)
        comparisons[term].append({
            "code": code, "date": date,
            "llm_score": llm_score, "det_score": det["score"],
            "diff": round(det["score"] - llm_score, 1),
            "det_rating": map_score_to_rating(det["score"]),
            "risk_flags": det.get("risk_gate_flags", []),
        })

    metrics: Dict[str, Any] = {}
    for term in TERMS:
        items = comparisons[term]
        if not items:
            metrics[term] = {"n": 0}
            continue
        llm_list = [i["llm_score"] for i in items]
        det_list = [i["det_score"] for i in items]
        diffs = [i["diff"] for i in items]
        # Top-N 重叠（按各自分数排序）
        llm_top = {i["code"] for i in sorted(items, key=lambda x: -x["llm_score"])[:top_n]}
        det_top = {i["code"] for i in sorted(items, key=lambda x: -x["det_score"])[:top_n]}
        overlap = len(llm_top & det_top) / top_n if top_n else 0.0
        metrics[term] = {
            "n": len(items),
            "spearman": round(spearman(llm_list, det_list), 4) if spearman(llm_list, det_list) is not None else None,
            "mae": round(sum(abs(d) for d in diffs) / len(diffs), 2),
            "mean_diff": round(sum(diffs) / len(diffs), 2),
            f"top{top_n}_overlap": round(overlap, 3),
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cache_dir": cache_dir,
        "pack_stocks": len({c for c, _ in packs.keys()}),
        "llm_score_records": len(llm_scores),
        "category_coverage": category_coverage,
        "metrics": metrics,
        "comparisons": comparisons,
        "strength_distribution": strength_distribution(packs),
        "report_audit": await audit_reports(packs),
    }


def render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# 离线比对报告：确定性 scorer vs 历史 LLM 分数", ""]
    lines.append(f"- 生成时间: {result['generated_at']}")
    lines.append(f"- 缓存目录: {result['cache_dir']}")
    lines.append(f"- signal_pack 覆盖股票数: {result['pack_stocks']}，LLM 分数记录数: {result['llm_score_records']}")
    lines.append("")
    coverage = result.get("category_coverage", 0.0)
    if coverage < 0.5:
        lines.append(
            f"> ⚠️ **category 覆盖率仅 {coverage:.1%}**：比对所用 signal_pack 大部分缺少 category 字段"
            "（旧 Schema），确定性 scorer 跳过未归类信号，分数退化为缺失惩罚分布。"
            "本次比对仅验证管线连通性，**不构成公式校准依据**。"
            "请先运行 `python scripts/backfill_signal_pack_category.py --apply` 回填后重新比对。"
        )
        lines.append("")
    lines.append("## 1. 汇总指标（验收门槛：Spearman≥0.9 / Top-N 重叠≥85% / |均值差|≤3）")
    lines.append("")
    lines.append("| 期限 | 样本数 | Spearman | MAE | 均值差 | Top-N 重叠 |")
    lines.append("|---|---|---|---|---|---|")
    for term in TERMS:
        m = result["metrics"].get(term, {})
        if not m.get("n"):
            lines.append(f"| {term} | 0 | — | — | — | — |")
            continue
        top_key = [k for k in m if k.startswith("top")][0]
        lines.append(
            f"| {term} | {m['n']} | {m.get('spearman')} | {m.get('mae')} "
            f"| {m.get('mean_diff')} | {m.get(top_key)} |"
        )
    lines.append("")
    lines.append("## 2. strength 字段分布（4.9-11 口径决策依据）")
    lines.append("")
    lines.append("| Agent | 信号总数 | strength=0且direction≠0 | 占比 | 1-10 | 11-50 | 51-100 |")
    lines.append("|---|---|---|---|---|---|---|")
    for agent, st in sorted(result["strength_distribution"].items()):
        total = st["total"] or 1
        lines.append(
            f"| {agent} | {st['total']} | {st['zero_with_direction']} "
            f"| {round(100 * st['zero_with_direction'] / total, 1)}% "
            f"| {st['bucket_1_10']} | {st['bucket_11_50']} | {st['bucket_51_100']} |"
        )
    lines.append("")
    audit = result["report_audit"]
    lines.append("## 3. 历史报告矛盾率审计（4.9-2）")
    lines.append("")
    lines.append(f"- 报告总数: {audit['total_reports']}，可比对（同股同日缓存）: {audit['comparable']}")
    if audit.get("contradiction_rate") is not None:
        lines.append(
            f"- 比对期限条数: {audit['compared_terms']}，矛盾: {audit['contradictory_terms']}"
            f"，矛盾率: {audit['contradiction_rate']:.1%}"
        )
    else:
        lines.append("- 无可比对样本（缓存日期与报告日期不重叠），转为对上线后新报告做前瞻性抽检。")
    lines.append("")
    lines.append("## 4. 偏差最大的 20 条记录")
    lines.append("")
    lines.append("| 期限 | 股票 | 日期 | LLM 分 | 确定性分 | 差值 |")
    lines.append("|---|---|---|---|---|---|")
    all_items = []
    for term in TERMS:
        all_items.extend({**i, "term": term} for i in result["comparisons"][term])
    for i in sorted(all_items, key=lambda x: -abs(x["diff"]))[:20]:
        lines.append(
            f"| {i['term']} | {i['code']} | {i['date']} "
            f"| {i['llm_score']} | {i['det_score']} | {i['diff']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性 scorer 离线比对")
    parser.add_argument("--cache-dir", default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    result = asyncio.run(run(args.cache_dir, args.top_n))

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = args.out or os.path.join(_OUTPUT_DIR, f"offline_comparison_{stamp}.md")
    json_path = md_path.replace(".md", ".json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"报告已生成: {md_path}")
    print(f"明细 JSON: {json_path}")
    for term in TERMS:
        m = result["metrics"].get(term, {})
        if m.get("n"):
            print(f"  [{term}] n={m['n']} spearman={m.get('spearman')} "
                  f"mae={m.get('mae')} mean_diff={m.get('mean_diff')}")


if __name__ == "__main__":
    main()
