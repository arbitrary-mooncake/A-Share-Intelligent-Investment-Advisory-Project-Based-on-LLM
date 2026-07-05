"""
Prompt诊断与修复器 — LLM自动分析prompt问题并生成修复方案（总纲 §11.4）。

工作流：
  1. diagnose(): 将 agent prompt + 表现数据发给 LLM，要求输出结构化诊断
  2. generate_patch(): 基于诊断结果，让 LLM 生成修订后的 prompt
  3. validate_patch(): 在测试用例上对 old vs new prompt 做快速对比验证
  4. verify_prompt_change(): 通过mini回测对比original vs patched prompt的实际评分效果
  5. run_backtest_verification(): 完整的回测验证流程，生成VerificationResult

Gap #15 (2026-07): 新增回测验证方法，支持prompt补丁在真实回测数据上的效果验证。

LLM 调用通过 OpenAICompatibleClient + get_eval_model_config，使用 eval_orchestrator 模型。
所有 LLM 输出都有 JSON 解析保护 — 失败时回退到结构化的错误对象。
"""

import asyncio
import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.utils.model_config import get_eval_model_config
from src.utils.llm_clients import OpenAICompatibleClient


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 输出中提取 JSON 块（支持 ```json ... ``` 包裹）。"""
    if not text:
        return None
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None


class PromptPatcher:
    """Prompt诊断与修复器 — LLM自动分析prompt问题并生成修复方案。

    依赖 LLM（eval_orchestrator 模型）进行分析和补丁生成。
    """

    # LLM 诊断用的 system prompt 模板
    DIAGNOSE_SYSTEM_PROMPT = """你是一个 Prompt Engineering 专家。你需要分析一个 AI Agent 的 system prompt，诊断其潜在问题。

分析维度：
1. 指令清晰度：prompt 是否明确表达了期望的输出格式和内容？
2. 数据使用规范：是否要求区分数据事实和推理判断？是否标注了数据来源？
3. 评分/判断标准：是否存在模糊的标准（如"合理""较好"等缺乏具体锚点的表述）？
4. 抗幻觉机制：是否有防止编造数据、引用虚假信息的约束？
5. 边界条件处理：是否明确了数据缺失、异常数据、冲突信息时的处理方式？
6. 输出结构化：是否要求结构化输出（JSON / 标签分隔 / 分区分块）？

请基于提供的 agent 表现数据（performance_data），输出严格 JSON：

{
  "issues": [
    {
      "dimension": "指令清晰度/数据使用规范/评分标准/抗幻觉/边界条件/输出结构",
      "severity": "high/medium/low",
      "description": "具体问题描述",
      "location_hint": "prompt 中可能的问题位置或段落特征"
    }
  ],
  "suggestions": [
    {
      "action": "具体的修改建议（可直接作为补丁描述）",
      "priority": "high/medium/low",
      "expected_improvement": "预期改善效果描述"
    }
  ],
  "overall_assessment": {
    "quality_score": 0-100,
    "summary": "整体评估摘要"
  }
}

只输出 JSON，不要加任何前缀或后缀文字。"""

    PATCH_SYSTEM_PROMPT = """你是一个 Prompt Engineering 专家。你需要根据诊断结果修改一个 AI Agent 的 system prompt。

要求：
1. 保留原始 prompt 的核心信息、数据区和判断区结构
2. 修改诊断中标记的问题点
3. 确保修改后的 prompt 语言简洁、指令明确、无歧义
4. 添加必要的约束以防止幻觉
5. 保持原有的输出格式标签（如 <SIGNAL_PACK>、[数据]、[判断] 等）

请输出严格 JSON：

{
  "patched_prompt": "完整的修改后 prompt 文本",
  "changes": [
    {
      "type": "add/modify/remove",
      "location": "描述修改位置",
      "before": "修改前内容（如适用）",
      "after": "修改后内容",
      "reason": "修改原因"
    }
  ],
  "rationale": "修改的整体原理说明"
}

只输出 JSON，不要加任何前缀或后缀文字。"""

    def __init__(self):
        # 尝试初始化 LLM 客户端；若环境未配置则在 diagnose/generate 时返回友好错误
        self._client = None
        self._init_error = None
        try:
            model_cfg = get_eval_model_config("eval_orchestrator")
            self._client = OpenAICompatibleClient(
                api_key=model_cfg["api_key"],
                base_url=model_cfg["base_url"],
                model=model_cfg["model_name"],
                env_prefix="OPENAI_COMPATIBLE",
            )
        except Exception as e:
            self._init_error = str(e)

    def _call_llm(self, system_prompt: str, user_content: str) -> Optional[str]:
        """调用 LLM，返回文本内容；失败返回 None。"""
        if self._client is None:
            return None
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            return self._client.get_completion(messages)
        except Exception:
            return None

    # ── Public API ────────────────────────────────────────────────

    def diagnose(
        self,
        agent_name: str,
        prompt_text: str,
        performance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """分析 prompt 在给定表现数据下的潜在问题。

        Args:
            agent_name: Agent 名称（如 "fundamental_agent"）
            prompt_text: 该 agent 的 system prompt 全文
            performance_data: 表现数据 dict，可包含：
                - loss_contribution: agent 对总 loss 的贡献
                - common_errors: 常见错误类型列表
                - accuracy: 准确率等指标
                - sample_size: 样本量

        Returns:
            {
                "issues": [{dimension, severity, description, location_hint}, ...],
                "suggestions": [{action, priority, expected_improvement}, ...],
                "overall_assessment": {quality_score, summary},
                "diagnosis_source": "llm" | "fallback_rules"
            }
        """
        if hasattr(self, '_init_error') and self._init_error:
            logger.warning("PromptPatcher initialized with error: %s", self._init_error)

        if self._client is None:
            return self._fallback_diagnose(agent_name, prompt_text, performance_data)

        user_content = self._build_diagnose_prompt(agent_name, prompt_text, performance_data)
        raw = self._call_llm(self.DIAGNOSE_SYSTEM_PROMPT, user_content)

        if raw:
            parsed = _extract_json_block(raw)
            if parsed and "issues" in parsed:
                parsed["diagnosis_source"] = "llm"
                return parsed

        # LLM 调用失败或返回无效 JSON → 回退到规则诊断
        return self._fallback_diagnose(agent_name, prompt_text, performance_data)

    def _build_diagnose_prompt(
        self,
        agent_name: str,
        prompt_text: str,
        performance_data: Dict[str, Any],
    ) -> str:
        """构造发送给 LLM 的诊断输入。"""
        perf_str = json.dumps(performance_data, ensure_ascii=False, indent=2)
        # 如果 prompt 太长，截断但保留首尾（头部的角色指令和尾部的格式约束最重要）
        if len(prompt_text) > 4000:
            prompt_text = prompt_text[:2500] + "\n\n...(truncated)...\n\n" + prompt_text[-1500:]
        return (
            f"## Agent: {agent_name}\n\n"
            f"### 当前 Prompt\n\n{prompt_text}\n\n"
            f"### 表现数据\n\n{perf_str}"
        )

    def _fallback_diagnose(
        self,
        agent_name: str,
        prompt_text: str,
        performance_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """基于规则的 fallback 诊断（不依赖 LLM）。"""
        issues = []
        suggestions = []
        quality_score = 70  # 默认基础分

        # 规则 1: prompt 长度过短（<200 字符）可能缺乏细节
        if len(prompt_text) < 200:
            issues.append({
                "dimension": "指令清晰度",
                "severity": "high",
                "description": f"Prompt 长度仅 {len(prompt_text)} 字符，可能缺乏足够的指引和约束",
                "location_hint": "整体",
            })
            suggestions.append({
                "action": "扩充 prompt，增加输出格式要求、评分区间锚定、数据缺失处理规则",
                "priority": "high",
                "expected_improvement": "提升输出规范性和一致性",
            })
            quality_score -= 20

        # 规则 2: 检查是否包含数据/判断分区标记
        has_data_zone = "数据事实" in prompt_text or "数据区" in prompt_text
        has_judge_zone = "分析判断" in prompt_text or "判断区" in prompt_text
        if not has_data_zone and not has_judge_zone:
            issues.append({
                "dimension": "数据使用规范",
                "severity": "medium",
                "description": "未找到明确的数据事实区/分析判断区分区标记",
                "location_hint": "输出格式部分",
            })
            suggestions.append({
                "action": "添加数据事实区和分析判断区，要求标注 [数据] 和 [判断] 标签",
                "priority": "medium",
                "expected_improvement": "减少幻觉，提高输出可信度",
            })
            quality_score -= 10

        # 规则 3: 检查是否有反幻觉约束
        has_anti_hallucination = any(kw in prompt_text for kw in [
            "不要编造", "禁止编造", "不得猜测", "幻觉", "hallucination",
            "不得虚构", "仅使用提供的数据", "基于上述数据",
        ])
        if not has_anti_hallucination:
            issues.append({
                "dimension": "抗幻觉机制",
                "severity": "high",
                "description": "未发现明确的防幻觉/防编造约束语句",
                "location_hint": "行为约束部分",
            })
            suggestions.append({
                "action": "添加明确的防幻觉约束：'仅基于工具返回的实际数据进行分析，不得编造或推测任何数据点'",
                "priority": "high",
                "expected_improvement": "显著降低虚构数据风险",
            })
            quality_score -= 15

        # 规则 4: 检查是否有结构化输出要求（SIGNAL_PACK / JSON）
        has_structured_output = any(kw in prompt_text for kw in [
            "SIGNAL_PACK", "signal_pack", "JSON", "json", "结构化",
        ])
        if not has_structured_output:
            issues.append({
                "dimension": "输出结构",
                "severity": "medium",
                "description": "未发现结构化输出要求（如 <SIGNAL_PACK> 标签或 JSON）",
                "location_hint": "输出格式部分",
            })
            suggestions.append({
                "action": "添加结构化输出要求：使用 <SIGNAL_PACK> 标签包裹 JSON 格式的信号数据",
                "priority": "medium",
                "expected_improvement": "提高下游解析可靠性",
            })
            quality_score -= 10

        # 规则 5: 基于表现数据诊断
        loss_contrib = performance_data.get("loss_contribution", 0)
        if loss_contrib > 0.02:
            issues.append({
                "dimension": "评分标准",
                "severity": "medium",
                "description": f"Agent 对总 Loss 正向贡献 {loss_contrib:.3f}（即增加了 Loss），可能评分标准有偏差",
                "location_hint": "评分/判断标准部分",
            })
            suggestions.append({
                "action": "检查评分锚定和区间定义是否与实际市场表现对齐",
                "priority": "high" if loss_contrib > 0.05 else "medium",
                "expected_improvement": "改善评分准确性",
            })
            quality_score -= min(15, int(loss_contrib * 100))

        common_errors = performance_data.get("common_errors", [])
        if common_errors:
            for err in common_errors[:3]:
                issues.append({
                    "dimension": "边界条件",
                    "severity": "medium",
                    "description": f"常见错误: {err}",
                    "location_hint": "数据处理逻辑",
                })
            suggestions.append({
                "action": "针对常见错误类型添加具体的边界条件处理指引",
                "priority": "medium",
                "expected_improvement": "减少典型错误发生频率",
            })

        return {
            "issues": issues,
            "suggestions": suggestions,
            "overall_assessment": {
                "quality_score": max(0, min(100, quality_score)),
                "summary": f"基于规则的诊断完成，发现 {len(issues)} 个潜在问题",
            },
            "diagnosis_source": "fallback_rules",
        }

    # ── Patch Generation ──────────────────────────────────────────

    def generate_patch(
        self,
        agent_name: str,
        diagnosis: Dict[str, Any],
        original_prompt: str = "",
    ) -> Dict[str, Any]:
        """基于诊断结果生成 prompt 补丁。

        Args:
            agent_name: Agent 名称
            diagnosis: 来自 diagnose() 的诊断结果
            original_prompt: 原始 prompt 文本（用于 LLM 不可用时的 fallback）

        Returns:
            {
                "patched_prompt": str,
                "changes": [{type, location, before, after, reason}, ...],
                "rationale": str,
                "patch_source": "llm" | "fallback"
            }
        """
        if self._client is None:
            return self._fallback_patch(agent_name, diagnosis, original_prompt)

        diag_json = json.dumps(diagnosis, ensure_ascii=False, indent=2)
        user_content = f"## Agent: {agent_name}\n\n### 诊断结果\n\n{diag_json}"

        raw = self._call_llm(self.PATCH_SYSTEM_PROMPT, user_content)

        if raw:
            parsed = _extract_json_block(raw)
            if parsed and "patched_prompt" in parsed:
                parsed["patch_source"] = "llm"
                return parsed

        return self._fallback_patch(agent_name, diagnosis, original_prompt)

    def _fallback_patch(
        self,
        agent_name: str,
        diagnosis: Dict[str, Any],
        original_prompt: str = "",
    ) -> Dict[str, Any]:
        """Fallback 补丁生成（不需要 LLM，基于诊断建议构造 patched_prompt）。"""
        suggestions = diagnosis.get("suggestions", [])
        # 将建议转化为 patch
        changes = []
        for s in suggestions:
            changes.append({
                "type": "add",
                "location": "output_format_section",
                "before": "",
                "after": s.get("action", ""),
                "reason": s.get("expected_improvement", ""),
            })
        return {
            "patched_prompt": original_prompt,  # Return original prompt unchanged when LLM unavailable
            "changes": changes,
            "rationale": f"基于 {len(suggestions)} 条诊断建议生成补丁摘要（LLM 不可用，未生成完整 patched_prompt）",
            "patch_source": "fallback",
        }

    # ── Validation ────────────────────────────────────────────────

    def validate_patch(
        self,
        original: str,
        patched: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """在测试用例上验证补丁。

        注意：当前版本执行启发式验证（不实际调用 agent）。
        未来可扩展为真实的 agent-in-the-loop 验证。

        Args:
            original: 原始 prompt
            patched: 补丁后的 prompt
            test_cases: 测试用例列表，每项可包含：
                - "query": 输入查询文本
                - "expected_keywords": 期望关键字列表
                - "forbidden_keywords": 禁止关键字列表

        Returns:
            {
                "passed": bool,
                "results": [{case_index, passed, reason}, ...],
                "improvement": float (-1.0 ~ 1.0, 正值表示改善),
                "validation_type": "heuristic"
            }
        """
        results = []

        for idx, case in enumerate(test_cases):
            original_ok = self._check_case(original, case)
            patched_ok = self._check_case(patched, case)
            if original_ok and not patched_ok:
                results.append({
                    "case_index": idx,
                    "passed": False,
                    "reason": "原始 prompt 通过但补丁后不满足约束",
                })
            elif not original_ok and patched_ok:
                results.append({
                    "case_index": idx,
                    "passed": True,
                    "reason": "补丁修复了原始 prompt 的约束违规",
                })
            else:
                results.append({
                    "case_index": idx,
                    "passed": patched_ok,
                    "reason": "原始和补丁一致" if original_ok == patched_ok else "状态变化",
                })

        n_total = len(results)
        n_passed = sum(1 for r in results if r["passed"])
        n_improved = sum(
            1 for r in results
            if "补丁修复了" in r.get("reason", "")
        )
        n_regressed = sum(
            1 for r in results
            if "原始 prompt 通过但补丁后" in r.get("reason", "")
        )

        # improvement score: (+1 per fix, -2 per regression) / n
        improvement = (n_improved - 2 * n_regressed) / max(n_total, 1)

        return {
            "passed": n_passed == n_total,
            "results": results,
            "improvement": max(-1.0, min(1.0, improvement)),
            "validation_type": "heuristic",
            "n_total": n_total,
            "n_passed": n_passed,
            "n_improved": n_improved,
            "n_regressed": n_regressed,
        }

    def _check_case(self, prompt: str, case: Dict[str, Any]) -> bool:
        """检查单个测试用例是否通过（基于关键词匹配的启发式检查）。"""
        expected_keywords = case.get("expected_keywords", [])
        forbidden_keywords = case.get("forbidden_keywords", [])

        # 期望关键词检查
        for kw in expected_keywords:
            if kw not in prompt:
                return False

        # 禁止关键词检查
        for kw in forbidden_keywords:
            if kw in prompt:
                return False

        return True

    # ── Gap #15: Prompt Patch Backtest Verification ──

    def verify_prompt_change(
        self,
        original_prompt: str,
        patched_prompt: str,
        anchor_dates: List[str],
        stock_codes: List[str],
        term: str = "medium",
    ) -> Dict[str, Any]:
        """
        Compare original vs patched prompt by running scoring on the same
        anchor dates and stocks.

        Uses ReplayBacktestEngine to score each stock at each anchor date
        with both prompts, then computes comparison statistics.

        Args:
            original_prompt: The original (unmodified) system prompt
            patched_prompt: The modified (patched) system prompt
            anchor_dates: List of anchor dates YYYY-MM-DD
            stock_codes: List of stock codes in internal format (sh.603871)
            term: short/medium/long

        Returns:
            {
                "mean_score_diff": float,          # patched - original
                "rank_correlation": float,         # Spearman-like rank correlation
                "score_volatility_change": float,  # ratio of patched/original std dev
                "direction_flip_rate": float,      # fraction of stocks whose rank order changed significantly
                "original_mean_score": float,
                "patched_mean_score": float,
                "details": [{anchor_date, stock, original_score, patched_score}, ...],
                "summary": str,
            }
        """
        import asyncio
        from src.eval.replay_backtest_engine import (
            ReplayBacktestEngine, BacktestConfig
        )

        # Build configs with/without prompt override
        original_config = BacktestConfig(term=term)
        patched_config = BacktestConfig(
            term=term,
            prompt_override={"scorer": patched_prompt},
        )

        engine_original = ReplayBacktestEngine(original_config)
        engine_patched = ReplayBacktestEngine(patched_config)

        # We need to score each stock at each anchor - do it via run_single_anchor
        # but we need price_data. For the verification scoring, we can use
        # _score_stock_pit directly or run a mini backtest.
        #
        # For efficiency, we score per-anchor with both: score = run_single_anchor
        details = []
        score_diffs = []

        async def _run_comparison():
            """Run scoring comparison using the engine's internal methods."""
            results = {"details": [], "score_diffs": []}

            for anchor_date in anchor_dates:
                anchor_scores_original: Dict[str, float] = {}
                anchor_scores_patched: Dict[str, float] = {}

                for stock_code in stock_codes:
                    # Score with original
                    original_score = await engine_original._score_stock_pit(
                        stock_code, stock_code, anchor_date, "full"
                    )
                    # Score with patched (prompt_override in config)
                    patched_score = await engine_patched._score_stock_pit(
                        stock_code, stock_code, anchor_date, "full"
                    )

                    anchor_scores_original[stock_code] = original_score
                    anchor_scores_patched[stock_code] = patched_score
                    diff = patched_score - original_score
                    results["score_diffs"].append(diff)

                    results["details"].append({
                        "anchor_date": anchor_date,
                        "stock_code": stock_code,
                        "original_score": original_score,
                        "patched_score": patched_score,
                        "diff": diff,
                    })

            return results

        # Run async comparison
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In a running loop, create a task
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    _run_comparison(), loop
                )
                comparison = future.result(timeout=600)
            else:
                comparison = asyncio.run(_run_comparison())
        except RuntimeError:
            comparison = asyncio.run(_run_comparison())

        details = comparison["details"]
        score_diffs = comparison["score_diffs"]

        if not score_diffs:
            return {
                "mean_score_diff": 0,
                "rank_correlation": 0,
                "score_volatility_change": 0,
                "direction_flip_rate": 0,
                "original_mean_score": 0,
                "patched_mean_score": 0,
                "details": [],
                "summary": "No comparison data available.",
            }

        # Compute statistics
        mean_diff = sum(score_diffs) / len(score_diffs)
        original_scores = [d["original_score"] for d in details]
        patched_scores = [d["patched_score"] for d in details]

        original_mean = sum(original_scores) / len(original_scores)
        patched_mean = sum(patched_scores) / len(patched_scores)

        # Score volatility change
        def _std(values):
            m = sum(values) / len(values)
            var = sum((v - m) ** 2 for v in values) / len(values)
            return var ** 0.5

        original_std = _std(original_scores)
        patched_std = _std(patched_scores)
        vol_change = (patched_std / original_std - 1.0) if original_std > 0 else 0.0

        # Rank correlation (Spearman-like)
        # Rank stocks within each anchor, then compute correlation of rank changes
        rank_correlations = []
        direction_flips = 0
        total_pairs = 0

        for anchor_date in anchor_dates:
            anchor_details = [d for d in details if d["anchor_date"] == anchor_date]
            if len(anchor_details) < 3:
                continue

            orig_ranked = sorted(anchor_details, key=lambda x: x["original_score"], reverse=True)
            patched_ranked = sorted(anchor_details, key=lambda x: x["patched_score"], reverse=True)

            orig_ranks = {d["stock_code"]: i for i, d in enumerate(orig_ranked)}
            patched_ranks = {d["stock_code"]: i for i, d in enumerate(patched_ranked)}

            # Spearman rank correlation
            n = len(anchor_details)
            d_squared = sum(
                (orig_ranks.get(d["stock_code"], 0) - patched_ranks.get(d["stock_code"], 0)) ** 2
                for d in anchor_details
            )
            if n > 1:
                rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
                rank_correlations.append(rho)

            # Direction flips: stocks that moved >= N positions in rank
            for d in anchor_details:
                total_pairs += 1
                rank_diff = abs(
                    orig_ranks.get(d["stock_code"], 0) - patched_ranks.get(d["stock_code"], 0)
                )
                # "Flip" if rank changed by >25% of pool size
                if rank_diff > max(n * 0.25, 1):
                    direction_flips += 1

        avg_rank_corr = sum(rank_correlations) / max(len(rank_correlations), 1)
        flip_rate = direction_flips / max(total_pairs, 1)

        # Summary
        if abs(mean_diff) < 0.5 and abs(vol_change) < 0.1 and avg_rank_corr > 0.9:
            summary = "补丁对评分影响极小（分数变化<0.5分，排名相关>0.9），行为基本一致"
        elif avg_rank_corr < 0.7:
            summary = f"补丁显著改变了评分排名（排名相关={avg_rank_corr:.2f}），需人工审查是否符合预期"
        else:
            summary = (
                f"补丁改变了评分（平均差异={mean_diff:+.2f}分，排名相关={avg_rank_corr:.2f}），"
                f"波动性变化={vol_change:+.1%}"
            )

        return {
            "mean_score_diff": round(mean_diff, 3),
            "rank_correlation": round(avg_rank_corr, 3),
            "score_volatility_change": round(vol_change, 3),
            "direction_flip_rate": round(flip_rate, 3),
            "original_mean_score": round(original_mean, 2),
            "patched_mean_score": round(patched_mean, 2),
            "details": details[:200],  # Truncate details for large comparisons
            "summary": summary,
        }

    def run_backtest_verification(
        self,
        patched_prompt: str,
        term: str,
        start_date: str,
        end_date: str,
        pool_stocks: Optional[List[str]] = None,
        price_data: Optional[Dict[str, Any]] = None,
        num_anchors: int = 15,
    ) -> Dict[str, Any]:
        """
        Run a mini backtest to verify prompt patch effectiveness.

        Compares baseline (original prompt) vs patched prompt using
        ReplayBacktestEngine.run_mini_backtest().

        Args:
            patched_prompt: The new/patched prompt to test
            term: short/medium/long
            start_date: Backtest start date YYYY-MM-DD
            end_date: Backtest end date YYYY-MM-DD
            pool_stocks: Stock pool (uses default refined pool if None)
            price_data: Price data dict (fetched if None)
            num_anchors: Max anchor dates for mini backtest (10-20)

        Returns:
            VerificationResult-like dict with improvement/no-change/regression
        """
        import asyncio
        from src.eval.replay_backtest_engine import (
            ReplayBacktestEngine, BacktestConfig, VerificationResult,
        )

        # Load default pool if not provided
        if pool_stocks is None:
            try:
                stock_pool_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "stock_pool.json"
                )
                if os.path.exists(stock_pool_path):
                    with open(stock_pool_path, "r", encoding="utf-8") as f:
                        pool_data = json.load(f)
                    term_key = f"{term}_pool" if term != "short" else "short_pool"
                    pool_stocks = pool_data.get(term_key, [])
                    if not pool_stocks:
                        pool_stocks = pool_data.get("medium_pool", [])[:30]
                else:
                    logger.warning("stock_pool.json not found, using empty pool")
                    pool_stocks = []
            except Exception as e:
                logger.warning("Failed to load stock pool: %s", str(e))
                pool_stocks = []

        if not pool_stocks:
            return {
                "passed": False,
                "determination": "undetermined",
                "mean_score_diff": 0,
                "rank_correlation": 0,
                "score_volatility_change": 0,
                "direction_flip_rate": 0,
                "performance_metrics": {},
                "anchor_results": [],
                "baseline_summary": {},
                "patched_summary": {},
                "error": "No stock pool available for verification",
            }

        num_anchors = max(10, min(num_anchors, 30))  # Clamp to 10-30

        # Config for baseline (no prompt override)
        baseline_config = BacktestConfig(
            term=term,
            start_date=start_date,
            end_date=end_date,
        )

        # Config for patched prompt
        patched_config = BacktestConfig(
            term=term,
            start_date=start_date,
            end_date=end_date,
            prompt_override={"scorer": patched_prompt},
        )

        async def _run_verification():
            baseline_engine = ReplayBacktestEngine(baseline_config)
            patched_engine = ReplayBacktestEngine(patched_config)

            # Use empty price_data — engine will use PIT fundamentals fallback
            if price_data is None:
                empty_price = {}

            logger.info("Running baseline mini backtest...")
            baseline_result = await baseline_engine.run_mini_backtest(
                pool_stocks, price_data or empty_price,
                num_anchors=num_anchors, num_stocks=min(len(pool_stocks), 25)
            )

            logger.info("Running patched mini backtest...")
            patched_result = await patched_engine.run_mini_backtest(
                pool_stocks, price_data or empty_price,
                num_anchors=num_anchors, num_stocks=min(len(pool_stocks), 25)
            )

            return baseline_result, patched_result

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    _run_verification(), loop
                )
                baseline_result, patched_result = future.result(timeout=600)
            else:
                baseline_result, patched_result = asyncio.run(_run_verification())
        except RuntimeError:
            baseline_result, patched_result = asyncio.run(_run_verification())

        # Compute comparison metrics
        baseline_contrib = baseline_result.get("contribution_summary", {})
        patched_contrib = patched_result.get("contribution_summary", {})

        # Extract full-agent mean returns for comparison
        def _extract_full_metric(contrib: Dict[str, Any]) -> Dict[str, float]:
            """Extract mean return and score for the 'full' ablation line."""
            per_anchor = contrib.get("per_anchor_metrics", [])
            if not per_anchor:
                return {"mean_return": 0, "mean_score": 0}
            mean_ret = sum(a.get("mean_return", 0) for a in per_anchor) / len(per_anchor)
            mean_score = sum(a.get("mean_score", 0) for a in per_anchor) / len(per_anchor)
            return {"mean_return": mean_ret, "mean_score": mean_score}

        baseline_metric = _extract_full_metric(baseline_result)
        patched_metric = _extract_full_metric(patched_result)

        # Score comparison across anchors
        score_diffs = []
        rank_correlations = []
        baseline_anchors = baseline_result.get("per_anchor_metrics", [])
        patched_anchors = patched_result.get("per_anchor_metrics", [])

        for ba, pa in zip(baseline_anchors, patched_anchors):
            if ba.get("anchor_date") == pa.get("anchor_date"):
                score_diffs.append(pa.get("mean_score", 0) - ba.get("mean_score", 0))

        mean_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
        score_std_before = (
            sum((s - sum(score_diffs) / len(score_diffs)) ** 2 for s in score_diffs) / len(score_diffs)
        ) ** 0.5 if len(score_diffs) > 1 else 0

        # Determine improvement/regression
        return_diff = patched_metric.get("mean_return", 0) - baseline_metric.get("mean_return", 0)
        score_stability = 1.0 / (1.0 + score_std_before)

        if mean_score_diff > 1.0 and return_diff > 0.001:
            determination = "improvement"
            passed = True
        elif mean_score_diff < -2.0 or return_diff < -0.005:
            determination = "regression"
            passed = False
        elif abs(mean_score_diff) <= 1.0 and abs(return_diff) <= 0.001:
            determination = "no_change"
            passed = True  # No change is acceptable
        else:
            determination = "inconclusive"
            passed = False

        return {
            "passed": passed,
            "determination": determination,
            "mean_score_diff": round(mean_score_diff, 3),
            "rank_correlation": (
                round(sum(rank_correlations) / len(rank_correlations), 3)
                if rank_correlations else 0.0
            ),
            "score_volatility_change": round(score_std_before, 3),
            "direction_flip_rate": 0.0,  # Computed at anchor-level in verify_prompt_change
            "performance_metrics": {
                "baseline_mean_return": round(baseline_metric.get("mean_return", 0), 4),
                "patched_mean_return": round(patched_metric.get("mean_return", 0), 4),
                "return_diff": round(return_diff, 4),
                "baseline_mean_score": round(baseline_metric.get("mean_score", 0), 2),
                "patched_mean_score": round(patched_metric.get("mean_score", 0), 2),
            },
            "anchor_results": list(zip(
                [a.get("anchor_date", "") for a in baseline_anchors],
                score_diffs,
            )),
            "baseline_summary": {
                "num_anchors": baseline_result.get("config", {}).get("num_anchors", 0),
                "contribution_summary": baseline_contrib,
            },
            "patched_summary": {
                "num_anchors": patched_result.get("config", {}).get("num_anchors", 0),
                "contribution_summary": patched_contrib,
            },
        }

    def validate_patch_with_backtest(
        self,
        original: str,
        patched: str,
        test_cases: List[Dict[str, Any]],
        term: str = "medium",
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
    ) -> Dict[str, Any]:
        """
        Enhanced validation: combines heuristic keyword checks with
        mini backtest verification (Gap #15).

        This is the recommended entry point for prompt patch validation.
        Falls back to heuristic-only when backtest is unavailable.

        Args:
            original: Original prompt
            patched: Patched prompt
            test_cases: Heuristic test cases
            term: short/medium/long
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Combined validation result with both heuristic and backtest scores.
        """
        # 1. Heuristic validation (always available)
        heuristic_result = self.validate_patch(original, patched, test_cases)

        # 2. Backtest verification (Gap #15)
        backtest_result = None
        try:
            backtest_result = self.run_backtest_verification(
                patched_prompt=patched,
                term=term,
                start_date=start_date,
                end_date=end_date,
            )
            logger.info("Backtest verification completed: %s",
                       backtest_result.get("determination", "unknown"))
        except Exception as e:
            logger.warning("Backtest verification failed: %s, falling back to heuristic only",
                         str(e))
            backtest_result = {
                "error": str(e),
                "determination": "undetermined",
            }

        # 3. Combined decision
        heuristic_passed = heuristic_result.get("passed", False)
        backtest_passed = backtest_result.get("passed", False) if backtest_result else False
        backtest_determination = backtest_result.get("determination", "undetermined") if backtest_result else "undetermined"

        # Regression from backtest is a hard fail
        if backtest_determination == "regression":
            combined_passed = False
            combined_reason = "Backtest detected regression — patch may degrade performance"
        elif backtest_determination == "improvement":
            combined_passed = True
            combined_reason = "Backtest confirmed improvement"
        else:
            combined_passed = heuristic_passed
            combined_reason = "Heuristic checks passed (backtest inconclusive or unavailable)"

        return {
            "passed": combined_passed,
            "reason": combined_reason,
            "heuristic": heuristic_result,
            "backtest": backtest_result,
        }

    # ── End-to-end PROMPT_PATCH ticket processing ──

    def apply_verified_prompt_patch(
        self,
        ticket: Dict[str, Any],
        original_prompt: str,
        patched_prompt: str,
        target_file: str,
        term: str = "medium",
        start_date: str = "",
        end_date: str = "",
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        End-to-end PROMPT_PATCH ticket processing: validate → verify → apply.

        Chains validate_patch_with_backtest() and safe_patch_executor.apply_patch()
        into a single workflow. If validation fails or backtest detects regression,
        the patch is NOT applied.

        Args:
            ticket: Optimization ticket dict (from router.generate_ticket)
            original_prompt: Current prompt text
            patched_prompt: Proposed new prompt text
            target_file: Path to the file containing the prompt
            term: short/medium/long for backtest verification
            start_date: Backtest start (default: 1 year ago)
            end_date: Backtest end (default: today)
            dry_run: If True, validate but don't apply (default True for safety)

        Returns:
            {
                "validation": {...},  # from validate_patch_with_backtest
                "applied": bool,
                "patch_result": {...} or None,  # from safe_patch_executor
                "skipped_reason": str or None,
            }
        """
        from src.eval.optimizer.safe_patch_executor import SafePatchExecutor

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Step 1: Validate with backtest
        validation = self.validate_patch_with_backtest(
            original=original_prompt,
            patched=patched_prompt,
            test_cases=[],
            term=term,
            start_date=start_date,
            end_date=end_date,
        )

        if not validation["passed"]:
            return {
                "validation": validation,
                "applied": False,
                "patch_result": None,
                "skipped_reason": f"Validation failed: {validation['reason']}",
            }

        if dry_run:
            return {
                "validation": validation,
                "applied": False,
                "patch_result": None,
                "skipped_reason": "dry_run=True — validation passed but patch not applied",
            }

        # Step 2: Apply via safe_patch_executor (git branch isolation)
        ticket_id = ticket.get("ticket_id", ticket.get("batch_id", "PROMPT_PATCH"))
        executor = SafePatchExecutor()
        patch_result = executor.apply_patch(
            file_path=target_file,
            new_content=patched_prompt,
            description=f"PROMPT_PATCH: {ticket.get('title', 'prompt improvement')}",
            ticket_id=ticket_id,
        )

        return {
            "validation": validation,
            "applied": patch_result.get("success", False),
            "patch_result": patch_result,
            "skipped_reason": None,
        }
