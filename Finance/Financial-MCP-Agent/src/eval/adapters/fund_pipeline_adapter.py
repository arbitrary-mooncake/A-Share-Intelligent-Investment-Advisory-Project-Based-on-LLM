"""
基金管线适配器 -- 总纲 §17.1
将评测系统的信号包和评分适配到基金分析管道。

参考 stock_pipeline_adapter 的设计模式：
  - 非侵入式包装现有基金分析管线
  - 通过直接调用基金 agent + merge node + scoring agent 完成分析
  - 为评测系统提供统一的 DecisionPack 输出格式
  - 支持分析日期注入和模型覆盖
  - 支持缓存和错误处理

基金管线工作流（fund_main.py）:
  start_node → [7 fund agents parallel] → fund_merge_node → fund_scoring_agent → END
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
import asyncio
import logging

logger = logging.getLogger(__name__)

# ── Fund agent keys (match fund_main.py and fund_merge_node.py) ──
FUND_AGENT_KEYS = [
    "fund_product_doc",
    "fund_perf_risk",
    "fund_holdings",
    "fund_manager",
    "fund_benchmark",
    "fund_fee",
    "fund_event",
]

FUND_AGENT_FUNCTIONS = {
    "fund_product_doc": None,
    "fund_perf_risk": None,
    "fund_holdings": None,
    "fund_manager": None,
    "fund_benchmark": None,
    "fund_fee": None,
    "fund_event": None,
}

# ── Cache TTL (days) matches A-stock agent cadence ──
FUND_CACHE_TTL_DAYS = {
    "fund_product_doc": 7,
    "fund_perf_risk": 7,
    "fund_holdings": 15,
    "fund_manager": 7,
    "fund_benchmark": 7,
    "fund_fee": 7,
    "fund_event": 1,
}


class FundPipelineAdapter:
    """基金分析管线适配器 -- 连接评测系统和基金分析管道。

    提供与 stock_pipeline_adapter 一致的接口，使评测系统可以
    统一处理 A 股和基金两种资产类型。

    用法:
        adapter = FundPipelineAdapter()
        result = await adapter.run_fund_analysis("sh.510050", "华夏上证50ETF")
        score = adapter.get_fund_score("sh.510050", term="medium")
    """

    def __init__(self, model_override: Optional[Dict[str, str]] = None):
        """初始化适配器。

        Args:
            model_override: 可选模型覆盖，格式与 stock_pipeline_adapter 一致
                            {"model_name": ..., "model_api_key": ..., "model_base_url": ...}
        """
        self.model_override = model_override or {}
        self._agent_funcs_loaded = False

    # ── Lazy import of fund agent functions ──────────────────────────────────

    def _ensure_agent_funcs(self):
        """延迟加载基金 agent 函数，避免启动时循环导入。"""
        if self._agent_funcs_loaded:
            return

        from src.agents.fund_product_doc_agent import fund_product_doc_agent
        from src.agents.fund_perf_risk_agent import fund_perf_risk_agent
        from src.agents.fund_holdings_agent import fund_holdings_analysis
        from src.agents.fund_manager_agent import fund_manager_agent
        from src.agents.fund_benchmark_agent import fund_benchmark_agent
        from src.agents.fund_fee_agent import fund_fee_agent
        from src.agents.fund_event_agent import fund_event_agent

        FUND_AGENT_FUNCTIONS["fund_product_doc"] = fund_product_doc_agent
        FUND_AGENT_FUNCTIONS["fund_perf_risk"] = fund_perf_risk_agent
        FUND_AGENT_FUNCTIONS["fund_holdings"] = fund_holdings_analysis
        FUND_AGENT_FUNCTIONS["fund_manager"] = fund_manager_agent
        FUND_AGENT_FUNCTIONS["fund_benchmark"] = fund_benchmark_agent
        FUND_AGENT_FUNCTIONS["fund_fee"] = fund_fee_agent
        FUND_AGENT_FUNCTIONS["fund_event"] = fund_event_agent

        self._agent_funcs_loaded = True

    # ── Core: Run Fund Analysis ──────────────────────────────────────────────

    async def run_fund_analysis(
        self,
        fund_code: str,
        fund_name: str = "",
        as_of_date: str = "",
        eval_mode: bool = True,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """运行完整基金分析管线并返回结构化结果。

        管线流程:
          1. 并行运行 7 个基金分析 agent
          2. merge node 合并为 fund_analysis_package
          3. fund_scoring_agent 打分
          4. 返回结构化的 score + signal_packs + analysis_texts

        Args:
            fund_code: 基金代码 (e.g. "sh.510050")
            fund_name: 基金名称
            as_of_date: 评测时点（YYYY-MM-DD格式），空串则用当前时间
            eval_mode: 是否评测模式
            skip_cache: 是否跳过缓存

        Returns:
            {
                "fund_code": ...,
                "fund_name": ...,
                "as_of_date": ...,
                "fund_score": {...},          # 评分结果 dict
                "fund_decision": DecisionPack, # 结构化决策
                "signal_packs": {...},         # {agent_key: signal_pack_dict}
                "analysis_texts": {...},       # {agent_key: analysis_text}
                "fund_analysis_package": {...}, # 合并分析包
                "execution_time": float,
                "error": str or None,
            }
        """
        self._ensure_agent_funcs()

        as_of_date = as_of_date or datetime.now().strftime("%Y-%m-%d")

        result = {
            "fund_code": fund_code,
            "fund_name": fund_name,
            "as_of_date": as_of_date,
            "fund_score": {},
            "fund_decision": None,
            "signal_packs": {},
            "analysis_texts": {},
            "fund_analysis_package": {},
            "execution_time": 0.0,
            "error": None,
        }

        start_time = datetime.now()

        try:
            from src.utils.state_definition import AgentState

            # ---- Step 1: 构建初始状态 ----
            initial_state = AgentState(
                messages=[],
                data={
                    "fund_code": fund_code,
                    "fund_name": fund_name,
                    "current_date": as_of_date,
                    "fund_intent": "score",
                },
                metadata={},
            )

            # ---- Step 2: 并行运行 7 个基金 analysis agent ----
            agent_tasks = {}
            for agent_key in FUND_AGENT_KEYS:
                func = FUND_AGENT_FUNCTIONS.get(agent_key)
                if func:
                    agent_tasks[agent_key] = asyncio.create_task(
                        self._run_agent_safe(func, initial_state, agent_key)
                    )

            agent_results = {}
            for agent_key, task in agent_tasks.items():
                try:
                    agent_results[agent_key] = await task
                except Exception as e:
                    logger.warning(f"基金 agent {agent_key} 执行失败: {e}")
                    agent_results[agent_key] = {}

            # ---- Step 3: 收集 agent 输出到 state.data ----
            merged_data = {
                "fund_code": fund_code,
                "fund_name": fund_name,
                "current_date": as_of_date,
                "fund_intent": "score",
            }

            for agent_key in FUND_AGENT_KEYS:
                agent_data = agent_results.get(agent_key, {})
                if agent_data and isinstance(agent_data, dict):
                    inner = agent_data.get("data", agent_data)
                    # 提取分析文本
                    text = inner.get("analysis", "") or inner.get(f"{agent_key}_analysis", "") or ""
                    if not text:
                        # Agent 可能直接返回了字符串
                        for v in inner.values():
                            if isinstance(v, str) and len(v) > 100:
                                text = str(v)
                                break
                    merged_data[agent_key] = text
                    result["analysis_texts"][agent_key] = text
                    # 提取 signal_pack
                    sp = inner.get(f"{agent_key}_signal_pack") or inner.get("signal_pack")
                    if sp and isinstance(sp, dict):
                        result["signal_packs"][agent_key] = sp

            # ---- Step 4: 构建 merge state 并调用 fund_merge_node ----
            merge_state = AgentState(
                messages=[],
                data=merged_data,
                metadata={},
            )

            from src.agents.fund_merge_node import fund_merge_node
            merge_result = await fund_merge_node(merge_state)

            fund_analysis_package = (
                merge_result.get("data", {}).get("fund_analysis_package", {})
                if isinstance(merge_result, dict) else {}
            )
            result["fund_analysis_package"] = fund_analysis_package

            # ---- Step 5: 确定基金类型 ----
            fund_type = "ETF"
            if fund_analysis_package:
                profile = fund_analysis_package.get("fund_profile", {})
                fund_type = profile.get("fund_type", "") or "ETF"

            # ---- Step 6: 调用 fund_scoring_agent ----
            from src.agents.fund_scoring_agent import fund_scoring_agent
            score_result = await fund_scoring_agent(
                fund_analysis_package=fund_analysis_package,
                fund_code=fund_code,
                fund_name=fund_name,
                fund_type=fund_type,
                current_date=as_of_date,
            )
            result["fund_score"] = score_result if isinstance(score_result, dict) else {}

            # ---- Step 7: 构建 DecisionPack ----
            result["fund_decision"] = _build_fund_decision_pack(
                fund_code=fund_code,
                fund_name=fund_name,
                as_of_date=as_of_date,
                score_data=score_result if isinstance(score_result, dict) else {},
                eval_mode=eval_mode,
            )

            # ---- Step 8: 缓存结果供后续 get_fund_score() 查询 ----
            self._cache_result(fund_code, result)

        except Exception as e:
            logger.error(f"基金分析管线执行失败 ({fund_code}): {e}", exc_info=True)
            result["error"] = str(e)

        result["execution_time"] = (datetime.now() - start_time).total_seconds()
        return result

    async def _run_agent_safe(
        self, func, state: 'AgentState', agent_key: str
    ) -> Dict[str, Any]:
        """安全运行单个基金 agent，捕获异常并记录。

        Args:
            func: agent 函数 (接受 state，返回 dict)
            state: AgentState
            agent_key: agent key (用于日志)

        Returns:
            agent 返回的 dict（可能为 {data: {agent_key: "...", ...}}）
        """
        try:
            result = await func(state)
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.warning(f"{agent_key} 执行异常: {e}")
            return {}

    # ── Get Fund Score ───────────────────────────────────────────────────────

    def get_fund_score(
        self, fund_code: str, term: str = "medium"
    ) -> float:
        """从缓存的分析结果中提取基金评分。

        Note: 基金评分不像 A 股那样有 short/medium/long 三个期限，
        只有一个综合评分。term 参数仅用于接口兼容性。

        Args:
            fund_code: 基金代码
            term: 期限标签（兼容性参数，基金评分不区分期限）

        Returns:
            评分 float (0-100)，无数据时返回 50.0
        """
        return self._cached_scores.get(fund_code, 50.0)

    # ── Get Fund Signal Packs ────────────────────────────────────────────────

    def get_fund_signal_packs(
        self, fund_code: str
    ) -> Dict[str, Dict[str, Any]]:
        """获取基金所有 agent 的 signal_pack。

        Args:
            fund_code: 基金代码

        Returns:
            {agent_key: signal_pack_dict}
        """
        return self._cached_signal_packs.get(fund_code, {})

    # ── Adapt for Backtest ───────────────────────────────────────────────────

    async def adapt_for_backtest(
        self,
        fund_code: str,
        anchor_date: str,
        fund_name: str = "",
    ) -> Dict[str, Any]:
        """准备 PIT（Point-In-Time）感知的回测数据。

        使用指定的历史锚点日期运行完整分析管线，
        确保不使用未来信息。

        Args:
            fund_code: 基金代码
            anchor_date: 历史锚点日期 (YYYY-MM-DD)
            fund_name: 基金名称

        Returns:
            与 run_fund_analysis 相同格式的结果 dict
        """
        return await self.run_fund_analysis(
            fund_code=fund_code,
            fund_name=fund_name,
            as_of_date=anchor_date,
            eval_mode=True,
            skip_cache=False,
        )

    # ── Adapt for Eval ───────────────────────────────────────────────────────

    def adapt_for_eval(
        self, fund_code: str, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """将基金分析结果格式化为评测系统期望的统一格式。

        格式对齐 stock_pipeline_adapter 的 DecisionPack 接口，
        使评测系统可以用同一套逻辑处理 A 股和基金。

        Args:
            fund_code: 基金代码
            analysis_result: run_fund_analysis() 的返回值

        Returns:
            {
                "asset_type": "fund",
                "symbol": ...,
                "name": ...,
                "short_term_score": None,
                "medium_term_score": ...,   # 基金评分映射到 medium
                "long_term_score": None,
                "short_term_decision": None,
                "medium_term_decision": DecisionPack,
                "long_term_decision": None,
                "signal_packs": ...,
                "analysis_texts": ...,
                "execution_time": ...,
                "error": ...,
            }
        """
        # 基金评分映射到 medium_term（评测系统的主要参考期限）
        fund_score = analysis_result.get("fund_score", {})
        fund_decision = analysis_result.get("fund_decision")

        return {
            "asset_type": "fund",
            "symbol": fund_code,
            "name": analysis_result.get("fund_name", ""),
            "as_of_date": analysis_result.get("as_of_date", ""),
            "short_term_score": None,
            "medium_term_score": fund_score,
            "long_term_score": None,
            "short_term_decision": None,
            "medium_term_decision": fund_decision,
            "long_term_decision": None,
            "signal_packs": analysis_result.get("signal_packs", {}),
            "analysis_texts": analysis_result.get("analysis_texts", {}),
            "fund_analysis_package": analysis_result.get("fund_analysis_package", {}),
            "execution_time": analysis_result.get("execution_time", 0.0),
            "error": analysis_result.get("error"),
        }

    # ── Validate Fund Result ─────────────────────────────────────────────────

    def validate_fund_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证基金分析结果的结构完整性和数据质量。

        检查项:
          - 必要字段是否存在
          - 评分是否在有效范围 (0-100)
          - 是否有足够多的 agent 输出
          - signal_packs 是否包含必要字段

        Args:
            result: run_fund_analysis() 的返回值

        Returns:
            {
                "valid": bool,
                "issues": [str],      # 问题列表
                "warnings": [str],    # 警告列表
                "agent_coverage": int, # 成功执行的 agent 数量
                "data_quality_score": float,  # 0-1
            }
        """
        issues = []
        warnings = []
        agent_coverage = 0

        # 检查必要字段
        if not result:
            issues.append("result 为空")
            return {"valid": False, "issues": issues, "warnings": warnings,
                    "agent_coverage": 0, "data_quality_score": 0.0}

        # 检查 fund_code
        if not result.get("fund_code"):
            issues.append("缺少 fund_code")

        # 检查 execution_time
        if result.get("execution_time", 0) > 600:
            warnings.append(f"执行时间过长: {result['execution_time']:.0f}s")

        # 检查 agent 覆盖
        analysis_texts = result.get("analysis_texts", {})
        for agent_key in FUND_AGENT_KEYS:
            text = analysis_texts.get(agent_key, "")
            if text and len(str(text)) > 100:
                agent_coverage += 1

        data_quality = agent_coverage / len(FUND_AGENT_KEYS) if FUND_AGENT_KEYS else 0.0

        if agent_coverage < 3:
            issues.append(f"agent 覆盖率过低: {agent_coverage}/{len(FUND_AGENT_KEYS)}")
            data_quality = max(0.1, data_quality)  # at least flag as "very low"

        if agent_coverage < 5:
            warnings.append(f"agent 覆盖率偏低: {agent_coverage}/{len(FUND_AGENT_KEYS)}")

        # 检查评分
        fund_score = result.get("fund_score", {})
        if fund_score:
            overall = fund_score.get("overall_score", {})
            score_val = overall.get("score") if isinstance(overall, dict) else None
            if score_val is not None:
                try:
                    s = float(score_val)
                    if not (0 <= s <= 100):
                        issues.append(f"评分超出范围: {s}")
                except (ValueError, TypeError):
                    issues.append(f"评分不可解析: {score_val}")
        else:
            warnings.append("fund_score 为空")

        # 检查 signal_packs
        signal_packs = result.get("signal_packs", {})
        if not signal_packs:
            warnings.append("signal_packs 为空")
        else:
            for agent_key, sp in signal_packs.items():
                if isinstance(sp, dict):
                    if "bias" not in sp:
                        warnings.append(f"{agent_key} signal_pack 缺少 bias")
                    if "confidence" not in sp:
                        warnings.append(f"{agent_key} signal_pack 缺少 confidence")

        # 检查 fund_analysis_package
        fund_pkg = result.get("fund_analysis_package", {})
        if not fund_pkg:
            warnings.append("fund_analysis_package 为空")
        else:
            if not fund_pkg.get("normalized_subscores"):
                warnings.append("fund_analysis_package 缺少 normalized_subscores")
            if fund_pkg.get("merge_error"):
                issues.append(f"merge 错误: {fund_pkg['merge_error']}")

        valid = len(issues) == 0 and result.get("error") is None

        return {
            "valid": valid,
            "issues": issues,
            "warnings": warnings,
            "agent_coverage": agent_coverage,
            "data_quality_score": round(data_quality, 2),
        }

    # ── Compute Fund Contribution ────────────────────────────────────────────

    def compute_fund_contribution(
        self,
        fund_scores: Dict[str, float],
        benchmark_returns: List[float],
        term: str = "medium",
    ) -> Dict[str, Any]:
        """计算基金 agent 对组合表现的贡献。

        这是一个简化版的贡献计算，用于基金评测场景。
        完整的消融分析由 contribution_engine 处理。

        Args:
            fund_scores: {fund_code: score}
            benchmark_returns: 基准收益序列
            term: 期限标签

        Returns:
            {
                "term": str,
                "fund_count": int,
                "avg_score": float,
                "score_benchmark_correlation": float,
                "positive_contribution_count": int,
                "negative_contribution_count": int,
            }
        """
        from src.eval.loss_engine import spearman_rank_correlation

        n_funds = len(fund_scores)
        if n_funds == 0:
            return {
                "term": term,
                "fund_count": 0,
                "avg_score": 0.0,
                "score_benchmark_correlation": 0.0,
                "positive_contribution_count": 0,
                "negative_contribution_count": 0,
            }

        scores = list(fund_scores.values())
        avg_score = sum(scores) / n_funds if n_funds > 0 else 0.0

        # Score-benchmark correlation (if data available)
        corr = 0.0
        if len(benchmark_returns) >= 2:
            # Use first n scores vs benchmark returns
            n = min(len(scores), len(benchmark_returns))
            if n >= 2:
                corr = spearman_rank_correlation(scores[:n], benchmark_returns[:n])

        # Simple contribution classification
        positive = sum(1 for s in scores if s >= 60)
        negative = n_funds - positive

        return {
            "term": term,
            "fund_count": n_funds,
            "avg_score": round(avg_score, 2),
            "score_benchmark_correlation": round(corr, 4),
            "positive_contribution_count": positive,
            "negative_contribution_count": negative,
        }

    # ── Internal cache ────────────────────────────────────────────────────────

    @property
    def _cached_scores(self) -> Dict[str, float]:
        """返回最近分析结果的评分缓存"""
        if not hasattr(self, '_score_cache'):
            self._score_cache: Dict[str, float] = {}
        return self._score_cache

    @property
    def _cached_signal_packs(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """返回最近分析结果的 signal_pack 缓存"""
        if not hasattr(self, '_signal_pack_cache'):
            self._signal_pack_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
        return self._signal_pack_cache

    def _cache_result(self, fund_code: str, result: Dict[str, Any]):
        """缓存分析结果到实例内存。"""
        fund_score = result.get("fund_score", {})
        overall = fund_score.get("overall_score", {}) if isinstance(fund_score, dict) else {}
        score_val = overall.get("score", 50)
        try:
            self._cached_scores[fund_code] = float(score_val)
        except (ValueError, TypeError):
            self._cached_scores[fund_code] = 50.0

        signal_packs = result.get("signal_packs", {})
        if signal_packs:
            self._cached_signal_packs[fund_code] = signal_packs


# ── Helper: Build Fund DecisionPack ───────────────────────────────────────


def _build_fund_decision_pack(
    fund_code: str,
    fund_name: str,
    as_of_date: str,
    score_data: Dict[str, Any],
    eval_mode: bool = True,
) -> 'DecisionPack':
    """从基金评分 JSON 输出构建 DecisionPack。

    基金评分使用与 A-share 相同的 DecisionPack schema，
    确保评测系统统一处理。

    Args:
        fund_code: 基金代码
        fund_name: 基金名称
        as_of_date: 评测时点
        score_data: fund_scoring_agent 的输出
        eval_mode: 是否评测模式
    """
    from src.utils.analysis_schema import DecisionPack

    if not score_data or not isinstance(score_data, dict):
        return DecisionPack(
            asset_type="fund",
            symbol=fund_code,
            name=fund_name,
            term="medium",
            as_of_date=as_of_date,
            action="hold",
            model_profile="eval_analysis" if eval_mode else "production",
        )

    # 提取评分
    overall = score_data.get("overall_score", {})
    if not isinstance(overall, dict):
        overall = {}

    try:
        score = float(overall.get("score", 50))
    except (ValueError, TypeError):
        score = 50.0

    # 提取投资观点并映射为 action
    investment_view = overall.get("investment_view", "")
    action_map = {
        "可重点关注": "strong_buy",
        "可关注": "buy",
        "谨慎关注": "cautious_buy",
        "暂不建议配置": "sell",
    }
    action = action_map.get(investment_view, "hold")

    # 提取置信度
    score_meta = score_data.get("score_meta", {})
    if not isinstance(score_meta, dict):
        score_meta = {}
    try:
        confidence = float(overall.get("confidence", 0.7))
    except (ValueError, TypeError):
        confidence = 0.7

    # 数据质量：基于 subscores 中非默认值的比例
    subscores = score_data.get("subscores", {})
    if not isinstance(subscores, dict):
        subscores = {}
    valid_subs = sum(1 for v in subscores.values() if isinstance(v, (int, float)) and v > 0)
    data_quality = valid_subs / max(len(subscores), 1) if subscores else 0.5

    # 提取关键信号
    highlights = score_data.get("highlights", {})
    if not isinstance(highlights, dict):
        highlights = {}
    strengths = highlights.get("strengths", []) or []
    risks = highlights.get("risks", []) or []

    # 提取评分解释
    explanation = score_data.get("score_explanation", {})
    if not isinstance(explanation, dict):
        explanation = {}

    # 持有期建议
    holding = score_data.get("holding_period_suggestion", {})
    if not isinstance(holding, dict):
        holding = {}

    # 评级标签
    rating_label = overall.get("rating_label", "")

    return DecisionPack(
        asset_type="fund",
        symbol=fund_code,
        name=fund_name,
        task_type="eval" if eval_mode else "fund_analysis",
        term="medium",  # 基金评分映射到中线
        as_of_date=as_of_date,
        action=action,
        score=score,
        confidence=confidence,
        data_quality_score=data_quality,
        risk_gate_applied=False,
        risk_gate_result=None,
        key_positive_signals=strengths[:5] if strengths else None,
        key_negative_signals=risks[:5] if risks else None,
        model_profile="eval_analysis" if eval_mode else "production",
        version_hash="",
        meta={
            "raw_investment_view": investment_view,
            "rating_label": rating_label,
            "holding_period": holding.get("label", "") if isinstance(holding, dict) else "",
            "subscores": subscores,
            "score_explanation": {
                "why": explanation.get("why_this_score", ""),
                "holding": explanation.get("why_this_holding_period", ""),
            },
        },
    )


# ── Standalone utility functions ──────────────────────────────────────────


def extract_fund_signal_packs_from_state(state_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """从 AgentState.data 中提取所有基金 agent 的 signal_pack。

    Args:
        state_data: AgentState.data 字典

    Returns:
        {fund_agent_key: signal_pack_dict}
    """
    signal_packs = {}
    for agent_key in FUND_AGENT_KEYS:
        sp_key = f"{agent_key}_signal_pack"
        if sp_key in state_data and state_data[sp_key]:
            signal_packs[agent_key] = state_data[sp_key]
    return signal_packs


def extract_fund_analysis_texts_from_state(state_data: Dict[str, Any]) -> Dict[str, str]:
    """从 AgentState.data 中提取所有基金 agent 的分析文本。

    Args:
        state_data: AgentState.data 字典

    Returns:
        {fund_agent_key: analysis_text}
    """
    texts = {}
    for agent_key in FUND_AGENT_KEYS:
        if agent_key in state_data and state_data[agent_key]:
            texts[agent_key] = str(state_data[agent_key])
    return texts
