"""
股票推荐引擎 — 基于 n/x/5 规则的推荐引擎。

从精筛池（stock_pool.json）挑选股票，查生产缓存辅助决策。
只读缓存，不触发分析 Agent，不调用 LLM。
"""
from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 默认路径 ──

_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_DEFAULT_POOL_PATH = os.path.join(_BASE_DIR, "stock_pool.json")
_DEFAULT_CACHE_DIR = os.path.join(_BASE_DIR, "data", "intermediate_cache")

# ── 全局单例 ──

_engine_instance: Optional["RecommendationEngine"] = None


class RecommendationEngine:
    """股票推荐引擎 — 基于精筛池 + 生产缓存的 n/x/5 规则推荐。

    只读 intermediate_cache/，绝不碰 eval/cache/。
    全程不触发任何分析 Agent，不调用 LLM。
    """

    def __init__(
        self, pool_path: Optional[str] = None, cache_dir: Optional[str] = None
    ) -> None:
        """初始化推荐引擎。

        Args:
            pool_path: 精筛池 JSON 路径，默认项目根目录 stock_pool.json。
            cache_dir:  生产缓存目录，默认 data/intermediate_cache/。
        """
        self.pool_path: str = pool_path or _DEFAULT_POOL_PATH
        self.cache_dir: str = cache_dir or _DEFAULT_CACHE_DIR

    # ── 公共方法 ──

    def load_pool_stocks(self) -> List[Dict[str, Any]]:
        """从 stock_pool.json 加载 short/medium/long 三个期限池，去重。

        Returns:
            每只股票的 dict 列表，包含 stock_code / company_name / score /
            term / term_score / recommendation / detected_industry。
        """
        if not os.path.isfile(self.pool_path):
            logger.warning("精筛池文件不存在: %s", self.pool_path)
            return []

        try:
            with open(self.pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("精筛池文件读取失败: %s", e)
            return []

        pools = data.get("pools", {})
        seen: set[str] = set()
        result: List[Dict[str, Any]] = []

        for term in ("short", "medium", "long"):
            term_pool = pools.get(term, {})
            stocks = term_pool.get("stocks", {})
            for code, info in stocks.items():
                if code in seen:
                    continue
                seen.add(code)

                term_score_data = info.get("term_score") or {}

                result.append({
                    "stock_code": info.get("stock_code", code),
                    "company_name": info.get("company_name", ""),
                    "score": info.get("score"),
                    "term": term,
                    "term_score": term_score_data,
                    "recommendation": info.get("recommendation", ""),
                    "detected_industry": term_score_data.get(
                        "detected_industry",
                        info.get("detected_industry", ""),
                    ),
                })

        return result

    def check_score_cache(self, stock_code: str) -> Optional[float]:
        """检查 intermediate_cache/ 中的打分缓存文件。

        搜索模式涵盖实际缓存文件命名：
          - short_term_scorer_{safe_code}_{date}.json
          - medium_term_scorer_{safe_code}_{date}.json
          - long_term_scorer_{safe_code}_{date}.json
        同时兼容 scoring_ 前缀模式（future proof）。

        Returns:
            score 数值，或 None（无缓存 / 解析失败）。
        """
        if not os.path.isdir(self.cache_dir):
            return None

        safe_code = stock_code.replace(".", "_")

        # 主搜索：生产环境现有的 scorer 缓存
        # 文件名如: short_term_scorer_{safe_code}_{date}.json
        pattern = os.path.join(self.cache_dir, f"*scorer*{safe_code}*.json")
        matching: List[str] = glob.glob(pattern)

        # 备选搜索：brief 中描述的 scoring_ 前缀模式
        alt_pattern = os.path.join(self.cache_dir, f"*scoring*{safe_code}*score*.json")
        matching.extend(glob.glob(alt_pattern))

        if not matching:
            return None

        # 按文件名降序排列（最新优先）
        matching.sort(reverse=True)

        for filepath in matching:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # 尝试从 content 字段中提取（content 是 JSON 字符串的常见模式）
            content_str = cache.get("content")
            if content_str and isinstance(content_str, str):
                try:
                    content = json.loads(content_str)
                    score = content.get("score")
                    if score is not None:
                        return float(score)
                except (json.JSONDecodeError, TypeError):
                    pass

            # 直接从缓存顶层提取
            score = cache.get("score")
            if score is not None:
                return float(score)

        return None

    def check_signal_pack_cache(
        self, stock_code: str, agent: str
    ) -> Optional[Dict[str, Any]]:
        """检查某个 agent 的 signal_pack 缓存。

        匹配 {agent}_analysis_signal_pack_{safe_code}_{date}.json 模式。
        agent 为代理简称（如 'fundamental', 'technical' 等）。

        Args:
            stock_code: 股票代码，如 'sz.000498'。
            agent: 代理简称。

        Returns:
            signal_pack dict，或 None。
        """
        if not os.path.isdir(self.cache_dir):
            return None

        safe_code = stock_code.replace(".", "_")
        # 生产缓存命名: {agent}_analysis_signal_pack_{safe_code}_{date}.json
        pattern = os.path.join(
            self.cache_dir, f"{agent}_analysis_signal_pack_{safe_code}_*.json"
        )
        matching = glob.glob(pattern)

        if not matching:
            return None

        matching.sort(reverse=True)
        latest = matching[0]

        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("signal_pack 缓存读取失败 %s: %s", latest, e)
            return None

    def apply_nx5_rule(
        self,
        candidate_codes: List[str],
        pool_stocks: List[Dict[str, Any]],
    ) -> List[str]:
        """应用 n/x/5 推荐裁剪规则。

        规则逻辑：
          - n <= 5: 直接返回 n 只
          - n > 5: x = 有打分缓存的股票数
            - x = 0: 返回 ["__LLM_PICK__"] + candidate_codes
            - x <= 5: 返回这 x 只
            - x > 5: 返回评分最高的 5 只

        Args:
            candidate_codes: 候选股票代码列表（n 只）。
            pool_stocks:     load_pool_stocks() 返回的完整池数据，用作评分降级后备。

        Returns:
            裁剪后的股票代码列表。
        """
        n = len(candidate_codes)
        if n <= 5:
            return list(candidate_codes)

        # 构建池评分查找表（作为 cache miss 时的降级后备）
        pool_lookup: Dict[str, float] = {}
        for s in pool_stocks:
            sc = s.get("stock_code", "")
            score = s.get("score")
            if sc and score is not None:
                pool_lookup[sc] = float(score)

        # 逐只检查打分缓存
        scored: List[tuple] = []
        for code in candidate_codes:
            score = self.check_score_cache(code)
            if score is not None:
                scored.append((code, score))
            else:
                # 后备：使用精筛池内评分
                pool_score = pool_lookup.get(code)
                if pool_score is not None:
                    scored.append((code, pool_score))

        x = len(scored)

        if x == 0:
            return ["__LLM_PICK__"] + list(candidate_codes)

        # 有评分数据时按分数排序
        scored.sort(key=lambda t: t[1], reverse=True)

        if x <= 5:
            return [code for code, _ in scored]

        # x > 5: 返回评分最高的 5 只
        return [code for code, _ in scored[:5]]

    def build_recommendation_context(
        self,
        stocks: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建供 LLM 使用的纯文本上下文。

        Args:
            stocks:        load_pool_stocks() 返回的股票列表（或子集）。
            user_profile:  可选用户画像字典。

        Returns:
            格式化的纯文本上下文字符串。
        """
        lines: List[str] = ["## 候选股票推荐上下文\n"]

        if user_profile:
            lines.append("### 用户投资偏好")
            lines.append("```json")
            lines.append(json.dumps(user_profile, ensure_ascii=False, indent=2))
            lines.append("```\n")

        for i, s in enumerate(stocks, 1):
            code = s.get("stock_code", "")
            name = s.get("company_name", "")
            lines.append(f"### 候选 {i}: {name}（{code}）")
            lines.append(f"- 所属期限池: {s.get('term', 'N/A')}")
            lines.append(f"- 精筛池综合评分: {s.get('score', 'N/A')}")
            lines.append(f"- 精筛池推荐: {s.get('recommendation', 'N/A')}")
            lines.append(f"- 归属行业: {s.get('detected_industry', 'N/A')}")

            # 缓存评分辅助信息
            cache_score = self.check_score_cache(code)
            if cache_score is not None:
                lines.append(f"- 生产缓存最近评分: {cache_score:.1f}")
            else:
                lines.append("- 生产缓存: 无近期评分缓存")

            # 各 agent 缓存状况
            agent_hits = 0
            for agent in (
                "fundamental",
                "technical",
                "value",
                "news",
                "event",
                "quality_risk",
                "moneyflow",
            ):
                if self.check_signal_pack_cache(code, agent) is not None:
                    agent_hits += 1
            lines.append(f"- 分析缓存: {agent_hits}/7 agent 缓存命中")
            lines.append("")

        return "\n".join(lines)


def get_recommendation_engine(
    pool_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> RecommendationEngine:
    """懒加载单例工厂函数。

    后续调用返回同一个 RecommendationEngine 实例。
    仅在未初始化或手动重置后创建新实例。

    Args:
        pool_path: 精筛池 JSON 路径。
        cache_dir: 生产缓存目录。

    Returns:
        RecommendationEngine 单例。
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RecommendationEngine(
            pool_path=pool_path, cache_dir=cache_dir
        )
    return _engine_instance
