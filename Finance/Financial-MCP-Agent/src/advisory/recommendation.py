"""基于已发布精筛池和生产缓存的股票推荐引擎。"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.eval.refined_pool_repository import RefinedPoolError, RefinedPoolRepository

logger = logging.getLogger(__name__)

# ── 默认路径 ──

_BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_DEFAULT_POOL_PATH = os.path.join(_BASE_DIR, "data", "eval", "refined_pools.json")
_DEFAULT_CACHE_DIR = os.path.join(_BASE_DIR, "data", "intermediate_cache")
_TERMS = ("short", "medium", "long")
_SCORER_BY_TERM = {
    "short": "short_term_scorer",
    "medium": "medium_term_scorer",
    "long": "long_term_scorer",
}
_POOL_MAX_AGE_DAYS = {"short": 30, "medium": 60, "long": 90}

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
            pool_path: 精筛池 JSON 路径，默认 data/eval/refined_pools.json。
            cache_dir:  生产缓存目录，默认 data/intermediate_cache/。
        """
        self.pool_path: str = pool_path or _DEFAULT_POOL_PATH
        self.cache_dir: str = cache_dir or _DEFAULT_CACHE_DIR
        self.repository = RefinedPoolRepository(self.pool_path)
        self._last_pool_status: Dict[str, Any] = {
            "status": "unknown", "generation": 0
        }

    # ── 公共方法 ──

    def load_pool_stocks(self) -> List[Dict[str, Any]]:
        """从已发布精筛池加载三个期限，合并同股多期限信号。

        Returns:
            每只股票只出现一次；``terms`` 和 ``term_scores`` 保留所有期限。
        """
        if not os.path.isfile(self.pool_path):
            logger.warning("精筛池文件不存在: %s", self.pool_path)
            self._last_pool_status = self.repository.status()
            return []

        try:
            data = self.repository.read()
            self._last_pool_status = self.repository.status()
        except (RefinedPoolError, json.JSONDecodeError, OSError) as e:
            logger.error("精筛池文件读取失败: %s", e)
            self._last_pool_status = {"status": "error", "error": str(e)}
            return []

        # The real eval pool uses top-level term keys.  Keep a read-only adapter
        # for the previous advisory fixture shape during migration.
        pools = data.get("pools", data)
        by_code: Dict[str, Dict[str, Any]] = {}
        generation = self.repository.generation(data)
        published_at = data.get("_repository", {}).get("published_at", "")

        for term in _TERMS:
            term_pool = pools.get(term, {})
            if not isinstance(term_pool, dict):
                continue
            stocks = term_pool.get("stocks", [])
            iterable = stocks.items() if isinstance(stocks, dict) else (
                (None, item) for item in stocks if isinstance(item, (dict, str))
            )
            for mapped_code, raw_info in iterable:
                info = raw_info if isinstance(raw_info, dict) else {"code": raw_info}
                code = str(
                    info.get("stock_code") or info.get("code") or mapped_code or ""
                )
                if not code:
                    continue
                score_data = info.get("term_score") or {}
                pool_score = info.get("final_score", info.get("score"))
                cache = self.get_score_cache_info(code, term)
                score_time = (
                    info.get("scored_at")
                    or info.get("score_time")
                    or term_pool.get("updated_at", "")
                )
                membership = {
                    "score": pool_score,
                    "validity": info.get(
                        "validity",
                        score_data.get("validity", "legacy_non_actionable")
                        if isinstance(score_data, dict)
                        else "legacy_non_actionable",
                    ),
                    "coverage": info.get(
                        "coverage",
                        score_data.get("coverage", 0.0)
                        if isinstance(score_data, dict)
                        else 0.0,
                    ),
                    "score_time": score_time,
                    "recommendation": info.get("recommendation", info.get("rating", "")),
                    "pool_updated_at": term_pool.get("updated_at", ""),
                    "pool_version": term_pool.get("version", 0),
                    "cache_score": cache.get("score"),
                    "cache_date": cache.get("cache_date"),
                    "cache_fresh": cache.get("is_fresh", False),
                    "cache_validity": cache.get("validity", "missing"),
                }
                pool_age = self._age_days(term_pool.get("updated_at", ""))
                membership["pool_age_days"] = pool_age
                membership["pool_fresh"] = (
                    pool_age is not None and pool_age <= _POOL_MAX_AGE_DAYS[term]
                )
                membership.update(score_data if isinstance(score_data, dict) else {})

                existing = by_code.get(code)
                if existing is None:
                    existing = {
                        "stock_code": code,
                        "company_name": info.get("company_name", info.get("name", "")),
                        "score": pool_score,
                        "validity": membership["validity"],
                        "coverage": membership["coverage"],
                        "term": term,  # compatibility: first/primary membership
                        "term_score": score_data,
                        "terms": [],
                        "term_scores": {},
                        "recommendation": membership["recommendation"],
                        "detected_industry": score_data.get(
                            "detected_industry", info.get("detected_industry", "")
                        ) if isinstance(score_data, dict) else info.get("detected_industry", ""),
                        "pool_generation": generation,
                        "pool_published_at": published_at,
                    }
                    by_code[code] = existing
                existing["terms"].append(term)
                existing["term_scores"][term] = membership

        return list(by_code.values())

    def get_pool_status(self) -> Dict[str, Any]:
        """Status of the publication last observed by this engine."""
        try:
            status = self.repository.status()
            data = self.repository.read()
            term_status: Dict[str, Any] = {}
            has_stale = False
            for term in _TERMS:
                pool = data.get(term, {})
                stocks = pool.get("stocks", []) if isinstance(pool, dict) else []
                age = self._age_days(pool.get("updated_at", "")) if isinstance(pool, dict) else None
                state = "empty" if not stocks else (
                    "stale" if age is None or age > _POOL_MAX_AGE_DAYS[term] else "fresh"
                )
                has_stale = has_stale or state == "stale"
                term_status[term] = {"status": state, "age_days": age, "size": len(stocks)}
            if status["status"] == "current" and has_stale:
                status["status"] = "stale"
            status["terms"] = term_status
            return status
        except RefinedPoolError as exc:
            return {"status": "error", "error": str(exc), "generation": 0}

    @staticmethod
    def _age_days(value: Any) -> Optional[int]:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
        return (datetime.now().date() - parsed.date()).days

    def get_score_cache_info(
        self, stock_code: str, term: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return score plus the exact scorer/date/freshness provenance."""
        if not os.path.isdir(self.cache_dir):
            return {
                "score": None,
                "term": term,
                "cache_date": None,
                "is_fresh": False,
                "validity": "missing",
            }
        if term is not None and term not in _SCORER_BY_TERM:
            raise ValueError(f"unknown investment term: {term}")

        safe_code = stock_code.replace(".", "_")
        scorer_names = (
            [_SCORER_BY_TERM[term]] if term else list(_SCORER_BY_TERM.values())
        )
        candidates: List[tuple] = []
        for scorer in scorer_names:
            for path in glob.glob(os.path.join(self.cache_dir, f"{scorer}_{safe_code}_*.json")):
                candidates.append((path, scorer))

        def cache_sort_key(item: tuple) -> tuple:
            match = re.search(r"_(\d{4}-\d{2}-\d{2})(?:_eval)?\.json$", item[0])
            return (match.group(1) if match else "", item[0])

        candidates.sort(key=cache_sort_key, reverse=True)

        rejected_validity: Optional[str] = None
        for filepath, scorer in candidates:
            try:
                with open(filepath, "r", encoding="utf-8") as handle:
                    cache = json.load(handle)
            except (json.JSONDecodeError, OSError):
                continue
            content = cache.get("content")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = None
            source = content if isinstance(content, dict) else cache
            score = self._extract_cache_score(cache)
            if score is None:
                if isinstance(source, dict):
                    rejected_validity = str(
                        source.get("validity", "legacy_non_actionable")
                    )
                continue
            cache_date = self._cache_date(filepath, cache)
            age_days: Optional[int] = None
            fresh = False
            if cache_date:
                age_days = (datetime.now().date() - cache_date.date()).days
                fresh = 0 <= age_days <= 1
            resolved_term = next(
                (key for key, value in _SCORER_BY_TERM.items() if value == scorer), None
            )
            return {
                "score": score,
                "term": resolved_term,
                "scorer": scorer,
                "cache_date": cache_date.date().isoformat() if cache_date else None,
                "age_days": age_days,
                "is_fresh": fresh,
                "validity": "valid",
                "coverage": source.get("coverage", 0.0),
            }
        return {
            "score": None,
            "term": term,
            "cache_date": None,
            "is_fresh": False,
            "validity": rejected_validity or "missing",
        }

    @staticmethod
    def _extract_cache_score(cache: Dict[str, Any]) -> Optional[float]:
        content = cache.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = None
        for source in (content, cache):
            if not isinstance(source, dict):
                continue
            if source.get("validity") != "valid":
                continue
            if source.get("missing_core_fields"):
                continue
            coverage = source.get("coverage")
            if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
                continue
            if not 0.0 < float(coverage) <= 1.0:
                continue
            if source.get("score") is not None:
                try:
                    return float(source["score"])
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _cache_date(filepath: str, cache: Dict[str, Any]) -> Optional[datetime]:
        for key in ("as_of_date", "date", "created_at", "updated_at"):
            value = cache.get(key)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass
        match = re.search(r"_(\d{4}-\d{2}-\d{2})(?:_eval)?\.json$", filepath)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d")
            except ValueError:
                pass
        try:
            return datetime.fromtimestamp(os.path.getmtime(filepath))
        except OSError:
            return None

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
        return self.get_score_cache_info(stock_code).get("score")

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
        # 构建池评分查找表（作为 cache miss 时的降级后备）。旧池中只有
        # final_score 而没有有效性契约的值仅用于展示，不得参与推荐。
        pool_lookup: Dict[str, float] = {}
        for s in pool_stocks:
            sc = s.get("stock_code", "")
            score = s.get("score")
            if sc and score is not None and s.get("validity") == "valid":
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
            terms = s.get("terms") or [s.get("term", "N/A")]
            lines.append(f"- 所属期限池: {', '.join(terms)}")
            if s.get("term_scores"):
                lines.append(f"- 各期限评分与新鲜度: {json.dumps(s['term_scores'], ensure_ascii=False)}")
            if s.get("validity") == "valid":
                lines.append(f"- 精筛池综合评分: {s.get('score', 'N/A')}")
            else:
                lines.append(
                    f"- 精筛池综合评分: 不可用于决策（{s.get('validity', 'legacy_non_actionable')}）"
                )
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
