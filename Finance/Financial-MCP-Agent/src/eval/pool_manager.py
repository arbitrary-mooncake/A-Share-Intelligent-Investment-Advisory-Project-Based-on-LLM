"""
精筛池管理器 — 管理短/中/长线三个精筛股票池的创建、更新、查询。
"""
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


POOL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval"
)
POOL_FILE = os.path.join(POOL_DIR, "refined_pools.json")

DEFAULT_POOL_CONFIG = {
    "short": {"target_size": 100, "max_size": 110, "min_size": 90},
    "medium": {"target_size": 80, "max_size": 88, "min_size": 72},
    "long": {"target_size": 60, "max_size": 66, "min_size": 54},
}


class PoolManager:
    """精筛池管理器"""

    def __init__(self):
        os.makedirs(POOL_DIR, exist_ok=True)
        self.pools = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(POOL_FILE):
            try:
                with open(POOL_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "short": {"stocks": [], "updated_at": "", "version": 0},
            "medium": {"stocks": [], "updated_at": "", "version": 0},
            "long": {"stocks": [], "updated_at": "", "version": 0},
            "blacklist": [],
        }

    def _save(self):
        self.pools["updated_at"] = datetime.now().isoformat()
        with open(POOL_FILE, "w", encoding="utf-8") as f:
            json.dump(self.pools, f, ensure_ascii=False, indent=2)

    def get_pool(self, term: str) -> List[str]:
        """获取精筛池股票列表"""
        pool_data = self.pools.get(term, {})
        stocks = pool_data.get("stocks", [])
        return [s if isinstance(s, str) else s.get("code", "") for s in stocks]

    def get_pool_with_scores(self, term: str) -> List[Dict[str, Any]]:
        """获取带评分的精筛池"""
        pool_data = self.pools.get(term, {})
        return pool_data.get("stocks", [])

    def update_pool(self, term: str, stocks: List[Dict[str, Any]]):
        """更新精筛池"""
        self.pools[term] = {
            "stocks": stocks,
            "updated_at": datetime.now().isoformat(),
            "version": self.pools.get(term, {}).get("version", 0) + 1,
        }
        self._save()

    def get_pool_summary(self, term: str) -> Dict[str, Any]:
        pool = self.pools.get(term, {})
        stocks = pool.get("stocks", [])
        scores = [s.get("score", 0) if isinstance(s, dict) else 50 for s in stocks]
        return {
            "term": term,
            "size": len(stocks),
            "target_size": DEFAULT_POOL_CONFIG[term]["target_size"],
            "avg_score": round(sum(scores) / max(len(scores), 1), 1),
            "updated_at": pool.get("updated_at", "从未更新"),
            "version": pool.get("version", 0),
        }

    def get_all_summaries(self) -> Dict[str, Any]:
        return {term: self.get_pool_summary(term) for term in ["short", "medium", "long"]}

    def add_to_blacklist(self, stock_code: str, reason: str = "", expiry_days: int = 120):
        """添加到黑名单"""
        self.pools.setdefault("blacklist", [])
        self.pools["blacklist"].append({
            "code": stock_code,
            "reason": reason,
            "added_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=expiry_days)).isoformat(),
        })
        self._save()

    def is_blacklisted(self, stock_code: str) -> bool:
        """检查股票是否在黑名单中（自动过滤过期条目）"""
        now = datetime.now()
        active = []
        for entry in self.pools.get("blacklist", []):
            if entry.get("code") == stock_code:
                expires_str = entry.get("expires_at", "")
                if expires_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_str)
                        if now > expires_at:
                            continue  # 已过期，视为不在黑名单
                    except (ValueError, TypeError):
                        pass
                return True
        return False

    def clean_expired_blacklist(self):
        """清理过期的黑名单条目"""
        now = datetime.now()
        blacklist = self.pools.get("blacklist", [])
        cleaned = []
        removed = 0
        for entry in blacklist:
            expires_str = entry.get("expires_at", "")
            if expires_str:
                try:
                    if now > datetime.fromisoformat(expires_str):
                        removed += 1
                        continue
                except (ValueError, TypeError):
                    pass
            cleaned.append(entry)
        if removed > 0:
            self.pools["blacklist"] = cleaned
            self._save()

    def needs_update(self, term: str) -> Dict[str, Any]:
        """检查精筛池是否需要更新"""
        summary = self.get_pool_summary(term)
        days_since_update = 999
        if summary["updated_at"] and summary["updated_at"] != "从未更新":
            try:
                updated = datetime.fromisoformat(summary["updated_at"])
                days_since_update = (datetime.now() - updated).days
            except Exception:
                pass

        config = DEFAULT_POOL_CONFIG[term]
        triggers = {
            "short": 30,
            "medium": 60,
            "long": 90,
        }

        needs = days_since_update >= triggers.get(term, 30)

        return {
            "term": term,
            "needs_update": needs,
            "days_since_update": days_since_update,
            "max_days": triggers.get(term, 30),
            "current_size": summary["size"],
            "target_size": config["target_size"],
        }

    def get_pool_health(self,
                        lines_status: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        检查各期限精筛池的健康状态，返回红/黄/绿状态及触发原因。

        总纲§4.2 触发条件：
          RED   — >30%股票评分<50（连续5日）、LLM自主线持续优于所有Agent线、
                   黑名单股票有重大基本面变化
          YELLOW — >20%股票未被任何线路持有（连续10日）、距上次全量更新超限
          GREEN  — 无任何触发条件

        Args:
            lines_status: 可选，来自LineManager.get_all_status()的线路状态列表。
                          提供后可检查持仓覆盖率与LLM自主线对比；
                          不提供则仅基于精筛池自身数据判断。

        Returns:
            {term: {status, triggers, last_update, days_since_update,
                    suggested_action, details}}
        """
        max_days_map = {"short": 30, "medium": 60, "long": 90}

        # ── 预计算：从 lines_status 提取持仓覆盖和LLM自主线对比 ──
        held_by_term: Dict[str, set] = {}
        llm_free_exceeds: Dict[str, bool] = {}

        if lines_status:
            for term in ["short", "medium", "long"]:
                held_stocks: set = set()
                for line in lines_status:
                    if line.get("term") == term:
                        for code, shares in line.get("holdings", {}).items():
                            if shares > 0:
                                held_stocks.add(code)
                held_by_term[term] = held_stocks

                # LLM自主线 vs 最强Agent线
                # 短线: S-L8(agent longhold) vs S-L9(LLM free)
                # 中线: M-L0(agent) vs M-L1(LLM free)
                # 长线: L-L0(agent) vs L-L1(LLM free)
                agent_lines = [
                    l for l in lines_status
                    if l.get("term") == term
                    and l.get("type") in ("full_agent", "longhold", "ablation_base")
                ]
                llm_free_lines = [
                    l for l in lines_status
                    if l.get("term") == term and l.get("type") == "llm_free"
                ]
                if agent_lines and llm_free_lines:
                    best_agent_ret = max(
                        (l.get("cumulative_return_pct", -999) for l in agent_lines),
                        default=-999,
                    )
                    llm_free_ret = llm_free_lines[0].get("cumulative_return_pct", 0)
                    llm_free_exceeds[term] = (
                        llm_free_ret > best_agent_ret and best_agent_ret > -999
                    )

        # ── 逐期限检查 ──
        results = {}
        for term in ["short", "medium", "long"]:
            summary = self.get_pool_summary(term)
            stocks_with_scores = self.get_pool_with_scores(term)
            pool_size = len(stocks_with_scores)

            days_since_update = 999
            if summary["updated_at"] and summary["updated_at"] != "从未更新":
                try:
                    updated = datetime.fromisoformat(summary["updated_at"])
                    days_since_update = (datetime.now() - updated).days
                except Exception:
                    pass

            max_days = max_days_map[term]

            # ── 指标计算 ──
            low_score_count = sum(
                1 for s in stocks_with_scores
                if (isinstance(s, dict) and s.get("score", 50) < 50)
            )
            low_score_pct = (low_score_count / max(pool_size, 1)) * 100

            pool_codes = [
                s.get("code", "") if isinstance(s, dict) else s
                for s in stocks_with_scores
            ]

            not_held_pct = 0.0
            held_stocks_set = held_by_term.get(term, set())
            if held_stocks_set and pool_codes:
                not_held = [c for c in pool_codes if c not in held_stocks_set]
                not_held_pct = len(not_held) / max(len(pool_codes), 1) * 100

            # 黑名单检查
            active_blacklist_codes: set = set()
            now = datetime.now()
            for entry in self.pools.get("blacklist", []):
                expires_str = entry.get("expires_at", "")
                if expires_str:
                    try:
                        if now <= datetime.fromisoformat(expires_str):
                            active_blacklist_codes.add(entry.get("code", ""))
                    except (ValueError, TypeError):
                        active_blacklist_codes.add(entry.get("code", ""))
            pool_codes_set = set(pool_codes)
            blacklist_in_pool = active_blacklist_codes & pool_codes_set

            # ── 判断状态（RED优先） ──
            triggers = []
            status = "green"

            if low_score_pct > 30:
                triggers.append(
                    f"池内{low_score_pct:.0f}%的股票连续5日评分<50，"
                    f"建议部分更新（替换评分垫底的20%股票）"
                )
                status = "red"

            if llm_free_exceeds.get(term, False):
                triggers.append("LLM自主线累计收益超过所有Agent线，系统评分策略需审查")
                status = "red"

            if blacklist_in_pool:
                blacklist_descs = []
                for entry in self.pools.get("blacklist", []):
                    if entry.get("code", "") in blacklist_in_pool:
                        blacklist_descs.append(
                            f"{entry.get('code', '?')}"
                            f"({entry.get('reason', '未知原因')})"
                        )
                triggers.append(
                    f"黑名单股票仍在精筛池内: {', '.join(blacklist_descs[:3])}"
                )
                status = "red"

            # ── 欠填充检测（当前池容量显著低于目标） ──
            min_size = DEFAULT_POOL_CONFIG[term]["min_size"]
            severe_underfill = pool_size < min_size * 0.5

            # YELLOW（仅当非RED时）
            if status != "red":
                if severe_underfill:
                    triggers.append(
                        f"精筛池严重欠填充: 当前{pool_size}只 < 下限{min_size}只"
                        f"（目标{summary['target_size']}只），建议立即全量更新"
                    )
                    status = "yellow"

                if not_held_pct > 20:
                    triggers.append(
                        f"池内{not_held_pct:.0f}%的股票未被任何线路持有（连续10日）"
                    )
                    status = "yellow"

                if days_since_update >= max_days:
                    triggers.append(
                        f"距上次全量更新已{days_since_update}天（上限{max_days}天）"
                    )
                    status = "yellow"

            # ── 建议动作 ──
            if status == "red":
                suggested_action = "建议立即全量更新精筛池"
            elif status == "yellow":
                if severe_underfill or days_since_update >= max_days:
                    suggested_action = "建议全量更新精筛池"
                else:
                    suggested_action = "建议部分更新（替换评分垫底的20%股票）"
            else:
                suggested_action = "池状态健康，无需操作"

            results[term] = {
                "status": status,
                "triggers": triggers,
                "last_update": summary["updated_at"],
                "days_since_update": days_since_update,
                "max_days": max_days,
                "suggested_action": suggested_action,
                "details": {
                    "size": pool_size,
                    "target_size": summary["target_size"],
                    "avg_score": summary["avg_score"],
                    "low_score_pct": round(low_score_pct, 1),
                    "not_held_pct": round(not_held_pct, 1),
                    "blacklist_count": len(blacklist_in_pool),
                },
            }

        return results
