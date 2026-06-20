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
