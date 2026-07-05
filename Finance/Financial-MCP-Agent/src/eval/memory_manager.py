"""
记忆管理器 — 维护历史评估趋势，支持趋势查询和摘要。
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "eval"
)
TREND_FILE = os.path.join(MEMORY_DIR, "trend_cache.json")


class MemoryManager:
    """长期记忆与趋势管理器"""

    def __init__(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self.trends = self._load_trends()

    def _load_trends(self) -> Dict[str, Any]:
        defaults = {"score_history": [], "loss_history": [], "contribution_history": [],
                    "fidelity_history": [], "runtime_history": [], "batches": [],
                    "stock_scores": {}}
        if os.path.exists(TREND_FILE):
            try:
                with open(TREND_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # 合并：确保所有默认key存在（兼容旧版本缓存文件）
                for key, default_val in defaults.items():
                    if key not in loaded:
                        loaded[key] = default_val
                return loaded
            except Exception:
                pass
        return defaults

    def _save_trends(self):
        with open(TREND_FILE, "w", encoding="utf-8") as f:
            json.dump(self.trends, f, ensure_ascii=False, indent=2, default=str)

    def record_batch(self, batch_id: str, report_data: Dict[str, Any]):
        """记录一个批次的摘要数据"""
        entry = {
            "batch_id": batch_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
        }

        # 提取关键指标
        exec_summary = report_data.get("executive_summary", {})
        if exec_summary:
            entry["avg_return"] = exec_summary.get("avg_return_all_lines", 0)

        term_summaries = report_data.get("term_summaries", {})
        for term, summary in term_summaries.items():
            entry[f"{term}_avg_return"] = summary.get("avg_cumulative_return_pct", 0)

        # Loss趋势
        loss_data = report_data.get("loss_analysis", {})
        if loss_data:
            entry["L_total"] = loss_data.get("L_total", 0)
            entry["score_total"] = loss_data.get("score_total", 0)
            self.trends["loss_history"].append({
                "date": entry["date"],
                "L_total": entry["L_total"],
                "score_total": entry["score_total"],
                "batch_id": batch_id,
            })

        self.trends["score_history"].append(entry)
        self.trends["batches"].append(batch_id)

        # 只保留最近200条记录
        if len(self.trends["score_history"]) > 200:
            self.trends["score_history"] = self.trends["score_history"][-200:]
        if len(self.trends["batches"]) > 200:
            self.trends["batches"] = self.trends["batches"][-200:]
        if len(self.trends["loss_history"]) > 200:
            self.trends["loss_history"] = self.trends["loss_history"][-200:]

        self._save_trends()

    def record_contributions(self, batch_id: str, contributions: List[Dict]):
        """记录agent贡献趋势"""
        for c in contributions:
            self.trends["contribution_history"].append({
                "batch_id": batch_id,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "agent_name": c.get("agent_name", ""),
                "delta_L_total": c.get("delta_L_total", 0),
                "stars": c.get("stars", ""),
            })

        # 只保留最近500条
        if len(self.trends["contribution_history"]) > 500:
            self.trends["contribution_history"] = self.trends["contribution_history"][-500:]

        self._save_trends()

    def record_fidelity(self, batch_id: str, fidelity_data: Dict[str, Any]):
        """记录保真度趋势（总纲 §9.5）"""
        self.trends["fidelity_history"].append({
            "batch_id": batch_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "action_flip_rate": fidelity_data.get("action_flip_rate", 0),
            "score_drift": fidelity_data.get("score_drift", 0),
            "topK_overlap": fidelity_data.get("topK_overlap", 1.0),
            "rank_drift": fidelity_data.get("rank_drift", 0),
            "fidelity_loss": fidelity_data.get("fidelity_loss", 0),
            "warnings": fidelity_data.get("warnings", []),
        })
        if len(self.trends["fidelity_history"]) > 200:
            self.trends["fidelity_history"] = self.trends["fidelity_history"][-200:]
        self._save_trends()

    def record_runtime(self, batch_id: str, runtime_data: Dict[str, Any]):
        """记录运行耗时趋势"""
        self.trends["runtime_history"].append({
            "batch_id": batch_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_duration_seconds": runtime_data.get("total_duration_seconds", 0),
            "agent_calls": runtime_data.get("agent_calls", 0),
            "cache_hits": runtime_data.get("cache_hits", 0),
            "cache_misses": runtime_data.get("cache_misses", 0),
            "estimated_tokens": runtime_data.get("estimated_tokens", 0),
            "line_count": runtime_data.get("line_count", 0),
        })
        if len(self.trends["runtime_history"]) > 200:
            self.trends["runtime_history"] = self.trends["runtime_history"][-200:]
        self._save_trends()

    def get_fidelity_trend(self, days: int = 90) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self.trends["fidelity_history"] if e.get("date", "") >= cutoff]

    def get_runtime_trend(self, days: int = 90) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self.trends["runtime_history"] if e.get("date", "") >= cutoff]

    def get_score_trend(self, days: int = 90) -> List[Dict]:
        """获取评分趋势"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self.trends["score_history"] if e.get("date", "") >= cutoff]

    def get_loss_trend(self, days: int = 90) -> List[Dict]:
        """获取Loss趋势"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self.trends["loss_history"] if e.get("date", "") >= cutoff]

    def get_agent_trend(self, agent_name: str, days: int = 90) -> List[Dict]:
        """获取特定Agent的贡献趋势"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [e for e in self.trends["contribution_history"]
                if e.get("agent_name") == agent_name and e.get("date", "") >= cutoff]

    def record_stock_score(self, stock_code: str, score: float, date: str, term: str):
        """
        记录单只股票的单次评分，用于动态频率升级/降级判断（总纲 §14.4 扩展）。

        每个 stock_code + term 组合保留最近30条评分记录。

        Args:
            stock_code: 股票代码 (如 sh.603871)
            score: 综合评分 (0-100)
            date: 评分日期 YYYY-MM-DD
            term: short/medium/long
        """
        storage = self.trends.setdefault("stock_scores", {})
        key = f"{stock_code}|{term}"
        if key not in storage:
            storage[key] = []
        # 避免同一天重复记录
        if storage[key] and storage[key][-1].get("date") == date:
            storage[key][-1] = {"date": date, "score": score}
        else:
            storage[key].append({"date": date, "score": score})
        # 只保留最近30条
        if len(storage[key]) > 30:
            storage[key] = storage[key][-30:]
        self._save_trends()

    def get_stock_score_history(self, stock_code: str, term: str,
                                days: int = 30) -> List[Dict]:
        """
        获取单只股票的评分历史。

        Args:
            stock_code: 股票代码
            term: short/medium/long
            days: 回溯天数，0 表示返回全部

        Returns:
            [{"date": "YYYY-MM-DD", "score": 85.5}, ...] 按日期升序
        """
        storage = self.trends.get("stock_scores", {})
        key = f"{stock_code}|{term}"
        history = storage.get(key, [])
        if days and history:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            history = [e for e in history if e.get("date", "") >= cutoff]
        return history

    def get_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        score_history = self.trends["score_history"]
        loss_history = self.trends["loss_history"]

        recent_scores = [e.get("avg_return", 0) for e in score_history[-10:]]

        return {
            "total_batches": len(self.trends["batches"]),
            "recent_10_avg_return": round(sum(recent_scores) / max(len(recent_scores), 1), 2),
            "total_contribution_records": len(self.trends["contribution_history"]),
            "latest_batch": self.trends["batches"][-1] if self.trends["batches"] else None,
        }
