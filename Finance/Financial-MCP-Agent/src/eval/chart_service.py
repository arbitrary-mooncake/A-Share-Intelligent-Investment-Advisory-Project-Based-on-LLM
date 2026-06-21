"""
图表数据服务 — 生成前端绘图所需的标准化时序JSON。
不依赖任何图表库，只输出数据结构。
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class ChartService:
    """趋势图数据生成器"""

    def __init__(self, memory_manager=None):
        from src.eval.memory_manager import MemoryManager
        self.memory = memory_manager or MemoryManager()

    def get_score_trend_data(self, days: int = 90) -> Dict[str, Any]:
        """Score趋势数据（用于折线图）"""
        trend = self.memory.get_score_trend(days)
        return {
            "title": f"Score趋势 (近{days}天)",
            "x_axis": "日期",
            "y_axis": "Score",
            "data": [{"date": e.get("date", ""), "value": e.get("avg_return", 0)} for e in trend],
            "type": "line",
        }

    def get_loss_trend_data(self, days: int = 90) -> Dict[str, Any]:
        """Loss趋势数据"""
        trend = self.memory.get_loss_trend(days)
        return {
            "title": f"Loss趋势 (近{days}天)",
            "x_axis": "日期",
            "y_axis": "Loss",
            "data": [{"date": e.get("date", ""), "value": e.get("L_total", 1.0)} for e in trend],
            "type": "line",
        }

    def get_agent_contribution_bar_data(self, contributions: List[Dict]) -> Dict[str, Any]:
        """Agent贡献柱状图数据"""
        return {
            "title": "Agent贡献分",
            "x_axis": "Agent",
            "y_axis": "ΔLoss",
            "data": [
                {
                    "agent": c.get("agent_name", ""),
                    "delta_L": c.get("delta_L_total", 0),
                    "ci_low": c.get("ci_95_lower", 0),
                    "ci_high": c.get("ci_95_upper", 0),
                    "stars": c.get("stars", ""),
                }
                for c in contributions
            ],
            "type": "bar",
        }

    def get_agent_trend_data(self, agent_name: str, days: int = 90) -> Dict[str, Any]:
        """单个Agent的贡献趋势"""
        trend = self.memory.get_agent_trend(agent_name, days)
        return {
            "title": f"{agent_name} 贡献趋势 (近{days}天)",
            "x_axis": "日期",
            "y_axis": "ΔL",
            "data": [{"date": e.get("date", ""), "value": e.get("delta_L_total", 0)} for e in trend],
            "type": "line",
        }

    def get_fidelity_trend_data(self, days: int = 90) -> Dict[str, Any]:
        """保真度趋势数据（总纲 §9.5: action_flip_rate/topK_overlap等）"""
        trend = self.memory.get_fidelity_trend(days)
        data = []
        for e in trend:
            data.append({
                "date": e.get("date", ""),
                "action_flip_rate": e.get("action_flip_rate", 0),
                "topK_overlap": e.get("topK_overlap", 1.0),
                "score_drift": e.get("score_drift", 0),
                "fidelity_loss": e.get("fidelity_loss", 0),
            })
        return {
            "title": f"保真度趋势 (近{days}天)",
            "x_axis": "日期",
            "y_axis": "保真度指标",
            "data": data,
            "type": "line",
            "metrics": ["action_flip_rate", "topK_overlap", "score_drift", "fidelity_loss"],
        }

    def get_runtime_trend_data(self, days: int = 90) -> Dict[str, Any]:
        """运行耗时/Token趋势数据"""
        trend = self.memory.get_runtime_trend(days)
        data = []
        for e in trend:
            data.append({
                "date": e.get("date", ""),
                "duration_minutes": round(e.get("total_duration_seconds", 0) / 60, 1),
                "estimated_tokens": e.get("estimated_tokens", 0),
                "agent_calls": e.get("agent_calls", 0),
                "cache_hit_rate": round(
                    e.get("cache_hits", 0) / max(e.get("cache_hits", 0) + e.get("cache_misses", 0), 1) * 100, 1
                ),
            })
        return {
            "title": f"运行耗时/Token趋势 (近{days}天)",
            "x_axis": "日期",
            "y_axis": "指标",
            "data": data,
            "type": "line",
            "metrics": ["duration_minutes", "estimated_tokens", "agent_calls", "cache_hit_rate"],
        }

    def get_line_comparison_data(self, lines_status: List[Dict]) -> Dict[str, Any]:
        """各线路对比数据"""
        return {
            "title": "各线路累计收益对比",
            "x_axis": "线路",
            "y_axis": "累计收益%",
            "data": [
                {
                    "line_id": l.get("line_id", ""),
                    "return": l.get("cumulative_return_pct", 0),
                    "mdd": l.get("max_drawdown_pct", 0),
                    "term": l.get("term", ""),
                }
                for l in lines_status
            ],
            "type": "bar",
        }
