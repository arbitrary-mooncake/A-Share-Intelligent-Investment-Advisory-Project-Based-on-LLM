"""
基金管线适配器 — 总纲 §17.1
将评测系统的信号包和评分适配到基金分析管道。
"""
from typing import Dict, Any, Optional


class FundPipelineAdapter:
    """基金分析管线适配器"""

    def __init__(self):
        pass

    def adapt_signal_pack(self, fund_code: str, signal_pack: Dict[str, Any]) -> Dict[str, Any]:
        """将A股信号包适配为基金分析格式"""
        return {
            "fund_code": fund_code,
            "bias": signal_pack.get("bias", "neutral"),
            "confidence": signal_pack.get("confidence", 0.5),
            "source": "eval_system",
            "signals": signal_pack.get("signals", []),
            "risk_flags": signal_pack.get("risk_flags", []),
        }

    def adapt_scoring_result(self, fund_code: str, score_data: Dict[str, Any]) -> Dict[str, Any]:
        """适配基金评分结果"""
        return {
            "fund_code": fund_code,
            "score": score_data.get("score", 50),
            "term": score_data.get("term", "medium"),
            "rating": score_data.get("recommendation", ""),
            "source": "eval_scoring",
        }

    def merge_with_fund_analysis(
        self, fund_code: str, eval_data: Dict[str, Any], fund_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """合并评测数据和基金分析结果"""
        return {
            "fund_code": fund_code,
            "eval_score": eval_data.get("score", 50),
            "fund_analysis_score": fund_analysis.get("score", 50),
            "merged_score": (eval_data.get("score", 50) + fund_analysis.get("score", 50)) / 2,
            "eval_signals": eval_data.get("signals", []),
            "fund_signals": fund_analysis.get("signals", []),
        }
