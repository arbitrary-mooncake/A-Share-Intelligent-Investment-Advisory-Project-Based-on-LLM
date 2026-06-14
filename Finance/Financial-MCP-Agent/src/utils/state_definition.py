from typing import TypedDict, Sequence, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator


def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, d2 values overwrite d1."""
    return {**d1, **d2}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]

    # 以下字段由 merge_dicts 自动合并，无需预先声明
    # 分析中间产物（由4个分析Agent写入）:
    #   fundamental_analysis, technical_analysis, value_analysis, news_analysis
    # 评分结果（由3个打分Agent写入）:
    #   short_term_score, medium_term_score, long_term_score
    # 报告产物（由summary_agent写入）:
    #   final_report, report_path
    # 架构升级 v2 (2026-06):
    #   信号包（结构化中间产物）:
    #     fundamental_signal_pack, technical_signal_pack, value_signal_pack,
    #     news_signal_pack, event_signal_pack, quality_risk_signal_pack,
    #     moneyflow_signal_pack
    #   合并产物: analysis_package
    #   metadata 新增: analysis_version="a_share_v2", executed_agents,
    #     missing_agents, data_quality_summary, warnings
