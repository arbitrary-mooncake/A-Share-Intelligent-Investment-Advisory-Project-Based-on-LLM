"""
ScoringEngine: 完整Pipeline评分引擎

工作流架构:
    start_node
      ├── fundamental_analyst ─────┐
      ├── technical_analyst ───────┤
      ├── value_analyst ───────────┤  4个分析Agent并行
      ├── news_analyst ────────────┘
      ├── short_term_scorer ───────→ state.data.short_term_score
      ├── medium_term_scorer ──────→ state.data.medium_term_score
      └── long_term_scorer ────────→ state.data.long_term_score

评分直接来自专用打分Agent，不再从Markdown报告提取。
"""
import os
import sys
import time
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

# 抑制无关输出
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, END

from src.utils.state_definition import AgentState
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.agents.fundamental_agent import fundamental_agent
from src.agents.technical_agent import technical_agent
from src.agents.value_agent import value_agent
from src.agents.news_agent import news_agent
from src.agents.scoring_nodes import (
    short_term_scorer_node,
    medium_term_scorer_node,
    long_term_scorer_node,
)
from src.stock_pool.stock_pool_manager import StockPoolManager

logger = setup_logger(__name__)


class ScoringEngine:
    """
    评分引擎：运行完整分析+打分Pipeline为股票评分

    工作流:
    start_node → [fundamental, technical, value, news] → [short_term, medium_term, long_term] → END

    评分直接来自3个打分Agent，以中线评分为主评分写入股票池。
    """

    def __init__(self, pool_manager: Optional[StockPoolManager] = None):
        self.pool_manager = pool_manager or StockPoolManager()
        self._workflow = None

    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        if self._workflow is None:
            workflow = StateGraph(AgentState)

            # 分析节点（4个并行）
            workflow.add_node("start_node", lambda state: state)
            workflow.add_node("fundamental_analyst", fundamental_agent)
            workflow.add_node("technical_analyst", technical_agent)
            workflow.add_node("value_analyst", value_agent)
            workflow.add_node("news_analyst", news_agent)

            # 打分节点（3个并行）
            workflow.add_node("short_term_scorer", short_term_scorer_node)
            workflow.add_node("medium_term_scorer", medium_term_scorer_node)
            workflow.add_node("long_term_scorer", long_term_scorer_node)

            # 入口
            workflow.set_entry_point("start_node")

            # 分析节点并行
            workflow.add_edge("start_node", "fundamental_analyst")
            workflow.add_edge("start_node", "technical_analyst")
            workflow.add_edge("start_node", "value_analyst")
            workflow.add_edge("start_node", "news_analyst")

            # 分析节点 → 打分节点
            # short_term 只需要 technical + news
            workflow.add_edge("technical_analyst", "short_term_scorer")
            workflow.add_edge("news_analyst", "short_term_scorer")

            # medium_term 需要全部4个
            workflow.add_edge("fundamental_analyst", "medium_term_scorer")
            workflow.add_edge("technical_analyst", "medium_term_scorer")
            workflow.add_edge("value_analyst", "medium_term_scorer")
            workflow.add_edge("news_analyst", "medium_term_scorer")

            # long_term 需要全部4个
            workflow.add_edge("fundamental_analyst", "long_term_scorer")
            workflow.add_edge("technical_analyst", "long_term_scorer")
            workflow.add_edge("value_analyst", "long_term_scorer")
            workflow.add_edge("news_analyst", "long_term_scorer")

            # 打分节点 → END
            workflow.add_edge("short_term_scorer", END)
            workflow.add_edge("medium_term_scorer", END)
            workflow.add_edge("long_term_scorer", END)

            self._workflow = workflow.compile()
        return self._workflow

    @staticmethod
    def _build_initial_state(stock_code: str, company_name: str) -> AgentState:
        """构建Agent初始状态"""
        current_datetime = datetime.now()
        current_date_cn = current_datetime.strftime("%Y年%m月%d日")
        current_date_en = current_datetime.strftime("%Y-%m-%d")
        current_weekday_cn = ["星期一", "星期二", "星期三", "星期四",
                              "星期五", "星期六", "星期日"][current_datetime.weekday()]
        current_time = current_datetime.strftime("%H:%M:%S")
        current_time_info = f"{current_date_cn} ({current_date_en}) {current_weekday_cn} {current_time}"

        initial_data = {
            "query": f"分析{company_name}",
            "stock_code": stock_code,
            "company_name": company_name,
            "current_date": current_date_en,
            "current_date_cn": current_date_cn,
            "current_time": current_time,
            "current_weekday_cn": current_weekday_cn,
            "current_time_info": current_time_info,
            "analysis_timestamp": current_datetime.isoformat()
        }

        return AgentState(
            messages=[],
            data=initial_data,
            metadata={}
        )

    async def score_stock(self, stock_code: str, company_name: str) -> Dict[str, Any]:
        """
        对指定股票运行完整分析+打分Pipeline

        Args:
            stock_code: 股票代码(带交易所前缀)
            company_name: 公司名称

        Returns:
            评分结果字典
        """
        if self.pool_manager:
            self.pool_manager.update_stock_status(stock_code, "scoring")

        logger.info(f"{WAIT_ICON} 开始对 {company_name}({stock_code}) 运行完整评分Pipeline...")
        start_time = time.time()

        try:
            # 1. 构建工作流
            app = self._build_workflow()

            # 2. 构建初始状态
            initial_state = self._build_initial_state(stock_code, company_name)

            # 3. 执行Pipeline: 4个分析Agent并行 → 3个打分Agent并行
            logger.info(f"{WAIT_ICON} 正在运行分析+打分Pipeline...")
            final_state = await app.ainvoke(initial_state)

            # 4. 直接从状态中获取三种评分
            data = final_state.get("data", {})
            short_term_score = data.get("short_term_score", {})
            medium_term_score = data.get("medium_term_score", {})
            long_term_score = data.get("long_term_score", {})

            if not medium_term_score:
                raise ValueError("Pipeline未生成中线评分")

            # 5. 构建存储数据：以中线评分为主评分
            score_data = {
                # 主评分（中线，用于排序和显示）
                "score": medium_term_score.get("score"),
                "recommendation": medium_term_score.get("rating", ""),
                # 三种完整评分
                "short_term_score": short_term_score,
                "medium_term_score": medium_term_score,
                "long_term_score": long_term_score,
                "status": "scored",
            }

            elapsed = time.time() - start_time
            st = short_term_score.get("score", "-")
            mt = medium_term_score.get("score", "-")
            lt = long_term_score.get("score", "-")
            logger.info(
                f"{SUCCESS_ICON} {company_name} 评分完成: "
                f"短线={st}, 中线={mt}, 长线={lt}, 耗时={elapsed:.1f}s"
            )

            # 6. 更新股票池
            if self.pool_manager:
                self.pool_manager.update_stock_score(stock_code, score_data)

            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "score_data": score_data,
                "execution_time": elapsed
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{ERROR_ICON} {company_name} 评分失败: {e}")
            if self.pool_manager:
                self.pool_manager.update_stock_status(stock_code, "failed")
                self.pool_manager.update_stock_score(stock_code, {
                    "score": None,
                    "recommendation": "",
                    "short_term_score": {},
                    "medium_term_score": {},
                    "long_term_score": {},
                    "status": "failed",
                })

            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "score_data": None,
                "error": str(e),
                "execution_time": elapsed
            }

    async def score_all_pending(self) -> List[Dict[str, Any]]:
        """
        对所有待评分的股票逐一评分

        Returns:
            评分结果列表
        """
        pending = self.pool_manager.get_pending_stocks()
        if not pending:
            logger.info(f"{SUCCESS_ICON} 没有待评分的股票")
            return []

        results = []
        for stock in pending:
            result = await self.score_stock(
                stock["stock_code"], stock["company_name"]
            )
            results.append(result)

        return results
