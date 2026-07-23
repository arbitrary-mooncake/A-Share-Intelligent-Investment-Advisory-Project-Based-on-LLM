"""
ScoringEngine: 完整Pipeline评分引擎

工作流架构:
    start_node
      ├── fundamental_analyst ──────┐
      ├── technical_analyst ────────┤
      ├── value_analyst ────────────┤
      ├── news_analyst ─────────────┤  7个分析Agent并行
      ├── event_analyst ────────────┤
      ├── quality_risk_analyst ─────┤
      ├── moneyflow_analyst ────────┘
      ├── short_term_scorer ────────→ state.data.short_term_score
      ├── medium_term_scorer ───────→ state.data.medium_term_score
      └── long_term_scorer ─────────→ state.data.long_term_score

评分直接来自专用打分Agent，不再从Markdown报告提取。
"""
import os
import sys
import time
import asyncio
import math
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
from src.agents.event_analyst_agent import event_analyst_agent
from src.agents.quality_risk_analyst_agent import quality_risk_analyst_agent
from src.agents.moneyflow_analyst_agent import moneyflow_analyst_agent
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
    start_node → [fundamental, technical, value, news, event, quality_risk, moneyflow] → [short_term, medium_term, long_term] → END

    评分直接来自3个打分Agent，以中线评分为主评分写入股票池。
    """

    def __init__(self, pool_manager: Optional[StockPoolManager] = None):
        """
        Args:
            pool_manager: StockPoolManager 实例。
                - None (默认): 自动创建，评分结果写入 stock_pool.json
                - StockPoolManager 实例: 使用指定实例
                - False: 禁用池写入（pool screening 等外部调用场景）
        """
        if pool_manager is False:
            self.pool_manager = None
        else:
            self.pool_manager = pool_manager or StockPoolManager()
        self._workflow = None
        self._term_workflows: Dict[str, Any] = {}

    @staticmethod
    async def _prefetch(term: str, stock_code: str, initial_state: AgentState) -> None:
        """4.1 DataGateway：打分前按期限统一预取数据（失败不影响主流程）。"""
        from src.data.data_gateway import prefetch_term_bundle

        data = initial_state.get("data", {})
        await prefetch_term_bundle(
            term,
            stock_code,
            current_date=data.get("current_date", ""),
            is_etf=data.get("is_etf", False),
        )

    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        if self._workflow is None:
            workflow = StateGraph(AgentState)

            # 分析节点（7个并行）
            workflow.add_node("start_node", lambda state: state)
            workflow.add_node("fundamental_analyst", fundamental_agent)
            workflow.add_node("technical_analyst", technical_agent)
            workflow.add_node("value_analyst", value_agent)
            workflow.add_node("news_analyst", news_agent)
            workflow.add_node("event_analyst", event_analyst_agent)
            workflow.add_node("quality_risk_analyst", quality_risk_analyst_agent)
            workflow.add_node("moneyflow_analyst", moneyflow_analyst_agent)

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
            workflow.add_edge("start_node", "event_analyst")
            workflow.add_edge("start_node", "quality_risk_analyst")
            workflow.add_edge("start_node", "moneyflow_analyst")

            # 分析节点 → 打分节点
            # short_term 只需要 technical + news
            workflow.add_edge("technical_analyst", "short_term_scorer")
            workflow.add_edge("news_analyst", "short_term_scorer")
            workflow.add_edge("event_analyst", "short_term_scorer")
            workflow.add_edge("moneyflow_analyst", "short_term_scorer")

            # medium_term 需要全部7个
            workflow.add_edge("fundamental_analyst", "medium_term_scorer")
            workflow.add_edge("technical_analyst", "medium_term_scorer")
            workflow.add_edge("value_analyst", "medium_term_scorer")
            workflow.add_edge("news_analyst", "medium_term_scorer")
            workflow.add_edge("event_analyst", "medium_term_scorer")
            workflow.add_edge("quality_risk_analyst", "medium_term_scorer")
            workflow.add_edge("moneyflow_analyst", "medium_term_scorer")

            # long_term 需要全部7个
            workflow.add_edge("fundamental_analyst", "long_term_scorer")
            workflow.add_edge("technical_analyst", "long_term_scorer")
            workflow.add_edge("value_analyst", "long_term_scorer")
            workflow.add_edge("news_analyst", "long_term_scorer")
            workflow.add_edge("event_analyst", "long_term_scorer")
            workflow.add_edge("quality_risk_analyst", "long_term_scorer")
            workflow.add_edge("moneyflow_analyst", "long_term_scorer")

            # 打分节点 → END
            workflow.add_edge("short_term_scorer", END)
            workflow.add_edge("medium_term_scorer", END)
            workflow.add_edge("long_term_scorer", END)

            self._workflow = workflow.compile()
        return self._workflow

    # 各期限打分实际依赖的分析 Agent（4.2 期限子图）
    _TERM_AGENTS = {
        "short": ["technical", "news", "event", "moneyflow"],
        "medium": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
        "long": ["fundamental", "technical", "value", "news", "event", "quality_risk", "moneyflow"],
    }
    _AGENT_NODE_SPECS = {
        "fundamental": ("fundamental_analyst", fundamental_agent),
        "technical": ("technical_analyst", technical_agent),
        "value": ("value_analyst", value_agent),
        "news": ("news_analyst", news_agent),
        "event": ("event_analyst", event_analyst_agent),
        "quality_risk": ("quality_risk_analyst", quality_risk_analyst_agent),
        "moneyflow": ("moneyflow_analyst", moneyflow_analyst_agent),
    }

    def _build_term_workflow(self, term: str) -> StateGraph:
        """构建单期限评分子图（4.2）：只挂该期限实际需要的分析 Agent 和 scorer。

        - short: technical/news/event/moneyflow + short_term_scorer
        - medium/long: 全部 7 个分析 Agent + 对应期限 scorer

        score_stock() 使用的全图保持不变；本方法仅服务于 score_stock_for_term()
        与 score_for_quick_screen()，消除单期限打分白跑另外两个 scorer 及
        短期不需要的三个财务 Agent 的浪费。
        """
        if term not in self._TERM_AGENTS:
            raise ValueError(f"未知期限: {term} (应为 short/medium/long)")
        if term in self._term_workflows:
            return self._term_workflows[term]

        from src.agents.scoring_nodes import (
            short_term_scorer_node,
            medium_term_scorer_node,
            long_term_scorer_node,
        )
        scorer_specs = {
            "short": ("short_term_scorer", short_term_scorer_node),
            "medium": ("medium_term_scorer", medium_term_scorer_node),
            "long": ("long_term_scorer", long_term_scorer_node),
        }

        workflow = StateGraph(AgentState)
        workflow.add_node("start_node", lambda state: state)

        agent_nodes = []
        for agent_key in self._TERM_AGENTS[term]:
            node_name, node_fn = self._AGENT_NODE_SPECS[agent_key]
            workflow.add_node(node_name, node_fn)
            agent_nodes.append(node_name)

        scorer_name, scorer_fn = scorer_specs[term]
        workflow.add_node(scorer_name, scorer_fn)

        workflow.set_entry_point("start_node")
        for node_name in agent_nodes:
            workflow.add_edge("start_node", node_name)
            workflow.add_edge(node_name, scorer_name)
        workflow.add_edge(scorer_name, END)

        self._term_workflows[term] = workflow.compile()
        return self._term_workflows[term]

    @staticmethod
    def _build_initial_state(
        stock_code: str, company_name: str,
        model_config: Optional[Dict[str, str]] = None,
        skip_cache: bool = False,
        thinking_enabled: bool = True,
    ) -> AgentState:
        """构建Agent初始状态

        Args:
            stock_code: 股票代码
            company_name: 公司名称
            model_config: 可选的模型覆盖配置 {"model_name", "model_api_key", "model_base_url"}
            skip_cache: 是否跳过中间产物缓存（快筛模式）
            thinking_enabled: 打分Agent是否启用思考模式（快筛模式关闭以提速）
        """
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
            "analysis_timestamp": current_datetime.isoformat(),
            "skip_cache": skip_cache,
            "thinking_enabled": thinking_enabled,
            "is_etf": ScoringEngine._is_etf(stock_code),
            "analysis_version": "a_share_v2",
        }
        # 注入模型覆盖配置（快筛模式使用不同模型）
        if model_config:
            initial_data["model_name"] = model_config.get("model_name", "")
            initial_data["model_api_key"] = model_config.get("model_api_key", "")
            initial_data["model_base_url"] = model_config.get("model_base_url", "")

        return AgentState(
            messages=[],
            data=initial_data,
            metadata={}
        )

    @staticmethod
    def _is_etf(stock_code: str) -> bool:
        """检测股票代码是否为ETF/基金类产品（上交所51/58，深交所15/16/18）"""
        code = stock_code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()
        return code.startswith(("51", "58", "15", "16", "18"))

    @staticmethod
    def _has_valid_score(score_payload: Any) -> bool:
        """Return True only for a finite, bounded numeric scorer result."""
        if not isinstance(score_payload, dict):
            return False
        score = score_payload.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        return math.isfinite(float(score)) and 0.0 <= float(score) <= 100.0

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
            for t in ("short", "medium", "long"):
                self.pool_manager.update_stock_status(t, stock_code, "scoring")

        logger.info(f"{WAIT_ICON} 开始对 {company_name}({stock_code}) 运行完整评分Pipeline...")
        start_time = time.time()

        try:
            # 1. 构建工作流
            app = self._build_workflow()

            # 2. 构建初始状态
            initial_state = self._build_initial_state(stock_code, company_name)

            # 2.5 统一预取（4.1）：全图所需数据一次取齐，Agent 共享快照
            await self._prefetch("full", stock_code, initial_state)

            # 3. 执行Pipeline: 7个分析Agent并行 → 3个打分Agent并行
            logger.info(f"{WAIT_ICON} 正在运行分析+打分Pipeline...")
            final_state = await asyncio.wait_for(
                app.ainvoke(initial_state), timeout=2400.0
            )

            # 4. 直接从状态中获取三种评分
            data = final_state.get("data", {})
            short_term_score = data.get("short_term_score", {})
            medium_term_score = data.get("medium_term_score", {})
            long_term_score = data.get("long_term_score", {})

            missing_scores = [
                term for term, payload in (
                    ("短线", short_term_score),
                    ("中线", medium_term_score),
                    ("长线", long_term_score),
                )
                if not self._has_valid_score(payload)
            ]
            if missing_scores:
                raise ValueError(
                    f"Pipeline未生成有效评分: {', '.join(missing_scores)}"
                )

            # 5. 构建存储数据：以中线评分为主评分
            score_data = {
                # 主评分（中线，用于排序和显示）
                "score": medium_term_score.get("score"),
                "recommendation": medium_term_score.get("rating", ""),
                # 三种完整评分
                "short_term_score": short_term_score,
                "medium_term_score": medium_term_score,
                "long_term_score": long_term_score,
                "company_name": company_name,
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

            # 提取所有agent的signal_pack和分析文本
            signal_packs = {}
            analysis_texts = {}
            for agent in ("fundamental", "technical", "value", "news",
                          "event", "quality_risk", "moneyflow"):
                sp_key = f"{agent}_signal_pack"
                if sp_key in data and data[sp_key]:
                    signal_packs[agent] = data[sp_key]
                txt_key = f"{agent}_analysis"
                if txt_key in data and data[txt_key]:
                    analysis_texts[agent] = data[txt_key]

            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "score_data": score_data,
                "signal_packs": signal_packs,
                "analysis_texts": analysis_texts,
                "execution_time": elapsed
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{ERROR_ICON} {company_name} 评分失败: {e}")
            if self.pool_manager:
                for t in ("short", "medium", "long"):
                    self.pool_manager.update_stock_status(t, stock_code, "failed")
                # Failure must not overwrite a previously valid score with None.
                # The explicit failed status above is sufficient for retry/UI
                # visibility; score persistence is reserved for valid results.

            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "score_data": None,
                "error": str(e),
                "execution_time": elapsed
            }

    async def score_stock_for_term(self, term: str, stock_code: str, company_name: str) -> Dict[str, Any]:
        """
        对指定股票运行完整分析+打分，但只存储指定期限的评分到池中

        Args:
            term: short/medium/long
            stock_code: 股票代码(带交易所前缀)
            company_name: 公司名称

        Returns:
            评分结果字典
        """
        if self.pool_manager:
            self.pool_manager.update_stock_status(term, stock_code, "scoring")

        logger.info(f"{WAIT_ICON} 开始对 {company_name}({stock_code}) 运行{term}评分Pipeline...")
        start_time = time.time()

        try:
            # 4.2 期限子图：只运行该期限实际需要的 Agent 和 scorer
            app = self._build_term_workflow(term)
            initial_state = self._build_initial_state(stock_code, company_name)
            await self._prefetch(term, stock_code, initial_state)
            final_state = await asyncio.wait_for(
                app.ainvoke(initial_state), timeout=2400.0
            )

            data = final_state.get("data", {})
            term_scores = {
                "short": data.get("short_term_score", {}),
                "medium": data.get("medium_term_score", {}),
                "long": data.get("long_term_score", {}),
            }

            target_score = term_scores.get(term, {})
            if not self._has_valid_score(target_score):
                raise ValueError(f"Pipeline未生成有效{term}评分")

            score_data = {
                "score": target_score.get("score"),
                "recommendation": target_score.get("rating", target_score.get("recommendation", "")),
                "status": "scored",
            }

            elapsed = time.time() - start_time
            logger.info(
                f"{SUCCESS_ICON} {company_name} {term}评分完成: "
                f"{target_score.get('score', '-')} 分, 耗时={elapsed:.1f}s"
            )

            if self.pool_manager:
                self.pool_manager.update_term_score(term, stock_code, target_score)

            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "term": term,
                "term_score": target_score,
                "all_scores": term_scores,
                "execution_time": elapsed,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{ERROR_ICON} {company_name} {term}评分失败: {e}")
            if self.pool_manager:
                self.pool_manager.update_stock_status(term, stock_code, "failed")
            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "term": term,
                "error": str(e),
                "execution_time": elapsed,
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

    async def score_for_quick_screen(
        self, term: str, stock_code: str, company_name: str
    ) -> Dict[str, Any]:
        """
        快筛股票池打分：使用 qwen3.6-flash 模型，不缓存中间产物。

        Args:
            term: short/medium/long
            stock_code: 股票代码
            company_name: 公司名称

        Returns:
            {score, score_time, term, stock_code, company_name}
        """
        model_config = {
            "model_name": os.getenv("OPENAI_COMPATIBLE_MODEL_2", "qwen3.6-flash"),
            "model_api_key": os.getenv("OPENAI_COMPATIBLE_API_KEY_2", ""),
            "model_base_url": os.getenv("OPENAI_COMPATIBLE_BASE_URL_2", ""),
        }

        logger.info(
            f"{WAIT_ICON} QuickScreen: 开始对 {company_name}({stock_code}) 进行{term}打分 "
            f"(模型={model_config['model_name']}, 跳过缓存)"
        )
        start_time = time.time()

        try:
            # 4.2 期限子图：快筛同样只运行目标期限所需节点
            app = self._build_term_workflow(term)
            initial_state = self._build_initial_state(
                stock_code, company_name,
                model_config=model_config,
                skip_cache=True,
                thinking_enabled=True,  # MCP已绕过，thinking对速度影响可忽略
            )
            await self._prefetch(term, stock_code, initial_state)
            final_state = await asyncio.wait_for(
                app.ainvoke(initial_state), timeout=2400.0
            )

            data = final_state.get("data", {})
            term_scores = {
                "short": data.get("short_term_score", {}),
                "medium": data.get("medium_term_score", {}),
                "long": data.get("long_term_score", {}),
            }
            target_score = term_scores.get(term, {})

            if not self._has_valid_score(target_score):
                raise ValueError(f"Pipeline未生成有效{term}评分")

            elapsed = time.time() - start_time
            logger.info(
                f"{SUCCESS_ICON} QuickScreen: {company_name} {term}评分="
                f"{target_score.get('score', '-')} 分, 耗时={elapsed:.1f}s"
            )

            return {
                "score": target_score.get("score"),
                "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "term": term,
                "stock_code": stock_code,
                "company_name": company_name,
            }

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{ERROR_ICON} QuickScreen {company_name} {term}打分失败: {e}")
            return {
                "stock_code": stock_code,
                "company_name": company_name,
                "term": term,
                "error": str(e),
                "execution_time": elapsed,
            }
