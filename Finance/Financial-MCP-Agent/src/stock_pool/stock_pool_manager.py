"""
StockPoolManager: 股票池持久化与管理
提供股票的增删改查操作，使用JSON文件持久化存储
"""
import os
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)


class StockPoolManager:
    """
    股票池管理器，负责股票的CRUD操作和持久化存储

    每只股票的数据结构:
    {
        "stock_code": "sh.603871",
        "company_name": "嘉友国际",
        "score": null,                    # 主评分(=中线评分,用于排序显示)
        "recommendation": "",             # 主评级(=中线评级)
        "short_term_score": {},           # 短线完整评分对象(short_term_scorer输出)
        "medium_term_score": {},          # 中线完整评分对象(medium_term_scorer输出)
        "long_term_score": {},            # 长线完整评分对象(long_term_scorer输出)
        "last_updated": "",               # 最后更新时间
        "status": "pending",              # pending|scoring|scored|failed
        "score_history": []               # 历史评分记录
    }
    """

    def __init__(self, pool_path: Optional[str] = None):
        if pool_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pool_path = os.path.join(project_root, "stock_pool.json")

        self.pool_path = pool_path
        self.stocks: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        """从JSON文件加载股票池"""
        if os.path.exists(self.pool_path):
            try:
                with open(self.pool_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stocks = data.get("stocks", {})
                logger.info(f"{SUCCESS_ICON} 股票池已加载: {self.pool_path}, 共{len(self.stocks)}只股票")
            except Exception as e:
                logger.error(f"{ERROR_ICON} 加载股票池失败: {e}")
                self.stocks = {}
        else:
            logger.info(f"{WAIT_ICON} 股票池文件不存在，将创建新的: {self.pool_path}")
            self.stocks = {}

    def _save(self):
        """保存股票池到JSON文件"""
        try:
            os.makedirs(os.path.dirname(self.pool_path) or ".", exist_ok=True)
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump({"stocks": self.stocks, "updated_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"{ERROR_ICON} 保存股票池失败: {e}")
            raise

    def add_stock(self, stock_code: str, company_name: str) -> Dict[str, Any]:
        """
        添加股票到股票池

        Args:
            stock_code: 股票代码(带交易所前缀, 如 sh.603871)
            company_name: 公司名称

        Returns:
            股票信息字典
        """
        if stock_code in self.stocks:
            logger.info(f"{WAIT_ICON} 股票 {company_name}({stock_code}) 已在池中，更新名称")
            self.stocks[stock_code]["company_name"] = company_name
            self._save()
            return self.stocks[stock_code]

        self.stocks[stock_code] = {
            "stock_code": stock_code,
            "company_name": company_name,
            "score": None,
            "recommendation": "",
            "short_term_score": {},
            "medium_term_score": {},
            "long_term_score": {},
            "last_updated": "",
            "status": "pending",
            "score_history": []
        }
        self._save()
        logger.info(f"{SUCCESS_ICON} 已添加股票: {company_name}({stock_code})")
        return self.stocks[stock_code]

    def remove_stock(self, stock_code: str) -> bool:
        """
        从股票池中删除股票

        Args:
            stock_code: 股票代码

        Returns:
            是否删除成功
        """
        if stock_code not in self.stocks:
            logger.warning(f"{ERROR_ICON} 股票 {stock_code} 不在池中")
            return False

        name = self.stocks[stock_code]["company_name"]
        del self.stocks[stock_code]
        self._save()
        logger.info(f"{SUCCESS_ICON} 已删除股票: {name}({stock_code})")
        return True

    def get_stock(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取单只股票信息"""
        return self.stocks.get(stock_code)

    def list_stocks(self) -> List[Dict[str, Any]]:
        """
        列出股票池中所有股票

        Returns:
            股票信息列表，按评分降序排列(无评分的排最后)
        """
        stocks_list = list(self.stocks.values())
        stocks_list.sort(key=lambda s: s.get("score") or 0, reverse=True)
        return stocks_list

    def update_stock_score(self, stock_code: str, score_data: Dict[str, Any]):
        """
        更新股票评分数据

        Args:
            stock_code: 股票代码
            score_data: 评分数据字典，包含score, recommendation, short_term_score,
                       medium_term_score, long_term_score等字段
        """
        if stock_code not in self.stocks:
            raise ValueError(f"股票 {stock_code} 不在池中")

        stock = self.stocks[stock_code]
        # 保存旧评分到历史
        if stock["score"] is not None:
            stock["score_history"].append({
                "score": stock["score"],
                "recommendation": stock["recommendation"],
                "updated_at": stock["last_updated"]
            })

        # 更新新评分
        stock["score"] = score_data.get("score")
        stock["recommendation"] = score_data.get("recommendation", "")
        stock["short_term_score"] = score_data.get("short_term_score", {})
        stock["medium_term_score"] = score_data.get("medium_term_score", {})
        stock["long_term_score"] = score_data.get("long_term_score", {})
        stock["last_updated"] = datetime.now().isoformat()
        stock["status"] = score_data.get("status", "scored")

        self._save()
        logger.info(f"{SUCCESS_ICON} 已更新 {stock['company_name']} 评分: {stock['score']} ({stock['recommendation']})")

    def update_stock_status(self, stock_code: str, status: str):
        """更新股票状态(pending/scoring/scored/failed)"""
        if stock_code in self.stocks:
            self.stocks[stock_code]["status"] = status
            self._save()

    def get_scored_stocks(self) -> List[Dict[str, Any]]:
        """获取已评分的股票列表(按评分降序)"""
        return [s for s in self.list_stocks() if s["score"] is not None]

    def get_pending_stocks(self) -> List[Dict[str, Any]]:
        """获取待评分的股票列表"""
        return [s for s in self.list_stocks() if s["status"] == "pending"]

    def count(self) -> int:
        """返回股票池中股票数量"""
        return len(self.stocks)
