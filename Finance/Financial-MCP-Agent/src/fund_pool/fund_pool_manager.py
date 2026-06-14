"""
FundPoolManager: 基金池持久化与管理
提供基金的增删改查操作，使用JSON文件持久化存储

数据架构：两个独立的池（已评分/观察名单），每只基金拥有独立的评分和状态
"""
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

VALID_POOLS = ("scored", "watchlist")

POOL_LABELS = {
    "scored": "已评分",
    "watchlist": "观察名单",
}


class FundPoolManager:
    """
    基金池管理器，负责两个独立池的CRUD操作和持久化存储

    数据架构:
    {
        "updated_at": "...",
        "pools": {
            "scored":    { "funds": { "sh.510050": {...}, ... } },
            "watchlist": { "funds": { "sh.510050": {...}, ... } }
        }
    }

    每只基金在池内的数据结构:
    {
        "fund_code": "sh.510050",
        "fund_name": "华夏上证50ETF",
        "fund_type": "ETF",
        "score": 82,
        "rating": "较优",
        "investment_view": "可关注",
        "holding_period": "1年以上",
        "subscores": {
            "product_positioning": 85,
            "performance_risk": 78,
            ...
        },
        "strengths": [],
        "risks": [],
        "last_updated": "",
        "status": "pending",                # pending|scoring|scored|failed
        "score_history": []                 # 历史评分记录
    }
    """

    def __init__(self, pool_file: str = "fund_pool.json"):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.pool_path = os.path.join(project_root, pool_file)

        self.pools: Dict[str, Dict[str, Any]] = {
            "scored": {"funds": {}},
            "watchlist": {"funds": {}},
        }
        self._load()

    def _load(self):
        """从JSON文件加载基金池"""
        if not os.path.exists(self.pool_path):
            logger.info(f"{WAIT_ICON} 基金池文件不存在，将创建新的: {self.pool_path}")
            return

        try:
            with open(self.pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"{ERROR_ICON} 加载基金池失败: {e}")
            return

        if "pools" in data:
            for pool in VALID_POOLS:
                if pool in data["pools"]:
                    self.pools[pool]["funds"] = data["pools"][pool].get("funds", {})
            logger.info(f"{SUCCESS_ICON} 基金池已加载: {self.pool_path}")
            for pool in VALID_POOLS:
                count = len(self.pools[pool]["funds"])
                logger.info(f"  {POOL_LABELS[pool]}: {count} 只基金")

    def _save(self):
        """保存基金池到JSON文件"""
        try:
            os.makedirs(os.path.dirname(self.pool_path) or ".", exist_ok=True)
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.now().isoformat(),
                    "pools": self.pools,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"{ERROR_ICON} 保存基金池失败: {e}")
            raise

    def _fund_entry(self, fund_code: str, fund_name: str, fund_type: Optional[str] = None) -> Dict[str, Any]:
        """构建基金标准数据条目"""
        return {
            "fund_code": fund_code,
            "fund_name": fund_name,
            "fund_type": fund_type or "",
            "score": None,
            "rating": "",
            "investment_view": "",
            "holding_period": "",
            "subscores": {},
            "strengths": [],
            "risks": [],
            "last_updated": "",
            "status": "pending",
            "score_history": [],
        }

    # ─── CRUD 操作 ───

    def add_fund(self, fund_code: str, fund_name: str, fund_type: Optional[str] = None, pool: str = "watchlist") -> Dict[str, Any]:
        """
        添加基金到指定池

        Args:
            fund_code: 基金代码（带交易所前缀，如 sh.510050）
            fund_name: 基金名称
            fund_type: 基金类型（ETF/LOF/主动权益/债券/货币/QDII/FOF）
            pool: 目标池（scored/watchlist），默认 watchlist

        Returns:
            基金信息字典
        """
        if pool not in VALID_POOLS:
            raise ValueError(f"无效的池名称: {pool}，可选: {VALID_POOLS}")

        funds = self.pools[pool]["funds"]
        if fund_code in funds:
            logger.info(f"{WAIT_ICON} 基金 {fund_name}({fund_code}) 已在{POOL_LABELS[pool]}池中，更新信息")
            funds[fund_code]["fund_name"] = fund_name
            if fund_type:
                funds[fund_code]["fund_type"] = fund_type
            self._save()
            return funds[fund_code]

        entry = self._fund_entry(fund_code, fund_name, fund_type)
        funds[fund_code] = entry
        self._save()
        logger.info(f"{SUCCESS_ICON} 已添加 {fund_name}({fund_code}) 到{POOL_LABELS[pool]}池")
        return entry

    def remove_fund(self, fund_code: str, pool: str = "watchlist") -> bool:
        """
        从指定池删除基金

        Args:
            fund_code: 基金代码
            pool: 目标池（scored/watchlist），默认 watchlist

        Returns:
            是否成功删除
        """
        if pool not in VALID_POOLS:
            raise ValueError(f"无效的池名称: {pool}，可选: {VALID_POOLS}")

        funds = self.pools[pool]["funds"]
        if fund_code not in funds:
            logger.warning(f"{ERROR_ICON} 基金 {fund_code} 不在{POOL_LABELS[pool]}池中")
            return False

        name = funds[fund_code]["fund_name"]
        del funds[fund_code]
        self._save()
        logger.info(f"{SUCCESS_ICON} 已从{POOL_LABELS[pool]}池删除 {name}({fund_code})")
        return True

    def update_score(self, fund_code: str, score_data: Dict[str, Any], pool: str = "scored") -> bool:
        """
        更新基金评分

        Args:
            fund_code: 基金代码
            score_data: 评分数据，包含:
                - score: int
                - rating: str
                - investment_view: str
                - holding_period: str
                - subscores: dict (7 dimensions)
                - strengths: list[str]
                - risks: list[str]
            pool: 目标池（scored/watchlist），默认 scored

        Returns:
            是否成功更新
        """
        if pool not in VALID_POOLS:
            raise ValueError(f"无效的池名称: {pool}，可选: {VALID_POOLS}")

        funds = self.pools[pool]["funds"]
        if fund_code not in funds:
            logger.warning(f"{ERROR_ICON} 基金 {fund_code} 不在{POOL_LABELS[pool]}池中")
            return False

        fund = funds[fund_code]

        # 保存历史评分
        if fund["score"] is not None:
            fund["score_history"].append({
                "score": fund["score"],
                "rating": fund["rating"],
                "investment_view": fund["investment_view"],
                "updated_at": fund["last_updated"],
            })

        # 更新评分字段
        fund["score"] = score_data.get("score")
        fund["rating"] = score_data.get("rating", "")
        fund["investment_view"] = score_data.get("investment_view", "")
        fund["holding_period"] = score_data.get("holding_period", "")
        fund["subscores"] = score_data.get("subscores", {})
        fund["strengths"] = score_data.get("strengths", [])
        fund["risks"] = score_data.get("risks", [])
        fund["last_updated"] = datetime.now().isoformat()
        fund["status"] = score_data.get("status", "scored")

        if score_data.get("fund_type") and not fund["fund_type"]:
            fund["fund_type"] = score_data["fund_type"]

        self._save()
        logger.info(f"{SUCCESS_ICON} 已更新基金评分: {fund_code} 得分={fund['score']} 评级={fund['rating']}")
        return True

    def get_fund(self, fund_code: str) -> Optional[Dict[str, Any]]:
        """
        获取基金信息（跨池查找）

        Args:
            fund_code: 基金代码

        Returns:
            基金信息字典，未找到返回 None
        """
        for pool in VALID_POOLS:
            if fund_code in self.pools[pool]["funds"]:
                return dict(self.pools[pool]["funds"][fund_code])
        return None

    def list_funds(self, pool: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出基金

        Args:
            pool: 指定池名称（scored/watchlist），None 表示合并所有池

        Returns:
            基金信息列表（scored 池按评分降序排列）
        """
        if pool is not None:
            if pool not in VALID_POOLS:
                raise ValueError(f"无效的池名称: {pool}，可选: {VALID_POOLS}")
            funds_list = list(self.pools[pool]["funds"].values())
            if pool == "scored":
                funds_list.sort(key=lambda f: f.get("score") or 0, reverse=True)
            return funds_list

        # 合并所有池（去重，scored 优先）
        seen = {}
        for pool in ("scored", "watchlist"):
            for code, fund in self.pools[pool]["funds"].items():
                if code not in seen:
                    seen[code] = dict(fund)
        result = list(seen.values())
        return result

    def move_to_scored(self, fund_code: str) -> bool:
        """
        将基金从观察名单移至已评分池

        Args:
            fund_code: 基金代码

        Returns:
            是否成功移动
        """
        if fund_code not in self.pools["watchlist"]["funds"]:
            logger.warning(f"{ERROR_ICON} 基金 {fund_code} 不在观察名单中")
            return False

        fund = self.pools["watchlist"]["funds"].pop(fund_code)
        self.pools["scored"]["funds"][fund_code] = fund
        fund["last_updated"] = datetime.now().isoformat()
        self._save()
        logger.info(f"{SUCCESS_ICON} 已将 {fund['fund_name']}({fund_code}) 移至已评分池")
        return True

    def get_scored_funds_sorted(self) -> List[Dict[str, Any]]:
        """
        获取已评分基金，按评分降序排列

        Returns:
            已评分基金列表
        """
        funds_list = list(self.pools["scored"]["funds"].values())
        funds_list.sort(key=lambda f: f.get("score") or 0, reverse=True)
        return funds_list

    def get_fund_in_pool(self, fund_code: str, pool: str) -> Optional[Dict[str, Any]]:
        """
        获取指定池中某基金

        Args:
            fund_code: 基金代码
            pool: 池名称

        Returns:
            基金信息字典，未找到返回 None
        """
        if pool not in VALID_POOLS:
            raise ValueError(f"无效的池名称: {pool}，可选: {VALID_POOLS}")
        return self.pools[pool]["funds"].get(fund_code)

    def update_fund_status(self, pool: str, fund_code: str, status: str):
        """
        更新基金状态

        Args:
            pool: 池名称
            fund_code: 基金代码
            status: 状态值（pending/scoring/scored/failed）
        """
        if pool not in VALID_POOLS:
            raise ValueError(f"无效的池名称: {pool}，可选: {VALID_POOLS}")
        if fund_code in self.pools[pool]["funds"]:
            self.pools[pool]["funds"][fund_code]["status"] = status
            self._save()

    def count(self) -> int:
        """返回基金池中基金总数（去重）"""
        all_codes = set()
        for pool in VALID_POOLS:
            all_codes.update(self.pools[pool]["funds"].keys())
        return len(all_codes)
