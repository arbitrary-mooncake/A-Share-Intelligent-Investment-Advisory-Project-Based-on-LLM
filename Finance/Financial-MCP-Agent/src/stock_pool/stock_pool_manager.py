"""
StockPoolManager: 股票池持久化与管理
提供股票的增删改查操作，使用JSON文件持久化存储

数据架构：三个独立的期限池（短线/中线/长线），每只股票在各池中拥有独立的评分和状态
"""
import os
import json
import time
import math
from numbers import Real
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON

logger = setup_logger(__name__)

VALID_TERMS = ("short", "medium", "long", "quick_screen", "fine")

# 期限显示名称
TERM_LABELS = {
    "short": "短线",
    "medium": "中线",
    "long": "长线",
    "quick_screen": "快筛",
    "fine": "精筛",
}


class StockPoolManager:
    """
    股票池管理器，负责三个独立期限池的CRUD操作和持久化存储

    数据架构:
    {
        "updated_at": "...",
        "pools": {
            "short":  { "stocks": { "sh.603871": {...}, ... } },
            "medium": { "stocks": { "sh.603871": {...}, ... } },
            "long":   { "stocks": { "sh.603871": {...}, ... } }
        }
    }

    每只股票在池内的数据结构:
    {
        "stock_code": "sh.603871",
        "company_name": "嘉友国际",
        "score": null,                    # 当前池的评分
        "recommendation": "",             # 当前池的评级
        "term_score": {},                 # 当前期限的完整评分对象
        "last_updated": "",
        "status": "pending",              # pending|scoring|scored|failed
        "score_history": []               # 历史评分记录
    }
    """

    def __init__(self, pool_path: Optional[str] = None):
        if pool_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            pool_path = os.path.join(project_root, "stock_pool.json")

        self.pool_path = pool_path
        self.pools: Dict[str, Dict[str, Any]] = {
            "short": {"stocks": {}},
            "medium": {"stocks": {}},
            "long": {"stocks": {}},
            "quick_screen": {"stocks": {}},
            "fine": {"stocks": {}},
        }
        self._load()
        self._migrate_to_fine()

    def _load(self):
        """从JSON文件加载股票池，自动迁移旧格式"""
        if not os.path.exists(self.pool_path):
            logger.info(f"{WAIT_ICON} 股票池文件不存在，将创建新的: {self.pool_path}")
            return

        try:
            with open(self.pool_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"{ERROR_ICON} 加载股票池失败: {e}")
            return

        if "pools" in data:
            # 新格式：直接加载
            for term in VALID_TERMS:
                if term in data["pools"]:
                    self.pools[term]["stocks"] = data["pools"][term].get("stocks", {})
            logger.info(f"{SUCCESS_ICON} 股票池已加载: {self.pool_path}")
            for term in VALID_TERMS:
                count = len(self.pools[term]["stocks"])
                logger.info(f"  {TERM_LABELS[term]}: {count} 只股票")
        else:
            # 旧格式迁移：将扁平 stocks dict 复制到三个独立池
            old_stocks = data.get("stocks", {})
            for term in ("short", "medium", "long"):
                for code, stock_data in old_stocks.items():
                    entry = {
                        "stock_code": stock_data.get("stock_code", code),
                        "company_name": stock_data.get("company_name", ""),
                        "score": stock_data.get("score"),
                        "recommendation": stock_data.get("recommendation", ""),
                        "term_score": stock_data.get(f"{term}_term_score", {}),
                        "last_updated": stock_data.get("last_updated", ""),
                        "status": stock_data.get("status", "pending"),
                        "score_history": list(stock_data.get("score_history", [])),
                    }
                    self.pools[term]["stocks"][code] = entry
            logger.info(f"{SUCCESS_ICON} 股票池已从旧格式迁移: {len(old_stocks)} 只股票 → 3个独立池")
            self._save()

    @staticmethod
    def _is_bad_name(name: str, stock_code: str) -> bool:
        """检测 company_name 是否为无效占位（股票代码本身或空）"""
        if not name:
            return True
        bare = stock_code.replace("sh.", "").replace("sz.", "")
        return name == stock_code or name == bare

    def _migrate_to_fine(self):
        """将 short/medium/long 三池股票合并到 fine（精筛）池，去重，保留已有评分"""
        fine_stocks = self.pools["fine"]["stocks"]
        migrated = 0
        updated = 0
        name_fixed = 0

        for src_term in ("short", "medium", "long"):
            for code, stock in self.pools[src_term]["stocks"].items():
                if code in fine_stocks:
                    existing = fine_stocks[code]
                    if "scores" not in existing:
                        existing["scores"] = {}
                    ts = stock.get("term_score", {})
                    if ts:
                        existing["scores"][src_term] = {
                            "score": ts.get("score"),
                            "rating": ts.get("rating", ts.get("recommendation", "")),
                            "reasoning": ts.get("reasoning", ""),
                            "risk_warning": ts.get("risk_warning", ""),
                            "suggested_action": ts.get("suggested_action", ""),
                        }
                        updated += 1
                    src_name = stock.get("company_name", "")
                    existing_name = existing.get("company_name", "")
                    if src_name and self._is_bad_name(existing_name, code) and not self._is_bad_name(src_name, code):
                        existing["company_name"] = src_name
                        name_fixed += 1
                    existing["last_updated"] = stock.get("last_updated") or existing.get("last_updated", "")
                    if stock.get("status") == "scored":
                        existing["status"] = "scored"
                else:
                    entry = {
                        "stock_code": code,
                        "company_name": stock.get("company_name", ""),
                        "status": stock.get("status", "pending"),
                        "scores": {},
                        "last_updated": stock.get("last_updated", ""),
                        "score_history": list(stock.get("score_history", [])),
                    }
                    ts = stock.get("term_score", {})
                    if ts:
                        entry["scores"][src_term] = {
                            "score": ts.get("score"),
                            "rating": ts.get("rating", ts.get("recommendation", "")),
                            "reasoning": ts.get("reasoning", ""),
                            "risk_warning": ts.get("risk_warning", ""),
                            "suggested_action": ts.get("suggested_action", ""),
                        }
                    fine_stocks[code] = entry
                    migrated += 1

        if migrated > 0 or updated > 0 or name_fixed > 0:
            logger.info(f"{SUCCESS_ICON} 精筛池迁移完成: 新{migrated}只, 更新{updated}只, 名称修复{name_fixed}只")
            self._save()

    # ─── 精筛池专用操作 ───

    def add_stock_to_fine(self, stock_code: str, company_name: str) -> Dict[str, Any]:
        """添加股票到精筛池"""
        stocks = self.pools["fine"]["stocks"]
        if stock_code in stocks:
            stocks[stock_code]["company_name"] = company_name
            self._save()
            return stocks[stock_code]

        entry = {
            "stock_code": stock_code,
            "company_name": company_name,
            "status": "pending",
            "scores": {},
            "last_updated": "",
            "score_history": [],
        }
        stocks[stock_code] = entry
        self._save()
        logger.info(f"{SUCCESS_ICON} 已添加 {company_name}({stock_code}) 到精筛池")
        return entry

    def remove_stock_from_fine(self, stock_code: str) -> bool:
        """从精筛池删除股票"""
        stocks = self.pools["fine"]["stocks"]
        if stock_code not in stocks:
            return False
        name = stocks[stock_code]["company_name"]
        del stocks[stock_code]
        self._save()
        logger.info(f"{SUCCESS_ICON} 已从精筛池删除 {name}({stock_code})")
        return True

    def get_fine_pool(self) -> List[Dict[str, Any]]:
        """获取精筛池所有股票，按中线评分降序"""
        stocks_list = list(self.pools["fine"]["stocks"].values())
        stocks_list.sort(key=lambda s: (
            s.get("scores", {}).get("medium", {}).get("score") or 0
        ), reverse=True)
        return stocks_list

    def get_stock_in_fine(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取精筛池中某股票"""
        return self.pools["fine"]["stocks"].get(stock_code)

    def update_fine_scores(self, stock_code: str, scores_data: Dict[str, Any],
                           company_name: str = ""):
        """一次性更新精筛池三期限评分

        Args:
            stock_code: 股票代码
            scores_data: {short_term_score: {...}, medium_term_score: {...}, long_term_score: {...}}
            company_name: 公司名称（可选，用于修复缺失名称）
        """
        if stock_code not in self.pools["fine"]["stocks"]:
            logger.warning(f"{stock_code} 不在精筛池中")
            return

        stock = self.pools["fine"]["stocks"][stock_code]
        if company_name and self._is_bad_name(stock.get("company_name", ""), stock_code):
            stock["company_name"] = company_name
        term_map = {"short": "short_term_score", "medium": "medium_term_score", "long": "long_term_score"}

        for term, key in term_map.items():
            ts = scores_data.get(key, {})
            if ts:
                old = stock["scores"].get(term, {})
                if old.get("score") is not None:
                    stock["score_history"].append({
                        "term": term,
                        "score": old["score"],
                        "rating": old.get("rating", ""),
                        "updated_at": stock["last_updated"],
                    })
                stock["scores"][term] = {
                    "score": ts.get("score"),
                    "rating": ts.get("rating", ts.get("recommendation", "")),
                    "reasoning": ts.get("reasoning", ""),
                    "risk_warning": ts.get("risk_warning", ""),
                    "suggested_action": ts.get("suggested_action", ""),
                }

        stock["status"] = "scored"
        stock["last_updated"] = datetime.now().isoformat()
        self._save()

        mid_score = stock["scores"].get("medium", {}).get("score", "-")
        logger.info(f"{SUCCESS_ICON} 精筛池已更新评分: {stock_code} 中线={mid_score}")

    def _save(self):
        """保存股票池到JSON文件"""
        try:
            os.makedirs(os.path.dirname(self.pool_path) or ".", exist_ok=True)
            with open(self.pool_path, "w", encoding="utf-8") as f:
                json.dump({
                    "updated_at": datetime.now().isoformat(),
                    "pools": self.pools,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"{ERROR_ICON} 保存股票池失败: {e}")
            raise

    def _term_stock_entry(self, term: str, stock_code: str) -> Dict[str, Any]:
        """构建某期限池中某股票的标准数据结构"""
        return {
            "stock_code": stock_code,
            "company_name": "",
            "score": None,
            "recommendation": "",
            "term_score": {},
            "last_updated": "",
            "status": "pending",
            "score_history": [],
        }

    # ─── 期限感知的 CRUD 操作 ───

    def add_stock_to_term(self, term: str, stock_code: str, company_name: str) -> Dict[str, Any]:
        """
        添加股票到指定期限池

        Args:
            term: short/medium/long/fine
            stock_code: 股票代码(带交易所前缀)
            company_name: 公司名称

        Returns:
            股票信息字典
        """
        if term == "fine":
            return self.add_stock_to_fine(stock_code, company_name)
        stocks = self.pools[term]["stocks"]
        if stock_code in stocks:
            logger.info(f"{WAIT_ICON} 股票 {company_name}({stock_code}) 已在{TERM_LABELS[term]}池中，更新名称")
            stocks[stock_code]["company_name"] = company_name
            self._save()
            return stocks[stock_code]

        entry = self._term_stock_entry(term, stock_code)
        entry["company_name"] = company_name
        stocks[stock_code] = entry
        self._save()
        logger.info(f"{SUCCESS_ICON} 已添加 {company_name}({stock_code}) 到{TERM_LABELS[term]}池")
        return entry

    def remove_stock_from_term(self, term: str, stock_code: str) -> bool:
        """从指定期限池删除股票"""
        if term == "fine":
            return self.remove_stock_from_fine(stock_code)
        stocks = self.pools[term]["stocks"]
        if stock_code not in stocks:
            logger.warning(f"{ERROR_ICON} 股票 {stock_code} 不在{TERM_LABELS[term]}池中")
            return False

        name = stocks[stock_code]["company_name"]
        del stocks[stock_code]
        self._save()
        logger.info(f"{SUCCESS_ICON} 已从{TERM_LABELS[term]}池删除 {name}({stock_code})")
        return True

    def get_stock_in_term(self, term: str, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取指定期限池中某股票"""
        return self.pools[term]["stocks"].get(stock_code)

    def list_stocks_by_term(self, term: str) -> List[Dict[str, Any]]:
        """列出指定期限池中的所有股票，按评分降序"""
        if term == "fine":
            return self.get_fine_pool()
        stocks_list = list(self.pools[term]["stocks"].values())
        stocks_list.sort(key=lambda s: s.get("score") or 0, reverse=True)
        return stocks_list

    # ─── CLI 向后兼容：跨池操作 ───

    def list_stocks(self) -> List[Dict[str, Any]]:
        """列出所有池中所有股票（合并，按中线评分降序）— CLI 兼容"""
        seen = {}
        for term in VALID_TERMS:
            for code, stock in self.pools[term]["stocks"].items():
                if code not in seen:
                    entry = dict(stock)
                    # 附加三个期限的评分
                    entry["short_term_score"] = self.pools["short"]["stocks"].get(code, {}).get("term_score", {})
                    entry["medium_term_score"] = self.pools["medium"]["stocks"].get(code, {}).get("term_score", {})
                    entry["long_term_score"] = self.pools["long"]["stocks"].get(code, {}).get("term_score", {})
                    entry["score"] = entry["medium_term_score"].get("score", entry.get("score"))
                    entry["recommendation"] = entry["medium_term_score"].get("rating", entry.get("recommendation", ""))
                    seen[code] = entry
                else:
                    # 合并状态：任一池有评分则标记
                    if stock.get("score") is not None:
                        seen[code]["score"] = stock.get("score")
        result = list(seen.values())
        result.sort(key=lambda s: s.get("score") or 0, reverse=True)
        return result

    def get_stock(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """获取股票信息（查找所有池，返回合并结果）— CLI 兼容"""
        for term in VALID_TERMS:
            if stock_code in self.pools[term]["stocks"]:
                stock = dict(self.pools[term]["stocks"][stock_code])
                stock["short_term_score"] = self.pools["short"]["stocks"].get(stock_code, {}).get("term_score", {})
                stock["medium_term_score"] = self.pools["medium"]["stocks"].get(stock_code, {}).get("term_score", {})
                stock["long_term_score"] = self.pools["long"]["stocks"].get(stock_code, {}).get("term_score", {})
                return stock
        return None

    def count(self) -> int:
        """返回股票池中股票总数（去重）"""
        all_codes = set()
        for term in VALID_TERMS:
            all_codes.update(self.pools[term]["stocks"].keys())
        return len(all_codes)

    # ─── 评分更新 ───

    @staticmethod
    def _is_valid_score_value(value: Any) -> bool:
        """只接受有限的 0-100 数值，避免无效分数进入股票池。"""
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        value = float(value)
        return math.isfinite(value) and 0.0 <= value <= 100.0

    def update_stock_score(self, stock_code: str, score_data: Dict[str, Any]):
        """
        更新股票评分（CLI 兼容：写入所有三个池的评分）

        股票不在任何 term 池时自动添加到所有三个 term。

        Args:
            stock_code: 股票代码
            score_data: 包含 score, recommendation, short_term_score,
                       medium_term_score, long_term_score 等字段
        """
        if not isinstance(score_data, dict):
            logger.warning("跳过非字典评分结果: %s", stock_code)
            return

        status = str(score_data.get("status") or "").lower()
        main_score = score_data.get("score")
        if status in {"failed", "invalid", "abstain"} or not self._is_valid_score_value(main_score):
            # Failure/invalid payloads may update an existing status, but must
            # never add a stock or overwrite a previous valid score.
            changed = False
            if status in {"failed", "invalid", "abstain"}:
                for term in ("short", "medium", "long"):
                    stock = self.pools[term]["stocks"].get(stock_code)
                    if stock is not None and stock.get("status") != status:
                        stock["status"] = status
                        changed = True
            if changed:
                self._save()
            logger.warning(
                "跳过无效评分写入: %s status=%s score=%r",
                stock_code, status or "unspecified", main_score,
            )
            return

        term_scores = {
            "short": score_data.get("short_term_score", {}),
            "medium": score_data.get("medium_term_score", {}),
            "long": score_data.get("long_term_score", {}),
        }
        if any(
            not isinstance(term_score, dict)
            or not self._is_valid_score_value(term_score.get("score"))
            for term_score in term_scores.values()
        ):
            logger.warning("跳过不完整三期限评分写入: %s", stock_code)
            return

        new_company_name = score_data.get("company_name", "")

        exists_in_any = any(
            stock_code in self.pools[term]["stocks"]
            for term in ("short", "medium", "long")
        )
        if not exists_in_any and new_company_name:
            for term in ("short", "medium", "long"):
                self.add_stock_to_term(term, stock_code, new_company_name)

        for term in ("short", "medium", "long"):
            ts = term_scores[term]
            if stock_code in self.pools[term]["stocks"]:
                stock = self.pools[term]["stocks"][stock_code]
                if new_company_name and self._is_bad_name(stock.get("company_name", ""), stock_code):
                    stock["company_name"] = new_company_name
                if stock["score"] is not None:
                    stock["score_history"].append({
                        "score": stock["score"],
                        "recommendation": stock["recommendation"],
                        "updated_at": stock["last_updated"],
                    })
                stock["score"] = ts.get("score")
                stock["recommendation"] = ts.get("rating", ts.get("recommendation", ""))
                stock["term_score"] = ts
                stock["last_updated"] = datetime.now().isoformat()
                stock["status"] = score_data.get("status", "scored")

        if new_company_name and stock_code in self.pools.get("fine", {}).get("stocks", {}):
            fine_stock = self.pools["fine"]["stocks"][stock_code]
            if self._is_bad_name(fine_stock.get("company_name", ""), stock_code):
                fine_stock["company_name"] = new_company_name

        self._save()
        main_rec = score_data.get("recommendation", "")
        logger.info(f"{SUCCESS_ICON} 已更新评分: {main_score} ({main_rec})")

    def update_term_score(self, term: str, stock_code: str, score_data: Dict[str, Any]):
        """
        更新指定期限池的评分

        Args:
            term: short/medium/long
            stock_code: 股票代码
            score_data: 评分数据（包含 score, rating/recommendation 等）
        """
        if stock_code not in self.pools[term]["stocks"]:
            raise ValueError(f"股票 {stock_code} 不在{TERM_LABELS[term]}池中")

        if not isinstance(score_data, dict):
            logger.warning("跳过非字典%s评分结果: %s", TERM_LABELS[term], stock_code)
            return
        status = str(score_data.get("status") or "").lower()
        score = score_data.get("score")
        if status in {"failed", "invalid", "abstain"} or not self._is_valid_score_value(score):
            stock = self.pools[term]["stocks"][stock_code]
            if status in {"failed", "invalid", "abstain"}:
                stock["status"] = status
                self._save()
            logger.warning(
                "跳过无效%s评分写入: %s status=%s score=%r",
                TERM_LABELS[term], stock_code, status or "unspecified", score,
            )
            return

        stock = self.pools[term]["stocks"][stock_code]
        if stock["score"] is not None:
            stock["score_history"].append({
                "score": stock["score"],
                "recommendation": stock["recommendation"],
                "updated_at": stock["last_updated"],
            })

        stock["score"] = score_data.get("score")
        stock["recommendation"] = score_data.get("rating", score_data.get("recommendation", ""))
        stock["term_score"] = score_data
        stock["last_updated"] = datetime.now().isoformat()
        stock["status"] = score_data.get("status", "scored")
        self._save()
        logger.info(f"{SUCCESS_ICON} 已更新{TERM_LABELS[term]}评分: {stock['score']} ({stock['recommendation']})")

    def update_stock_status(self, term: str, stock_code: str, status: str):
        """更新指定期限池中股票的状态"""
        if stock_code in self.pools[term]["stocks"]:
            self.pools[term]["stocks"][stock_code]["status"] = status
            self._save()

    def update_quick_screen_score(self, stock_code: str, term: str, score_data: Dict[str, Any]):
        """持久化快筛股票池单期打分结果

        Args:
            stock_code: 股票代码
            term: short/medium/long
            score_data: {score, score_time, recommendation, suggested_action, reasoning, risk_warning}
        """
        pool = self.pools["quick_screen"]["stocks"]
        if stock_code not in pool:
            logger.warning(f"{stock_code} 不在快筛池中，跳过持久化")
            return
        stock = pool[stock_code]
        if "quick_scores" not in stock:
            stock["quick_scores"] = {}
        stock["quick_scores"][term] = {
            "score": score_data.get("score"),
            "score_time": score_data.get("score_time", ""),
            "recommendation": score_data.get("recommendation", ""),
            "suggested_action": score_data.get("suggested_action", ""),
            "reasoning": score_data.get("reasoning", ""),
            "risk_warning": score_data.get("risk_warning", ""),
        }
        stock["last_updated"] = datetime.now().isoformat()
        self._save()

    # ─── 查询 ───

    def get_scored_stocks(self, term: str) -> List[Dict[str, Any]]:
        """获取指定期限池中已评分的股票"""
        return [s for s in self.list_stocks_by_term(term) if s["score"] is not None]

    def get_pending_stocks(self) -> List[Dict[str, Any]]:
        """获取所有池中待评分的股票（跨池去重，CLI 兼容）"""
        seen = {}
        for term in VALID_TERMS:
            for code, stock in self.pools[term]["stocks"].items():
                if code not in seen and stock.get("status") == "pending":
                    seen[code] = stock
        return list(seen.values())
