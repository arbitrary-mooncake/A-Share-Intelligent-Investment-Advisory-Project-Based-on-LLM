"""
精筛池管理器 — 管理短/中/长线三个精筛股票池的创建、更新、查询。
"""
import json
import logging
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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

DEFAULT_BLACKLIST_TTL = {
    "short": 14,
    "medium": 60,
    "long": 365,
}


class PoolManager:
    """精筛池管理器"""

    def __init__(self):
        os.makedirs(POOL_DIR, exist_ok=True)
        self.pools = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(POOL_FILE):
            try:
                with open(POOL_FILE, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                self._migrate_blacklist(data)
                return data
            except Exception as e:
                logger.error("[PoolManager] Failed to load %s: %s", POOL_FILE, e)
        return {
            "short": {"stocks": [], "updated_at": "", "version": 0},
            "medium": {"stocks": [], "updated_at": "", "version": 0},
            "long": {"stocks": [], "updated_at": "", "version": 0},
            "blacklist": {"short": [], "medium": [], "long": []},
            "pool_health_history": {"short": [], "medium": [], "long": []},
        }

    def _migrate_blacklist(self, data: Dict[str, Any]):
        """Migrate blacklist from old flat-list format to per-term dict.

        Old: "blacklist": [{code, reason, ...}, ...]
        New: "blacklist": {"short": [...], "medium": [...], "long": [...]}
        """
        bl = data.get("blacklist")
        if isinstance(bl, list):
            import shutil
            backup_path = POOL_FILE + f".bl_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                shutil.copy2(POOL_FILE, backup_path)
            except Exception:
                pass
            new_bl = {"short": [], "medium": [], "long": []}
            for entry in bl:
                term = entry.get("term", "long")
                if term not in new_bl:
                    term = "long"
                entry.setdefault("term", term)
                new_bl[term].append(entry)
            data["blacklist"] = new_bl
            logger.info("[PoolManager] Blacklist migrated from flat list to per-term "
                       "(short=%d, medium=%d, long=%d), backup: %s",
                       len(new_bl["short"]), len(new_bl["medium"]),
                       len(new_bl["long"]), backup_path)
        elif isinstance(bl, dict):
            for term in ("short", "medium", "long"):
                bl.setdefault(term, [])
        else:
            data["blacklist"] = {"short": [], "medium": [], "long": []}

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

    def save_reserve(self, term: str, candidates: List[Dict[str, Any]]):
        """保存候补名单到 refined_pools.json

        Args:
            term: short/medium/long
            candidates: [{code, name, final_score, recommendation}, ...]
        """
        pool_data = self.pools.get(term, {})
        pool_data["reserve"] = candidates
        pool_data["reserve_updated_at"] = datetime.now().isoformat()
        self._save()
        logger.info("[PoolManager] %s 候补名单已保存: %d只", term, len(candidates))

    def get_reserve(self, term: str) -> List[Dict[str, Any]]:
        """获取候补名单"""
        pool_data = self.pools.get(term, {})
        return pool_data.get("reserve", [])

    def update_pool_partial(self, term: str, new_stocks: List[Dict[str, Any]],
                            removed_codes: List[str]) -> Dict[str, Any]:
        """部分更新: 替换底部股票为新股票

        Args:
            term: short/medium/long
            new_stocks: 新入池的股票列表 (已按 final_score 排序)
            removed_codes: 被替换掉的股票代码列表

        Returns:
            {kept: int, added: int, removed: int, new_pool_size: int}
        """
        pool_data = self.pools.get(term, {})
        stocks = pool_data.get("stocks", [])

        # 保留非移除股票
        kept_stocks = [s for s in stocks if s.get("code", "") not in set(removed_codes)]

        # 合并保留 + 新增
        merged = kept_stocks + new_stocks

        # 按 final_score 降序排序
        merged.sort(key=lambda s: s.get("final_score", 0), reverse=True)

        # 更新池
        self.pools[term] = {
            "stocks": merged,
            "updated_at": datetime.now().isoformat(),
            "version": pool_data.get("version", 0) + 1,
        }
        self._save()

        return {
            "kept": len(kept_stocks),
            "added": len(new_stocks),
            "removed": len(removed_codes),
            "new_pool_size": len(merged),
        }

    def get_pool_summary(self, term: str) -> Dict[str, Any]:
        pool = self.pools.get(term, {})
        stocks = pool.get("stocks", [])
        scores = [s.get("final_score", s.get("score", 0)) if isinstance(s, dict) else 50 for s in stocks]
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

    def add_to_blacklist(self, stock_code: str, term: str = "long",
                         reason: str = "", expiry_days: int = None):
        """添加到黑名单 (按期限隔离)

        Args:
            stock_code: 股票代码
            term: short/medium/long
            reason: 加入原因
            expiry_days: 过期天数, None 时使用 DEFAULT_BLACKLIST_TTL[term]
        """
        if expiry_days is None:
            expiry_days = DEFAULT_BLACKLIST_TTL.get(term, 365)
        bl = self.pools.get("blacklist", {})
        if isinstance(bl, list):
            bl = {"short": [], "medium": [], "long": []}
            self.pools["blacklist"] = bl
        bl.setdefault(term, [])
        bl[term].append({
            "code": stock_code,
            "term": term,
            "reason": reason,
            "added_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=expiry_days)).isoformat(),
        })
        self._save()

    def is_blacklisted(self, stock_code: str, term: str = None) -> bool:
        """检查股票是否在黑名单中（自动过滤过期条目）

        Args:
            stock_code: 股票代码
            term: 仅查对应 term 的黑名单; None 时查所有 term
        """
        now = datetime.now()
        bl = self.pools.get("blacklist", {})
        if isinstance(bl, list):
            terms_to_check = [None]
            entries_iter = bl
        else:
            terms_to_check = [term] if term else ["short", "medium", "long"]
            entries_iter = []
            for t in terms_to_check:
                entries_iter.extend(bl.get(t, []))
        for entry in entries_iter:
            if entry.get("code") == stock_code:
                expires_str = entry.get("expires_at", "")
                if expires_str:
                    try:
                        expires_at = datetime.fromisoformat(expires_str)
                        if now > expires_at:
                            continue
                    except (ValueError, TypeError):
                        pass
                return True
        return False

    def clean_expired_blacklist(self):
        """清理过期的黑名单条目 (按 term 分别清理)"""
        now = datetime.now()
        bl = self.pools.get("blacklist", {})
        removed = 0

        if isinstance(bl, list):
            cleaned = []
            for entry in bl:
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
        else:
            for term in ("short", "medium", "long"):
                entries = bl.get(term, [])
                cleaned = []
                for entry in entries:
                    expires_str = entry.get("expires_at", "")
                    if expires_str:
                        try:
                            if now > datetime.fromisoformat(expires_str):
                                removed += 1
                                continue
                        except (ValueError, TypeError):
                            pass
                    cleaned.append(entry)
                bl[term] = cleaned
            if removed > 0:
                self.pools["blacklist"] = bl
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

    def record_pool_health_snapshot(self, date: str, term: str,
                                     low_score_pct: float,
                                     not_held_pct: float):
        """
        记录池健康快照，用于连续交易日检查（总纲 §4.2）。

        每个期限保留最近30个交易日的快照历史。

        Args:
            date: 交易日 YYYY-MM-DD
            term: short/medium/long
            low_score_pct: 评分<50的股票占比（%）
            not_held_pct: 未被任何线路持有的股票占比（%）
        """
        history = self.pools.setdefault("pool_health_history", {})
        term_history = history.setdefault(term, [])
        # 避免同日重复记录（覆盖更新）
        if term_history and term_history[-1].get("date") == date:
            term_history[-1] = {
                "date": date, "low_score_pct": round(low_score_pct, 1),
                "not_held_pct": round(not_held_pct, 1),
            }
        else:
            term_history.append({
                "date": date, "low_score_pct": round(low_score_pct, 1),
                "not_held_pct": round(not_held_pct, 1),
            })
        # 保留最近30天
        if len(term_history) > 30:
            history[term] = term_history[-30:]
        self._save()

    @staticmethod
    def _count_consecutive_days_above(history: List[Dict], field: str,
                                       threshold: float) -> int:
        """
        从最新日期往回数，统计连续满足 field > threshold 的天数。

        历史按日期降序扫描，直到遇到不满足条件的那天为止。

        Args:
            history: 按日期降序排列的健康快照列表
            field: 检查的字段名（low_score_pct / not_held_pct）
            threshold: 阈值

        Returns:
            连续满足条件的天数（>=0）
        """
        if not history:
            return 0
        # 按日期降序排列
        sorted_hist = sorted(history, key=lambda x: x.get("date", ""), reverse=True)
        count = 0
        for entry in sorted_hist:
            if entry.get(field, 0) > threshold:
                count += 1
            else:
                break
        return count

    # ── 黑名单基本面改善检查的静态缓存 ──
    _BL_IMPROVEMENT_CACHE: Dict[str, tuple] = {}  # code → (timestamp, result_dict)
    _BL_IMPROVEMENT_CACHE_TTL = 86400  # 24小时

    def check_blacklist_fundamental_improvements(
        self, max_entries: int = 20, per_entry_timeout: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        检查黑名单中的股票是否有基本面显著改善，值得提前解除黑名单。

        对每个未过期的黑名单股票，查询最新财务数据，检查3项指标：
          1. PE回到行业正常范围（pe_cheap_threshold <= PE <= pe_expensive_threshold）
          2. ROE转正（>0）
          3. 营收增速转正（or_yoy > 0）

        3项全部满足 → 标记为"建议提前解除黑名单"。

        Args:
            max_entries: 最多检查的条目数（避免页面加载时大量API调用）
            per_entry_timeout: 单条检查超时（秒）

        Returns:
            [{code, term, reason（入黑原因）, improvements: [描述, ...]}, ...]
            按改善项数量降序排列
        """
        now = datetime.now()
        bl = self.pools.get("blacklist", {})

        # 收集所有未过期的黑名单条目
        active_entries = []
        if isinstance(bl, dict):
            for t, entries in bl.items():
                for entry in entries:
                    expires_str = entry.get("expires_at", "")
                    if expires_str:
                        try:
                            if now > datetime.fromisoformat(expires_str):
                                continue
                        except (ValueError, TypeError):
                            pass
                    active_entries.append(entry)
        elif isinstance(bl, list):
            for entry in bl:
                expires_str = entry.get("expires_at", "")
                if expires_str:
                    try:
                        if now > datetime.fromisoformat(expires_str):
                            continue
                    except (ValueError, TypeError):
                        pass
                active_entries.append(entry)

        if not active_entries:
            return []

        # 去重 + 跳过测试代码
        _TEST_CODE_PREFIXES = ("sh.999999", "sz.999999", "sh.000000", "sz.000000")
        seen_codes = set()
        unique_entries = []
        for entry in active_entries:
            code = entry.get("code", "")
            if code in seen_codes:
                continue
            if any(code.startswith(p) for p in _TEST_CODE_PREFIXES):
                continue
            seen_codes.add(code)
            unique_entries.append(entry)

        # 限制检查数量，优先检查短期限（short > medium > long）
        term_priority = {"short": 0, "medium": 1, "long": 2}
        unique_entries.sort(key=lambda e: term_priority.get(e.get("term", "long"), 99))
        unique_entries = unique_entries[:max_entries]

        candidates = []

        for entry in unique_entries:
            code = entry.get("code", "")
            if not code:
                continue

            # 代码转换为Tushare格式
            if code.startswith("sh."):
                ts_code = code[3:] + ".SH"
            elif code.startswith("sz."):
                ts_code = code[3:] + ".SZ"
            else:
                ts_code = code

            # ── 缓存检查 ──
            cache_key = f"bl_improve_{ts_code}"
            if cache_key in self._BL_IMPROVEMENT_CACHE:
                cache_ts, cached_result = self._BL_IMPROVEMENT_CACHE[cache_key]
                if now.timestamp() - cache_ts < self._BL_IMPROVEMENT_CACHE_TTL:
                    # 与首次检查阈值一致: imp_count >= 3 才加入 candidates。
                    # 此前误用 "improvements 非空" 判断, 导致仅 1-2 项改善的股票
                    # 在缓存命中时被错误标记为"建议解除黑名单"。
                    if cached_result and cached_result.get("imp_count", 0) >= 3:
                        candidates.append({
                            "code": cached_result.get("code", code),
                            "term": cached_result.get("term", entry.get("term", "")),
                            "reason": cached_result.get("reason", entry.get("reason", "未知原因")),
                            "improvements": cached_result.get("improvements", []),
                        })
                    continue

            try:
                import concurrent.futures

                def _fetch_data():
                    from src.utils.tushare_client import _call as tushare_call
                    from src.utils.industry_knowledge import get_industry_info
                    from src.utils.tushare_client import get_stock_info

                    improvements = []

                    # ── 指标1: PE检查（daily_basic） ──
                    from src.eval.data_fetcher import fetch_latest_trading_day
                    trade_date = fetch_latest_trading_day().replace("-", "")
                    pe_basic = tushare_call("daily_basic", {
                        "ts_code": ts_code,
                        "trade_date": trade_date,
                    }, fields="pe_ttm")
                    pe = None
                    if pe_basic and pe_basic.get("items"):
                        field_names = pe_basic["fields"]
                        row = dict(zip(field_names, pe_basic["items"][0]))
                        pe_val = row.get("pe_ttm")
                        if pe_val is not None:
                            pe = float(pe_val)

                    # ── 指标2 & 3: ROE和营收增速（fina_indicator） ──
                    fina = tushare_call("fina_indicator", {
                        "ts_code": ts_code,
                        "start_date": f"{now.year - 1}0101",
                        "end_date": now.strftime("%Y1231"),
                    }, fields="roe,or_yoy,end_date")
                    roe = None
                    rev_growth = None
                    if fina and fina.get("items"):
                        field_names = fina["fields"]
                        latest = None
                        for row_data in fina["items"]:
                            row = dict(zip(field_names, row_data))
                            if latest is None or row.get("end_date", "") > latest.get("end_date", ""):
                                latest = row
                        if latest:
                            roe_val = latest.get("roe")
                            if roe_val is not None:
                                roe = float(roe_val)
                            rg_val = latest.get("or_yoy")
                            if rg_val is not None:
                                rev_growth = float(rg_val)

                    # ── 获取行业基准 ──
                    stock_info = None
                    try:
                        stock_info = get_stock_info(ts_code)
                    except Exception:
                        pass
                    industry = stock_info.get("industry", "") if stock_info else ""
                    benchmark = get_industry_info(industry) if industry else None

                    # ── 检查3项条件 ──
                    if pe is not None and benchmark:
                        pe_cheap = benchmark.get("pe_cheap_threshold", 0)
                        pe_expensive = benchmark.get("pe_expensive_threshold", float("inf"))
                        if pe_cheap <= pe <= pe_expensive:
                            improvements.append(
                                f"PE={pe:.1f}回到行业合理区间({pe_cheap}-{pe_expensive})"
                            )
                    elif pe is not None:
                        if 0 < pe < 200:
                            improvements.append(f"PE={pe:.1f}在基本合理范围")

                    if roe is not None and roe > 0:
                        improvements.append(f"ROE转正={roe:.1f}%")

                    if rev_growth is not None and rev_growth > 0:
                        improvements.append(f"营收增速转正={rev_growth:.1f}%")

                    return {
                        "code": code,
                        "term": entry.get("term", ""),
                        "reason": entry.get("reason", "未知原因"),
                        "improvements": improvements,
                        "imp_count": len(improvements),
                    }

                # 带超时的执行（防止单条 Tushare 调用卡死页面）
                # 不用 with 语句: ThreadPoolExecutor.__exit__ 调 shutdown(wait=True),
                # 会阻塞到底层线程完成, 使 per_entry_timeout 失效（Tushare 卡死时页面仍卡）。
                # 手动管理, 超时/异常后 shutdown(wait=False, cancel_futures=True) 立即返回。
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(_fetch_data)
                try:
                    result = future.result(timeout=per_entry_timeout)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "[PoolManager] 黑名单改善检查超时 %s (%.1fs)", code, per_entry_timeout
                    )
                    # 缓存超时结果（避免重复尝试）
                    self._BL_IMPROVEMENT_CACHE[cache_key] = (
                        now.timestamp(),
                        {"code": code, "improvements": [], "imp_count": 0},
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    continue
                except Exception as e:
                    logger.warning(
                        "[PoolManager] 黑名单改善检查失败 %s: %s", code, e
                    )
                    # 缓存失败结果（避免重复尝试浪费 API 配额）
                    self._BL_IMPROVEMENT_CACHE[cache_key] = (
                        now.timestamp(),
                        {"code": code, "improvements": [], "imp_count": 0},
                    )
                    executor.shutdown(wait=False, cancel_futures=True)
                    continue
                # 正常完成: future 已结束, wait=True 可安全清理线程资源
                executor.shutdown(wait=True)

                # ── 缓存结果 ──
                self._BL_IMPROVEMENT_CACHE[cache_key] = (now.timestamp(), result)

                if result and result.get("imp_count", 0) >= 3:
                    candidates.append({
                        "code": result["code"],
                        "term": result["term"],
                        "reason": result["reason"],
                        "improvements": result["improvements"],
                    })

            except Exception as e:
                logger.warning(
                    "[PoolManager] 检查黑名单基本面改善失败 %s: %s", code, e
                )
                # 缓存失败结果（避免重复尝试）
                try:
                    self._BL_IMPROVEMENT_CACHE[cache_key] = (
                        now.timestamp(),
                        {"code": code, "improvements": [], "imp_count": 0},
                    )
                except Exception:
                    pass
                continue

        # 按改善项数量降序
        candidates.sort(key=lambda x: len(x["improvements"]), reverse=True)
        return candidates

    def suggest_blacklist_removal(self, term: str = None) -> List[Dict[str, Any]]:
        """
        获取建议提前解除黑名单的股票列表。

        Args:
            term: 限定期限；None 返回所有期限

        Returns:
            [{code, term, reason, improvements, ...}, ...]
        """
        candidates = self.check_blacklist_fundamental_improvements()
        if term:
            candidates = [c for c in candidates if c.get("term") == term]
        return candidates

    def get_pool_health(self,
                        lines_status: Optional[List[Dict[str, Any]]] = None,
                        quick: bool = True) -> Dict[str, Any]:
        """
        检查各期限精筛池的健康状态，返回红/黄/绿状态及触发原因。

        总纲§4.2 触发条件：
          RED   — >30%股票评分<50（连续5日）、LLM自主线持续优于所有Agent线、
                   黑名单股票有重大基本面变化
          YELLOW — >20%股票未被任何线路持有（连续10日）、距上次全量更新超限
          GREEN  — 无任何触发条件

        Args:
            lines_status: 可选，来自LineManager.get_all_status()的线路状态列表。
            quick: True=跳过Tushare API调用（页面加载用），False=完整检查含黑名单改善分析。
        Returns:
            {term: {status, triggers, last_update, days_since_update,
                    suggested_action, details}}
        """
        max_days_map = {"short": 30, "medium": 60, "long": 90}

        # ── 预计算：从 lines_status 提取持仓覆盖和LLM自主线对比 ──
        held_by_term: Dict[str, set] = {}
        llm_free_exceeds: Dict[str, bool] = {}

        if lines_status:
            for term in ["short", "medium", "long"]:
                held_stocks: set = set()
                for line in lines_status:
                    if line.get("term") == term:
                        for code, shares in line.get("holdings", {}).items():
                            if shares > 0:
                                held_stocks.add(code)
                held_by_term[term] = held_stocks

                # LLM自主线 vs 最强Agent线
                # 短线: S-L8(agent longhold) vs S-L9(LLM free)
                # 中线: M-L0(agent) vs M-L1(LLM free)
                # 长线: L-L0(agent) vs L-L1(LLM free)
                agent_lines = [
                    l for l in lines_status
                    if l.get("term") == term
                    and l.get("type") in ("full_agent", "longhold", "ablation_base")
                ]
                llm_free_lines = [
                    l for l in lines_status
                    if l.get("term") == term and l.get("type") == "llm_free"
                ]
                if agent_lines and llm_free_lines:
                    best_agent_ret = max(
                        (l.get("cumulative_return_pct", -999) for l in agent_lines),
                        default=-999,
                    )
                    llm_free_ret = llm_free_lines[0].get("cumulative_return_pct", 0)
                    llm_free_exceeds[term] = (
                        llm_free_ret > best_agent_ret and best_agent_ret > -999
                    )

        # ── 逐期限检查 ──
        results = {}
        for term in ["short", "medium", "long"]:
            summary = self.get_pool_summary(term)
            stocks_with_scores = self.get_pool_with_scores(term)
            pool_size = len(stocks_with_scores)

            days_since_update = 999
            if summary["updated_at"] and summary["updated_at"] != "从未更新":
                try:
                    updated = datetime.fromisoformat(summary["updated_at"])
                    days_since_update = (datetime.now() - updated).days
                except Exception:
                    pass

            max_days = max_days_map[term]

            # ── 指标计算 ──
            low_score_count = sum(
                1 for s in stocks_with_scores
                if (isinstance(s, dict) and s.get("final_score", s.get("score", 50)) < 50)
            )
            low_score_pct = (low_score_count / max(pool_size, 1)) * 100

            pool_codes = [
                s.get("code", "") if isinstance(s, dict) else s
                for s in stocks_with_scores
            ]

            not_held_pct = 0.0
            held_stocks_set = held_by_term.get(term, set())
            if held_stocks_set and pool_codes:
                not_held = [c for c in pool_codes if c not in held_stocks_set]
                not_held_pct = len(not_held) / max(len(pool_codes), 1) * 100

            # 黑名单检查 (仅查当前期限，短线/中线/长线独立隔离)
            active_blacklist_codes: set = set()
            now = datetime.now()
            bl = self.pools.get("blacklist", {})
            if isinstance(bl, dict):
                term_entries = bl.get(term, [])
            else:
                term_entries = bl
            for entry in term_entries:
                expires_str = entry.get("expires_at", "")
                if expires_str:
                    try:
                        if now <= datetime.fromisoformat(expires_str):
                            active_blacklist_codes.add(entry.get("code", ""))
                    except (ValueError, TypeError):
                        active_blacklist_codes.add(entry.get("code", ""))
            pool_codes_set = set(pool_codes)
            blacklist_in_pool = active_blacklist_codes & pool_codes_set

            # ── 判断状态（RED优先） ──
            triggers = []
            status = "green"

            # ── 读取健康历史用于连续交易日检查 ──
            health_history = self.pools.get("pool_health_history", {}).get(term, [])

            if low_score_pct > 30:
                # 总纲 §4.2 RED条件1: >30%评分<50 连续5个交易日
                consecutive_low = self._count_consecutive_days_above(
                    health_history, "low_score_pct", 30
                )
                if consecutive_low >= 5:
                    triggers.append(
                        f"池内{low_score_pct:.0f}%的股票连续{consecutive_low}日评分<50，"
                        f"建议部分更新（替换评分垫底的20%股票）"
                    )
                    status = "red"
                elif consecutive_low > 0:
                    triggers.append(
                        f"池内{low_score_pct:.0f}%的股票评分<50"
                        f"（连续{consecutive_low}/5天，尚未触发RED）"
                    )
                    if status != "red":
                        status = "yellow"
                else:
                    triggers.append(
                        f"池内{low_score_pct:.0f}%的股票评分<50"
                        f"（快照首次触发，需连续5个交易日确认）"
                    )
                    if status != "red":
                        status = "yellow"

            if llm_free_exceeds.get(term, False):
                triggers.append("LLM自主线累计收益超过所有Agent线，系统评分策略需审查")
                status = "red"

            if blacklist_in_pool:
                blacklist_descs = []
                for entry in term_entries:
                    if entry.get("code", "") in blacklist_in_pool:
                        blacklist_descs.append(
                            f"{entry.get('code', '?')}"
                            f"({entry.get('reason', '未知原因')})"
                        )
                triggers.append(
                    f"黑名单股票仍在精筛池内: {', '.join(blacklist_descs[:3])}"
                )
                status = "red"

                # ── 检查黑名单股是否有基本面改善（Gap #3）──
                # quick 模式下跳过（避免 Tushare API 调用拖慢页面加载）
                if not quick:
                    try:
                        improved = self.suggest_blacklist_removal(term)
                        pool_improved = [
                            c for c in improved
                            if c.get("code", "") in blacklist_in_pool
                        ]
                        if pool_improved:
                            for c in pool_improved[:3]:
                                triggers.append(
                                    f"黑名单股{c['code']}基本面已改善"
                                    f"（{'; '.join(c['improvements'][:2])}），"
                                    f"建议提前解除黑名单"
                                )
                    except Exception as e:
                        logger.warning(
                            "[PoolManager] 黑名单改善检查失败: %s", e
                        )

            # ── 欠填充检测（当前池容量显著低于目标） ──
            min_size = DEFAULT_POOL_CONFIG[term]["min_size"]
            severe_underfill = pool_size < min_size * 0.5

            # YELLOW（仅当非RED时）
            if status != "red":
                if severe_underfill:
                    triggers.append(
                        f"精筛池严重欠填充: 当前{pool_size}只 < 下限{min_size}只"
                        f"（目标{summary['target_size']}只），建议立即全量更新"
                    )
                    status = "yellow"

                if not_held_pct > 20:
                    # 总纲 §4.2 YELLOW条件4: >20%未被持有 连续10个交易日
                    consecutive_not_held = self._count_consecutive_days_above(
                        health_history, "not_held_pct", 20
                    )
                    if consecutive_not_held >= 10:
                        triggers.append(
                            f"池内{not_held_pct:.0f}%的股票未被任何线路持有"
                            f"（连续{consecutive_not_held}日）"
                        )
                    elif consecutive_not_held > 0:
                        triggers.append(
                            f"池内{not_held_pct:.0f}%的股票未被任何线路持有"
                            f"（连续{consecutive_not_held}/10天，进度中）"
                        )
                    else:
                        triggers.append(
                            f"池内{not_held_pct:.0f}%的股票未被任何线路持有"
                            f"（快照首次触发，需连续10个交易日确认）"
                        )
                    status = "yellow"

                if days_since_update >= max_days:
                    triggers.append(
                        f"距上次全量更新已{days_since_update}天（上限{max_days}天）"
                    )
                    status = "yellow"

            # ── 建议动作 ──
            if status == "red":
                suggested_action = "建议立即全量更新精筛池"
            elif status == "yellow":
                if severe_underfill or days_since_update >= max_days:
                    suggested_action = "建议全量更新精筛池"
                else:
                    suggested_action = "建议部分更新（替换评分垫底的20%股票）"
            else:
                suggested_action = "池状态健康，无需操作"

            results[term] = {
                "status": status,
                "triggers": triggers,
                "last_update": summary["updated_at"],
                "days_since_update": days_since_update,
                "max_days": max_days,
                "suggested_action": suggested_action,
                "details": {
                    "size": pool_size,
                    "target_size": summary["target_size"],
                    "avg_score": summary["avg_score"],
                    "low_score_pct": round(low_score_pct, 1),
                    "not_held_pct": round(not_held_pct, 1),
                    "blacklist_count": len(blacklist_in_pool),
                },
            }

        return results
