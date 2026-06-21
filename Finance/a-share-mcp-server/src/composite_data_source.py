"""
CompositeDataSource: AKshare data source with in-memory cache.
Includes in-memory cache (5-min TTL) to avoid redundant calls within a single session.
"""
import logging
import time
import pandas as pd
from typing import List, Optional

from .data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError

logger = logging.getLogger(__name__)

# 内存缓存 TTL（秒）
_MEMORY_CACHE_TTL = 300  # 5 分钟


class CompositeDataSource(FinancialDataSource):
    """
    AKshare data source with in-memory cache.
    Maintains an in-memory cache keyed by (method_name, args, kwargs)
    with a 5-min TTL to avoid redundant upstream calls within a session.
    """

    def __init__(self, primary: FinancialDataSource, fallback: FinancialDataSource):
        self.fallback = fallback    # AKshare (sole active data source)
        self._memory_cache: dict = {}  # key → (timestamp, result)

    @staticmethod
    def _make_cache_key(method_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a hashable cache key from method name and arguments."""
        # Sort kwargs keys for deterministic key
        kw_items = tuple(sorted(kwargs.items())) if kwargs else ()
        return f"{method_name}:{args}:{kw_items}"

    def _try(self, method_name: str, *args, **kwargs):
        """
        Try AKshare as the sole data source.
        Results are cached in-memory for 5 minutes to avoid redundant upstream calls.
        """
        # 0. Check memory cache
        cache_key = self._make_cache_key(method_name, args, kwargs)
        if cache_key in self._memory_cache:
            ts, result = self._memory_cache[cache_key]
            if time.time() - ts < _MEMORY_CACHE_TTL:
                logger.debug(f"[Composite] Cache hit: {method_name}")
                return result
            del self._memory_cache[cache_key]

        # 1. Try AKshare (sole data source)
        try:
            fallback_method = getattr(self.fallback, method_name)
            result = fallback_method(*args, **kwargs)
            logger.debug(f"[Composite] AKshare {method_name} succeeded")
            self._memory_cache[cache_key] = (time.time(), result)
            return result
        except LoginError:
            logger.warning(f"[Composite] AKshare {method_name} login error (unexpected)")
        except NoDataFoundError as e:
            logger.info(f"[Composite] AKshare {method_name} returned no data ({e})")
        except DataSourceError as e:
            logger.warning(f"[Composite] AKshare {method_name} error ({e})")
        except Exception as e:
            logger.warning(f"[Composite] AKshare {method_name} unexpected error ({e})")

        raise DataSourceError(
            f"AKshare {method_name} failed. "
            f"Use Tushare MCP tools (tushare_*) for equivalent data."
        )

    # --- Interface methods ---
    def get_historical_k_data(
        self, code: str, start_date: str, end_date: str,
        frequency: str = "d", adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        return self._try(
            "get_historical_k_data", code, start_date, end_date, frequency, adjust_flag, fields
        )

    def get_stock_basic_info(self, code: str) -> pd.DataFrame:
        return self._try("get_stock_basic_info", code)

    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_trade_dates", start_date=start_date, end_date=end_date)

    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_all_stock", date=date)

    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_deposit_rate_data", start_date=start_date, end_date=end_date)

    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_loan_rate_data", start_date=start_date, end_date=end_date)

    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        return self._try("get_required_reserve_ratio_data", start_date=start_date, end_date=end_date, year_type=year_type)

    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_money_supply_data_month", start_date=start_date, end_date=end_date)

    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_money_supply_data_year", start_date=start_date, end_date=end_date)

    # --- Additional methods (not in interface but used by tools) ---
    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._try("get_profit_data", code, year, quarter)

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._try("get_operation_data", code, year, quarter)

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._try("get_growth_data", code, year, quarter)

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._try("get_balance_data", code, year, quarter)

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._try("get_cash_flow_data", code, year, quarter)

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._try("get_dupont_data", code, year, quarter)

    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_sz50_stocks", date=date)

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_hs300_stocks", date=date)

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_zz500_stocks", date=date)

    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        return self._try("get_dividend_data", code, year, year_type)

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._try("get_adjust_factor_data", code, start_date, end_date)

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._try("get_performance_express_report", code, start_date, end_date)

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self._try("get_forecast_report", code, start_date, end_date)

    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        return self._try("get_stock_industry", code=code, date=date)

    def crawl_news(self, query: str, top_k: int = 10) -> str:
        """News via AkShare East Money (Baostock news crawler disabled)."""
        try:
            return self.fallback.crawl_news(query, top_k)
        except Exception as e:
            logger.warning(f"[Composite] crawl_news failed: {e}")
            return f"新闻获取失败: {e}。请使用 tushare_news 工具获取新闻数据。"
