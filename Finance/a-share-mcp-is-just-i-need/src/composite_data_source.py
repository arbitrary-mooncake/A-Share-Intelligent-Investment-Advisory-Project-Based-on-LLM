"""
CompositeDataSource: Primary-Fallback data source pattern.
Tries Baostock first; on NoDataFoundError / DataSourceError, falls back to AKshare.
"""
import logging
import pandas as pd
from typing import List, Optional

from .data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError

logger = logging.getLogger(__name__)


class CompositeDataSource(FinancialDataSource):
    """
    Wraps a primary data source (Baostock) and a fallback (AKshare).
    For every method, it tries the primary first.
    If the primary raises NoDataFoundError, DataSourceError, or an unexpected exception,
    it falls back to the secondary source.

    crawl_news is a special case: AKshare does not support it,
    so it only runs on the primary and does NOT fall back.
    """

    def __init__(self, primary: FinancialDataSource, fallback: FinancialDataSource):
        self.primary = primary
        self.fallback = fallback

    def _try(self, method_name: str, *args, **kwargs):
        """
        Generic fallback executor.
        Tries primary, falls back to fallback on recoverable errors.
        """
        try:
            method = getattr(self.primary, method_name)
            return method(*args, **kwargs)
        except LoginError:
            # LoginError on primary: try fallback (which doesn't need login)
            logger.warning(
                f"[Composite] Primary {method_name} login failed, falling back to AKshare"
            )
        except NoDataFoundError as e:
            logger.info(
                f"[Composite] Primary {method_name} returned no data ({e}), falling back to AKshare"
            )
        except DataSourceError as e:
            logger.warning(
                f"[Composite] Primary {method_name} error ({e}), falling back to AKshare"
            )
        except Exception as e:
            logger.warning(
                f"[Composite] Primary {method_name} unexpected error ({e}), falling back to AKshare"
            )

        # Fallback
        try:
            fallback_method = getattr(self.fallback, method_name)
            result = fallback_method(*args, **kwargs)
            logger.info(f"[Composite] Fallback {method_name} succeeded via AKshare")
            return result
        except Exception as fallback_err:
            logger.error(
                f"[Composite] Fallback {method_name} also failed: {fallback_err}"
            )
            raise DataSourceError(
                f"Both primary and fallback data sources failed for {method_name}: "
                f"fallback error={fallback_err}"
            ) from fallback_err

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
        """
        News crawler is only available on Baostock primary.
        No AKshare fallback.
        """
        try:
            return self.primary.crawl_news(query, top_k)
        except Exception as e:
            logger.warning(f"[Composite] Primary crawl_news failed: {e}, no AKshare fallback available")
            return f"新闻爬取失败: {e} (AKshare不支持此功能)"
