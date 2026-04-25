"""
AKshare data source implementation of FinancialDataSource interface.
Used as fallback when Baostock data is insufficient.
"""
import logging
import re
import pandas as pd
from typing import List, Optional

from .data_source_interface import FinancialDataSource, DataSourceError, NoDataFoundError, LoginError

logger = logging.getLogger(__name__)


def _parse_chinese_date(date_str: str) -> str:
    """Convert Chinese date format like '2023年09月14日' to '2023-09-14'."""
    m = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", str(date_str))
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"
    return str(date_str)


def _strip_exchange_prefix(code: str) -> str:
    """Remove exchange prefix like 'sh.' / 'sz.' for AKshare (uses pure digits)."""
    code = code.strip().lower()
    if code.startswith("sh.") or code.startswith("sz."):
        return code[3:]
    return code


def _ensure_akshare():
    """Lazy import check for akshare."""
    try:
        import akshare
        return akshare
    except ImportError:
        raise DataSourceError("akshare is not installed. Run: pip install akshare")


class AkshareDataSource(FinancialDataSource):
    """
    AKshare-based implementation of FinancialDataSource.
    AKshare does not require login, so all methods are direct API calls.
    """

    # ---- K-line data ----
    def get_historical_k_data(
        self,
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        logger.info(f"[AKshare] K-line data for {code}, {start_date}~{end_date}, freq={frequency}, adjust={adjust_flag}")

        # Map adjust_flag: Baostock 1=前复权, 2=后复权, 3=不复权
        # AKshare Sina: "qfq"=前复权, "hfq"=后复权, ""=不复权
        adjust_map = {"1": "qfq", "2": "hfq", "3": ""}
        adjust = adjust_map.get(adjust_flag, "")

        # Try Sina first (more reliable in some environments)
        last_error = None
        try:
            # Sina API uses symbol with exchange prefix: sh603871
            sina_symbol = symbol
            if not sina_symbol.startswith(("sh", "sz")):
                # Determine exchange from code
                if symbol.startswith("6"):
                    sina_symbol = f"sh{symbol}"
                else:
                    sina_symbol = f"sz{symbol}"

            df = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start_date, end_date=end_date, adjust=adjust)
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No K-line data for {code} in {start_date}~{end_date}")
            # Sina returns date as index or 'date' column
            if "date" not in df.columns and df.index.name == "date":
                df = df.reset_index()
            df["code"] = code
            logger.info(f"[AKshare] Retrieved {len(df)} K-line records for {code} (Sina)")
            return self._normalize_kline_fields(df, fields)
        except Exception as e:
            last_error = e
            logger.debug(f"[AKshare] Sina K-line failed: {e}, trying East Money...")

        # Fallback to East Money
        try:
            period_map = {"d": "daily", "w": "weekly", "m": "monthly"}
            period = period_map.get(frequency, "daily")
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adjust,
            )
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No K-line data for {code} in {start_date}~{end_date}")
            df["code"] = code
            logger.info(f"[AKshare] Retrieved {len(df)} K-line records for {code} (East Money)")
            return self._normalize_kline_fields(df, fields)
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] K-line data error for {code}: {e}")

    def _normalize_kline_fields(self, df: pd.DataFrame, fields: Optional[List[str]]) -> pd.DataFrame:
        """Normalize K-line column names from various AKshare sources."""
        col_rename = {
            "日期": "date", "股票代码": "code", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
            "涨跌幅": "pctChg", "涨跌额": "change", "换手率": "turn",
            "市盈率-动态": "peTTM", "市净率": "pbMRQ", "总市值": "total_mv",
            "流通市值": "circ_mv", "振幅": "amplitude", "涨幅": "pctChg",
            "前收盘价": "preclose",
        }
        df.rename(columns=col_rename, inplace=True)
        for col in ["date", "code", "open", "high", "low", "close", "volume", "amount"]:
            if col not in df.columns:
                df[col] = None
        if fields:
            available = [c for c in fields if c in df.columns]
            if available:
                df = df[available]
        return df

    # ---- Stock basic info ----
    def get_stock_basic_info(self, code: str) -> pd.DataFrame:
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        logger.info(f"[AKshare] Basic info for {code}")

        # Try East Money first
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                if "item" in df.columns and "value" in df.columns:
                    df = df.set_index("item")["value"].to_frame().T
                    df.index = [0]
                    df.rename(columns={"股票代码": "code", "股票简称": "code_name"}, inplace=True)
                    if "code" not in df.columns:
                        df["code"] = code
                    if "code_name" not in df.columns:
                        df["code_name"] = None
                    if "tradeStatus" not in df.columns:
                        df["tradeStatus"] = "1"
                logger.info(f"[AKshare] Retrieved basic info for {code} (East Money)")
                return df
        except Exception as e:
            logger.debug(f"[AKshare] East Money basic info failed: {e}")

        # Fallback: use stock_info_a_code_name
        try:
            all_stocks = ak.stock_info_a_code_name()
            if all_stocks is not None and not all_stocks.empty:
                row = all_stocks[all_stocks["code"] == symbol]
                if not row.empty:
                    df = pd.DataFrame({
                        "code": [code],
                        "code_name": [row.iloc[0]["name"]],
                        "tradeStatus": ["1"],
                    })
                    logger.info(f"[AKshare] Retrieved basic info for {code} (Sina)")
                    return df
        except Exception as e:
            logger.debug(f"[AKshare] Sina basic info failed: {e}")

        raise NoDataFoundError(f"[AKshare] No basic info for {code}")

    # ---- Trade dates ----
    def get_trade_dates(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] Trade dates {start_date or 'default'} ~ {end_date or 'default'}")
        try:
            df = ak.tool_trade_date_hist_sina()
            if df is None or df.empty:
                raise NoDataFoundError("[AKshare] No trade date data available")

            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
                if start_date:
                    df = df[df["trade_date"] >= start_date]
                if end_date:
                    df = df[df["trade_date"] <= end_date]

            if df.empty:
                raise NoDataFoundError("[AKshare] No trade dates in specified range")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] Trade dates error: {e}")

    # ---- All stocks ----
    def get_all_stock(self, date: Optional[str] = None) -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] All stocks for date={date or 'latest'}")

        # Try Sina first (more reliable in some environments)
        try:
            df = ak.stock_zh_a_spot()
            if df is not None and not df.empty:
                col_rename = {
                    "代码": "code", "名称": "code_name", "最新价": "close",
                    "涨跌幅": "pctChg", "涨跌额": "change", "成交量": "volume",
                    "成交额": "amount", "今开": "open", "最高": "high",
                    "最低": "low", "昨收": "preclose",
                }
                df.rename(columns=col_rename, inplace=True)
                if "tradeStatus" not in df.columns:
                    df["tradeStatus"] = "1"
                logger.info(f"[AKshare] Retrieved {len(df)} stock records (Sina)")
                return df
        except Exception as e:
            logger.debug(f"[AKshare] Sina all-stock failed: {e}")

        # Fallback to East Money
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                col_rename = {
                    "代码": "code", "名称": "code_name", "最新价": "close",
                    "涨跌幅": "pctChg", "换手率": "turn", "成交量": "volume",
                    "成交额": "amount", "总市值": "total_mv", "流通市值": "circ_mv",
                }
                df.rename(columns=col_rename, inplace=True)
                if "tradeStatus" not in df.columns:
                    df["tradeStatus"] = "1"
                logger.info(f"[AKshare] Retrieved {len(df)} stock records (East Money)")
                return df
        except Exception as e:
            logger.debug(f"[AKshare] East Money all-stock failed: {e}")

        # Final fallback: stock_info_a_code_name
        try:
            df = ak.stock_info_a_code_name()
            if df is not None and not df.empty:
                df["tradeStatus"] = "1"
                logger.info(f"[AKshare] Retrieved {len(df)} stock records (code name list)")
                return df
        except Exception as e:
            logger.debug(f"[AKshare] code_name list failed: {e}")

        raise NoDataFoundError("[AKshare] No stock spot data available from any source")

    # ---- Macro: deposit rate ----
    def get_deposit_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] Deposit rates {start_date or 'default'} ~ {end_date or 'default'}")
        try:
            df = ak.macro_bank_china_interest_rate()
            if df is None or df.empty:
                raise NoDataFoundError("[AKshare] No interest rate data available")

            if start_date or end_date:
                date_col = None
                for c in df.columns:
                    if "日期" in str(c) or "date" in str(c).lower() or "年" in str(c):
                        date_col = c
                        break
                if date_col:
                    df[date_col] = df[date_col].astype(str)
                    if start_date:
                        df = df[df[date_col] >= start_date]
                    if end_date:
                        df = df[df[date_col] <= end_date]

            if df.empty:
                raise NoDataFoundError("[AKshare] No deposit rate data in specified range")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] Deposit rate error: {e}")

    # ---- Macro: loan rate ----
    def get_loan_rate_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        # Same API as deposit rate for AKshare
        return self.get_deposit_rate_data(start_date, end_date)

    # ---- Macro: reserve ratio ----
    def get_required_reserve_ratio_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, year_type: str = '0') -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] Reserve ratio {start_date or 'default'} ~ {end_date or 'default'}")
        try:
            df = ak.macro_china_reserve_requirement_ratio()
            if df is None or df.empty:
                raise NoDataFoundError("[AKshare] No reserve ratio data available")

            if start_date or end_date:
                date_col = None
                for c in df.columns:
                    if "公布时间" in str(c) or "日期" in str(c) or "date" in str(c).lower():
                        date_col = c
                        break
                if date_col:
                    # Convert Chinese dates to ISO format for comparison
                    df["_parsed_date"] = df[date_col].apply(_parse_chinese_date)
                    if start_date:
                        df = df[df["_parsed_date"] >= start_date]
                    if end_date:
                        df = df[df["_parsed_date"] <= end_date]
                    df.drop(columns=["_parsed_date"], inplace=True)

            if df.empty:
                raise NoDataFoundError("[AKshare] No reserve ratio data in specified range")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] Reserve ratio error: {e}")

    # ---- Macro: money supply monthly ----
    def get_money_supply_data_month(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] Monthly money supply {start_date or 'default'} ~ {end_date or 'default'}")
        try:
            df = ak.macro_china_money_supply()
            if df is None or df.empty:
                raise NoDataFoundError("[AKshare] No money supply data available")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] Monthly money supply error: {e}")

    # ---- Macro: money supply yearly ----
    def get_money_supply_data_year(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self.get_money_supply_data_month(start_date, end_date)

    # ---- Index constituents ----
    def get_sz50_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._get_index_constituents("000016", "SZSE 50")

    def get_hs300_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._get_index_constituents("000300", "CSI 300")

    def get_zz500_stocks(self, date: Optional[str] = None) -> pd.DataFrame:
        return self._get_index_constituents("000905", "CSI 500")

    def _get_index_constituents(self, index_symbol: str, index_name: str) -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] {index_name} constituents")
        try:
            df = ak.index_stock_cons(symbol=index_symbol)
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No constituent data for {index_name}")

            col_rename = {
                "品种代码": "code", "证券代码": "code", "股票代码": "code",
                "品种简称": "code_name", "证券简称": "code_name", "股票名称": "code_name",
            }
            df.rename(columns=col_rename, inplace=True)

            if "code" not in df.columns:
                # Try first column as code
                df.rename(columns={df.columns[0]: "code"}, inplace=True)

            logger.info(f"[AKshare] Retrieved {len(df)} {index_name} constituents")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] {index_name} constituents error: {e}")

    # ---- Financial statements (quarterly) ----
    def get_profit_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._get_financial_statement(code, year, quarter, "profit")

    def get_operation_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._get_financial_ratios(code, "operation")

    def get_growth_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._get_financial_ratios(code, "growth")

    def get_balance_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._get_financial_statement(code, year, quarter, "balance")

    def get_cash_flow_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._get_financial_statement(code, year, quarter, "cashflow")

    def get_dupont_data(self, code: str, year: str, quarter: int) -> pd.DataFrame:
        return self._get_financial_ratios(code, "dupont")

    def _get_financial_statement(self, code: str, year: str, quarter: int, stmt_type: str) -> pd.DataFrame:
        """Get income statement / balance sheet / cash flow from Sina via AKshare."""
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        # Map statement type
        type_map = {
            "profit": "利润表",
            "balance": "资产负债表",
            "cashflow": "现金流量表",
        }
        report_type = type_map.get(stmt_type, "利润表")
        logger.info(f"[AKshare] {report_type} for {code}, {year}Q{quarter}")
        try:
            df = ak.stock_financial_report_sina(stock=symbol, symbol=report_type)
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No {report_type} data for {code}")
            logger.info(f"[AKshare] Retrieved {len(df)} {report_type} records for {code}")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] {report_type} error for {code}: {e}")

    def _get_financial_ratios(self, code: str, ratio_type: str) -> pd.DataFrame:
        """Get financial analysis indicators (DuPont, operation, growth)."""
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        logger.info(f"[AKshare] {ratio_type} ratios for {code}")
        try:
            df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year="2020")
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No {ratio_type} ratios for {code}")
            logger.info(f"[AKshare] Retrieved {len(df)} {ratio_type} ratio records for {code}")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] {ratio_type} ratios error for {code}: {e}")

    # ---- Additional data ----
    def get_dividend_data(self, code: str, year: str, year_type: str = "report") -> pd.DataFrame:
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        logger.info(f"[AKshare] Dividend data for {code}, year={year}")
        try:
            df = ak.stock_history_dividend_detail(symbol=symbol, indicator="分红", date="")
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No dividend data for {code}")

            # Filter by year if possible (the date columns are usually "公告日期" or "除权除息日")
            for col in ["公告日期", "除权除息日", "股权登记日"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
                    df = df[df[col].str.startswith(year)]
                    break

            if df.empty:
                raise NoDataFoundError(f"[AKshare] No dividend data for {code} year {year}")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] Dividend error for {code}: {e}")

    def get_adjust_factor_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        logger.info(f"[AKshare] Adjust factor for {code}, {start_date}~{end_date}")

        # Try Sina daily data which includes adjusted prices
        try:
            sina_symbol = symbol
            if not sina_symbol.startswith(("sh", "sz")):
                if symbol.startswith("6"):
                    sina_symbol = f"sh{symbol}"
                else:
                    sina_symbol = f"sz{symbol}"

            # Get both non-adjusted and forward-adjusted data to compute factor
            df_hfq = ak.stock_zh_a_daily(symbol=sina_symbol, start_date=start_date, end_date=end_date, adjust="hfq")
            if df_hfq is not None and not df_hfq.empty:
                df_hfq["code"] = code
                logger.info(f"[AKshare] Retrieved {len(df_hfq)} adjusted records for {code} (Sina)")
                return df_hfq
        except Exception as e:
            logger.debug(f"[AKshare] Sina adjust factor failed: {e}")

        # Fallback to East Money
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="hfq",
            )
            if df is not None and not df.empty:
                df["code"] = code
                return df
        except Exception as e:
            logger.debug(f"[AKshare] East Money adjust factor failed: {e}")

        raise NoDataFoundError(f"[AKshare] No adjust factor data for {code}")

    def get_performance_express_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        ak = _ensure_akshare()
        symbol = _strip_exchange_prefix(code)
        logger.info(f"[AKshare] Performance express report for {code}")
        try:
            df = ak.stock_yjbb_em(date=start_date[:4] if start_date else "20241231")
            if df is None or df.empty:
                raise NoDataFoundError(f"[AKshare] No performance express data for {code}")
            if "股票代码" in df.columns:
                df = df[df["股票代码"] == symbol]
            if df.empty:
                raise NoDataFoundError(f"[AKshare] No performance express data for {code}")
            return df
        except NoDataFoundError:
            raise
        except Exception as e:
            raise DataSourceError(f"[AKshare] Performance express error for {code}: {e}")

    def get_forecast_report(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Similar to performance express
        return self.get_performance_express_report(code, start_date, end_date)

    def get_stock_industry(self, code: Optional[str] = None, date: Optional[str] = None) -> pd.DataFrame:
        ak = _ensure_akshare()
        logger.info(f"[AKshare] Stock industry for {code or 'all'}")

        # Try East Money industry name list
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                logger.info(f"[AKshare] Retrieved {len(df)} industry records (East Money)")
                return df
        except Exception as e:
            logger.debug(f"[AKshare] East Money industry failed: {e}")

        # Fallback: use combined SH/SZ stock list with basic info
        try:
            sh_df = ak.stock_info_sh_name_code()
            sz_df = ak.stock_info_sz_name_code()
            # Combine both exchanges
            if sh_df is not None and not sh_df.empty:
                rename_map = {"证券代码": "code", "证券简称": "code_name"}
                sh_df.rename(columns=rename_map, inplace=True)
                sh_df["exchange"] = "sh"
            if sz_df is not None and not sz_df.empty:
                sz_df.rename(columns=rename_map, inplace=True)
                sz_df["exchange"] = "sz"
            if sh_df is not None and sz_df is not None:
                combined = pd.concat([sh_df, sz_df], ignore_index=True)
                if code:
                    symbol = _strip_exchange_prefix(code)
                    combined = combined[combined["code"] == symbol]
                if not combined.empty:
                    logger.info(f"[AKshare] Retrieved {len(combined)} stock records (SH/SZ exchange)")
                    return combined
        except Exception as e:
            logger.debug(f"[AKshare] SH/SZ exchange info failed: {e}")

        raise NoDataFoundError(f"[AKshare] No industry data for {code or 'all'}")

    # ---- News crawler (AKshare does not have equivalent; delegate to Baostock) ----
    def crawl_news(self, query: str, top_k: int = 10) -> str:
        raise DataSourceError(
            "[AKshare] crawl_news is not supported by AKshare. "
            "This method should fall back to Baostock via CompositeDataSource."
        )
