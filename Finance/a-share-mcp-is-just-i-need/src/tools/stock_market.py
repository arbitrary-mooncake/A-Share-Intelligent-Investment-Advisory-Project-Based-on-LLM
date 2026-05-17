"""
股票市场数据工具，用于MCP服务器
"""
import logging
from typing import List, Optional, Callable, Any

from mcp.server.fastmcp import FastMCP
from src.data_source_interface import FinancialDataSource, NoDataFoundError, LoginError, DataSourceError
from src.formatting.markdown_formatter import format_df_to_markdown

logger = logging.getLogger(__name__)


def safe_data_fetch(
    func_name: str,
    data_source_func: Callable,
    *args,
    **kwargs
) -> str:
    """
    安全的数据获取函数，统一处理所有异常和错误情况
    
    参数:
        func_name: 函数名称，用于日志记录
        data_source_func: 数据源函数
        *args: 传递给数据源函数的参数
        **kwargs: 传递给数据源函数的关键字参数
        
    返回:
        Markdown格式的数据表格或错误消息
    """
    try:
        # 调用数据源函数
        df = data_source_func(*args, **kwargs)
        
        # 格式化结果
        logger.info(f"Successfully retrieved data for {func_name}, formatting to Markdown.")
        return format_df_to_markdown(df)
        
    except NoDataFoundError as e:
        logger.warning(f"NoDataFoundError for {func_name}: {e}")
        return f"Error: {e}"
    except LoginError as e:
        logger.error(f"LoginError for {func_name}: {e}")
        return f"Error: Could not connect to data source. {e}"
    except DataSourceError as e:
        logger.error(f"DataSourceError for {func_name}: {e}")
        return f"Error: An error occurred while fetching data. {e}"
    except ValueError as e:
        logger.warning(f"ValueError processing request for {func_name}: {e}")
        return f"Error: Invalid input parameter. {e}"
    except Exception as e:
        logger.exception(f"Unexpected Exception processing {func_name}: {e}")
        return f"Error: An unexpected error occurred: {e}"


def register_stock_market_tools(app: FastMCP, active_data_source: FinancialDataSource):
    """
    向MCP应用注册股票市场数据工具

    参数:
        app: FastMCP应用实例
        active_data_source: 活跃的金融数据源
    """

    @app.tool()
    def get_historical_k_data(
        code: str,
        start_date: str,
        end_date: str,
        frequency: str = "d",
        adjust_flag: str = "3",
        fields: Optional[List[str]] = None,
    ) -> str:
        """
        获取中国A股股票的历史K线（OHLCV）数据

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'
            frequency: 数据频率。有效选项（来自Baostock）：
                         'd': 日线
                         'w': 周线
                         'm': 月线
                         '5': 5分钟
                         '15': 15分钟
                         '30': 30分钟
                         '60': 60分钟
                       默认为'd'
            adjust_flag: 价格/成交量调整标志。有效选项（来自Baostock）：
                           '1': 前复权
                           '2': 后复权
                           '3': 不复权
                         默认为'3'
            fields: 可选的具体数据字段列表（必须是有效的Baostock字段）
                    如果为None或空，将使用默认字段（例如：date, code, open, high, low, close, volume, amount, pctChg）

        返回:
            包含K线数据表的Markdown格式字符串，或错误消息
            如果结果集太大，表格可能会被截断
        """
        logger.info(
            f"Tool 'get_historical_k_data' called for {code} ({start_date}-{end_date}, freq={frequency}, adj={adjust_flag}, fields={fields})")
        
        # 验证频率和调整标志
        valid_freqs = ['d', 'w', 'm', '5', '15', '30', '60']
        valid_adjusts = ['1', '2', '3']
        if frequency not in valid_freqs:
            logger.warning(f"Invalid frequency requested: {frequency}")
            return f"Error: Invalid frequency '{frequency}'. Valid options are: {valid_freqs}"
        if adjust_flag not in valid_adjusts:
            logger.warning(f"Invalid adjust_flag requested: {adjust_flag}")
            return f"Error: Invalid adjust_flag '{adjust_flag}'. Valid options are: {valid_adjusts}"

        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_historical_k_data",
            active_data_source.get_historical_k_data,
            code=code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjust_flag=adjust_flag,
            fields=fields,
        )

    @app.tool()
    def get_stock_basic_info(code: str, fields: Optional[List[str]] = None) -> str:
        """
        获取给定中国A股股票的基本信息

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            fields: 可选列表，用于从可用的基本信息中选择特定列
                    （例如：['code', 'code_name', 'industry', 'listingDate']）
                    如果为None或空，返回Baostock中所有可用的基本信息列

        返回:
            包含基本股票信息表的Markdown格式字符串，或错误消息
        """
        logger.info(
            f"Tool 'get_stock_basic_info' called for {code} (fields={fields})")
        
        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_stock_basic_info",
            active_data_source.get_stock_basic_info,
            code=code,
            fields=fields,
        )

    @app.tool()
    def get_dividend_data(code: str, year: str, year_type: str = "report") -> str:
        """
        获取给定股票代码和年份的分红信息

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            year: 查询年份（例如：'2023'）
            year_type: 年份类型。有效选项（来自Baostock）：
                         'report': 预案公告年份
                         'operate': 除权除息年份
                       默认为'report'

        返回:
            包含分红数据表的Markdown格式字符串，或错误消息
        """
        logger.info(
            f"Tool 'get_dividend_data' called for {code}, year={year}, year_type={year_type}")
        
        # 基本验证
        if year_type not in ['report', 'operate']:
            logger.warning(f"Invalid year_type requested: {year_type}")
            return f"Error: Invalid year_type '{year_type}'. Valid options are: 'report', 'operate'"
        if not year.isdigit() or len(year) != 4:
            logger.warning(f"Invalid year format requested: {year}")
            return f"Error: Invalid year '{year}'. Please provide a 4-digit year."

        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_dividend_data",
            active_data_source.get_dividend_data,
            code=code,
            year=year,
            year_type=year_type,
        )

    @app.tool()
    def get_adjust_factor_data(code: str, start_date: str, end_date: str) -> str:
        """
        获取给定股票代码和日期范围的复权因子数据
        使用Baostock的"涨跌幅复权算法"因子。用于计算复权价格

        参数:
            code: Baostock格式的股票代码（例如：'sh.600000', 'sz.000001'）
            start_date: 开始日期，格式为'YYYY-MM-DD'
            end_date: 结束日期，格式为'YYYY-MM-DD'

        返回:
            包含复权因子数据表的Markdown格式字符串，或错误消息
        """
        logger.info(
            f"Tool 'get_adjust_factor_data' called for {code} ({start_date} to {end_date})")
        
        # 使用通用函数处理数据获取
        return safe_data_fetch(
            "get_adjust_factor_data",
            active_data_source.get_adjust_factor_data,
            code=code,
            start_date=start_date,
            end_date=end_date,
        )

    @app.tool()
    def get_st_risk_data(code: str) -> str:
        """
        检测A股股票的ST风险状态。
        通过AkShare/Sina接口获取股票当前名称，判断是否包含ST/*ST标记。

        参数:
            code: 股票代码（例如：'sh.600000', 'sz.000001'）

        返回:
            Markdown格式的ST风险数据表格，包含当前ST状态、ST类型、风险等级
        """
        import akshare as ak

        symbol = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()
        logger.info(f"Tool 'get_st_risk_data' called for {code} (symbol={symbol})")

        try:
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return (
                    "| 项目 | 值 |\\n"
                    "|------|----|\\n"
                    "| ST状态 | 数据不可用 |\\n"
                    "| 数据来源 | AkShare/Sina返回空数据 |\\n"
                    "| 说明 | ST风险数据暂时不可用，请通过其他渠道核实 |"
                )

            row = df[df["code"] == symbol]
            if row.empty:
                return (
                    f"| 项目 | 值 |\\n"
                    f"|------|----|\\n"
                    f"| 股票代码 | {symbol} |\\n"
                    f"| ST状态 | 正常（未在ST名单中） |\\n"
                    f"| 数据来源 | AkShare/Sina stock_info_a_code_name |\\n"
                    f"| 说明 | 当前股票名称不含ST/*ST标记，未进入风险警示板 |"
                )

            name = str(row.iloc[0]["name"])
            is_st = "ST" in name or "*ST" in name

            if is_st:
                st_type = "退市风险警示（*ST）" if "*ST" in name else "其他风险警示（ST）"
                return (
                    f"| 项目 | 值 |\\n"
                    f"|------|----|\\n"
                    f"| 股票代码 | {symbol} |\\n"
                    f"| 当前名称 | {name} |\\n"
                    f"| ST状态 | ⚠️ **已标记ST** |\\n"
                    f"| ST类型 | {st_type} |\\n"
                    f"| 数据来源 | AkShare/Sina stock_info_a_code_name |\\n"
                    f"| 风险提示 | 该股票当前处于风险警示板，存在退市风险，投资需极度谨慎 |"
                )

            return (
                f"| 项目 | 值 |\\n"
                f"|------|----|\\n"
                f"| 股票代码 | {symbol} |\\n"
                f"| 当前名称 | {name} |\\n"
                f"| ST状态 | 正常 |\\n"
                f"| 数据来源 | AkShare/Sina stock_info_a_code_name |"
            )

        except Exception as e:
            logger.warning(f"get_st_risk_data failed for {code}: {e}")
            return (
                f"| 项目 | 值 |\\n"
                f"|------|----|\\n"
                f"| ST状态 | 查询失败 |\\n"
                f"| 错误信息 | {str(e)[:100]} |\\n"
                f"| 数据来源 | AkShare/Sina |\\n"
                f"| 说明 | ST风险数据暂时不可用，请通过其他渠道核实 |"
            )
