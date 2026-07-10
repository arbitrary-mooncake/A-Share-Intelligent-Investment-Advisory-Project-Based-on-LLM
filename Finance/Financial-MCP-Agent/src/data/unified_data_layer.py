"""
统一数据接口层：根据 Tushare 积分自动选择数据源。

策略：
- Tushare 积分充足 -> 调用 Tushare（更权威、更稳定）
- Tushare 积分不足 -> 回退 AKShare（免费、覆盖面广）
- 两者都失败 -> 返回 None，Agent 标记数据缺失

Full 模式下全部使用 Tushare（因为 Full 模式要求 5000+ 积分）。
Lite 模式下自动检测积分并回退 AKShare。
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from src.data.akshare_client import call_akshare, dataframe_to_dicts
from src.data import data_adapter

logger = logging.getLogger(__name__)

# Tushare 接口积分要求
TUSHARE_POINTS_REQUIRED = {
    "daily": 120,
    "daily_basic": 120,
    "stock_basic": 120,
    "trade_cal": 120,
    "fina_indicator": 500,
    "income": 500,
    "balancesheet": 500,
    "cashflow": 500,
    "moneyflow": 2000,
    "margin_detail": 2000,
    "top10_holders": 2000,
    "block_trade": 2000,
    "stk_limit": 2000,
    "concept": 5000,
    "concept_detail": 5000,
}


def _to_akshare_code(tushare_code: str) -> str:
    """Tushare 代码 -> AKShare 代码"""
    c = tushare_code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
    c = c.replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()
    return c


def _to_tushare_code(code: str) -> str:
    """AKShare 代码 -> Tushare 代码"""
    c = code.strip()
    if c.startswith(("6", "5")):
        return f"{c}.SH"
    elif c.startswith(("0", "3", "1")):
        return f"{c}.SZ"
    elif c.startswith(("4", "8", "9")):
        return f"{c}.BJ"
    return f"{c}.SZ"


class UnifiedDataLayer:
    """
    统一数据层：根据 Tushare 积分自动选择数据源。

    用法：
        data_layer = UnifiedDataLayer()
        fina_data = await data_layer.get_financial_indicators("603871.SH")
    """

    def __init__(self, tushare_call_func=None, tushare_points: int = 0):
        """
        Args:
            tushare_call_func: Tushare _call 函数引用（从 MCP server 或 tushare_client 传入）
            tushare_points: 当前 Tushare 积分（如果已知）
        """
        self._tushare_call = tushare_call_func
        self._tushare_points = tushare_points

    def _has_access(self, required_points: int) -> bool:
        """检查是否有权限访问某个 Tushare 接口"""
        return self._tushare_points >= required_points

    async def _call_tushare(self, api_name: str, params: dict = None, fields: str = "") -> Optional[list]:
        """调用 Tushare API（在线程池中执行同步调用，避免阻塞事件循环）"""
        if self._tushare_call is None:
            return None
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._tushare_call, api_name, params or {}, fields
            )
            if result is None:
                return None
            if isinstance(result, dict):
                f = result.get("fields", [])
                items = result.get("items", [])
                return [dict(zip(f, row)) for row in items] if items else []
            if isinstance(result, list):
                return result
            return None
        except Exception as e:
            logger.warning(f"Tushare call failed: {api_name} -> {e}")
            return None

    async def _try_tushare_first(self, api_name: str, params: dict, fields: str,
                                  required_points: int) -> Optional[list]:
        """先尝试 Tushare，积分不足或失败时返回 None"""
        if not self._has_access(required_points):
            logger.info(f"Tushare {api_name} requires {required_points} points, skipping")
            return None

        result = await self._call_tushare(api_name, params, fields)
        if result:
            for item in result:
                item["_source"] = "tushare"
            return result
        return None

    # ─── 财务数据 ─────────────────────────────────────────

    async def get_financial_indicators(self, stock_code: str, years: int = 3) -> Optional[list]:
        """获取财务指标（ROE、毛利率、净利率等）"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        end_date = datetime.now().strftime("%Y1231")
        start_date = f"{datetime.now().year - years}0101"

        result = await self._try_tushare_first(
            "fina_indicator",
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
            "ts_code,ann_date,end_date,roe,roe_dt,roa,grossprofit_margin,netprofit_margin,"
            "debt_to_assets,current_ratio,quick_ratio,inv_turn,ar_turn,assets_turn,"
            "or_yoy,profit_yoy",
            500,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_financial_analysis_indicator", symbol=ak_code)
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_financial_indicators(dicts)
        return None

    async def get_income_statement(self, stock_code: str) -> Optional[list]:
        """获取利润表"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        end_date = f"{datetime.now().year}1231"

        result = await self._try_tushare_first(
            "income",
            {"ts_code": ts_code, "end_date": end_date},
            "ts_code,end_date,revenue,operate_profit,total_profit,n_income,ebit,ebitda,"
            "total_cogs,interest_expense",
            500,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_financial_report_sina", stock=ak_code, symbol="利润表")
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_income(dicts)
        return None

    async def get_balance_sheet(self, stock_code: str) -> Optional[list]:
        """获取资产负债表"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        end_date = f"{datetime.now().year}1231"

        result = await self._try_tushare_first(
            "balancesheet",
            {"ts_code": ts_code, "end_date": end_date},
            "ts_code,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int,"
            "money_cap,accounts_receiv,inventories,notes_receiv",
            500,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_financial_report_sina", stock=ak_code, symbol="资产负债表")
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_balance_sheet(dicts)
        return None

    async def get_cashflow(self, stock_code: str) -> Optional[list]:
        """获取现金流量表"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        end_date = f"{datetime.now().year}1231"

        result = await self._try_tushare_first(
            "cashflow",
            {"ts_code": ts_code, "end_date": end_date},
            "ts_code,end_date,n_cashflow_act,n_cashflow_inv_act,n_cashflow_fin_act,"
            "c_fr_oth_operate_a",
            500,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_financial_report_sina", stock=ak_code, symbol="现金流量表")
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_cashflow(dicts)
        return None

    # ─── 资金流向 ─────────────────────────────────────────

    async def get_money_flow(self, stock_code: str, days: int = 30) -> Optional[list]:
        """获取个股资金流向"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        start_date = (datetime.now() - timedelta(days=days + 5)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        result = await self._try_tushare_first(
            "moneyflow",
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
            "ts_code,trade_date,buy_elg_vol,sell_elg_vol,net_mf_vol,net_mf_amount",
            2000,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        market = "bj" if ts_code.endswith(".BJ") else ("sh" if ts_code.endswith(".SH") else "sz")
        raw = await call_akshare("stock_individual_fund_flow", stock=ak_code, market=market)
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_money_flow(dicts)
        return None

    async def get_margin_trading(self, stock_code: str) -> Optional[list]:
        """获取融资融券数据"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code

        result = await self._try_tushare_first(
            "margin_detail",
            {"ts_code": ts_code},
            "ts_code,trade_date,rzye,rqye,rzmre,rqmcl",
            2000,
        )
        if result:
            return result

        if ts_code.endswith(".BJ"):
            return None  # 北交所无融资融券数据
        exchange = "szse" if ts_code.endswith(".SZ") else "sse"
        raw = await call_akshare(f"stock_margin_detail_{exchange}")
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            adapted = data_adapter.adapt_margin(dicts)
            # AKShare 返回全市场数据，按个股代码过滤
            ak_code = _to_akshare_code(stock_code)
            return [r for r in adapted if r.get("ts_code") == ak_code]
        return None

    async def get_block_trade(self, stock_code: str) -> Optional[list]:
        """获取大宗交易数据"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code

        result = await self._try_tushare_first(
            "block_trade",
            {"ts_code": ts_code},
            "ts_code,trade_date,price,vol,amount,buyer,seller",
            2000,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_dzjy_detail", symbol=ak_code)
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_block_trade(dicts)
        return None

    # ─── 持仓与概念 ───────────────────────────────────────

    async def get_top_holders(self, stock_code: str) -> Optional[list]:
        """获取十大股东"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        end_date = f"{datetime.now().year - 1}1231"

        result = await self._try_tushare_first(
            "top10_holders",
            {"ts_code": ts_code, "end_date": end_date},
            "ts_code,end_date,holder_name,hold_amount,hold_ratio",
            2000,
        )
        if result:
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_gdfx_holding_detail_em", symbol=ak_code)
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_top_holders(dicts)
        return None

    async def get_concept_data(self, stock_code: str) -> Optional[list]:
        """获取概念板块信息"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code

        result = await self._try_tushare_first(
            "concept",
            {"ts_code": ts_code},
            "ts_code,name",
            5000,
        )
        if result:
            return result

        raw = await call_akshare("stock_board_concept_name")
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_concept(dicts, ts_code)
        return None

    # ─── 涨跌停统计 ──────────────────────────────────────

    async def get_limit_statistics(self, stock_code: str) -> Optional[list]:
        """获取涨跌停统计"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code

        result = await self._try_tushare_first(
            "stk_limit",
            {"ts_code": ts_code},
            "ts_code,zt_count,dt_count",
            2000,
        )
        if result:
            return result

        zt_raw = await call_akshare("stock_zt_pool_em")
        dt_raw = await call_akshare("stock_dt_pool_em")
        zt_dicts = dataframe_to_dicts(zt_raw) if zt_raw is not None else []
        dt_dicts = dataframe_to_dicts(dt_raw) if dt_raw is not None else []
        return data_adapter.adapt_limit_stats(zt_dicts, dt_dicts, ts_code)

    # ─── 基础数据（Tushare 免费可用，无需 AKShare 回退）────

    async def get_daily_quotes(self, stock_code: str, days: int = 250) -> Optional[list]:
        """获取日线行情（120 积分即可）"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        result = await self._call_tushare(
            "daily",
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
            "trade_date,open,high,low,close,vol,amount,pct_chg",
        )
        if result:
            for item in result:
                item["_source"] = "tushare"
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare(
            "stock_zh_a_hist",
            symbol=ak_code, period="daily",
            start_date=start_date, end_date=end_date, adjust="",
        )
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_daily(dicts)
        return None

    async def get_daily_basic(self, stock_code: str, days: int = 500) -> Optional[list]:
        """获取基础估值指标（120 积分即可）"""
        ts_code = _to_tushare_code(stock_code) if "." not in stock_code else stock_code
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        result = await self._call_tushare(
            "daily_basic",
            {"ts_code": ts_code, "start_date": start_date, "end_date": end_date},
            "trade_date,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate",
        )
        if result:
            for item in result:
                item["_source"] = "tushare"
            return result

        ak_code = _to_akshare_code(stock_code)
        raw = await call_akshare("stock_a_indicator_lg", symbol=ak_code)
        if raw is not None:
            dicts = dataframe_to_dicts(raw)
            return data_adapter.adapt_daily_basic(dicts)
        return None
