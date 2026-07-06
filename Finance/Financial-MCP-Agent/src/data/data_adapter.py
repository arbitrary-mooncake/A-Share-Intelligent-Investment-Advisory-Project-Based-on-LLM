"""
AKShare -> Tushare 格式适配器。

目的：让 Agent 代码无需关心数据来源，统一使用 Tushare 格式处理数据。
当 Lite 模式下 Tushare 积分不足时，AKShare 数据经过适配器转换后，
字段名和数据结构与 Tushare 返回完全一致。
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_float(val) -> Optional[float]:
    """安全转换为 float"""
    if val is None or val == "" or val == "--":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_str(val) -> str:
    """安全转换为字符串"""
    if val is None:
        return ""
    return str(val).strip()


def adapt_financial_indicators(akshare_data: list) -> list:
    """
    AKShare stock_financial_analysis_indicator -> Tushare fina_indicator 格式

    字段映射：
    - 净资产收益率(%) -> roe
    - 销售毛利率(%) -> grossprofit_margin
    - 销售净利率(%) -> netprofit_margin
    - 资产负债率(%) -> debt_to_assets
    - 流动比率 -> current_ratio
    - 速动比率 -> quick_ratio
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "ann_date": _safe_str(row.get("公告日期", "")).replace("-", ""),
            "end_date": _safe_str(row.get("报告期", "")).replace("-", ""),
            "roe": _safe_float(row.get("净资产收益率(%)")),
            "roe_dt": _safe_float(row.get("净资产收益率(扣除非经常性损益)(%)")),
            "roa": _safe_float(row.get("总资产净利率(%)")),
            "grossprofit_margin": _safe_float(row.get("销售毛利率(%)")),
            "netprofit_margin": _safe_float(row.get("销售净利率(%)")),
            "debt_to_assets": _safe_float(row.get("资产负债率(%)")),
            "current_ratio": _safe_float(row.get("流动比率")),
            "quick_ratio": _safe_float(row.get("速动比率")),
            "inv_turn": _safe_float(row.get("存货周转率(次)")),
            "ar_turn": _safe_float(row.get("应收账款周转率(次)")),
            "assets_turn": _safe_float(row.get("总资产周转率(次)")),
            "or_yoy": _safe_float(row.get("主营业务收入增长率(%)")),
            "profit_yoy": _safe_float(row.get("净利润增长率(%)")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_income(akshare_data: list) -> list:
    """
    AKShare stock_financial_report_sina (利润表) -> Tushare income 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "end_date": _safe_str(row.get("报告期", "")).replace("-", ""),
            "revenue": _safe_float(row.get("一、营业总收入")),
            "operate_profit": _safe_float(row.get("三、营业利润")),
            "total_profit": _safe_float(row.get("四、利润总额")),
            "n_income": _safe_float(row.get("五、净利润")),
            "n_income_attr_p": _safe_float(row.get("归属于母公司所有者的净利润")),
            "ebit": _safe_float(row.get("息税前利润")),
            "ebitda": _safe_float(row.get("息税折旧摊销前利润")),
            "total_cogs": _safe_float(row.get("二、营业总成本")),
            "interest_expense": _safe_float(row.get("利息费用")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_balance_sheet(akshare_data: list) -> list:
    """
    AKShare stock_financial_report_sina (资产负债表) -> Tushare balancesheet 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "end_date": _safe_str(row.get("报告期", "")).replace("-", ""),
            "total_assets": _safe_float(row.get("资产总计")),
            "total_liab": _safe_float(row.get("负债合计")),
            "total_hldr_eqy_exc_min_int": _safe_float(row.get("所有者权益合计(不含少数股东权益)")),
            "total_hldr_eqy_inc_min_int": _safe_float(row.get("所有者权益合计(含少数股东权益)")),
            "money_cap": _safe_float(row.get("货币资金")),
            "accounts_receiv": _safe_float(row.get("应收账款")),
            "inventories": _safe_float(row.get("存货")),
            "goodwill": _safe_float(row.get("商誉")),
            "notes_receiv": _safe_float(row.get("应收票据")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_cashflow(akshare_data: list) -> list:
    """
    AKShare stock_financial_report_sina (现金流量表) -> Tushare cashflow 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "end_date": _safe_str(row.get("报告期", "")).replace("-", ""),
            "n_cashflow_act": _safe_float(row.get("经营活动产生的现金流量净额")),
            "n_cashflow_inv_act": _safe_float(row.get("投资活动产生的现金流量净额")),
            "n_cashflow_fin_act": _safe_float(row.get("筹资活动产生的现金流量净额")),
            "c_fr_oth_operate_a": _safe_float(row.get("收到的其他与经营活动有关的现金")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_money_flow(akshare_data: list) -> list:
    """
    AKShare stock_individual_fund_flow -> Tushare moneyflow 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "trade_date": _safe_str(row.get("日期", "")).replace("-", ""),
            "buy_elg_vol": _safe_float(row.get("超大单净流入-净量")),
            "sell_elg_vol": None,
            "net_mf_vol": _safe_float(row.get("主力净流入-净量")),
            "net_mf_amount": _safe_float(row.get("主力净流入-净额")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_top_holders(akshare_data: list) -> list:
    """
    AKShare stock_gdfx_holding_detail_em -> Tushare top10_holders 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "end_date": _safe_str(row.get("报告期", "")).replace("-", ""),
            "holder_name": _safe_str(row.get("股东名称")),
            "hold_amount": _safe_float(row.get("持股数量")),
            "hold_ratio": _safe_float(row.get("持股比例")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_margin(akshare_data: list) -> list:
    """
    AKShare stock_margin_detail -> Tushare margin_detail 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("标的证券", "")),
            "trade_date": _safe_str(row.get("日期", "")).replace("-", ""),
            "rzye": _safe_float(row.get("融资余额")),
            "rqye": _safe_float(row.get("融券余额")),
            "rzmre": _safe_float(row.get("融资买入额")),
            "rqmcl": _safe_float(row.get("融券卖出量")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_block_trade(akshare_data: list) -> list:
    """
    AKShare stock_dzjy_detail -> Tushare block_trade 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("证券代码", "")),
            "trade_date": _safe_str(row.get("交易日期", "")).replace("-", ""),
            "price": _safe_float(row.get("成交价")),
            "vol": _safe_float(row.get("成交量")),
            "amount": _safe_float(row.get("成交额")),
            "buyer": _safe_str(row.get("买方营业部")),
            "seller": _safe_str(row.get("卖方营业部")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_limit_stats(zt_data: list, dt_data: list, stock_code: str) -> list:
    """
    AKShare stock_zt_pool_em + stock_dt_pool_em -> Tushare stk_limit 格式
    """
    zt_count = len(zt_data) if zt_data else 0
    dt_count = len(dt_data) if dt_data else 0

    return [{
        "ts_code": stock_code,
        "zt_count": zt_count,
        "dt_count": dt_count,
        "_source": "akshare",
    }]


def adapt_concept(akshare_data: list, stock_code: str) -> list:
    """
    AKShare stock_board_concept_name -> Tushare concept 格式
    """
    if not akshare_data:
        return []

    return [{
        "ts_code": stock_code,
        "concepts": akshare_data,
        "_source": "akshare",
    }]


def adapt_daily(akshare_data: list) -> list:
    """
    AKShare stock_zh_a_hist -> Tushare daily 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("股票代码", "")),
            "trade_date": _safe_str(row.get("日期", "")).replace("-", ""),
            "open": _safe_float(row.get("开盘")),
            "high": _safe_float(row.get("最高")),
            "low": _safe_float(row.get("最低")),
            "close": _safe_float(row.get("收盘")),
            "vol": _safe_float(row.get("成交量")),
            "amount": _safe_float(row.get("成交额")),
            "pct_chg": _safe_float(row.get("涨跌幅")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result


def adapt_daily_basic(akshare_data: list) -> list:
    """
    AKShare stock_a_indicator_lg -> Tushare daily_basic 格式
    """
    if not akshare_data:
        return []

    result = []
    for row in akshare_data:
        adapted = {
            "ts_code": _safe_str(row.get("code", "")),
            "trade_date": _safe_str(row.get("trade_date", "")).replace("-", ""),
            "pe": _safe_float(row.get("pe")),
            "pe_ttm": _safe_float(row.get("pe_ttm")),
            "pb": _safe_float(row.get("pb")),
            "ps": _safe_float(row.get("ps")),
            "ps_ttm": _safe_float(row.get("ps_ttm")),
            "total_mv": _safe_float(row.get("total_mv")),
            "circ_mv": _safe_float(row.get("circ_mv")),
            "turnover_rate": _safe_float(row.get("turnover_rate")),
            "_source": "akshare",
        }
        result.append(adapted)

    return result
