"""
Tushare HTTP API 客户端 — 轻量封装，支持 200次/分钟 调用频率
提供: 历史PE/PB分位、PS/PCF、财务指标、行业分类、资金流向、分红历史
"""
import time
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

TUSHARE_TOKEN = "fd4ff6e84626d2e63616ec08769f99110d626a91856036c30cb34818"
TUSHARE_URL = "https://api.tushare.pro"

# 速率控制: 200次/分钟 → 最小间隔 0.3s
_last_call_time = 0.0
_MIN_INTERVAL = 0.35


def _rate_limit():
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.time()


def _call(api_name: str, params: dict = None, fields: str = "",
          max_retries: int = 3) -> Optional[dict]:
    """调用 Tushare HTTP API，返回 {fields: [...], items: [[...], ...]}

    瞬态失败（网络超时、限流、服务端错误）自动重试，指数退避 1s/2s/4s。
    """
    last_error = None
    for attempt in range(max_retries + 1):
        _rate_limit()
        try:
            resp = requests.post(TUSHARE_URL, json={
                "api_name": api_name,
                "token": TUSHARE_TOKEN,
                "params": params or {},
                "fields": fields,
            }, timeout=(3, 10))  # connect=3s, read=10s
            data = resp.json()
            if data.get("code") != 0:
                err_msg = data.get("msg", "未知错误")
                # 限流错误 (code=-104) 或服务端错误 (-1) 可重试
                if data.get("code") in (-104, -1, -2) and attempt < max_retries:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                return None
            result = data.get("data", {})
            return result if result.get("items") else None
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            return None
    return None


def _items_to_dicts(result: dict) -> List[Dict]:
    """将 Tushare 返回的 {fields, items} 转为 [{field: value}, ...]"""
    if not result:
        return []
    fields = result.get("fields", [])
    items = result.get("items", [])
    return [dict(zip(fields, row)) for row in items]


def get_stock_info(ts_code: str) -> Optional[Dict]:
    """股票基本信息: 名称、行业、上市日期"""
    r = _call("stock_basic", {"ts_code": ts_code}, "ts_code,name,industry,list_date,area")
    items = _items_to_dicts(r)
    return items[0] if items else None


def get_all_stocks() -> List[Dict]:
    """全A股列表（带行业分类）"""
    r = _call("stock_basic", {"list_status": "L"}, "ts_code,name,industry,area,list_date")
    return _items_to_dicts(r)


def get_stock_info_batch(ts_codes: List[str]) -> Dict[str, Dict]:
    """批量获取股票基本信息。

    Args:
        ts_codes: ['603871.SH', '000858.SZ', ...]

    Returns:
        {ts_code: {name, industry, list_date, area}, ...}
        缺失的 ts_code 不出现在返回中。
    """
    if not ts_codes:
        return {}
    ts_str = ",".join(ts_codes)
    r = _call("stock_basic", {"ts_code": ts_str},
              "ts_code,name,industry,list_date,area")
    items = _items_to_dicts(r)
    return {d["ts_code"]: d for d in items if d.get("ts_code")}


def get_kline(ts_code: str, start_date: str = None, end_date: str = None, days: int = 500) -> List[Dict]:
    """日K线数据"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    r = _call("daily", {
        "ts_code": ts_code, "start_date": start_date, "end_date": end_date
    }, "trade_date,open,high,low,close,vol,amount,pct_chg")
    return _items_to_dicts(r)


def get_daily_basic(ts_code: str, start_date: str = None, end_date: str = None, days: int = 500) -> List[Dict]:
    """日频估值指标: PE/PB/PS/总市值/流通市值/换手率"""
    if not start_date:
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y%m%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y%m%d")
    r = _call("daily_basic", {
        "ts_code": ts_code, "start_date": start_date, "end_date": end_date
    }, "trade_date,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate,turnover_rate_f,volume_ratio,dv_ratio")
    return _items_to_dicts(r)


def get_fina_indicator(ts_code: str, years: int = 3) -> List[Dict]:
    """财务指标: ROE/ROA/毛利率/净利率/资产负债率/流动比率/速动比率/周转率"""
    end_date = datetime.now().strftime("%Y1231")
    start_date = f"{datetime.now().year - years}0101"
    r = _call("fina_indicator", {
        "ts_code": ts_code, "start_date": start_date, "end_date": end_date
    }, "ts_code,ann_date,end_date,roe,roe_dt,roa,grossprofit_margin,netprofit_margin,"
       "debt_to_assets,current_ratio,quick_ratio,inv_turn,ar_turn,assets_turn,"
       "or_yoy,profit_yoy,ocf_yoy,ocf_netprofit")
    return _items_to_dicts(r)


_FINA_BATCH_SIZE = 40


def _chunk_ts_codes(ts_codes: List[str], size: int = _FINA_BATCH_SIZE) -> List[List[str]]:
    """将 ts_code 列表拆分为固定大小的批次"""
    return [ts_codes[i:i + size] for i in range(0, len(ts_codes), size)]


def get_fina_indicator_batch(ts_codes: List[str], years: int = 2) -> Dict[str, Dict]:
    """批量获取最新财务指标，每批最多 40 只。

    Args:
        ts_codes: ['603871.SH', '000858.SZ', ...]
        years: 查询最近几年的数据

    Returns:
        {ts_code: {roe, grossprofit_margin, ...}, ...}
        只返回每个 ts_code 的最新一期数据。
    """
    if not ts_codes:
        return {}

    end_date = datetime.now().strftime("%Y1231")
    start_date = f"{datetime.now().year - years}0101"
    fields = ("ts_code,ann_date,end_date,roe,roe_dt,roa,grossprofit_margin,"
              "netprofit_margin,debt_to_assets,current_ratio,quick_ratio,"
              "inv_turn,ar_turn,assets_turn,or_yoy,profit_yoy,ocf_yoy,ocf_netprofit")

    result: Dict[str, Dict] = {}
    chunks = _chunk_ts_codes(ts_codes, _FINA_BATCH_SIZE)

    for chunk in chunks:
        ts_str = ",".join(chunk)
        r = _call("fina_indicator", {
            "ts_code": ts_str,
            "start_date": start_date,
            "end_date": end_date,
        }, fields)
        items = _items_to_dicts(r)

        # 按 ts_code 分组，选最新一期 (end_date 最大)
        by_code: Dict[str, Dict] = {}
        for d in items:
            code = d.get("ts_code", "")
            if not code:
                continue
            if code not in by_code or d.get("end_date", "") > by_code[code].get("end_date", ""):
                by_code[code] = d
        result.update(by_code)

    return result


def get_dividend(ts_code: str) -> List[Dict]:
    """分红历史"""
    r = _call("dividend", {"ts_code": ts_code},
              "ts_code,end_date,div_proc,cash_div,cash_div_tax,stk_div,record_date,ex_date")
    return _items_to_dicts(r)


def get_moneyflow(ts_code: str, days: int = 30) -> List[Dict]:
    """个股资金流向"""
    start_date = (datetime.now() - timedelta(days=days + 5)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    r = _call("moneyflow", {
        "ts_code": ts_code, "start_date": start_date, "end_date": end_date
    }, "ts_code,trade_date,buy_elg_vol,sell_elg_vol,net_mf_vol,net_mf_amount")
    return _items_to_dicts(r)


def get_hsgt_top10(ts_code: str, days: int = 30) -> List[Dict]:
    """沪深股通十大成交股"""
    start_date = (datetime.now() - timedelta(days=days + 5)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    r = _call("hsgt_top10", {
        "ts_code": ts_code, "start_date": start_date, "end_date": end_date
    }, "ts_code,trade_date,name,close,buy,sell")
    return _items_to_dicts(r)


def compute_pe_percentile(ts_code: str, current_pe: float, years: int = 5) -> Optional[Dict]:
    """计算当前PE在历史中的分位数"""
    start_date = f"{datetime.now().year - years}0101"
    items = get_daily_basic(ts_code, start_date=start_date)
    if not items or len(items) < 100:
        return None
    pe_list = []
    for d in items:
        pe_val = d.get("pe_ttm")
        if pe_val and pe_val != "None":
            try:
                pe_list.append(float(pe_val))
            except (ValueError, TypeError):
                pass
    if not pe_list:
        return None
    pe_list_sorted = sorted(pe_list)
    rank = sum(1 for p in pe_list_sorted if p < current_pe)
    percentile = rank / len(pe_list_sorted) * 100 if pe_list_sorted else 0
    return {
        "current_pe": current_pe,
        "years": years,
        "min_pe": min(pe_list_sorted),
        "max_pe": max(pe_list_sorted),
        "median_pe": pe_list_sorted[len(pe_list_sorted) // 2],
        "percentile": round(percentile, 1),
        "data_points": len(pe_list_sorted),
    }


def get_top_list(trade_date: str = None) -> List[Dict]:
    """龙虎榜: 当日上榜股票列表"""
    if not trade_date:
        trade_date = datetime.now().strftime("%Y%m%d")
    r = _call("top_list", {"trade_date": trade_date},
              "trade_date,ts_code,name,close,pct_change,reason,l_buy,l_sell,net")
    return _items_to_dicts(r)


def get_top_inst(trade_date: str = None) -> List[Dict]:
    """龙虎榜机构明细: 机构买卖席位"""
    if not trade_date:
        trade_date = datetime.now().strftime("%Y%m%d")
    r = _call("top_inst", {"trade_date": trade_date},
              "trade_date,ts_code,exalter,buy,sel,net_buy,side,inst_name")
    return _items_to_dicts(r)


def get_top10_holders(ts_code: str, end_date: str = None) -> List[Dict]:
    """十大股东"""
    if not end_date:
        end_date = f"{datetime.now().year - 1}1231"
    r = _call("top10_holders", {"ts_code": ts_code, "end_date": end_date},
              "ts_code,end_date,holder_name,hold_amount,hold_ratio")
    return _items_to_dicts(r)


def get_stk_holdernumber(ts_code: str) -> List[Dict]:
    """股东户数趋势"""
    start = f"{datetime.now().year - 2}0101"
    end = datetime.now().strftime("%Y%m%d")
    r = _call("stk_holdernumber", {"ts_code": ts_code, "start_date": start, "end_date": end},
              "ts_code,ann_date,end_date,holder_num")
    return _items_to_dicts(r)


def get_balance_sheet(ts_code: str, end_date: str = None) -> Optional[Dict]:
    """资产负债表（单期）: 总资产/总负债/归母权益/货币资金/存货/应收"""
    if not end_date:
        end_date = f"{datetime.now().year}1231"
    r = _call("balancesheet", {"ts_code": ts_code, "end_date": end_date},
              "ts_code,end_date,total_assets,total_liab,total_hldr_eqy_exc_min_int,"
              "money_cap,accounts_receiv,inventories,notes_receiv")
    items = _items_to_dicts(r)
    return items[0] if items else None


def get_income_latest(ts_code: str) -> Optional[Dict]:
    """最近一期利润表: revenue/ebit/ebitda/n_income"""
    end_date = f"{datetime.now().year}1231"
    r = _call("income", {"ts_code": ts_code, "end_date": end_date},
              "ts_code,end_date,revenue,operate_profit,total_profit,n_income,ebit,ebitda,"
              "total_cogs,interest_expense")
    items = _items_to_dicts(r)
    # Get the latest non-None ebitda row
    for item in items:
        if item.get("ebitda") and item["ebitda"] != "None":
            return item
    return items[0] if items else None


def compute_ev_ebitda(ts_code: str, current_mv: float) -> Optional[Dict]:
    """计算 EV/EBITDA:
    EV = 总市值 + 总负债 - 货币资金
    EBITDA = ebitda (from income) 或 ebit + 折旧摊销(估算)
    """
    # Get balance sheet
    bs = get_balance_sheet(ts_code)
    if not bs or not bs.get("total_liab") or not bs.get("money_cap"):
        return None

    total_liab = float(bs["total_liab"])
    money_cap = float(bs["money_cap"])
    ev = current_mv + total_liab - money_cap

    # Get income for ebitda
    inc = get_income_latest(ts_code)
    if not inc:
        return None

    ebitda = inc.get("ebitda")
    ebit = inc.get("ebit")
    n_income = inc.get("n_income")

    # If no ebitda, estimate: EBITDA ≈ EBIT + Depreciation (use revenue * 5% as rough approx for depreciation)
    ebitda_is_estimated = False
    if not ebitda or ebitda == "None":
        if ebit and ebit != "None":
            ebit_val = float(ebit)
            ebitda = ebit_val * 1.08  # rough 8% uplift for D&A
            ebitda_is_estimated = True
        else:
            return None

    ebitda_val = float(ebitda)
    if ebitda_val <= 0:
        return None

    result = {
        "ev": round(ev / 1e8, 2),           # 亿元
        "ebitda": round(ebitda_val / 1e8, 2),  # 亿元
        "ev_ebitda": round(ev / ebitda_val, 2),
        "ebitda_estimated": ebitda_is_estimated,
    }
    if n_income and n_income != "None":
        result["pe"] = round(current_mv / 1e8 / float(n_income) * 1e8 if float(n_income) > 0 else 0, 2)  # just for reference
    return result


def get_fund_holders(ts_code: str, period: str = None) -> List[Dict]:
    """查询持有该股票的所有基金（通过 fund_portfolio 全量查询后过滤）"""
    if not period:
        period = f"{datetime.now().year - 1}1231"
    r = _call("fund_portfolio", {"period": period},
              "ts_code,ann_date,end_date,symbol,amount,market_val")
    if not r:
        return []
    items = _items_to_dicts(r)
    # Filter: symbol is the stock held by the fund
    pure = ts_code.replace(".SH", "").replace(".SZ", "")
    return [{
        "fund_code": d["ts_code"],
        "fund_name": "",  # would need fund_basic lookup
        "ann_date": d.get("ann_date", ""),
        "shares": d.get("amount", ""),
        "market_val": d.get("market_val", ""),
    } for d in items if d.get("symbol") == f"{pure}.SH" or d.get("symbol") == f"{pure}.SZ" or d.get("symbol") == pure]


def get_stk_holdertrade(ts_code: str, years: int = 3) -> List[Dict]:
    """股东增减持: 大股东/董监高增持减持记录"""
    start = f"{datetime.now().year - years}0101"
    end = datetime.now().strftime("%Y%m%d")
    r = _call("stk_holdertrade", {"ts_code": ts_code, "start_date": start, "end_date": end},
              "ts_code,ann_date,holder_name,holder_type,in_de,change_vol,"
              "change_ratio,after_share,after_ratio,avg_price,total_share")
    return _items_to_dicts(r)


def get_stock_news_em(symbol: str, limit: int = 10) -> List[Dict]:
    """AkShare 东方财富个股新闻"""
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return []
        news = []
        for _, row in df.head(limit).iterrows():
            news.append({
                "title": str(row.get("新闻标题", "")),
                "content": str(row.get("新闻内容", ""))[:200],
                "time": str(row.get("发布时间", "")),
                "source": str(row.get("文章来源", "")),
                "url": str(row.get("新闻链接", "")),
            })
        return news
    except Exception:
        return []
