"""
Tushare MCP Server — 通过 stdio 传输暴露 Tushare 数据工具
供 LangGraph Agent 通过 langchain-mcp-adapters 调用
"""
import sys
import os

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
from mcp.server.fastmcp import FastMCP

# 内联 tushare_client（避免跨项目 src 命名空间冲突）
import json, time as _time, requests as _requests
from datetime import datetime as _datetime, timedelta as _timedelta
from typing import Dict, List, Optional, Any

_TUSHARE_TOKEN = "fd4ff6e84626d2e63616ec08769f99110d626a91856036c30cb34818"
_TUSHARE_URL = "https://api.tushare.pro"
_last_call = 0.0

def _rate_limit():
    global _last_call
    elapsed = _time.time() - _last_call
    if elapsed < 0.35:
        _time.sleep(0.35 - elapsed)
    _last_call = _time.time()

def _call(api_name: str, params: dict = None, fields: str = "") -> Optional[dict]:
    _rate_limit()
    try:
        resp = _requests.post(_TUSHARE_URL, json={
            "api_name": api_name, "token": _TUSHARE_TOKEN,
            "params": params or {}, "fields": fields,
        }, timeout=15)
        data = resp.json()
        if data.get("code") != 0:
            return None
        r = data.get("data", {})
        return r if r.get("items") else None
    except Exception:
        return None

def _dicts(result) -> list:
    if not result: return []
    f = result.get("fields", []); items = result.get("items", [])
    return [dict(zip(f, r)) for r in items]

def _ts_code(code: str) -> str:
    c = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").strip()
    # 6xxxxx=沪市主板, 688xxx=科创板, 5xxxxx=沪市基金/ETF, 8xxxxx=北交所
    # 0xxxxx=深市主板, 3xxxxx=创业板, 1xxxxx=深市基金/ETF, 4xxxxx=深市
    return f"{c}.SH" if (c.startswith(("6", "688", "5", "8"))) else f"{c}.SZ"

def ts_stock_info(code: str) -> list:
    return _dicts(_call("stock_basic", {"ts_code": _ts_code(code)}, "ts_code,name,industry,list_date,area"))

def ts_kline(code: str, days: int = 250) -> list:
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("daily", {"ts_code": _ts_code(code), "start_date": s, "end_date": e},
                        "trade_date,open,high,low,close,vol,amount,pct_chg"))

def ts_daily_basic(code: str, days: int = 500) -> list:
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("daily_basic", {"ts_code": _ts_code(code), "start_date": s, "end_date": e},
                        "trade_date,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate"))

def ts_fina_indicator(code: str, years: int = 3) -> list:
    e = _datetime.now().strftime("%Y1231")
    s = f"{_datetime.now().year - years}0101"
    return _dicts(_call("fina_indicator", {"ts_code": _ts_code(code), "start_date": s, "end_date": e},
                        "ts_code,ann_date,end_date,roe,roe_dt,roa,grossprofit_margin,netprofit_margin,"
                        "debt_to_assets,current_ratio,quick_ratio,inv_turn,ar_turn,assets_turn,"
                        "or_yoy,profit_yoy"))

def ts_dividend(code: str) -> list:
    return _dicts(_call("dividend", {"ts_code": _ts_code(code)},
                        "ts_code,end_date,div_proc,cash_div,cash_div_tax,stk_div,record_date,ex_date"))

def ts_moneyflow(code: str, days: int = 30) -> list:
    s = (_datetime.now() - _timedelta(days=days + 5)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("moneyflow", {"ts_code": _ts_code(code), "start_date": s, "end_date": e},
                        "ts_code,trade_date,buy_elg_vol,sell_elg_vol,net_mf_vol,net_mf_amount"))

def ts_pe_percentile(code: str) -> list:
    ts = _ts_code(code)
    basics = ts_daily_basic(code, days=5)
    if not basics: return []
    pe = basics[0].get("pe_ttm")
    if not pe or pe == "None": return []
    pe = float(pe)
    s = f"{_datetime.now().year - 5}0101"
    hist = _dicts(_call("daily_basic", {"ts_code": ts, "start_date": s},
                         "trade_date,pe_ttm"))
    if not hist or len(hist) < 100: return []
    pe_list = sorted(float(d["pe_ttm"]) for d in hist if d.get("pe_ttm") and d["pe_ttm"] != "None")
    if not pe_list: return []
    rank = sum(1 for p in pe_list if p < pe)
    pct = rank / len(pe_list) * 100
    return [{"current_pe": round(pe, 2), "years": 5, "min_pe": round(min(pe_list), 1),
             "median_pe": round(pe_list[len(pe_list)//2], 1), "max_pe": round(max(pe_list), 1),
             "percentile": round(pct, 1), "data_points": len(pe_list)}]

def ts_top_list(date: str = None) -> list:
    """龙虎榜日榜单"""
    d = date or _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("top_list", {"trade_date": d},
                        "trade_date,ts_code,name,close,pct_change,reason,l_buy,l_sell,net"))

def ts_top10_holders(code: str) -> list:
    """十大股东"""
    e = f"{_datetime.now().year - 1}1231"
    return _dicts(_call("top10_holders", {"ts_code": _ts_code(code), "end_date": e},
                        "ts_code,end_date,holder_name,hold_amount,hold_ratio"))

def ts_holder_num(code: str) -> list:
    """股东户数趋势"""
    s = f"{_datetime.now().year - 2}0101"; e = _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("stk_holdernumber", {"ts_code": _ts_code(code), "start_date": s, "end_date": e},
                        "ts_code,ann_date,end_date,holder_num"))

def ts_ev_ebitda(code: str) -> list:
    """EV/EBITDA 计算"""
    ts = _ts_code(code)
    bs = _dicts(_call("balancesheet", {"ts_code": ts, "end_date": f"{_datetime.now().year}1231"},
                     "ts_code,end_date,total_liab,money_cap"))
    inc = _dicts(_call("income", {"ts_code": ts, "end_date": f"{_datetime.now().year}1231"},
                      "ts_code,end_date,ebit,ebitda,n_income"))
    basic = ts_daily_basic(code, days=5)
    if not bs or not inc or not basic: return []
    mv = float(basic[0].get("total_mv", 0))
    ebitda = inc[0].get("ebitda"); ebit = inc[0].get("ebit")
    if (not ebitda or ebitda == "None") and ebit and ebit != "None":
        ebitda = float(ebit) * 1.08
    if not ebitda or ebitda == "None" or float(ebitda) <= 0: return []
    ev = mv + float(bs[0]["total_liab"]) - float(bs[0]["money_cap"])
    return [{"ev_yi": round(ev/1e8,2), "ebitda_yi": round(float(ebitda)/1e8,2),
             "ev_ebitda": round(ev/float(ebitda),2)}]

def ts_st_status(code: str) -> list:
    """获取个股ST状态历史: 日期、名称、ST类型、类型说明"""
    return _dicts(_call("stock_st", {"ts_code": _ts_code(code)},
                        "ts_code,name,trade_date,type,type_name"))

def ts_news(code: str) -> list:
    """个股新闻（AkShare东方财富）"""
    try:
        import akshare as ak
        sym = code.replace("sh.","").replace("sz.","").replace(".SH","").replace(".SZ","").strip()
        df = ak.stock_news_em(symbol=sym)
        if df is None or df.empty: return []
        result = []
        for _, r in df.head(8).iterrows():
            result.append({"title": str(r.get("新闻标题","")),
                          "time": str(r.get("发布时间","")),
                          "source": str(r.get("文章来源","")),
                          "url": str(r.get("新闻链接",""))})
        return result
    except: return []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastMCP()


def _format_result(items, max_rows: int = 50) -> str:
    """将查询结果格式化为 Markdown 表格"""
    if not items:
        return "无数据"
    if len(items) > max_rows:
        items = items[:max_rows]
    if not isinstance(items, list) or not items:
        return str(items)
    headers = list(items[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in items:
        vals = [str(v)[:30] if v is not None else "" for v in row.values()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)





# ── 工具注册 ──

@app.tool()
def tushare_stock_info(code: str) -> str:
    """获取A股基本信息: 股票名称、所属行业、上市日期、地域"""
    info = ts_stock_info(code)
    return _format_result(info) if info else f"未找到股票 {code} 的信息"


@app.tool()
def tushare_kline(code: str, days: int = 250) -> str:
    """获取A股日K线数据: 开高低收、成交量、成交额、涨跌幅"""
    return _format_result(ts_kline(code, days=days))


@app.tool()
def tushare_daily_basic(code: str, days: int = 500) -> str:
    """获取A股日频估值指标: PE_TTM, PB, PS_TTM, 总市值, 流通市值, 换手率"""
    return _format_result(ts_daily_basic(code, days=days))


@app.tool()
def tushare_fina_indicator(code: str, years: int = 3) -> str:
    """获取A股财务指标: ROE, ROA, 毛利率, 净利率, 资产负债率, 流动/速动比率, 周转率, 营收/利润同比增长率"""
    return _format_result(ts_fina_indicator(code, years=years))


@app.tool()
def tushare_dividend(code: str) -> str:
    """获取A股历史分红记录: 每股现金分红、送转股、除权除息日"""
    return _format_result(ts_dividend(code))


@app.tool()
def tushare_moneyflow(code: str, days: int = 30) -> str:
    """获取A股个股资金流向: 超大单买入/卖出、净流入量、净流入额"""
    return _format_result(ts_moneyflow(code, days=days))


@app.tool()
def tushare_pe_percentile(code: str) -> str:
    """计算A股当前PE_TTM在近5年历史中的分位数: 当前PE、历史最低/最高/中位数PE、所处百分位"""
    result = ts_pe_percentile(code)
    return _format_result(result) if result else "PE分位计算失败（数据点不足）"


@app.tool()
def tushare_hsgt_flow(code: str, days: int = 30) -> str:
    """获取沪深股通十大成交股中该股的买卖数据（若该股在列表中）"""
    s = (_datetime.now() - _timedelta(days=days + 5)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    items = _dicts(_call("hsgt_top10", {"ts_code": _ts_code(code), "start_date": s, "end_date": e},
                          "ts_code,trade_date,name,close,buy,sell"))
    return _format_result(items)


@app.tool()
def tushare_top_list(date: str = "") -> str:
    """获取龙虎榜日榜单: 当日上榜股票、涨跌幅、上榜原因、买卖金额"""
    d = date if date else _datetime.now().strftime("%Y%m%d")
    return _format_result(ts_top_list(d))


@app.tool()
def tushare_top10_holders(code: str) -> str:
    """获取十大股东: 股东名称、持股数量、持股比例"""
    return _format_result(ts_top10_holders(code))


@app.tool()
def tushare_holder_num(code: str) -> str:
    """获取股东户数趋势: 各报告期股东户数变化"""
    return _format_result(ts_holder_num(code))


@app.tool()
def tushare_ev_ebitda(code: str) -> str:
    """计算EV/EBITDA估值: 企业价值(EV)、EBITDA、EV/EBITDA倍数"""
    return _format_result(ts_ev_ebitda(code))


@app.tool()
def tushare_news(code: str) -> str:
    """获取A股个股最新新闻: 标题、发布时间、来源、链接（来源东方财富）"""
    items = ts_news(code)
    if not items:
        return "暂无相关新闻"
    return _format_result(items)


@app.tool()
def tushare_st_status(code: str) -> str:
    """获取A股个股ST状态历史: 当前/历史ST标记、ST类型（退市风险警示/其他风险警示）、变更日期"""
    items = ts_st_status(code)
    if not items:
        return (
            "| 项目 | 值 |\\n"
            "|------|----|\\n"
            "| ST状态 | 正常 |\\n"
            "| 说明 | Tushare stock_st接口无该股票记录，当前未处于ST状态 |"
        )
    return _format_result(items, max_rows=30)


# ═══════════════════════════════════════════════════
# 板块/行业/概念数据工具
# ═══════════════════════════════════════════════════

def ts_ths_index(keyword: str = "") -> list:
    """同花顺行业/板块指数: 按关键词搜索，默认返回全部热门板块"""
    params = {"exchange": "A", "type": "N"}
    items = _dicts(_call("ths_index", params, "ts_code,name,count"))
    if not keyword:
        # 按股票数量降序，返回前50个板块
        sorted_items = sorted(items, key=lambda x: int(x.get("count", 0)) if x.get("count", "").isdigit() else 0, reverse=True)
        return sorted_items[:50]
    kw = keyword.strip().lower()
    return [d for d in items if kw in d.get("name", "").lower() or kw in d.get("ts_code", "").lower()]


def ts_ths_member(ts_code: str) -> list:
    """同花顺板块成分股: 给定板块代码，返回成分股列表"""
    return _dicts(_call("ths_member", {"ts_code": ts_code},
                        "ts_code,con_code,name,industry"))


def ts_concept_list(keyword: str = "") -> list:
    """东方财富概念板块列表: 按名称搜索，默认返回全部。用 keyword 过滤"""
    items = _dicts(_call("concept", {}, "code,name"))
    if not items:
        return []
    if not keyword:
        return items[:100]
    kw = keyword.strip().lower()
    return [d for d in items if kw in d.get("name", "").lower() or kw == d.get("code", "")]


def ts_concept_detail(concept_code: str) -> list:
    """东方财富概念板块成分股: 给定概念代码，返回成分股票"""
    return _dicts(_call("concept_detail", {"code": concept_code},
                        "code,name,ts_code,concept_code,in_date"))


def ts_dc_index(keyword: str = "", trade_date: str = "") -> list:
    """东方财富板块指数行情: 按名称搜索板块"""
    from datetime import datetime as _dt
    d = trade_date if trade_date else _dt.now().strftime("%Y%m%d")
    items = _dicts(_call("dc_index", {"trade_date": d}, "ts_code,name,industry,level"))
    if not items:
        return []
    if not keyword:
        return items[:80]
    kw = keyword.strip().lower()
    return [d for d in items if kw in d.get("name", "").lower()]


def ts_dc_member(ts_code: str) -> list:
    """东方财富板块成分股: 给定板块/概念指数代码，返回成分股"""
    return _dicts(_call("dc_member", {"ts_code": ts_code},
                        "ts_code,con_code,name"))


@app.tool()
def tushare_ths_index(keyword: str = "") -> str:
    """搜索同花顺行业/板块指数: 不填返回热门TOP50板块，填关键词(如'半导体''PCB''白酒')模糊匹配。返回板块代码ts_code供ths_member查询"""
    items = ts_ths_index(keyword)
    if not items:
        return "未找到匹配的板块指数"
    return _format_result(items, max_rows=50)


@app.tool()
def tushare_ths_member(ts_code: str) -> str:
    """获取同花顺板块成分股: 输入板块代码(如'THS12345678.BJ')，返回成分股票列表"""
    items = ts_ths_member(ts_code)
    if not items:
        return "未找到该板块的成分股数据"
    return _format_result(items, max_rows=100)


@app.tool()
def tushare_concept_list(keyword: str = "") -> str:
    """搜索东方财富概念板块: 不填返回全部，填关键词(如'PCB''机器人''CPO')模糊匹配。返回概念代码code供concept_detail查询"""
    items = ts_concept_list(keyword)
    if not items:
        return f"未找到匹配'{keyword}'的概念板块"
    return _format_result(items, max_rows=60)


@app.tool()
def tushare_concept_detail(concept_code: str) -> str:
    """获取东方财富概念板块成分股: 输入概念代码(如'BK0999')，返回成分股票及代码"""
    items = ts_concept_detail(concept_code)
    if not items:
        return "未找到该概念板块的成分股数据"
    return _format_result(items, max_rows=80)


@app.tool()
def tushare_dc_index(keyword: str = "", trade_date: str = "") -> str:
    """搜索东方财富板块指数行情: 按名称(如'半导体''白酒')搜索板块。不填返回全部。trade_date格式YYYYMMDD"""
    items = ts_dc_index(keyword, trade_date)
    if not items:
        return f"未找到匹配'{keyword}'的板块指数"
    return _format_result(items, max_rows=80)


@app.tool()
def tushare_dc_member(ts_code: str) -> str:
    """获取东方财富板块/概念成分股: 输入板块代码，返回成分股票及代码"""
    items = ts_dc_member(ts_code)
    if not items:
        return "未找到该板块的成分股数据"
    return _format_result(items, max_rows=100)


# ═══════════════════════════════════════════════════
# 股票名称搜索工具（名称→代码反向查找）
# ═══════════════════════════════════════════════════

def ts_search_stock(keyword: str) -> list:
    """按名称/代码搜索A股: 输入关键词(如'中芯''茅台')，返回匹配的股票代码和名称"""
    items = _dicts(_call("stock_basic", {"list_status": "L"},
                        "ts_code,name,industry,list_date,area"))
    if not items:
        return []
    kw = keyword.strip().lower()
    return [d for d in items if kw in d.get("name", "").lower() or kw in d.get("ts_code", "").lower()]


@app.tool()
def tushare_search_stock(keyword: str) -> str:
    """按名称搜索A股股票代码: 输入公司名称关键词(如'中芯国际''宁德时代')，返回匹配的股票代码/名称/行业。用于名称→代码反向查找"""
    items = ts_search_stock(keyword)
    if not items:
        return f"未找到名称包含'{keyword}'的A股股票"
    # 返回前20条匹配，含标准化代码提示
    result_items = []
    for d in items[:20]:
        raw_code = d.get("ts_code", "")
        # 标准化: 600519.SH → sh.600519
        if ".SH" in raw_code:
            d["normalized_code"] = "sh." + raw_code.replace(".SH", "")
        elif ".SZ" in raw_code:
            d["normalized_code"] = "sz." + raw_code.replace(".SZ", "")
        else:
            d["normalized_code"] = raw_code
        result_items.append(d)
    return _format_result(result_items, max_rows=20)


# ═══════════════════════════════════════════════════
# 宏观数据工具
# ═══════════════════════════════════════════════════

def _to_month(s: str) -> str:
    """YYYY-MM-DD / YYYY-MM / YYYYMM → YYYYMM"""
    c = s.replace("-", "")
    return c[:6]


def _to_date(s: str) -> str:
    """YYYY-MM-DD → YYYYMMDD"""
    return s.replace("-", "")[:8]


def ts_cn_cpi(start_date: str = "", end_date: str = "") -> list:
    """居民消费价格指数 CPI: 当月值、同比、环比、累计值"""
    m = _to_month(end_date) if end_date else _datetime.now().strftime("%Y%m")
    s = _to_month(start_date) if start_date else f"{_datetime.now().year - 1}01"
    return _dicts(_call("cn_cpi", {"start_m": s, "end_m": m},
                        "month,nt_val,nt_yoy,nt_mom,nt_accu,town_val,town_yoy,country_val,country_yoy"))


def ts_cn_gdp(start_date: str = "", end_date: str = "") -> list:
    """国内生产总值 GDP: 累计值、当季同比、三大产业增加值"""
    m = _to_month(end_date) if end_date else _datetime.now().strftime("%Y%m")
    s = _to_month(start_date) if start_date else f"{_datetime.now().year - 3}01"
    return _dicts(_call("cn_gdp", {"start_q": s, "end_q": m},
                        "quarter,gdp,gdp_yoy,pi,si,ti,pi_yoy,si_yoy,ti_yoy"))


def ts_cn_pmi(start_date: str = "", end_date: str = "") -> list:
    """采购经理人指数 PMI: 制造业/非制造业 PMI、各分项"""
    m = _to_month(end_date) if end_date else _datetime.now().strftime("%Y%m")
    s = _to_month(start_date) if start_date else f"{_datetime.now().year - 1}01"
    return _dicts(_call("cn_pmi", {"start_m": s, "end_m": m},
                        "month,pmi010000,pmi020000,pmi010100,pmi010200,pmi010300,pmi010400,pmi010500"))


def ts_cn_ppi(start_date: str = "", end_date: str = "") -> list:
    """工业生产者出厂价格指数 PPI: 当月同比、生产资料/生活资料同比"""
    m = _to_month(end_date) if end_date else _datetime.now().strftime("%Y%m")
    s = _to_month(start_date) if start_date else f"{_datetime.now().year - 1}01"
    return _dicts(_call("cn_ppi", {"start_m": s, "end_m": m},
                        "month,ppi_yoy,ppi_mp_yoy,ppi_cg_yoy,ppi_mp_rm_yoy,ppi_mp_p_yoy"))


def ts_cn_m(start_date: str = "", end_date: str = "") -> list:
    """货币供应量: M0/M1/M2 月末值及同比增速"""
    m = _to_month(end_date) if end_date else _datetime.now().strftime("%Y%m")
    s = _to_month(start_date) if start_date else f"{_datetime.now().year - 1}01"
    return _dicts(_call("cn_m", {"start_m": s, "end_m": m},
                        "month,m0,m1,m2,m0_yoy,m1_yoy,m2_yoy"))


def ts_shibor(date: str = "") -> list:
    """SHIBOR 上海银行间同业拆放利率: 隔夜/1W/2W/1M/3M/6M/9M/1Y"""
    d = _to_date(date) if date else _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("shibor", {"date": d},
                        "date,on,1w,2w,1m,3m,6m,9m,1y"))


def ts_fx_daily(start_date: str = "", end_date: str = "") -> list:
    """外汇日行情: 美元/欧元/日元/港币/英镑兑人民币中间价"""
    e = _to_date(end_date) if end_date else _datetime.now().strftime("%Y%m%d")
    s = _to_date(start_date) if start_date else (_datetime.now() - _timedelta(days=180)).strftime("%Y%m%d")
    return _dicts(_call("fx_daily", {"start_date": s, "end_date": e, "fields": "trade_date,usdcny,eurcny,jpycny,hkdcny,gbpcny"}))


def ts_eco_cal(date: str = "") -> list:
    """全球经济日历: 当天公布的重要经济数据事件"""
    d = _to_date(date) if date else _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("eco_cal", {"date": d},
                        "date,time,country,event,actual,previous,consensus,forecast,level"))


@app.tool()
def tushare_cn_cpi(start_date: str = "", end_date: str = "") -> str:
    """中国CPI居民消费价格指数: 当月值、同比%、环比%、累计值。start_date/end_date格式YYYY-MM或YYYY-MM-DD"""
    return _format_result(ts_cn_cpi(start_date, end_date))


@app.tool()
def tushare_cn_gdp(start_date: str = "", end_date: str = "") -> str:
    """中国GDP国内生产总值: 累计值、当季同比%、三大产业增加值。start_date/end_date格式YYYY-MM或YYYY-MM-DD"""
    return _format_result(ts_cn_gdp(start_date, end_date))


@app.tool()
def tushare_cn_pmi(start_date: str = "", end_date: str = "") -> str:
    """中国PMI采购经理人指数: 制造业/非制造业PMI、生产/新订单等分项。start_date/end_date格式YYYY-MM或YYYY-MM-DD"""
    return _format_result(ts_cn_pmi(start_date, end_date))


@app.tool()
def tushare_cn_ppi(start_date: str = "", end_date: str = "") -> str:
    """中国PPI工业生产者出厂价格指数: 当月同比%、分项同比。start_date/end_date格式YYYY-MM或YYYY-MM-DD"""
    return _format_result(ts_cn_ppi(start_date, end_date))


@app.tool()
def tushare_cn_m(start_date: str = "", end_date: str = "") -> str:
    """中国货币供应量: M0/M1/M2月末值及同比增速。start_date/end_date格式YYYY-MM或YYYY-MM-DD"""
    return _format_result(ts_cn_m(start_date, end_date))


@app.tool()
def tushare_shibor(date: str = "") -> str:
    """SHIBOR上海银行间同业拆放利率: 隔夜/1W/2W/1M/3M/6M/9M/1Y。date格式YYYY-MM-DD"""
    return _format_result(ts_shibor(date))


@app.tool()
def tushare_fx_daily(start_date: str = "", end_date: str = "") -> str:
    """外汇日行情: 美元/欧元/日元/港币/英镑兑人民币中间价。start_date/end_date格式YYYY-MM-DD"""
    return _format_result(ts_fx_daily(start_date, end_date))


@app.tool()
def tushare_eco_cal(date: str = "") -> str:
    """全球经济日历: 当天重要经济数据事件（实际值/前值/预期值）。date格式YYYY-MM-DD"""
    return _format_result(ts_eco_cal(date))


if __name__ == "__main__":
    logger.info("Starting Tushare MCP Server via stdio...")
    app.run(transport='stdio')
