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
import json, time as _time, threading as _threading, requests as _requests
from datetime import datetime as _datetime, timedelta as _timedelta
from typing import Dict, List, Optional, Any

_TUSHARE_TOKEN = "fd4ff6e84626d2e63616ec08769f99110d626a91856036c30cb34818"
_TUSHARE_URL = "https://api.tushare.pro"
_last_call = 0.0
_rate_lock = _threading.Lock()

def _rate_limit():
    """线程安全的速率限制，不阻塞事件循环"""
    global _last_call
    with _rate_lock:
        elapsed = _time.time() - _last_call
        if elapsed < 0.35:
            _time.sleep(0.35 - elapsed)
        _last_call = _time.time()

def _call(api_name: str, params: dict = None, fields: str = "", max_retries: int = 3) -> Optional[dict]:
    """调用 Tushare API，带退避重试（最多3次: 首次等待1.5s, 然后3s）。网络/连接异常和临时API错误自动重试"""
    log = logging.getLogger("tushare_mcp")
    last_error = ""
    for attempt in range(max_retries):
        _rate_limit()
        try:
            resp = _requests.post(_TUSHARE_URL, json={
                "api_name": api_name, "token": _TUSHARE_TOKEN,
                "params": params or {}, "fields": fields,
            }, timeout=15)
            data = resp.json()
            code_val = data.get("code")
            if code_val == 0:
                r = data.get("data", {})
                if r.get("items"):
                    if attempt > 0:
                        log.info(f"Tushare API {api_name} 重试成功 (第{attempt+1}次)")
                    return r
                # 空数据：可能是请求时间范围无数据，直接返回 None 不重试
                log.warning(f"Tushare API {api_name} 返回空数据 (params={params})")
                return None
            # API 返回错误码：可能临时故障，可重试
            last_error = f"code={code_val}, msg={data.get('msg', '')}"
            if code_val in (-1, -2, 2001, 2002):  # 临时错误码可重试
                log.warning(f"Tushare API {api_name} 临时错误 {last_error}，{1.5*(attempt+1):.1f}s后重试({attempt+1}/{max_retries})")
                _time.sleep(1.5 * (attempt + 1))
                continue
            log.warning(f"Tushare API {api_name} 返回错误: {last_error}")
            return None
        except (_requests.Timeout, _requests.ConnectionError) as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                wait = 1.5 * (attempt + 1)
                log.warning(f"Tushare API {api_name} 网络异常: {e}，{wait:.1f}s后重试({attempt+1}/{max_retries})")
                _time.sleep(wait)
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                log.warning(f"Tushare API {api_name} 调用异常: {e}，1s后重试({attempt+1}/{max_retries})")
                _time.sleep(1)
                continue
    log.warning(f"Tushare API {api_name} 最终失败: {last_error}（已重试{max_retries}次）")
    return None

def _dicts(result) -> list:
    if not result: return []
    f = result.get("fields", []); items = result.get("items", [])
    return [dict(zip(f, r)) for r in items]

def _is_bse(code: str) -> bool:
    """检测是否为北交所(BSE)代码"""
    c = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()
    return (c.startswith(("430", "431", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                          "870", "871", "872", "873", "920")) or
            (len(c) >= 3 and c[:3] in ("830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
                                        "870", "871", "872", "873")))

def _ts_code(code: str) -> str:
    c = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()
    # 北交所代码使用 .BJ 后缀
    if _is_bse(c):
        return f"{c}.BJ"
    # 6xxxxx=沪市主板, 688xxx=科创板, 5xxxxx=沪市基金/ETF
    # 0xxxxx=深市主板, 3xxxxx=创业板, 1xxxxx=深市基金/ETF, 4xxxxx=深市
    return f"{c}.SH" if (c.startswith(("6", "688", "5"))) else f"{c}.SZ"

def _ts_fund_code(code: str) -> str:
    """基金代码转 tushare ts_code 格式。ETF/LOF 用 .SH/.SZ，开放式基金用 .OF"""
    c = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").replace(".OF", "").replace(".BJ", "").strip()
    if c.startswith(("51", "58")):
        return f"{c}.SH"
    elif c.startswith(("15", "16", "18")):
        return f"{c}.SZ"
    else:
        return f"{c}.OF"

def _is_etf(code: str) -> bool:
    """检测是否为 ETF/基金代码（上交所51/58，深交所15/16/18）"""
    c = code.replace("sh.", "").replace("sz.", "").replace(".SH", "").replace(".SZ", "").replace(".BJ", "").strip()
    return c.startswith(("51", "58", "15", "16", "18"))

def ts_stock_info(code: str) -> list:
    # ETF 无 stock_basic 数据，尝试 fund_basic
    if _is_etf(code):
        return _dicts(_call("fund_basic", {"ts_code": _ts_code(code)}, "ts_code,name,fund_type,found_date,management"))
    return _dicts(_call("stock_basic", {"ts_code": _ts_code(code)}, "ts_code,name,industry,list_date,area"))

def ts_kline(code: str, days: int = 250) -> list:
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    ts = _ts_code(code)
    if _is_etf(code):
        # ETF/基金使用 fund_daily（Tushare stock daily API 不支持 ETF 代码）
        return _dicts(_call("fund_daily", {"ts_code": ts, "start_date": s, "end_date": e},
                            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"))
    return _dicts(_call("daily", {"ts_code": ts, "start_date": s, "end_date": e},
                        "trade_date,open,high,low,close,vol,amount,pct_chg"))

def ts_daily_basic(code: str, days: int = 500) -> list:
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    ts = _ts_code(code)
    if _is_etf(code):
        # ETF 无 PE/PB，用 fund_daily 返回价格+成交量+换手率（若可用）
        return _dicts(_call("fund_daily", {"ts_code": ts, "start_date": s, "end_date": e},
                            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"))
    return _dicts(_call("daily_basic", {"ts_code": ts, "start_date": s, "end_date": e},
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
    rank = sum(1 for p in pe_list if p <= pe)
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
        sym = code.replace("sh.","").replace("sz.","").replace(".SH","").replace(".SZ","").replace(".BJ","").strip()
        df = ak.stock_news_em(symbol=sym)
        if df is None or df.empty: return []
        result = []
        for _, r in df.head(8).iterrows():
            result.append({"title": str(r.get("新闻标题","")),
                          "time": str(r.get("发布时间","")),
                          "source": str(r.get("文章来源","")),
                          "url": str(r.get("新闻链接",""))})
        return result
    except Exception as e:
        logging.getLogger("tushare_mcp").warning(f"ts_news({code}) 获取失败: {e}")
        return []


# ── 基金专用数据函数 ──

def ts_fund_basic(code: str) -> list:
    """基金基本信息: 名称、类型、管理人、托管人、业绩基准、费率、申购赎回日期"""
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_basic", {"ts_code": ts},
                        "ts_code,name,management,custodian,fund_type,found_date,"
                        "due_date,list_date,issue_date,issue_amount,m_fee,c_fee,"
                        "duration_year,p_value,min_amount,exp_return,benchmark,"
                        "status,invest_type,type,trustee,purc_startdate,redm_startdate,market"))


def ts_fund_nav(code: str, days: int = 500) -> list:
    """基金净值历史: 单位净值、累计净值、累计分红、复权净值"""
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_nav", {"ts_code": ts, "start_date": s, "end_date": e},
                        "ts_code,nav_date,unit_nav,accum_nav,accum_div,adj_nav"))


def ts_fund_portfolio(code: str) -> list:
    """基金持仓明细: 最新一期持仓股票、市值、占净值比例"""
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_portfolio", {"ts_code": ts},
                        "ts_code,ann_date,end_date,symbol,mkv,amount,stk_mkv_ratio,stk_float_ratio"))


def ts_fund_manager(code: str) -> list:
    """基金经理信息: 姓名、性别、学历、任职期限、个人简历"""
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_manager", {"ts_code": ts},
                        "ts_code,ann_date,name,gender,birth_year,edu,nationality,begin_date,end_date,resume"))


def ts_fund_div(code: str) -> list:
    """基金分红记录: 公告日、除权日、分红金额"""
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_div", {"ts_code": ts},
                        "ts_code,ann_date,imp_anndate,base_date,div_proc,record_date,"
                        "ex_date,pay_date,div_cash,base_unit,ear_distr,ear_amount"))


def ts_fund_share(code: str) -> list:
    """基金份额变动: 近一年每日份额，判断规模变化趋势"""
    ts = _ts_fund_code(code)
    s = (_datetime.now() - _timedelta(days=400)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    return _dicts(_call("fund_share", {"ts_code": ts, "start_date": s, "end_date": e},
                        "ts_code,trade_date,fd_share"))


def ts_fund_company(code: str) -> list:
    """基金管理人信息: 获取基金对应的管理公司详情"""
    basic = ts_fund_basic(code)
    if not basic:
        return []
    mgmt = basic[0].get("management", "")
    if not mgmt:
        return []
    companies = _dicts(_call("fund_company", {},
                             "name,shortname,province,city,setup_date,employees,reg_capital,chairman,manager,website"))
    if companies:
        return [c for c in companies if mgmt in c.get("name", "") or (c.get("shortname") and mgmt in c.get("shortname", ""))]
    return []

def ts_fund_daily(code: str, days: int = 500) -> list:
    """基金日线行情: 开盘价/收盘价/最高/最低/成交量（仅适用于上市型基金 ETF/LOF）"""
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_daily", {"ts_code": ts, "start_date": s, "end_date": e},
                        "ts_code,trade_date,open,high,low,close,vol,amount"))

def ts_fund_adj(code: str, days: int = 500) -> list:
    """基金复权因子: 前/后复权因子（仅适用于上市型基金 ETF/LOF）"""
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    ts = _ts_fund_code(code)
    return _dicts(_call("fund_adj", {"ts_code": ts, "start_date": s, "end_date": e},
                        "ts_code,trade_date,adj_factor"))


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
def tushare_major_news(keyword: str = "", days: int = 90) -> str:
    """获取主流财经新闻: 搜索近N天的重大财经/证券市场新闻。keyword为空则返回全部新闻，可传入基金公司名称、基金经理姓名等进行过滤。返回标题(title)、发布时间(pub_time)、来源(src)、链接(url)"""
    s = (_datetime.now() - _timedelta(days=days + 5)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    items = _dicts(_call("major_news", {"src": "", "start_date": s, "end_date": e},
                        "title,pub_time,src,url"))
    if not items:
        return "暂无相关新闻"
    if keyword:
        kw = keyword.strip()
        items = [i for i in items if kw in str(i.get("title", "")) or kw in str(i.get("src", ""))]
    if not items:
        return f"未找到与'{keyword}'相关的新闻"
    return _format_result(items, max_rows=30)


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


@app.tool()
def tushare_latest_trading_date() -> str:
    """获取最近的交易日期。返回最新交易日、最近5个交易日列表，用于确定数据查询的时间范围"""
    today = _datetime.now().strftime("%Y%m%d")
    week_ago = (_datetime.now() - _timedelta(days=10)).strftime("%Y%m%d")
    r = _call("trade_cal", {"exchange": "SSE", "start_date": week_ago, "end_date": today, "is_open": "1"},
              "cal_date")
    if not r:
        # 兜底：返回今天
        fallback = _datetime.now().strftime("%Y-%m-%d")
        return f"无法获取交易日历，使用当前日期: {fallback}"
    dates = [d["cal_date"] for d in _dicts(r)]
    latest = max(dates) if dates else today
    latest_fmt = f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}" if len(latest) == 8 else latest
    recent = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in sorted(dates, reverse=True)[:5]]
    lines = [f"| 字段 | 值 |", "|------|----|",
             f"| 最新交易日 | {latest_fmt} |",
             f"| 近5个交易日 | {', '.join(recent)} |"]
    return "\n".join(lines)


@app.tool()
def tushare_adj_factor(code: str, days: int = 500) -> str:
    """获取A股复权因子: 前/后复权因子（qfq/hfq），用于精确计算复权价格"""
    ts = _ts_code(code)
    s = (_datetime.now() - _timedelta(days=days + 30)).strftime("%Y%m%d")
    e = _datetime.now().strftime("%Y%m%d")
    if _is_etf(code):
        # ETF/基金使用 fund_adj 接口
        items = _dicts(_call("fund_adj", {"ts_code": ts, "start_date": s, "end_date": e},
                            "ts_code,trade_date,adj_factor"))
        if not items:
            return "ETF/基金暂无复权因子数据"
        return _format_result(items, max_rows=60)
    items = _dicts(_call("adj_factor", {"ts_code": ts, "start_date": s, "end_date": e},
                        "ts_code,trade_date,adj_factor"))
    if not items:
        return "未找到该股票的复权因子数据"
    return _format_result(items, max_rows=60)


@app.tool()
def tushare_income(code: str) -> str:
    """获取A股利润表: 营业收入、营业成本、三费、营业利润、净利润、EBIT、EBITDA等（最新4期）"""
    ts = _ts_code(code)
    end_dates = []
    now = _datetime.now()
    # 从最新已完成的季度开始回溯
    current_q = (now.month - 1) // 3 + 1
    q = current_q - 1
    y = now.year
    if q <= 0:
        q = 4
        y -= 1
    for i in range(4):
        month = q * 3
        last_day = 30 if month in (6, 9) else 31
        end_dates.append(f"{y}{month:02d}{last_day}")
        q -= 1
        if q <= 0:
            q = 4
            y -= 1
    all_items = []
    for ed in end_dates[:4]:
        items = _dicts(_call("income", {"ts_code": ts, "end_date": ed},
                            "ts_code,end_date,revenue,oper_cost,sell_exp,admin_exp,fin_exp,"
                            "oper_profit,total_profit,n_income,ebit,ebitda"))
        if items:
            for d in items:
                d["period"] = ed
            all_items.extend(items)
    if not all_items:
        return "未找到该股票的利润表数据"
    return _format_result(all_items, max_rows=20)


@app.tool()
def tushare_balancesheet(code: str) -> str:
    """获取A股资产负债表: 总资产、总负债、股东权益、货币资金、应收/应付、存货等（最新4期）"""
    ts = _ts_code(code)
    end_dates = []
    now = _datetime.now()
    # 从最新已完成的季度开始回溯
    current_q = (now.month - 1) // 3 + 1
    q = current_q - 1
    y = now.year
    if q <= 0:
        q = 4
        y -= 1
    for i in range(4):
        month = q * 3
        last_day = 30 if month in (6, 9) else 31
        end_dates.append(f"{y}{month:02d}{last_day}")
        q -= 1
        if q <= 0:
            q = 4
            y -= 1
    all_items = []
    for ed in end_dates[:4]:
        items = _dicts(_call("balancesheet", {"ts_code": ts, "end_date": ed},
                            "ts_code,end_date,total_assets,total_liab,total_hldr_eqy,"
                            "money_cap,acct_reci,notes_reci,inventories,acct_payable,"
                            "notes_payable,goodwill,lt_borr,st_borr,bp_assets"))
        if items:
            for d in items:
                d["period"] = ed
            all_items.extend(items)
    if not all_items:
        return "未找到该股票的资产负债表数据"
    return _format_result(all_items, max_rows=20)


@app.tool()
def tushare_cashflow(code: str) -> str:
    """获取A股现金流量表: 经营活动/投资活动/筹资活动现金流净额、自由现金流等（最新4期）"""
    ts = _ts_code(code)
    end_dates = []
    now = _datetime.now()
    # 从最新已完成的季度开始回溯
    current_q = (now.month - 1) // 3 + 1
    q = current_q - 1
    y = now.year
    if q <= 0:
        q = 4
        y -= 1
    for i in range(4):
        month = q * 3
        last_day = 30 if month in (6, 9) else 31
        end_dates.append(f"{y}{month:02d}{last_day}")
        q -= 1
        if q <= 0:
            q = 4
            y -= 1
    all_items = []
    for ed in end_dates[:4]:
        items = _dicts(_call("cashflow", {"ts_code": ts, "end_date": ed},
                            "ts_code,end_date,n_cashflow_act,n_cashflow_inv_act,"
                            "n_cashflow_fin_act,free_cashflow,c_paid_for_assets"))
        if items:
            for d in items:
                d["period"] = ed
            all_items.extend(items)
    if not all_items:
        return "未找到该股票的现金流量表数据"
    return _format_result(all_items, max_rows=20)


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


# ═══════════════════════════════════════════════════
# 基金专用数据工具
# ═══════════════════════════════════════════════════

@app.tool()
def tushare_fund_basic(code: str) -> str:
    """获取公募基金/ETF基本信息: 基金名称、类型(ETF/LOF/封闭式等)、管理人、托管人、业绩比较基准、管理费率、托管费率、成立日期、申购开始日、赎回开始日、投资类型(主动/被动/指数增强)"""
    items = ts_fund_basic(code)
    return _format_result(items) if items else f"未找到基金 {code} 的信息"


@app.tool()
def tushare_fund_nav(code: str, days: int = 500) -> str:
    """获取公募基金净值历史数据: 单位净值(unit_nav)、累计净值(accum_nav)、累计分红(accum_div)、复权净值(adj_nav)。days默认500个交易日用于计算收益率/波动率/最大回撤等指标"""
    items = ts_fund_nav(code, days=days)
    return _format_result(items, max_rows=500) if items else f"未找到基金 {code} 的净值数据"


@app.tool()
def tushare_fund_portfolio(code: str) -> str:
    """获取公募基金最新持仓明细: 前十大重仓股代码(symbol)、市值(mkv)、持仓数量(amount)、占净值比例(stk_mkv_ratio)、占流通股比例(stk_float_ratio)。注意:持仓为最近一期季报/年报披露，存在时滞"""
    items = ts_fund_portfolio(code)
    return _format_result(items, max_rows=100) if items else f"未找到基金 {code} 的持仓数据"


@app.tool()
def tushare_fund_manager(code: str) -> str:
    """获取公募基金经理信息: 姓名(name)、性别(gender)、学历(edu)、国籍(nationality)、任职起始日(begin_date)、离任日期(end_date)、个人简历(resume)。可用于评估经理稳定性和经验"""
    items = ts_fund_manager(code)
    return _format_result(items, max_rows=10) if items else f"未找到基金 {code} 的经理信息"


@app.tool()
def tushare_fund_div(code: str) -> str:
    """获取公募基金历史分红记录: 公告日(ann_date)、除权日(ex_date)、分红金额(div_cash)、基准份额(base_unit)、再投资金额(ear_amount)。用于评估分红稳定性和现金回报能力"""
    items = ts_fund_div(code)
    return _format_result(items, max_rows=30) if items else f"未找到基金 {code} 的分红记录"


@app.tool()
def tushare_fund_share(code: str) -> str:
    """获取公募基金份额变动趋势: 每日份额数据(fd_share)，展示近一年规模变化趋势。份额持续下降可能反映投资者赎回压力"""
    items = ts_fund_share(code)
    return _format_result(items, max_rows=500) if items else f"未找到基金 {code} 的份额数据"


@app.tool()
def tushare_fund_company(code: str) -> str:
    """获取基金管理人(基金公司)信息: 公司全称(name)、简称(shortname)、所在省市(province/city)、成立日期(setup_date)、员工人数(employees)、注册资本(reg_capital)、董事长(chairman)、总经理(manager)、官网(website)"""
    items = ts_fund_company(code)
    return _format_result(items) if items else f"未找到基金 {code} 对应的管理人信息"

@app.tool()
def tushare_fund_daily(code: str, days: int = 500) -> str:
    """获取基金日线行情数据: 开盘价(open)、收盘价(close)、最高价(high)、最低价(low)、成交量(vol)、成交额(amount)。仅适用于上市交易型基金(ETF/LOF)，场外开放式基金无此数据。days默认500个交易日"""
    items = ts_fund_daily(code, days=days)
    return _format_result(items, max_rows=500) if items else f"基金 {code} 无日线数据（可能为场外基金，不上市交易）"

@app.tool()
def tushare_fund_adj(code: str, days: int = 500) -> str:
    """获取基金复权因子: 前复权/后复权因子(adj_factor)。仅适用于上市交易型基金(ETF/LOF)，用于精确计算复权价格和收益率"""
    items = ts_fund_adj(code, days=days)
    return _format_result(items, max_rows=500) if items else f"基金 {code} 无复权因子数据（可能为场外基金）"


@app.tool()
def tushare_fund_search(keyword: str) -> str:
    """按关键词搜索公募基金(含场内ETF/LOF和场外基金): 输入关键词(如'沪深300''科创50''新能源')，返回匹配的基金代码(ts_code)、名称(name)、管理公司(management)、基金类型(fund_type)、业绩基准(benchmark)、投资类型(invest_type)"""
    items_e = _dicts(_call("fund_basic", {"market": "E"},
                           "ts_code,name,management,fund_type,found_date,benchmark,invest_type,type"))
    items_o = _dicts(_call("fund_basic", {"market": "O"},
                           "ts_code,name,management,fund_type,found_date,benchmark,invest_type,type"))
    all_items = (items_e or []) + (items_o or [])
    if not all_items:
        return "未找到任何基金数据，请检查Tushare API权限"
    kw = keyword.strip().lower()
    matched = [d for d in all_items if kw in d.get("name", "").lower() or kw in d.get("ts_code", "").lower()]
    if not matched:
        return f"未找到名称包含'{keyword}'的基金，请尝试其他关键词(如'沪深300''科创50 ETF'等)"
    # 优先返回ETF/LOF等场内基金
    matched_sorted = sorted(matched, key=lambda x: (0 if x.get("type", "") in ("E", "LOF") else 1))
    return _format_result(matched_sorted[:30], max_rows=30)


if __name__ == "__main__":
    logger.info("Starting Tushare MCP Server via stdio...")
    app.run(transport='stdio')
