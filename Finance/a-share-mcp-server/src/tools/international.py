"""
国际宏观与商品数据工具 — AKShare 数据源
提供美国CPI、PMI、非农就业、COMEX库存、黄金基准价等非A股数据。
"""
import logging
from typing import Optional
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _to_markdown_table(data, title: str) -> str:
    """将 pandas DataFrame、dict 或 list-of-dict 转为 Markdown 表格"""
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict) and data:
        # 单行 dict：转为两列表格（指标 | 值）
        lines = [f"## {title}\n"]
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        for k, v in data.items():
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)
    elif isinstance(data, list) and data:
        df = pd.DataFrame(data)
    else:
        return f"## {title}\n\n无数据"

    if df.empty:
        return f"## {title}\n\n无数据"

    lines = [f"## {title}\n"]
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    # 优先展示最新数据：取最后50行（用户关心当前值，不关心10年前的历史）
    rows_to_show = min(50, len(df))
    for _, row in df.tail(rows_to_show).iterrows():
        cells = [str(v) if v is not None else "" for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _safe_ak_call(func_name: str, **kwargs):
    """安全调用 AKShare 函数，返回 DataFrame 或 None"""
    import akshare as ak
    try:
        func = getattr(ak, func_name)
        result = func(**kwargs)
        return result
    except Exception as e:
        logger.error(f"AKShare {func_name} 调用失败: {e}")
        return None


def register_international_tools(app: FastMCP):
    """向 MCP 应用注册国际宏观与商品数据工具"""

    @app.tool()
    def get_us_cpi() -> str:
        """
        获取美国消费者价格指数(CPI)年率数据。
        CPI是衡量通胀的核心指标，对黄金价格有重大影响：
        - CPI高于预期 → 通胀压力 → 美联储加息预期 → 利空黄金
        - CPI低于预期 → 通缩风险 → 美联储降息预期 → 利好黄金

        返回: Markdown 表格
        """
        result = _safe_ak_call("macro_usa_cpi_yoy")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 美国CPI数据\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "美国CPI年率")

    @app.tool()
    def get_us_pmi() -> str:
        """
        获取美国ISM制造业PMI数据。
        PMI是经济景气度的先行指标，50以上扩张、50以下收缩。

        返回: Markdown 表格
        """
        result = _safe_ak_call("macro_usa_ism_pmi")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 美国ISM PMI\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "美国ISM制造业PMI")

    @app.tool()
    def get_us_non_farm() -> str:
        """
        获取美国非农就业人数变化数据。
        非农是美联储决策的关键参考，数据强劲支持加息/维持高利率，利空黄金。

        返回: Markdown 表格
        """
        result = _safe_ak_call("macro_usa_non_farm")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 美国非农就业\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "美国非农就业变化")

    @app.tool()
    def get_us_unemployment() -> str:
        """
        获取美国失业率数据。
        失业率是劳动力市场健康度的核心指标，影响美联储政策路径。

        返回: Markdown 表格
        """
        result = _safe_ak_call("macro_usa_unemployment_rate")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 美国失业率\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "美国失业率")

    @app.tool()
    def get_us_gdp() -> str:
        """
        获取美国GDP月度数据。
        GDP反映经济总量增长，影响利率预期和美元走势。

        返回: Markdown 表格
        """
        result = _safe_ak_call("macro_usa_gdp_monthly")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 美国GDP\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "美国GDP月度")

    @app.tool()
    def get_us_retail_sales() -> str:
        """
        获取美国零售销售数据。
        零售销售反映消费支出强度，是经济健康度的重要指标。

        返回: Markdown 表格
        """
        result = _safe_ak_call("macro_usa_retail_sales")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 美国零售销售\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "美国零售销售")

    @app.tool()
    def get_comex_inventory() -> str:
        """
        获取COMEX黄金库存数据（金库注册仓单量）。
        COMEX是全球最大的黄金期货交易所，库存变化反映交割需求。

        返回: Markdown 表格
        """
        result = _safe_ak_call("futures_comex_inventory")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## COMEX黄金库存\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "COMEX黄金库存")

    @app.tool()
    def get_spot_gold_sge() -> str:
        """
        获取上海黄金交易所(SGE)黄金现货基准价。
        这是中国黄金现货市场的核心定价参考。

        返回: Markdown 表格
        """
        result = _safe_ak_call("spot_golden_benchmark_sge")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 上海金基准价\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "上海黄金交易所现货基准价")

    @app.tool()
    def get_sge_spot_prices() -> str:
        """
        获取上海黄金交易所(SGE)所有品种实时报价。
        包含 Au99.99（黄金）、Ag(T+D)（白银）等现货合约的实时价格、
        涨跌幅、成交量。数据源 AKShare，无速率限制。

        返回: Markdown 表格，含品种/最新价/涨跌幅
        """
        result = _safe_ak_call("spot_quotations_sge")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## SGE实时报价\n\n数据获取失败或暂无数据"

        import pandas as pd
        if isinstance(result, pd.DataFrame) and not result.empty:
            # 只返回最近时刻的数据，去重品种
            latest = result.drop_duplicates(subset=result.columns[0] if len(result.columns) > 0 else None, keep='last')
            return _to_markdown_table(latest.head(30), "上海黄金交易所实时报价")

        return _to_markdown_table(result, "上海黄金交易所实时报价")

    @app.tool()
    def get_global_futures_spot() -> str:
        """
        获取全球期货交易所实时行情（CME/COMEX/LME等）。
        包含 COMEX黄金、白银、铜、WTI原油、布伦特原油、天然气等
        国际大宗商品期货的最新价格、涨跌幅、成交量。
        数据源 AKShare（东方财富），无 Yahoo Finance 速率限制。

        返回: Markdown 表格，含品种/合约/最新价/涨跌幅/成交量
        """
        result = _safe_ak_call("futures_global_spot_em")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 全球期货行情\n\n数据获取失败或暂无数据"

        import pandas as pd
        if isinstance(result, pd.DataFrame) and not result.empty:
            # 筛选主要品种：黄金、白银、铜、原油、天然气
            gold_keywords = ['黄金', '金', 'Gold', 'GOLD', 'GC', 'QO']
            oil_keywords = ['原油', 'WTI', 'Brent', '布伦特', 'CL', 'BZ']
            silver_keywords = ['白银', '银', 'Silver', 'SI']
            copper_keywords = ['铜', 'Copper', 'HG']
            all_kw = gold_keywords + oil_keywords + silver_keywords + copper_keywords

            cols = list(result.columns)
            text_col = cols[0] if cols else ''
            # 在各列中搜索关键词
            mask = pd.Series([False] * len(result), index=result.index)
            for col in result.columns:
                mask |= result[col].astype(str).str.contains('|'.join(all_kw), case=False, na=False)
            filtered = result[mask].head(30)
            if not filtered.empty:
                return _to_markdown_table(filtered, "全球期货实时行情（黄金/白银/铜/原油）")
            # 过滤无结果时返回前20条
            return _to_markdown_table(result.head(20), "全球期货实时行情")

        return _to_markdown_table(result, "全球期货实时行情")

    @app.tool()
    def get_fx_rates() -> str:
        """
        获取主要外汇汇率（USD/CNY、EUR/CNY、JPY/CNY等）。
        包含买入报价和卖出报价，反映人民币对主要货币的汇率水平。
        美元走强通常利空黄金，人民币汇率影响国内金价。

        返回: Markdown 表格，含货币对/买入价/卖出价
        """
        result = _safe_ak_call("fx_spot_quote")
        if result is None or (hasattr(result, 'empty') and result.empty):
            return "## 外汇汇率\n\n数据获取失败或暂无数据"
        return _to_markdown_table(result, "主要外汇汇率")
