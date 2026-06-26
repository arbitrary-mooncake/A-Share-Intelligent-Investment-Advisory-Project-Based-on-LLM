"""
国际宏观与商品数据工具 — AKShare 数据源
提供美国CPI、PMI、非农就业、COMEX库存、黄金基准价等非A股数据。
"""
import logging
from typing import Optional
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def _to_markdown_table(data, title: str) -> str:
    """将 pandas DataFrame 或 list-of-dict 转为 Markdown 表格"""
    import pandas as pd
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, list) and data:
        df = pd.DataFrame(data)
    else:
        return f"## {title}\n\n无数据"

    if df.empty:
        return f"## {title}\n\n无数据"

    lines = [f"## {title}\n"]
    # Headers
    headers = list(df.columns)
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    # Rows (max 50)
    for _, row in df.head(50).iterrows():
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
