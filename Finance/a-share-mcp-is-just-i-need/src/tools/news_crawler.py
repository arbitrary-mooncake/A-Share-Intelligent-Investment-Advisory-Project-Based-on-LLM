"""
新闻爬虫工具模块
提供新闻搜索和爬取功能
"""

import logging
from typing import List, Dict
from mcp.server.fastmcp import FastMCP
from ..data_source_interface import FinancialDataSource

logger = logging.getLogger(__name__)

def register_news_crawler_tools(app: FastMCP, data_source: FinancialDataSource):
    """
    注册新闻爬虫工具
    
    Args:
        app: FastMCP应用实例
        data_source: 数据源实例
    """
    
    @app.tool()
    def crawl_news(query: str, top_k: int = 10) -> str:
        """
        获取个股或行业相关新闻（来源：东方财富）

        Args:
            query: 搜索查询词，如股票代码"600519"、公司名"嘉友国际"、行业名"白酒"等
            top_k: 返回的新闻数量，默认为10条

        Returns:
            格式化的新闻结果，包含标题、时间、来源、链接
        """
        try:
            import akshare as ak
            import re

            # 尝试多种方式获取新闻
            df = None
            search_method = ""

            # 方式1: 提取5-6位数字代码查询
            code_match = re.search(r'(\d{5,6})', query)
            if code_match:
                symbol = code_match.group(1)
                try:
                    df = ak.stock_news_em(symbol=symbol)
                    if df is not None and not df.empty:
                        search_method = f"代码({symbol})"
                except Exception:
                    df = None

            # 方式2: 如果代码查询失败或query不含代码，尝试用原始query直接搜索
            if (df is None or df.empty) and not code_match:
                try:
                    # 直接用原始查询词（可能是公司名）调用
                    df = ak.stock_news_em(symbol=query.strip())
                    if df is not None and not df.empty:
                        search_method = f"关键词({query})"
                except Exception:
                    df = None

            # 方式3: 如果仍无结果，尝试用纯数字部分（移除交易所前缀 sh./sz./.SH/.SZ）
            if df is None or df.empty:
                clean = re.sub(r'(sh\.|sz\.|\.SH|\.SZ)', '', query, flags=re.IGNORECASE).strip()
                if clean != query.strip():
                    try:
                        df = ak.stock_news_em(symbol=clean)
                        if df is not None and not df.empty:
                            search_method = f"清洗代码({clean})"
                    except Exception:
                        df = None

            # 格式化结果
            if df is not None and not df.empty:
                n = min(len(df), top_k)
                lines = [f"找到 {n} 条关于 {query} 的新闻（搜索方式: {search_method}）：\n"]
                for i, (_, row) in enumerate(df.head(top_k).iterrows()):
                    title = str(row.get("新闻标题", ""))
                    content = str(row.get("新闻内容", ""))[:200]
                    src = str(row.get("文章来源", ""))
                    url = str(row.get("新闻链接", ""))
                    dt = str(row.get("发布时间", ""))
                    lines.append(
                        f"{i+1}. **{title}**\n"
                        f"   来源: {src} | 时间: {dt}\n"
                        f"   摘要: {content}\n"
                        f"   链接: {url}\n"
                    )
                return "\n".join(lines)

            return f"未找到关于 '{query}' 的新闻。请尝试输入股票代码（如600519）或公司名称。"
        except Exception as e:
            logger.error(f"爬取新闻时出错: {e}")
            return f"新闻获取失败: {str(e)}"

    