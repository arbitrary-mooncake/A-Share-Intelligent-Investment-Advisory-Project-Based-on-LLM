"""
Web Search MCP Server — DuckDuckGo-based web search and fetch.
Provides real-time news, macro events, and geopolitical context.
"""
import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import logging
import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastMCP()

SEARCH_TIMEOUT = 15.0
FETCH_TIMEOUT = 20.0
FETCH_MAX_CHARS = 8000


@app.tool()
def web_search(query: str, max_results: int = 10) -> str:
    """
    使用 DuckDuckGo 搜索引擎搜索网络，获取最新新闻、事件和宏观信息。
    用于获取 A 股数据源无法提供的国际宏观事件、央行政策、地缘政治动态等。

    参数:
        query: 搜索关键词
        max_results: 最大返回结果数，默认 10

    返回:
        Markdown格式的搜索结果列表（标题+摘要+链接）
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=min(max_results, 10)))

        if not results:
            return f"搜索 '{query}' 未找到结果。"

        lines = [f"## 网络搜索结果：{query}\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            href = r.get("href", "无链接")
            body = r.get("body", "").strip()[:300]
            lines.append(f"**{i}. [{title}]({href})**")
            lines.append(f"> {body}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"网络搜索失败: {e}\n请稍后重试或调整搜索关键词。"


@app.tool()
def web_fetch(url: str) -> str:
    """
    抓取指定网页的完整文本内容。
    用于获取搜索结果的详细内容，如美联储声明、央行报告等。

    参数:
        url: 要抓取的网页链接

    返回:
        网页的纯文本内容（截断至 {FETCH_MAX_CHARS} 字符）
    """
    try:
        # SSRF 防护：仅允许 http/https
        parsed = httpx.URL(url)
        if parsed.scheme not in ("http", "https"):
            return f"不支持的协议: {parsed.scheme}，仅支持 http/https"

        resp = httpx.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; AI-Advisor/1.0)"},
            follow_redirects=True,
        )
        resp.raise_for_status()

        html = resp.text
        # Simple HTML tag stripping
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > FETCH_MAX_CHARS:
            text = text[:FETCH_MAX_CHARS] + "\n\n... (内容已截断)"

        return f"## 网页内容：{url}\n\n{text}"

    except Exception as e:
        logger.error(f"web_fetch failed: {e}")
        return f"网页抓取失败: {e}"


if __name__ == "__main__":
    logger.info("Starting Web Search MCP Server via stdio...")
    app.run(transport="stdio")
