"""
Web Search MCP Server — Bing-based web search and fetch.
Provides real-time news, macro events, and geopolitical context.
Uses cn.bing.com (accessible in mainland China), bypasses duckduckgo_search.
"""
import sys
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import re
import logging
import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastMCP()

SEARCH_TIMEOUT = 15.0
FETCH_TIMEOUT = 20.0
FETCH_MAX_CHARS = 8000

# Bing 搜索的 User-Agent（需要模拟浏览器，否则返回空白页）
_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _parse_bing_results(html: str, query: str) -> str:
    """从 Bing HTML 中提取搜索结果。"""
    # Bing 搜索结果通常在 <li class="b_algo"> 中
    # 标题在 <h2> → <a>，摘要紧随其后
    results = []
    # 匹配每个搜索结果块：<h2> 中的链接 + 紧随的摘要段落
    pattern = re.compile(
        r'<li[^>]*class="b_algo"[^>]*>.*?'
        r'<h2[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?</h2>'
        r'(.*?)'
        r'</li>',
        re.DOTALL,
    )
    matches = pattern.findall(html)

    for url, title, body in matches:
        title = re.sub(r'<[^>]+>', '', title).strip()
        # 摘要通常在 <p> 或 <div class="b_caption"> 中
        snippet_match = re.search(
            r'<(?:p|div)[^>]*class="(?:b_lineclamp|b_caption|b_snippet)[^"]*"[^>]*>(.*?)</(?:p|div)>',
            body, re.DOTALL,
        )
        if not snippet_match:
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', body, re.DOTALL)
        snippet = re.sub(r'<[^>]+>', '', snippet_match.group(1)).strip() if snippet_match else ""
        snippet = snippet[:300]

        if title and url:
            results.append((title, url, snippet))

    if not results:
        # 备选：更宽松的匹配
        alt_pattern = re.compile(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        alt_matches = alt_pattern.findall(html)
        seen_urls = set()
        for url, title in alt_matches:
            title = re.sub(r'<[^>]+>', '', title).strip()
            if title and url not in seen_urls and "bing.com" not in url and len(title) > 3:
                seen_urls.add(url)
                results.append((title, url, ""))
            if len(results) >= 10:
                break

    if not results:
        return f"搜索 '{query}' 未找到结果。请尝试更简洁的关键词。"

    lines = [f"## 网络搜索结果：{query}\n"]
    for i, (title, url, snippet) in enumerate(results[:10], 1):
        lines.append(f"**{i}. [{title}]({url})**")
        if snippet:
            lines.append(f"> {snippet}")
        lines.append("")

    return "\n".join(lines)


@app.tool()
def web_search(query: str, max_results: int = 10) -> str:
    """
    使用 Bing 搜索引擎搜索网络，获取最新新闻、事件和宏观信息。
    用于获取 A 股数据源无法提供的国际宏观事件、央行政策、地缘政治动态等。

    参数:
        query: 搜索关键词
        max_results: 最大返回结果数，默认 10

    返回:
        Markdown格式的搜索结果列表（标题+摘要+链接）
    """
    try:
        resp = httpx.get(
            "https://cn.bing.com/search",
            params={"q": query, "count": min(max_results, 20)},
            headers=_SEARCH_HEADERS,
            timeout=SEARCH_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()

        html = resp.text
        if len(html) < 500:
            return f"Bing 搜索返回内容过短（{len(html)} 字符），可能被反爬拦截。请稍后重试。"

        return _parse_bing_results(html, query)

    except httpx.TimeoutException:
        return f"Bing 搜索超时（{SEARCH_TIMEOUT}s）。请检查网络连接后重试。"
    except httpx.HTTPError as e:
        return f"Bing 搜索请求失败（HTTP {e.response.status_code}）。请稍后重试。"
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
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AI-Advisor/1.0)",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()

        html = resp.text
        # Simple HTML tag stripping
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
    logger.info("Starting Web Search MCP Server (Bing) via stdio...")
    app.run(transport="stdio")
