"""
Web Search MCP Server — Baidu + Bing dual-engine search and fetch.
Provides real-time news, macro events, and geopolitical context.
Baidu primary (works in mainland China), Bing fallback.
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

_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def _parse_baidu_results(html: str, query: str) -> str:
    """从百度搜索结果页提取结果。"""
    results = []
    # 百度结果在 div.result.c-container 中
    # 标题在 h3.t > a，摘要可能在不同位置
    blocks = re.split(
        r'<div[^>]*class="(?:result|c-container)[^"]*"[^>]*>',
        html,
    )
    for block in blocks[1:]:  # 第一个是分割前的无关内容
        # 提取标题链接
        title_match = re.search(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            block, re.DOTALL,
        )
        if not title_match:
            continue
        url = title_match.group(1)
        title = re.sub(r'<[^>]+>', '', title_match.group(2)).strip()
        # 跳过百度内链和非结果链接
        if not title or len(title) < 3:
            continue
        if any(d in url for d in ("baidu.com/cache", "baidu.com/aladdin", "m.baidu.com/s")):
            continue

        # 摘要：c-abstract / c-span / content 等类名
        snippet = ""
        for cls in ("c-abstract", "c-span-last", "content", "c-color-text"):
            sm = re.search(
                rf'<(?:div|span|p)[^>]*class="[^"]*{cls}[^"]*"[^>]*>(.*?)</(?:div|span|p)>',
                block, re.DOTALL,
            )
            if sm:
                snippet = re.sub(r'<[^>]+>', '', sm.group(1)).strip()
                break
        if not snippet:
            # 宽松匹配：取第一个足够长的纯文本段落
            text_parts = re.split(r'<[^>]+>', block)
            for part in text_parts:
                part = part.strip()
                if len(part) > 30:
                    snippet = part[:300]
                    break

        results.append((title, url, snippet[:300] if snippet else ""))
        if len(results) >= 10:
            break

    if not results:
        return ""

    lines = [f"## 网络搜索结果（百度）：{query}\n"]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"**{i}. [{title}]({url})**")
        if snippet:
            lines.append(f"> {snippet}")
        lines.append("")
    return "\n".join(lines)


def _parse_bing_results(html: str, query: str) -> str:
    """从 Bing 搜索结果页提取结果（备用引擎）。"""
    results = []
    pattern = re.compile(
        r'<li[^>]*class="b_algo"[^>]*>.*?'
        r'<h2[^>]*>.*?<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?</h2>'
        r'(.*?)'
        r'</li>',
        re.DOTALL,
    )
    for url, title, body in pattern.findall(html):
        title = re.sub(r'<[^>]+>', '', title).strip()
        sm = re.search(
            r'<(?:p|div)[^>]*class="(?:b_lineclamp|b_caption|b_snippet)[^"]*"[^>]*>(.*?)</(?:p|div)>',
            body, re.DOTALL,
        )
        snippet = re.sub(r'<[^>]+>', '', sm.group(1)).strip()[:300] if sm else ""
        if title and url:
            results.append((title, url, snippet))
        if len(results) >= 10:
            break

    if not results:
        return ""

    lines = [f"## 网络搜索结果（Bing）：{query}\n"]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"**{i}. [{title}]({url})**")
        if snippet:
            lines.append(f"> {snippet}")
        lines.append("")
    return "\n".join(lines)


def _try_baidu(query: str, max_results: int) -> str:
    """尝试百度搜索。"""
    resp = httpx.get(
        "https://www.baidu.com/s",
        params={"wd": query, "rn": min(max_results, 20)},
        headers=_SEARCH_HEADERS,
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    if len(resp.text) < 1000:
        return ""

    # 反爬检测
    html_lower = resp.text.lower()
    if any(kw in html_lower for kw in ("请输入验证码", "安全验证", "访问太过频繁")):
        logger.warning("Baidu anti-bot triggered, falling back to Bing")
        return ""

    return _parse_baidu_results(resp.text, query)


def _try_bing(query: str, max_results: int) -> str:
    """尝试 Bing 搜索（备用）。"""
    resp = httpx.get(
        "https://www.bing.com/search",
        params={"q": query, "count": min(max_results, 20)},
        headers=_SEARCH_HEADERS,
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    resp.raise_for_status()
    if len(resp.text) < 1000:
        return ""
    if "captcha" in resp.text.lower():
        logger.warning("Bing CAPTCHA triggered")
        return ""
    return _parse_bing_results(resp.text, query)


@app.tool()
def web_search(query: str, max_results: int = 10) -> str:
    """
    使用百度搜索引擎搜索网络（Bing 备用），获取最新新闻和宏观信息。
    用于获取 A 股数据源无法提供的国际宏观事件、央行政策、地缘政治动态等。

    参数:
        query: 搜索关键词
        max_results: 最大返回结果数，默认 10

    返回:
        Markdown格式的搜索结果列表（标题+摘要+链接）
    """
    # 百度优先（国内可用，反爬宽松）
    for attempt, engine_fn in enumerate((_try_baidu, _try_bing)):
        try:
            result = engine_fn(query, max_results)
            if result:
                return result
        except httpx.TimeoutException:
            logger.warning(f"{engine_fn.__name__} timeout ({SEARCH_TIMEOUT}s)")
        except httpx.HTTPError as e:
            logger.warning(f"{engine_fn.__name__} HTTP {e.response.status_code}")
        except Exception as e:
            logger.warning(f"{engine_fn.__name__} failed: {e}")

    return (
        f"网络搜索 '{query}' 失败：百度与 Bing 均不可用。"
        f"请检查网络连接或稍后重试。"
    )


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
    logger.info("Starting Web Search MCP Server (Baidu+Bing) via stdio...")
    app.run(transport="stdio")
