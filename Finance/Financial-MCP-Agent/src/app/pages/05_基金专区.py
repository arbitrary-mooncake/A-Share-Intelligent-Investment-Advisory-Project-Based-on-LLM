"""
基金专区页面 — 基金/ETF深度分析 + 多维度打分 + 基金池管理
"""

import asyncio
import os
import sys
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
import httpx

from api_client import APIError
from config import API_BASE_URL


# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="基金专区",
    page_icon="🏦",
    layout="wide",
)

# ──────────────────────────────────────────────
# 自定义样式
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .score-badge-green {
        background: #d4edda; color: #155724; padding: 0.25rem 0.75rem;
        border-radius: 12px; font-weight: 700; font-size: 0.9em;
        display: inline-block; border: 1px solid #c3e6cb;
    }
    .score-badge-teal {
        background: #d1ecf1; color: #0c5460; padding: 0.25rem 0.75rem;
        border-radius: 12px; font-weight: 700; font-size: 0.9em;
        display: inline-block; border: 1px solid #bee5eb;
    }
    .score-badge-yellow {
        background: #fff3cd; color: #856404; padding: 0.25rem 0.75rem;
        border-radius: 12px; font-weight: 700; font-size: 0.9em;
        display: inline-block; border: 1px solid #ffeeba;
    }
    .score-badge-orange {
        background: #ffe5cc; color: #b35c00; padding: 0.25rem 0.75rem;
        border-radius: 12px; font-weight: 700; font-size: 0.9em;
        display: inline-block; border: 1px solid #ffd19b;
    }
    .score-badge-red {
        background: #f8d7da; color: #721c24; padding: 0.25rem 0.75rem;
        border-radius: 12px; font-weight: 700; font-size: 0.9em;
        display: inline-block; border: 1px solid #f5c6cb;
    }
    .subscore-bar { margin-bottom: 0.3rem; }
    .subscore-label { font-size: 0.85em; color: #555; }
    .subscore-value { font-size: 0.85em; font-weight: 600; float: right; }
    .fund-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# API 客户端函数
# ──────────────────────────────────────────────

FUND_TIMEOUT = 300  # 基金分析整体超时（秒）
FUND_POLL_TIMEOUT = 10  # 轮询单次超时（秒）


async def fund_search(keyword: str) -> list:
    """搜索基金：输入关键词，返回匹配的基金列表"""
    if not keyword or not keyword.strip():
        return []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/fund/search",
                json={"keyword": keyword.strip()}
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("results", [])
    except httpx.HTTPStatusError as e:
        raise APIError(f"基金搜索失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_report(fund_code: str, fund_name: str, mode: str = "report") -> str:
    """触发基金分析（深度报告或综合打分），返回 task_id"""
    try:
        async with httpx.AsyncClient(timeout=FUND_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/fund/report",
                json={
                    "fund_code": fund_code,
                    "fund_name": fund_name,
                    "mode": mode,
                }
            )
            resp.raise_for_status()
            return resp.json()["task_id"]
    except httpx.HTTPStatusError as e:
        raise APIError(f"触发基金报告失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_report_poll(task_id: str) -> dict:
    """轮询基金分析任务结果"""
    for attempt in range(900):  # 最长约 30 分钟 (900 x 2s)
        try:
            async with httpx.AsyncClient(timeout=FUND_POLL_TIMEOUT) as client:
                resp = await client.get(f"{API_BASE_URL}/api/fund/report/{task_id}")
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") in ("completed", "failed"):
                    return data
        except httpx.RequestError as e:
            raise APIError(f"查询基金报告状态失败: {e}")
        await asyncio.sleep(2)
    raise APIError("基金分析超时")


async def fund_pool_list(pool: str = "scored") -> list:
    """获取基金池列表"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{API_BASE_URL}/api/fund/pool/{pool}"
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("funds", [])
    except httpx.HTTPStatusError as e:
        raise APIError(f"获取基金池失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_pool_add(fund_code: str, fund_name: str, pool: str = "watchlist") -> dict:
    """向基金池添加基金"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/fund/pool/{pool}",
                json={"fund_code": fund_code, "fund_name": fund_name}
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"添加基金失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_pool_remove(fund_code: str, pool: str = "scored") -> dict:
    """从基金池删除基金"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.delete(
                f"{API_BASE_URL}/api/fund/pool/{pool}/{fund_code}"
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"删除基金失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_pool_score(fund_code: str) -> str:
    """触发基金池打分，返回 task_id"""
    try:
        async with httpx.AsyncClient(timeout=FUND_TIMEOUT) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/fund/score/{fund_code}"
            )
            resp.raise_for_status()
            return resp.json()["task_id"]
    except httpx.HTTPStatusError as e:
        raise APIError(f"触发基金打分失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_quick_query(keyword: str) -> dict:
    """基金快速查询 — 返回关键指标+同类基准+LLM解读"""
    if not keyword or not keyword.strip():
        raise APIError("关键字不能为空")
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(
                f"{API_BASE_URL}/api/fund/query",
                json={"keyword": keyword.strip()},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise APIError(f"基金查询失败: {e.response.text}", e.response.status_code)
    except httpx.RequestError as e:
        raise APIError(f"连接后端失败: {e}")


async def fund_cache_status(fund_code: str) -> dict:
    """获取基金分析缓存状态"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{API_BASE_URL}/api/fund/cache/{fund_code}")
            resp.raise_for_status()
            return resp.json()
    except httpx.RequestError as e:
        raise APIError(f"缓存状态查询失败: {e}")


# ──────────────────────────────────────────────
# 同步包装器
# ──────────────────────────────────────────────
def _ar(coro):
    return asyncio.run(coro)


def handle_fund_search(keyword: str) -> list:
    return _ar(fund_search(keyword))


def handle_fund_report(fund_code: str, fund_name: str, mode: str) -> str:
    return _ar(fund_report(fund_code, fund_name, mode))


def handle_fund_report_poll(task_id: str) -> dict:
    return _ar(fund_report_poll(task_id))


def handle_fund_pool_list(pool: str = "scored") -> list:
    return _ar(fund_pool_list(pool))


def handle_fund_pool_add(fund_code: str, fund_name: str, pool: str = "watchlist") -> dict:
    return _ar(fund_pool_add(fund_code, fund_name, pool))


def handle_fund_pool_remove(fund_code: str, pool: str = "scored") -> dict:
    return _ar(fund_pool_remove(fund_code, pool))


def handle_fund_pool_score(fund_code: str) -> str:
    return _ar(fund_pool_score(fund_code))


def handle_fund_quick_query(keyword: str) -> dict:
    return _ar(fund_quick_query(keyword))


def handle_fund_cache_status(fund_code: str) -> dict:
    return _ar(fund_cache_status(fund_code))


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def get_score_badge_html(score) -> str:
    """根据分数返回彩色徽章 HTML"""
    try:
        s = int(score)
    except (ValueError, TypeError):
        return f'<span class="score-badge-yellow">N/A</span>'
    if s >= 90:
        cls = "score-badge-green"
        label = f"优秀 {s}分"
    elif s >= 80:
        cls = "score-badge-teal"
        label = f"较优 {s}分"
    elif s >= 70:
        cls = "score-badge-yellow"
        label = f"中等 {s}分"
    elif s >= 60:
        cls = "score-badge-orange"
        label = f"一般 {s}分"
    else:
        cls = "score-badge-red"
        label = f"谨慎 {s}分"
    return f'<span class="{cls}">{label}</span>'


def get_score_color(score) -> str:
    """根据分数返回 Streamlit metric 颜色"""
    try:
        s = int(score)
    except (ValueError, TypeError):
        return "off"
    if s >= 90: return "green"
    if s >= 80: return "teal"
    if s >= 70: return "yellow"
    if s >= 60: return "orange"
    return "red"


def get_rating_label(score) -> str:
    """分数 → 评级标签"""
    try:
        s = int(score)
    except (ValueError, TypeError):
        return "N/A"
    if s >= 90: return "优秀"
    if s >= 80: return "较优"
    if s >= 70: return "中等偏上"
    if s >= 60: return "一般"
    return "谨慎"


SUBSOCRE_LABELS = {
    "product_positioning": "产品定位与策略",
    "performance_risk": "业绩与风险",
    "portfolio_structure": "组合结构",
    "manager_team": "经理/团队",
    "benchmark_style_consistency": "基准一致性",
    "fee_liquidity": "费用与流动性",
    "event_risk": "事件风险",
}

SUBSOCRE_ORDER = [
    "product_positioning", "performance_risk", "portfolio_structure",
    "manager_team", "benchmark_style_consistency", "fee_liquidity", "event_risk",
]


def render_subscores(subscores: dict):
    """渲染7维度子得分条"""
    for key in SUBSOCRE_ORDER:
        val = subscores.get(key)
        if val is None:
            continue
        label = SUBSOCRE_LABELS.get(key, key)
        pct = max(0, min(int(val), 100))
        bar_color = "#28a745" if pct >= 80 else "#17a2b8" if pct >= 70 else "#ffc107" if pct >= 60 else "#dc3545"
        st.markdown(
            f"""
            <div class="subscore-bar">
                <span class="subscore-label">{label}</span>
                <span class="subscore-value" style="color:{bar_color};">{val}分</span>
                <div style="background:#e9ecef;border-radius:4px;height:6px;margin-top:3px;">
                    <div style="background:{bar_color};width:{pct}%;height:6px;border-radius:4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# 状态初始化
# ──────────────────────────────────────────────
if "_fund_busy" not in st.session_state:
    st.session_state["_fund_busy"] = False
if "_fund_search_results" not in st.session_state:
    st.session_state["_fund_search_results"] = []
if "_fund_selected" not in st.session_state:
    st.session_state["_fund_selected"] = None
if "_fund_task_id" not in st.session_state:
    st.session_state["_fund_task_id"] = None
if "_fund_result" not in st.session_state:
    st.session_state["_fund_result"] = None
if "_fund_error" not in st.session_state:
    st.session_state["_fund_error"] = None
if "_fund_pool_data" not in st.session_state:
    st.session_state["_fund_pool_data"] = None
if "_fund_pool_error" not in st.session_state:
    st.session_state["_fund_pool_error"] = None
if "_fund_score_tasks" not in st.session_state:
    st.session_state["_fund_score_tasks"] = {}
if "_fund_pool_action_msg" not in st.session_state:
    st.session_state["_fund_pool_action_msg"] = None


# ──────────────────────────────────────────────
# 页面标题
# ──────────────────────────────────────────────
st.markdown(
    '<div style="margin-bottom:4px;">'
    '<span style="font-size:1.7em;font-weight:800;color:#0f172a;">🏦 基金专区</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption("快速查询基金关键指标与同类基准，基金池支持深度打分与报告生成。")

# ──────────────────────────────────────────────
# 渲染辅助函数
# ──────────────────────────────────────────────

def _safe_str(val) -> str:
    """安全转换：None/空 → \"N/A\" """
    if val is None:
        return "N/A"
    s = str(val).strip()
    if s == "" or s == "None":
        return "N/A"
    return s


def _render_fund_query_result(data: dict) -> None:
    """渲染基金查询结果卡片 — 仿照股票查询 result_card 设计"""
    fund_code = data.get("fund_code", "")
    fund_name = data.get("fund_name", "未知")
    fund_type = data.get("fund_type", "")
    analysis = data.get("analysis", {})
    metrics = data.get("metrics", {})
    benchmark = data.get("fund_benchmark", {})

    # ── 标题行 ──
    st.markdown(
        f'<span style="font-size:1.25rem;font-weight:700;">{fund_name}</span>'
        f'<span style="font-size:1rem;color:#888;margin-left:0.5rem;">{fund_code}</span>',
        unsafe_allow_html=True,
    )
    if fund_type:
        st.caption(f"类型：{fund_type} | 管理人：{analysis.get('management', metrics.get('management', 'N/A'))}")
    st.markdown("---")

    # ── 关键指标卡片 ──
    st.markdown("#### 关键指标")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新净值", _safe_str(metrics.get("latest_nav", analysis.get("latest_nav", ""))))
    col2.metric("近1年收益", _safe_str(metrics.get("return_1y", analysis.get("return_1y", ""))))
    col3.metric("年化波动率", _safe_str(metrics.get("annual_volatility", analysis.get("annual_volatility", ""))))
    col4.metric("最大回撤(1Y)", _safe_str(metrics.get("max_drawdown_1y", analysis.get("max_drawdown_1y", ""))))
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("夏普比率", _safe_str(metrics.get("sharpe_ratio", analysis.get("sharpe_ratio", ""))))
    col6.metric("管理费率", _safe_str(metrics.get("m_fee", analysis.get("m_fee", ""))))
    col7.metric("托管费率", _safe_str(metrics.get("c_fee", analysis.get("c_fee", ""))))
    col8.metric("成立日期", _safe_str(metrics.get("found_date", analysis.get("found_date", ""))))

    # ── 同类基准对比 ──
    if benchmark and benchmark.get("type_name"):
        st.markdown("#### 同类基准对比")
        b_type = benchmark.get("type_name", "")
        b_desc = benchmark.get("description", "")
        st.caption(f"参考类型：{b_type} — {b_desc}")

        bcols = st.columns(6)
        b_items = [
            ("1年收益", "annual_return_1y"),
            ("3年收益", "annual_return_3y"),
            ("年化波动", "annual_volatility"),
            ("最大回撤", "max_drawdown"),
            ("夏普比率", "sharpe_ratio"),
            ("管理费率", "mgmt_fee"),
        ]
        for i, (label, key) in enumerate(b_items):
            bcols[i].metric(label, benchmark.get(key, "N/A"))

    # ── LLM 解读 ──
    if analysis.get("fund_intro") or analysis.get("performance_comment") or analysis.get("suitability"):
        st.markdown("#### 分析解读")
        if analysis.get("fund_intro"):
            st.caption(f"📌 {analysis['fund_intro']}")
        if analysis.get("performance_comment"):
            perf_color = "#28a745" if any(kw in str(analysis.get('performance_comment', '')) for kw in ["优于", "优秀", "良好", "突出"]) else "#ffc107" if any(kw in str(analysis.get('performance_comment', '')) for kw in ["一般", "中等", "偏弱"]) else "#888"
            st.markdown(
                f'<div style="font-size:0.9em;margin:0.3rem 0;color:{perf_color};">'
                f'📊 {analysis["performance_comment"]}</div>',
                unsafe_allow_html=True,
            )
        if analysis.get("suitability"):
            st.info(f"👤 {analysis['suitability']}")

    # ── 附加信息 ──
    with st.expander("更多信息"):
        benchmark_name = metrics.get("benchmark", analysis.get("benchmark", ""))
        invest_type = metrics.get("invest_type", analysis.get("invest_type", ""))
        status = metrics.get("status", analysis.get("status", ""))
        st.markdown(f"**业绩基准**: {benchmark_name or 'N/A'}")
        st.markdown(f"**投资类型**: {invest_type or 'N/A'}")
        st.markdown(f"**基金状态**: {status or 'N/A'}")
        nav_date = metrics.get("nav_date", analysis.get("nav_date", ""))
        if nav_date:
            st.caption(f"净值日期: {nav_date}")


# ──────────────────────────────────────────────
# Tab 结构
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 基金查询", "📊 基金池"])


# ══════════════════════════════════════════════
# Tab 1: 基金查询
# ══════════════════════════════════════════════
with tab1:
    st.subheader("🔍 基金查询")
    st.caption("输入基金代码或名称，快速获取关键指标与同类基准对比")

    is_busy = st.session_state.get("_fund_busy", False)

    # ── 查询输入 ──
    col_input, col_query, col_report = st.columns([3.5, 1, 1])
    with col_input:
        fund_input = st.text_input(
            "输入基金代码或名称",
            placeholder="如: 000390 / 华商优势行业 / 510050",
            key="fund_query_input",
            label_visibility="collapsed",
        )
    with col_query:
        query_clicked = st.button("🔍 快速查询", use_container_width=True, key="btn_fund_query",
                                   disabled=is_busy)
    with col_report:
        report_clicked = st.button("📝 生成报告", use_container_width=True, key="btn_fund_report_tab1",
                                    disabled=is_busy)

    # ── 快速查询 ──
    if query_clicked and fund_input:
        st.session_state["_fund_busy"] = True
        st.session_state["_fund_query_result"] = None
        st.session_state["_fund_query_error"] = None
        st.session_state["_fund_query_input"] = fund_input
        try:
            result = handle_fund_quick_query(fund_input)
            st.session_state["_fund_query_result"] = result
        except APIError as e:
            st.session_state["_fund_query_error"] = str(e)
        except Exception as e:
            st.session_state["_fund_query_error"] = f"查询异常: {e}"
        finally:
            st.session_state["_fund_busy"] = False
            st.rerun()

    # ── 生成报告 ──
    if report_clicked and fund_input:
        st.session_state["_fund_busy"] = True
        st.session_state["_fund_report_task_id"] = None
        st.session_state["_fund_report_result"] = None
        st.session_state["_fund_report_error"] = None
        st.session_state["_fund_report_input"] = fund_input
        try:
            resolved = handle_fund_quick_query(fund_input)
            fcode = resolved.get("fund_code", "")
            fname = resolved.get("fund_name", fund_input)
            task_id = handle_fund_report(fcode, fname, "report")
            st.session_state["_fund_report_task_id"] = task_id
            st.session_state["_fund_report_fname"] = fname
        except APIError as e:
            st.session_state["_fund_report_error"] = str(e)
            st.session_state["_fund_busy"] = False
        except Exception as e:
            st.session_state["_fund_report_error"] = f"触发报告异常: {e}"
            st.session_state["_fund_busy"] = False
        st.rerun()

    # ── 快速查询结果 ──
    query_result = st.session_state.get("_fund_query_result")
    if query_result:
        st.markdown("---")
        _render_fund_query_result(query_result)

    # ── 查询错误 ──
    if st.session_state.get("_fund_query_error"):
        st.error(st.session_state["_fund_query_error"])

    # ── 报告生成轮询 ──
    report_task_id = st.session_state.get("_fund_report_task_id")
    if report_task_id and not st.session_state.get("_fund_report_result") and not st.session_state.get("_fund_report_error"):
        _fname = st.session_state.get("_fund_report_fname", "该基金")
        with st.status(f"正在为 **{_fname}** 生成深度分析报告...", expanded=True, state="running") as rpt_status:
            rpt_progress = st.progress(0, text="初始化分析任务...")
            try:
                result = handle_fund_report_poll(report_task_id)
                if result.get("status") == "completed":
                    rpt_progress.progress(1.0, text="报告生成完成！")
                    rpt_status.update(label="报告已生成", state="complete", expanded=False)
                    st.session_state["_fund_report_result"] = result
                elif result.get("status") == "failed":
                    err = result.get("error", "未知错误")
                    rpt_status.update(label="报告生成失败", state="error", expanded=False)
                    st.session_state["_fund_report_error"] = err
            except APIError as e:
                rpt_status.update(label="报告生成失败", state="error", expanded=False)
                st.session_state["_fund_report_error"] = str(e)
            except Exception as e:
                rpt_status.update(label="报告生成失败", state="error", expanded=False)
                st.session_state["_fund_report_error"] = f"轮询异常: {e}"
            finally:
                st.session_state["_fund_busy"] = False

    # ── 报告结果 ──
    if st.session_state.get("_fund_report_result"):
        result = st.session_state["_fund_report_result"]
        st.success("深度分析报告已生成！")
        report_content = result.get("report_content", "")
        fname_rpt = result.get("fund_name", "基金")
        fcode_rpt = result.get("fund_code", "")
        date_str = datetime.now().strftime("%Y%m%d")

        pdf_path = result.get("report_pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label="📄 下载 PDF 报告",
                data=pdf_data,
                file_name=f"{fname_rpt}_分析报告_{date_str}.pdf",
                mime="application/pdf",
                type="primary",
            )
        elif report_content:
            st.download_button(
                label="📥 下载报告 (Markdown)",
                data=report_content,
                file_name=f"{fname_rpt}_分析报告_{date_str}.md",
                mime="text/markdown",
                type="primary",
            )
        with st.expander("预览报告内容"):
            st.markdown(report_content)
        if st.button("清除结果，开始新查询", key="btn_clear_fund_report"):
            for k in ["_fund_report_task_id", "_fund_report_result", "_fund_report_error",
                       "_fund_report_input", "_fund_report_fname",
                       "_fund_query_result", "_fund_query_error", "_fund_query_input"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── 报告错误 ──
    if st.session_state.get("_fund_report_error"):
        st.error(st.session_state["_fund_report_error"])
        if st.button("清除错误，重试", key="btn_clear_fund_rpt_err"):
            for k in ["_fund_report_task_id", "_fund_report_result", "_fund_report_error",
                       "_fund_report_input", "_fund_report_fname"]:
                st.session_state.pop(k, None)
            st.rerun()


# ══════════════════════════════════════════════
# 结果渲染辅助函数
# ══════════════════════════════════════════════
# Tab 2: 基金池
# ══════════════════════════════════════════════
with tab2:
    st.subheader("基金持仓池")

    # ── 状态初始化 ──
    if "_fund_pool_selected" not in st.session_state:
        st.session_state["_fund_pool_selected"] = None
    if "_fund_confirm_delete" not in st.session_state:
        st.session_state["_fund_confirm_delete"] = None
    if "_fund_report_tasks" not in st.session_state:
        st.session_state["_fund_report_tasks"] = {}

    # ── 刷新池数据 ──
    col_refresh, col_empty = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 刷新", use_container_width=True, key="btn_refresh_pool"):
            try:
                st.session_state["_fund_pool_data"] = handle_fund_pool_list("scored")
                st.session_state["_fund_pool_error"] = None
            except APIError as e:
                st.session_state["_fund_pool_error"] = str(e)
            except Exception as e:
                st.session_state["_fund_pool_error"] = f"获取基金池异常: {e}"
            st.rerun()

    # 首次加载
    if st.session_state.get("_fund_pool_data") is None:
        try:
            st.session_state["_fund_pool_data"] = handle_fund_pool_list("scored")
        except APIError as e:
            st.session_state["_fund_pool_error"] = str(e)
            st.session_state["_fund_pool_data"] = []
        except Exception as e:
            st.session_state["_fund_pool_error"] = f"获取基金池异常: {e}"
            st.session_state["_fund_pool_data"] = []

    # ── 操作结果提示 ──
    if st.session_state.get("_fund_pool_action_msg"):
        st.success(st.session_state["_fund_pool_action_msg"])
        st.session_state["_fund_pool_action_msg"] = None

    if st.session_state.get("_fund_pool_error"):
        st.error(st.session_state["_fund_pool_error"])
        st.session_state["_fund_pool_error"] = None

    # ── 添加基金表单 ──
    st.markdown("---")
    with st.expander("➕ 添加基金到池", expanded=False):
        col_code, col_name, col_btn = st.columns([2, 2, 1])
        with col_code:
            add_code = st.text_input("基金代码", placeholder="如: sh.510050", key="add_fund_code")
        with col_name:
            add_name = st.text_input("基金名称", placeholder="如: 华夏上证50ETF", key="add_fund_name")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("添加", use_container_width=True, type="primary",
                        key="btn_add", disabled=not (add_code and add_name)):
                try:
                    result = handle_fund_pool_add(add_code.strip(), add_name.strip(), "scored")
                    st.session_state["_fund_pool_action_msg"] = f"已添加 {add_name}({add_code})"
                    st.session_state["_fund_pool_data"] = handle_fund_pool_list("scored")
                except APIError as e:
                    st.session_state["_fund_pool_error"] = str(e)
                except Exception as e:
                    st.session_state["_fund_pool_error"] = f"添加异常: {e}"
                st.rerun()

    pool_data = st.session_state.get("_fund_pool_data", [])

    # ── 构建基金查找字典 ──
    _pool_map = {f.get("fund_code", ""): f for f in pool_data} if pool_data else {}

    # ── 操作行（选中基金后显示） ──
    selected_code = st.session_state.get("_fund_pool_selected")
    selected_fund = _pool_map.get(selected_code) if selected_code else None

    if selected_fund:
        st.markdown("---")
        sf_code = selected_fund.get("fund_code", "")
        sf_name = selected_fund.get("fund_name", "")
        sf_score = selected_fund.get("score")
        sf_rating = selected_fund.get("rating", "")
        sf_holding = selected_fund.get("holding_period", "-")
        if isinstance(sf_holding, dict):
            sf_holding = sf_holding.get("label", "-")

        # 操作行标题
        st.markdown(
            f'<div style="background:#f0f7ff;border-radius:10px;padding:0.8rem 1rem;'
            f'margin-bottom:0.5rem;border:1px solid #b8daff;">'
            f'<span style="font-weight:700;font-size:1.05em;">已选：{sf_name}</span> &nbsp; '
            f'<code>{sf_code}</code> &nbsp; '
            f'{get_score_badge_html(sf_score)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── 删除确认弹窗 ──
        if st.session_state.get("_fund_confirm_delete") == sf_code:
            st.markdown(
                '<div style="'
                'border:1.5px solid #fca5a5;'
                'border-radius:12px;'
                'padding:16px 20px;'
                'background: linear-gradient(145deg, #fff5f5 0%, #fef2f2 100%);'
                'box-shadow: 0 4px 12px rgba(220,38,38,0.08);'
                'margin-bottom:10px;'
                '">',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:1.1em;font-weight:700;color:#b91c1c;margin-bottom:10px;">'
                f'⚠️ 确定要删除 <span style="color:#7f1d1d;">{sf_name}</span>（{sf_code}）吗？此操作不可撤销。'
                f'</div>',
                unsafe_allow_html=True,
            )
            cc1, cc2, _ = st.columns([1, 1, 3])
            with cc1:
                if st.button("确认删除", type="primary", key=f"cfm_del_yes_{sf_code}"):
                    try:
                        handle_fund_pool_remove(sf_code, "scored")
                        st.session_state["_fund_pool_action_msg"] = f"已删除 {sf_name}({sf_code})"
                        st.session_state["_fund_pool_selected"] = None
                        st.session_state["_fund_confirm_delete"] = None
                        st.session_state["_fund_pool_data"] = handle_fund_pool_list("scored")
                    except APIError as e:
                        st.session_state["_fund_pool_error"] = str(e)
                    except Exception as e:
                        st.session_state["_fund_pool_error"] = f"删除异常: {e}"
                    st.rerun()
            with cc2:
                if st.button("取消", key=f"cfm_del_no_{sf_code}"):
                    st.session_state["_fund_confirm_delete"] = None
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        c_act1, c_act2, c_act3, c_act4, c_act5, c_act6 = st.columns([1, 1, 1, 1, 2, 0.4])
        with c_act1:
            if st.button("🔄 重新打分", key=f"act_score_{sf_code}", use_container_width=True):
                try:
                    task_id = handle_fund_pool_score(sf_code)
                    st.session_state["_fund_score_tasks"][task_id] = sf_code
                except APIError as e:
                    st.session_state["_fund_pool_error"] = str(e)
                except Exception as e:
                    st.session_state["_fund_pool_error"] = f"打分异常: {e}"
                st.rerun()
        with c_act2:
            if st.button("📝 生成报告", key=f"act_report_{sf_code}", use_container_width=True):
                try:
                    task_id = handle_fund_report(sf_code, sf_name, "report")
                    st.session_state["_fund_report_tasks"][task_id] = sf_name
                except APIError as e:
                    st.session_state["_fund_pool_error"] = str(e)
                except Exception as e:
                    st.session_state["_fund_pool_error"] = f"报告异常: {e}"
                st.rerun()
        with c_act3:
            if st.button("🗑 删除", key=f"act_del_{sf_code}", use_container_width=True):
                st.session_state["_fund_confirm_delete"] = sf_code
                st.rerun()
        with c_act4:
            detail_key = f"_fund_detail_{sf_code}"
            is_expanded = st.session_state.get(detail_key, False)
            if st.button(
                "📋 收起详情" if is_expanded else "📋 展开详情",
                key=f"act_detail_{sf_code}",
                use_container_width=True,
            ):
                st.session_state[detail_key] = not is_expanded
                st.rerun()
        with c_act5:
            st.caption(f"评级: {sf_rating or 'N/A'} | 建议持有: {sf_holding}")
        with c_act6:
            if st.button("✕", key=f"act_close_{sf_code}", help="取消选中", use_container_width=True):
                st.session_state["_fund_pool_selected"] = None
                st.session_state["_fund_confirm_delete"] = None
                st.session_state.pop(f"_fund_detail_{sf_code}", None)
                st.rerun()

        # ── 打分/报告进度条（内嵌在操作行下方，不跳出绿色框）──
        _score_tasks_copy = dict(st.session_state.get("_fund_score_tasks", {}))
        for tid, fcode_in_task in _score_tasks_copy.items():
            if fcode_in_task != sf_code:
                continue
            with st.status(f"正在对 **{sf_name}** 打分...", expanded=True, state="running") as score_status:
                try:
                    result = handle_fund_report_poll(tid)
                    if result.get("status") == "completed":
                        score_data = result.get("score", {})
                        overall = score_data.get("overall_score", {})
                        sc = overall.get("score", "-")
                        score_status.update(label=f"✅ 打分完成: {sf_name} → {sc}分", state="complete", expanded=False)
                        st.session_state["_fund_pool_action_msg"] = f"打分完成: {sf_name} → {sc}分"
                    elif result.get("status") == "failed":
                        err = result.get("error", "未知错误")
                        st.session_state["_fund_pool_error"] = f"打分失败: {err}"
                        score_status.update(label="打分失败", state="error", expanded=False)
                except APIError as e:
                    st.session_state["_fund_pool_error"] = str(e)
                    score_status.update(label="打分失败", state="error", expanded=False)
                except Exception as e:
                    st.session_state["_fund_pool_error"] = f"打分异常: {e}"
                    score_status.update(label="打分失败", state="error", expanded=False)
                finally:
                    st.session_state["_fund_score_tasks"].pop(tid, None)
                    try:
                        st.session_state["_fund_pool_data"] = handle_fund_pool_list("scored")
                    except Exception:
                        pass
                    st.rerun()

        _report_tasks_copy = dict(st.session_state.get("_fund_report_tasks", {}))
        for tid, fname_in_task in _report_tasks_copy.items():
            if fname_in_task != sf_name:
                continue
            with st.status(f"正在生成 **{sf_name}** 报告...", expanded=True, state="running") as report_status:
                try:
                    result = handle_fund_report_poll(tid)
                    if result.get("status") == "completed":
                        report_status.update(label=f"✅ 报告完成: {sf_name}", state="complete", expanded=False)
                        st.session_state["_fund_pool_action_msg"] = f"报告生成完成: {sf_name}"
                        # Store report for display
                        st.session_state["_fund_report_result"] = result
                    elif result.get("status") == "failed":
                        err = result.get("error", "未知错误")
                        st.session_state["_fund_pool_error"] = f"报告生成失败: {err}"
                        report_status.update(label="报告失败", state="error", expanded=False)
                except APIError as e:
                    st.session_state["_fund_pool_error"] = str(e)
                    report_status.update(label="报告失败", state="error", expanded=False)
                except Exception as e:
                    st.session_state["_fund_pool_error"] = f"报告异常: {e}"
                    report_status.update(label="报告失败", state="error", expanded=False)
                finally:
                    st.session_state["_fund_report_tasks"].pop(tid, None)
                    st.rerun()

        # ── 报告结果展示（如果有已完成报告）──
        report_result = st.session_state.get("_fund_report_result")
        if report_result:
            report_content = report_result.get("report_content", "")
            pdf_path = report_result.get("report_pdf_path", "")
            date_str = datetime.now().strftime("%Y%m%d")
            if report_content or pdf_path:
                st.markdown("---")
                col_title, col_close = st.columns([5, 1])
                with col_title:
                    st.markdown("#### 📄 分析报告")
                with col_close:
                    if st.button("✕ 关闭", key=f"close_report_{sf_code}"):
                        st.session_state.pop("_fund_report_result", None)
                        st.rerun()

                col_dl1, col_dl2 = st.columns(2)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    with col_dl1:
                        st.download_button(
                            label="📄 下载 PDF 报告",
                            data=pdf_data,
                            file_name=f"{sf_name}_基金分析报告_{date_str}.pdf",
                            mime="application/pdf",
                            type="primary",
                            key=f"dl_pdf_inline_{sf_code}",
                        )
                if report_content:
                    with col_dl2:
                        st.download_button(
                            label="📥 下载 Markdown",
                            data=report_content,
                            file_name=f"{sf_name}_基金分析报告_{date_str}.md",
                            mime="text/markdown",
                            key=f"dl_md_inline_{sf_code}",
                        )
                if report_content:
                    sections = report_content.split("\n## ")
                    if len(sections) > 1:
                        st.markdown(sections[0])
                        for i, sec in enumerate(sections[1:], 1):
                            lines = sec.strip().split("\n", 1)
                            title = lines[0].strip()
                            body = lines[1].strip() if len(lines) > 1 else ""
                            with st.expander(f"## {title}", expanded=(i <= 2)):
                                st.markdown(body)
                    else:
                        st.markdown(report_content)

        # ── 详情展开区域 ──
        if st.session_state.get(detail_key, False):
            with st.container():
                st.markdown("---")
                subscores = selected_fund.get("subscores", {})
                strengths = selected_fund.get("strengths", [])
                risks = selected_fund.get("risks", [])

                st.markdown(f"#### {sf_name} 详细信息")
                if subscores:
                    col_left, col_right = st.columns(2)
                    with col_left:
                        render_subscores(subscores)
                    with col_right:
                        st.markdown("**✅ 优势**")
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.caption("暂无优势数据")
                        st.markdown("**⚠️ 风险**")
                        if risks:
                            for r in risks:
                                st.markdown(f"- {r}")
                        else:
                            st.caption("暂无风险数据")

                iview = selected_fund.get("investment_view", "")
                if iview:
                    st.info(f"投资观点: {iview}")
                status_text = selected_fund.get("status", "")
                last_upd = selected_fund.get("last_updated", "")
                st.caption(f"状态: {status_text or 'N/A'} | 更新: {last_upd[:19] if last_upd else 'N/A'}")

    # ── 基金池列表 ──
    st.markdown("---")
    if not pool_data:
        st.info("基金池为空，请添加基金或等待后端评分完成。")
    else:
        st.caption(f"共 {len(pool_data)} 只基金，点击行选中操作")
        # 表头
        col_1, col_2, col_3, col_4, col_5, col_6 = st.columns(
            [1.5, 2.5, 1, 1, 1.2, 1.2]
        )
        with col_1: st.markdown("**基金代码**")
        with col_2: st.markdown("**基金名称**")
        with col_3: st.markdown("**得分**")
        with col_4: st.markdown("**评级**")
        with col_5: st.markdown("**持有期**")
        with col_6: st.markdown("**更新时间**")
        st.markdown("---")

        for fund in pool_data:
            fcode = fund.get("fund_code", "")
            fname = fund.get("fund_name", "")
            score = fund.get("score")
            rating = fund.get("rating", "")
            holding = fund.get("holding_period", "-")
            last_upd = fund.get("last_updated", "")
            if isinstance(holding, dict):
                holding = holding.get("label", "-")

            # 高亮当前选中行
            is_sel = (fcode == st.session_state.get("_fund_pool_selected"))
            row_bg = "#e8f4fd" if is_sel else "transparent"
            row_border = "2px solid #2196F3" if is_sel else "1px solid #e9ecef"

            c1, c2, c3, c4, c5, c6 = st.columns(
                [1.5, 2.5, 1, 1, 1.2, 1.2]
            )
            with c1:
                st.markdown(
                    f'<div style="background:{row_bg};border-left:{row_border};padding:0.3rem 0.4rem;">'
                    f'<code>{fcode}</code></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button(fname, key=f"sel_fund_{fcode}", help=f"点击选中 {fname}"):
                    st.session_state["_fund_pool_selected"] = fcode
                    st.session_state["_fund_confirm_delete"] = None
                    st.rerun()
            with c3:
                st.markdown(get_score_badge_html(score), unsafe_allow_html=True)
            with c4:
                st.caption(rating or "-")
            with c5:
                st.caption(holding)
            with c6:
                st.caption(last_upd[:10] if last_upd else "-")

            st.markdown("<div style='margin-bottom:0.3rem;'></div>", unsafe_allow_html=True)

