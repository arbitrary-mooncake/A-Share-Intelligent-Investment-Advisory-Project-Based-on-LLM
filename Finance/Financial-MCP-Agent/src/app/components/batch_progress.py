"""
批量打分进度组件 — 实时进度条 + 统计信息 + 结果表格 (st.dataframe)
"""
import html as _html
import streamlit as st
import pandas as pd
from theme import COLORS


# 5 级颜色映射（与后端 LEVELS 一致）
LEVEL_COLORS = {
    "强烈推荐": "#059669",
    "推荐": "#0891b2",
    "中性": "#d97706",
    "回避": "#ea580c",
    "卖出": "#dc2626",
}
LEVEL_BG = {
    "强烈推荐": "#d1fae5",
    "推荐": "#ccfbf1",
    "中性": "#fef3c7",
    "回避": "#ffedd5",
    "卖出": "#fee2e2",
}
LEVEL_EMOJI = {
    "强烈推荐": "🟢",
    "推荐": "🔵",
    "中性": "🟡",
    "回避": "🟠",
    "卖出": "🔴",
}
LEVEL_ORDER = {"强烈推荐": 0, "推荐": 1, "中性": 2, "回避": 3, "卖出": 4}


def render_progress_bar(progress_pct: float, status: str,
                         fetched: int, scored: int, total: int,
                         elapsed_seconds: int):
    """渲染批量打分进度条和统计信息"""
    status_labels = {
        "parsed": "解析完成",
        "fetching": "数据获取中",
        "fetched": "数据获取完成",
        "scoring": "LLM 打分中",
        "completed": "完成",
        "failed": "失败",
    }
    status_label = status_labels.get(status, status)

    pct_display = min(progress_pct, 100.0)
    bar_color = "#2563eb"
    if pct_display >= 100:
        bar_color = "#059669"
    elif status == "failed":
        bar_color = "#dc2626"

    st.markdown(f"""
    <div style="margin:12px 0;">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
      <span style="font-weight:700;color:#1e40af;">{status_label}</span>
      <span style="color:#64748b;">{pct_display:.1f}%</span>
    </div>
    <div style="background:#e2e8f0;border-radius:6px;height:10px;overflow:hidden;">
      <div style="background:{bar_color};width:{pct_display}%;height:100%;
        border-radius:6px;transition:width 0.3s;">
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    elapsed_min = elapsed_seconds // 60
    elapsed_sec = elapsed_seconds % 60
    cols = st.columns(5)
    with cols[0]:
        st.metric("总数", total)
    with cols[1]:
        st.metric("已获取数据", fetched)
    with cols[2]:
        st.metric("已打分", scored)
    with cols[3]:
        st.metric("耗时", f"{elapsed_min}分{elapsed_sec}秒")
    with cols[4]:
        if status == "fetching" and elapsed_seconds > 0 and fetched > 0:
            rate = fetched / elapsed_seconds * 60
            st.metric("速度", f"{rate:.0f}只/分")
        elif status == "scoring" and elapsed_seconds > 0 and scored > 0:
            rate = scored / elapsed_seconds * 60
            st.metric("速度", f"{rate:.0f}只/分")
        else:
            st.metric("速度", "—")


def render_level_badge(level: str):
    """渲染 5 级分类彩色徽章"""
    color = LEVEL_COLORS.get(level, "#6b7280")
    bg = LEVEL_BG.get(level, "#f3f4f6")
    st.markdown(
        f'<span style="display:inline-block;padding:3px 12px;'
        f'background:{bg};color:{color};border-radius:10px;'
        f'font-weight:700;font-size:0.88em;border:1.5px solid {color}25;">'
        f'{_html.escape(level)}</span>',
        unsafe_allow_html=True,
    )


def build_results_dataframe(stocks: list) -> pd.DataFrame:
    """将股票结果列表转为 pandas DataFrame，供 st.dataframe 展示。

    包含 emoji 级别列用于视觉区分。排序默认按推荐度（绿在前）。
    """
    if not stocks:
        return pd.DataFrame()

    rows = []
    for s in stocks:
        level = s.get("level", "中性")
        emoji = LEVEL_EMOJI.get(level, "⚪")
        pc = s.get("price_changes", {}) or {}

        # 格式化 PE/PB/ROE 为数值用于排序
        def _num(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        rows.append({
            "代码": s.get("code", "").replace("sh.", "").replace("sz.", ""),
            "名称": s.get("name") or "未知",
            "PE": _num(s.get("pe")),      # None → pandas NaN, NumberColumn handles
            "PB": _num(s.get("pb")),
            "ROE(%)": _num(s.get("roe")),
            "行业": s.get("industry") or "-",
            "评级": f"{emoji} {level}",
            "_level_raw": level,
            "置信度": s.get("confidence", ""),
            "理由": s.get("reason", ""),
            "风险": s.get("risk", ""),
            "市值": s.get("market_cap") or "-",
            "涨跌(1月)": pc.get("1m", "-"),
            "涨跌(1年)": pc.get("1y", "-"),
        })

    df = pd.DataFrame(rows)
    # 默认排序：推荐度从高到低
    df["_sort_order"] = df["_level_raw"].map(
        lambda x: LEVEL_ORDER.get(x, 2)
    )
    df = df.sort_values(["_sort_order", "PE"], ascending=[True, True])
    df = df.drop(columns=["_sort_order"])
    return df


def render_results_dataframe(stocks: list, key: str = "batch_results",
                              height: int = 500) -> tuple:
    """用 st.dataframe 渲染结果表格，返回 (selected_df, selection_state)。

    Args:
        stocks: 股票结果列表
        key: dataframe key for session state
        height: 表格高度（像素），超限自动滚动

    Returns:
        (filtered_df, selection) — filtered_df 为当前显示的 DataFrame,
        selection 为 st.dataframe 返回的选中状态
    """
    if not stocks:
        st.info("暂无结果")
        return pd.DataFrame(), None

    df = build_results_dataframe(stocks)

    # 自定义列配置：隐藏辅助列，控制宽度
    column_config = {
        "选中": st.column_config.CheckboxColumn(width="small"),
        "_level_raw": None,  # 隐藏
        "代码": st.column_config.TextColumn(width="small"),
        "名称": st.column_config.TextColumn(width="medium"),
        "PE": st.column_config.NumberColumn(format="%.1f", width="small"),
        "PB": st.column_config.NumberColumn(format="%.2f", width="small"),
        "ROE(%)": st.column_config.NumberColumn(format="%.1f", width="small"),
        "行业": st.column_config.TextColumn(width="small"),
        "评级": st.column_config.TextColumn(width="medium"),
        "置信度": st.column_config.TextColumn(width="small"),
        "理由": st.column_config.TextColumn(width="medium"),
        "风险": st.column_config.TextColumn(width="medium"),
        "市值": st.column_config.TextColumn(width="small"),
        "涨跌(1月)": st.column_config.TextColumn(width="small"),
        "涨跌(1年)": st.column_config.TextColumn(width="small"),
    }

    selection = st.dataframe(
        df,
        column_config=column_config,
        column_order=[
            "代码", "名称", "PE", "PB", "ROE(%)", "行业",
            "评级", "置信度", "理由", "风险", "市值",
            "涨跌(1月)", "涨跌(1年)",
        ],
        hide_index=True,
        height=height,
        use_container_width=True,
        selection_mode="multi-row",
        key=key,
        on_select="rerun",
    )

    return df, selection


def render_level_count_bar(stocks: list):
    """渲染评级分布统计条"""
    from collections import Counter
    counts = Counter(s.get("level", "中性") for s in stocks)
    count_str = " | ".join(
        f'<span style="color:{LEVEL_COLORS.get(k, "#6b7280")};font-weight:600;">'
        f'{k}: {v}</span>' for k, v in
        [("强烈推荐", counts.get("强烈推荐", 0)),
         ("推荐", counts.get("推荐", 0)),
         ("中性", counts.get("中性", 0)),
         ("回避", counts.get("回避", 0)),
         ("卖出", counts.get("卖出", 0))]
    )
    st.markdown(f'<div style="margin:8px 0;font-size:0.9em;">{count_str}</div>',
                unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 保留旧版 HTML 表格渲染（向后兼容，不再推荐使用）
# ──────────────────────────────────────────────

def render_results_table(stocks: list, max_display: int = 500):
    """[已废弃] 渲染批量打分结果表格 — 请使用 render_results_dataframe 替代"""
    return build_results_dataframe(stocks)
