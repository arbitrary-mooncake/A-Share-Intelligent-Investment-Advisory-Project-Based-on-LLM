"""精筛池健康卡片组件"""
import streamlit as st
from datetime import datetime


def render_pool_health(health: dict) -> None:
    """渲染精筛池健康状态卡片。

    Args:
        health: orch.pool_manager.get_pool_health() 返回的 dict
    """
    emoji_map = {"red": "\U0001f534", "yellow": "\U0001f7e1", "green": "\U0001f7e2"}
    status_label = {"red": "需更新", "yellow": "注意", "green": "健康"}
    pool_configs = [
        ("short", "短线", 100),
        ("medium", "中线", 80),
        ("long", "长线", 60),
    ]

    st.subheader("\U0001f4ca 精筛池健康")
    col_h1, col_h2, col_h3 = st.columns(3)
    cols_by_idx = {0: col_h1, 1: col_h2, 2: col_h3}

    for i, (term, label, target_size) in enumerate(pool_configs):
        h = health.get(term, {})
        emoji = emoji_map.get(h.get("status", "green"), "\U0001f7e2")
        s_label = status_label.get(h.get("status", "green"), "健康")
        with cols_by_idx[i]:
            st.markdown(f"### {emoji} {label}({target_size}只)")
            last_upd = h.get("last_update", "从未更新")
            if isinstance(last_upd, str) and len(last_upd) >= 10:
                last_upd = last_upd[:10]
            st.caption(
                f"状态: **{s_label}**  |  "
                f"更新于: {last_upd}  |  "
                f"平均分: {h.get('details', {}).get('avg_score', 'N/A')}"
            )
            if h.get("triggers"):
                for t in h["triggers"]:
                    if h["status"] == "red":
                        st.error(t)
                    else:
                        st.warning(t)
            else:
                st.success("状态健康，无需操作")
