"""各线路持仓与收益表格组件"""
import streamlit as st
import pandas as pd


def render_holdings_table(lines: list) -> None:
    """渲染14条模拟盘持仓与收益表格。

    Args:
        lines: status.get("lines", []) 返回的线路数据列表
    """
    st.markdown("---")
    st.subheader("\U0001f4ca 各线路持仓与收益 — 14条模拟盘实时状态")

    st.caption("""
    **线路说明**: S-L0~S-L7 = 短线消融实验（每天同一起点出发，各差1个Agent）。
    S-L8 = 短线正常交易（连续持仓）。S-L9/M-L1/L-L1 = LLM自主投资对照组。M-L0/L-L0 = 中/长线。
    """)

    tab_s, tab_m, tab_l = st.tabs(["⚡ 短线 (10条)", "\U0001f4c5 中线 (2条)", "\U0001f3db️ 长线 (2条)"])

    for tab, term in [(tab_s, "short"), (tab_m, "medium"), (tab_l, "long")]:
        with tab:
            term_lines = [l for l in lines if l.get("term") == term]
            if term_lines:
                df_data = []
                for l in term_lines:
                    df_data.append({
                        "线路": l["line_id"],
                        "类型": l.get("type", ""),
                        "持仓数": l.get("holdings_count", 0),
                        "总市值(万)": round(l.get("total_value", 0) / 10000, 1),
                        "累计收益%": l.get("cumulative_return_pct", 0),
                        "最大回撤%": l.get("max_drawdown_pct", 0),
                        "今日收益%": l.get("daily_return_pct", 0),
                    })
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)
            else:
                st.info("\U0001f446 还没数据。先更新精筛池，再点'一键完整检查'，这里就会显示各条线的实时持仓和收益率。")
