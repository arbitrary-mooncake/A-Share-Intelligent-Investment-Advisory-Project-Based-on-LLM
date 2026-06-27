"""Agent贡献榜 + 优化建议组件"""
import streamlit as st
import pandas as pd


def render_contribution_leaderboard(eval_ready: bool) -> None:
    """渲染 Agent 贡献榜 — 谁在帮忙，谁在拖后腿"""
    st.markdown("---")
    st.subheader("\U0001f3c6 Agent贡献分 — 谁在帮忙，谁在拖后腿")
    st.caption("ΔL > 0 = 正贡献（去掉它系统变差）。ΔL < 0 = 负贡献（去掉它系统反而变好）。★★★ = 统计显著。☆ = 不显著。")

    if eval_ready:
        try:
            from src.eval.memory_manager import MemoryManager
            mm = MemoryManager()
            contrib_trend = mm.trends.get("contribution_history", [])
            if contrib_trend:
                agent_stats = {}
                for c in contrib_trend[-50:]:
                    name = c.get("agent_name", "")
                    if name not in agent_stats:
                        agent_stats[name] = {"deltas": [], "stars": []}
                    agent_stats[name]["deltas"].append(c.get("delta_L_total", 0))
                    agent_stats[name]["stars"].append(c.get("stars", ""))

                contrib_data = []
                for name, stats in sorted(agent_stats.items(),
                                           key=lambda x: sum(x[1]["deltas"])/max(len(x[1]["deltas"]),1),
                                           reverse=True):
                    avg_delta = sum(stats["deltas"]) / max(len(stats["deltas"]), 1)
                    last_stars = stats["stars"][-1] if stats["stars"] else "☆"
                    contrib_data.append({
                        "Agent": name, "平均ΔL": round(avg_delta, 4),
                        "样本数": len(stats["deltas"]), "最新显著性": last_stars,
                        "评价": "\U0001f44d 正贡献" if avg_delta > 0.01 else ("\U0001f44e 负贡献" if avg_delta < -0.01 else "➖ 中性"),
                    })
                if contrib_data:
                    st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)
                else:
                    st.info("暂无Agent贡献数据。运行回测或积累足够实盘数据后这里会自动填充。")
            else:
                st.info("暂无贡献记录。点'历史回测'跑一次，或每天运行'一键完整检查'积累实盘数据。")
        except Exception as e:
            st.caption(f"加载中: {e}")
    else:
        st.caption("评测系统初始化中...")


def render_optimization_tickets(eval_ready: bool) -> None:
    """渲染优化建议列表"""
    st.markdown("---")
    st.subheader("\U0001f527 优化建议 — 基于Loss和贡献数据自动生成")
    st.caption("""
    \U0001f916 自动 = 参数调整类。\U0001f464\U0001f916 半自动 = 系统生成方案，需你审核后执行。\U0001f464 人工 = 系统出分析报告，需你手动改代码。
    运行 `python -m src.eval optimize --analyze` 生成新建议。
    """)

    if eval_ready:
        try:
            from src.eval.repositories import get_pending_tickets
            tickets = get_pending_tickets()
            if tickets:
                for t in tickets[:5]:
                    route_icon = {"auto": "\U0001f916", "semi_auto": "\U0001f464\U0001f916", "manual": "\U0001f464"}.get(t.get("route", ""), "❓")
                    with st.expander(f"{route_icon} [{t.get('ticket_type', '')}] {t.get('title', '未命名')}"):
                        st.write(t.get("summary", "无摘要"))
                        st.caption(f"路由: {t.get('route', '')} | 严重度: {t.get('severity', '')}")
            else:
                st.info("暂无优化建议。运行 `python -m src.eval optimize --analyze` 生成。")
        except Exception as e:
            st.caption(f"加载中: {e}")
    else:
        st.caption("评测系统初始化中...")
