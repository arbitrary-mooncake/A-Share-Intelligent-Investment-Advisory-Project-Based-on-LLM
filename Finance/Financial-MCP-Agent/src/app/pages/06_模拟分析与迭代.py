"""
模拟分析与迭代 — 第6大功能：评测控制台
Tab 1: 模拟盘运营（日常高频）
Tab 2: 回测与消融（研究分析）
Tab 3: 趋势与优化（长期追踪）
"""
import streamlit as st
from datetime import datetime
import sys, os, threading, time

st.set_page_config(page_title="模拟分析与迭代", page_icon="📈", layout="wide")
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from theme import inject_global_styles, page_title
from components.eval.pool_health import render_pool_health
from components.eval.ops_panel import render_ops_panel
from components.eval.holdings_table import render_holdings_table
from components.eval.backtest_panel import render_backtest_panel
from components.eval.contribution import render_contribution_leaderboard, render_optimization_tickets
from components.eval.trends_charts import render_trends_tabs
from components.shared_sidebar import render_sidebar

inject_global_styles()
render_sidebar()
page_title("📈 模拟分析与迭代")
st.caption("评分智能体 — 模拟盘评估、消融实验、回测、自动优化。所有结果仅供参考，使用评测模型(MiMo-V2.5)，目的是优化Agent系统。")


def _run_async(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


@st.cache_resource
def init_eval_system():
    from src.eval.database import init_db
    init_db()
    from src.eval.orchestrator import EvalOrchestrator
    return EvalOrchestrator()


try:
    orch = init_eval_system()
    status = orch.get_status()
    eval_ready = True
except Exception:
    eval_ready = False
    status = {}
    orch = None

# ── 顶部状态栏 ──
st.markdown("---")
col_v, col_b, col_t = st.columns(3)
with col_v:
    st.metric("系统版本", "agent-upgrade-v2")
with col_b:
    st.metric("最近检查", datetime.now().strftime("%Y-%m-%d %H:%M"))
with col_t:
    if eval_ready:
        latest = status.get("latest_batch")
        if latest:
            st.metric("最新批次", latest["batch_id"][:16], delta=latest["status"])
        else:
            st.metric("最新批次", "暂无批次")

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("总Score", status.get("total_score", "N/A"))
with col_s2:
    st.metric("短线Score", status.get("short_score", "N/A"))
with col_s3:
    st.metric("中线Score", status.get("medium_score", "N/A"))
with col_s4:
    st.metric("长线Score", status.get("long_score", "N/A"))

# ── 3 个顶层 Tabs ──
tab1, tab2, tab3 = st.tabs(["📊 模拟盘运营", "🔬 回测与消融", "📈 趋势与优化"])

with tab1:
    # 精筛池健康
    if eval_ready:
        health = orch.pool_manager.get_pool_health(
            lines_status=status.get("lines", [])
        )
        render_pool_health(health)

        # 对有问题的池显示快速更新入口
        problem_terms = [
            (term, label) for term, label, target_size in
            [("short", "短线", 100), ("medium", "中线", 80), ("long", "长线", 60)]
            if health.get(term, {}).get("status") in ("red", "yellow")
        ]
        if problem_terms:
            st.markdown("---")
            st.caption("以下精筛池需要关注，可快速更新：")
            action_cols = st.columns(len(problem_terms))
            emoji_map = {"red": "🔴", "yellow": "🟡", "green": "🟢"}
            for j, (term, label) in enumerate(problem_terms):
                h = health[term]
                with action_cols[j]:
                    st.caption(
                        f"{emoji_map[h['status']]} **{label}池**: "
                        f"{h.get('suggested_action', '')}"
                    )
                    if st.button(
                        f"🔄 全量更新{label}池", key=f"health_full_{term}",
                        use_container_width=True,
                    ):
                        # ── V3 流式进度 + ETA ──
                        progress_bar = st.progress(0, "Layer 0: 硬筛中...")
                        status_col1, status_col2, status_col3 = st.columns(3)
                        eta_display = st.empty()
                        stage_lines = st.empty()

                        latest_progress = {"overall_pct": 0, "eta_str": "计算中...",
                                          "stages": {}, "stall_s": 0}
                        progress_lock = threading.Lock()

                        def on_progress(data):
                            with progress_lock:
                                latest_progress.update(data)

                        def on_stage(stage_name, message):
                            pass  # stage transitions shown via progress data

                        from src.eval.web_adapter import WebAdapter
                        adapter = WebAdapter(orch)
                        holder = adapter.run_pool_update_streaming(
                            term, on_progress=on_progress, on_stage=on_stage
                        )

                        # 轮询进度, 更新 UI
                        while not holder.get("done", False):
                            with progress_lock:
                                pct = latest_progress.get("overall_pct", 0) / 100.0
                                eta = latest_progress.get("eta_str", "计算中...")
                                stages = latest_progress.get("stages", {})
                                stall = latest_progress.get("stall_s", 0)
                            progress_bar.progress(
                                min(pct, 1.0),
                                f"总进度 {pct*100:.0f}% | ETA: {eta}"
                                + (f" | ⚠️ 卡顿{stall:.0f}s" if stall > 60 else "")
                            )
                            # 阶段详情
                            stage_text = ""
                            for k, v in stages.items():
                                if v.get("done", 0) > 0:
                                    stage_text += (
                                        f"**{v['label']}**: {v['pct']:.0f}% "
                                        f"({v.get('done', 0)}/{v.get('total', 0)})  "
                                    )
                            stage_lines.markdown(stage_text or "准备中...")
                            eta_display.caption(
                                f"已运行 {(latest_progress.get('elapsed_s', 0) // 60)}min"
                                f" | ETA: {eta}"
                                f" | 队列: {latest_progress.get('queue_depth', 0)}只"
                            )
                            time.sleep(1.0)

                        # 完成
                        update_result = holder.get("result", {})
                        if holder.get("error"):
                            st.error(holder["error"])
                        elif "error" in update_result:
                            st.error(update_result["error"])
                        else:
                            progress_bar.progress(1.0, "完成!")
                            st.success(
                                f"更新完成！共 "
                                f"{update_result.get('final_pool_size', 0)} 只"
                                f" | 耗时 {update_result.get('stats', {}).get('elapsed_s', 0) // 60}min"
                            )
                            st.rerun()
    else:
        st.info("精筛池未初始化，请先运行评测系统。")

    # 操作面板
    st.markdown("---")
    render_ops_panel(orch, eval_ready)

    # 精筛池概览
    st.markdown("---")
    st.subheader("📋 精筛股票池 — 各期限的候选股票库")
    if eval_ready:
        pools = status.get("pools", {})
        col_sp, col_mp, col_lp = st.columns(3)
        for col, (term, label) in zip([col_sp, col_mp, col_lp],
                                       [("short", "短线"), ("medium", "中线"), ("long", "长线")]):
            with col:
                pool = pools.get(term, {})
                st.metric(f"{label}池", f"{pool.get('size', 0)}只",
                          delta=f"目标{pool.get('target_size', '-')}只")
        st.caption("精筛池 = 模拟盘的选股范围。14条线只能从这些池子里选股。")

        with st.expander("📋 精筛池 Top5 成分股", expanded=False):
            for term, label in [("short", "短线"), ("medium", "中线"), ("long", "长线")]:
                st.caption(f"**{label}精筛池 Top5:**")
                pool_data = orch.pool_manager.get_pool_with_scores(term)[:5]
                if pool_data:
                    for s in pool_data:
                        code = s.get("code", "?") if isinstance(s, dict) else s
                        name = s.get("name", "?") if isinstance(s, dict) else "?"
                        score = s.get("score", "?") if isinstance(s, dict) else "?"
                        st.text(f"  {code} ({name}) | {score}分")
                else:
                    st.text("  暂无数据")
    else:
        st.warning("评测系统初始化中...")

    # 持仓收益
    if eval_ready:
        lines = status.get("lines", [])
        render_holdings_table(lines)
    else:
        st.caption("评测系统初始化中...")

    # Tab 1 底部声明
    st.markdown("---")
    st.warning("⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考，目的是优化Agent系统而非追求极限投资收益。不构成任何投资建议。市场有风险，投资需谨慎。")

with tab2:
    if eval_ready:
        render_backtest_panel(orch, eval_ready)
    else:
        st.warning("评测系统初始化中...")

    render_contribution_leaderboard(eval_ready)

    # Tab 2 底部声明
    st.markdown("---")
    st.warning("⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考。不构成任何投资建议。市场有风险，投资需谨慎。")

with tab3:
    if eval_ready:
        render_trends_tabs(eval_ready, status)
    else:
        st.caption("评测系统初始化中...")

    render_optimization_tickets(eval_ready)

    # Tab 3 底部声明
    st.markdown("---")
    st.warning("⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考。不构成任何投资建议。市场有风险，投资需谨慎。")
