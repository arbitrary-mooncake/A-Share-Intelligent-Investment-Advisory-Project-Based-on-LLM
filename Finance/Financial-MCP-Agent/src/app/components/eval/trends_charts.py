"""趋势图组件 — Score / Loss / 收益对比 / 保真度 / 耗时"""
import streamlit as st
import pandas as pd


def render_trends_tabs(eval_ready: bool, status: dict) -> None:
    """渲染趋势图区域（5个子Tabs）"""
    st.markdown("---")
    st.subheader("\U0001f4c8 趋势图 — 系统表现随时间的变化")
    st.caption("每天运行'一键完整检查'会积累数据。至少积累10个批次后趋势图才会展示。")

    if eval_ready:
        try:
            from src.eval.chart_service import ChartService
            from src.eval.memory_manager import MemoryManager
            cs = ChartService()
            mm = MemoryManager()

            batch_count = len(mm.trends.get("score_history", []))
            min_batches = 10

            if batch_count < min_batches:
                st.info(
                    f"\U0001f4ca 数据积累中: 当前{batch_count}/{min_batches}批次 "
                    f"（至少需要{min_batches}批次才能展示趋势曲线）。"
                    f"每天运行'一键完整检查'会积累1个批次。"
                )
                st.progress(batch_count / min_batches, text=f"数据积累进度: {batch_count}/{min_batches}")
            else:
                tab_t1, tab_t2, tab_t3, tab_t4, tab_t5 = st.tabs([
                    "Score趋势", "Loss趋势", "线路收益对比", "保真度趋势", "运行耗时/Token"
                ])
                with tab_t1:
                    score_data = cs.get_score_trend_data(90)
                    if score_data["data"]:
                        df = pd.DataFrame(score_data["data"])
                        if not df.empty:
                            st.line_chart(df.set_index("date")["value"], use_container_width=True)
                            st.caption("Score越高越好 = 系统评分质量在提升。曲线向上 = 优化方向对了。")
                with tab_t2:
                    loss_data = cs.get_loss_trend_data(90)
                    if loss_data["data"]:
                        df = pd.DataFrame(loss_data["data"])
                        if not df.empty:
                            st.line_chart(df.set_index("date")["value"], use_container_width=True)
                            st.caption("Loss越低越好 = 预测与实际的差距在缩小。曲线向下 = 优化有效果。")
                with tab_t3:
                    if status.get("lines"):
                        line_data = cs.get_line_comparison_data(status["lines"])
                        if line_data["data"]:
                            df = pd.DataFrame(line_data["data"])
                            if not df.empty:
                                st.bar_chart(df.set_index("line_id")["return"], use_container_width=True)
                                st.caption("对比各条线的累计收益。消融线之间差异 = Agent贡献的直观体现。")
                with tab_t4:
                    fidelity_data = cs.get_fidelity_trend_data(90)
                    if fidelity_data["data"]:
                        df = pd.DataFrame(fidelity_data["data"])
                        if not df.empty and len(df) >= 2:
                            st.line_chart(df.set_index("date")["fidelity_loss"], use_container_width=True)
                            st.caption("保真度Loss越低 = 评测模型与生产模型输出越一致。")
                            with st.expander("\U0001f4cb 详细保真度指标", expanded=False):
                                st.dataframe(df.set_index("date"), use_container_width=True)
                        else:
                            st.info("保真度数据积累中（需要至少2个批次）。")
                with tab_t5:
                    runtime_data = cs.get_runtime_trend_data(90)
                    if runtime_data["data"]:
                        df = pd.DataFrame(runtime_data["data"])
                        if not df.empty and len(df) >= 2:
                            col_r1, col_r2 = st.columns(2)
                            with col_r1:
                                st.line_chart(df.set_index("date")["duration_minutes"], use_container_width=True)
                                st.caption("运行耗时（分钟）。持续上升 → 池子变大或Agent调用量增加。")
                            with col_r2:
                                st.line_chart(df.set_index("date")["cache_hit_rate"], use_container_width=True)
                                st.caption("缓存命中率%。越高越好 → 分析Agent结果被有效复用。")
                        else:
                            st.info("运行耗时数据积累中（需要至少2个批次）。")
        except Exception:
            st.caption("趋势数据加载中...")
    else:
        st.caption("评测系统初始化中...")
