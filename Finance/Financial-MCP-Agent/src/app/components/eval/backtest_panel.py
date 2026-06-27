"""回测面板组件 — 参数配置 + 运行 + 结果展示 + 市场环境切片"""
import streamlit as st
from datetime import datetime
import pandas as pd


def _run_async(coro):
    """Safely run async coroutine."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


def render_backtest_panel(orch, eval_ready: bool) -> None:
    """渲染回测面板：参数配置 + 运行按钮 + 结果表格 + 市场环境切片"""
    st.subheader("⚙️ 历史回测")
    st.caption("回测 = 在历史数据上重放系统，快速知道哪个Agent帮忙、哪个拖后腿。")
    st.caption("⚠️ 回测用当前精筛池，不含历史上退市/暴雷股。news/event agent在回测中不可用。结果带幸存者偏差，绝对收益仅供参考。")

    col_bt1, col_bt2, col_bt3 = st.columns(3)
    with col_bt1:
        bt_term = st.selectbox("回测期限", ["short", "medium", "long"], index=1)
    with col_bt2:
        bt_start = st.date_input("起始日期", value=datetime(2024, 1, 1))
    with col_bt3:
        bt_end = st.date_input("结束日期", value=datetime(2025, 12, 31))

    st.caption("消融对象: 勾选哪些Agent就测哪些（回测中news/event不可用）")
    col_ab1, col_ab2, col_ab3 = st.columns(3)
    with col_ab1:
        ab_fund = st.checkbox("fundamental", value=True)
        ab_tech = st.checkbox("technical", value=True)
    with col_ab2:
        ab_value = st.checkbox("value", value=True)
        ab_quality = st.checkbox("quality_risk", value=True)
    with col_ab3:
        ab_money = st.checkbox("moneyflow", value=True)

    regime_enabled = st.checkbox("\U0001f52c 按市场环境切片分析（牛/熊/震荡分别统计）", value=False,
                                help="总纲 §8.5: 对同一回测在牛市/熊市/震荡市分别验证")

    if st.button("▶️ 开始回测", type="primary"):
        if eval_ready:
            progress_bar = st.progress(0, text="回测初始化...")
            status_box = st.status("回测运行中...", expanded=True)
            try:
                from src.eval.replay_backtest_engine import ReplayBacktestEngine, BacktestConfig
                term_map = {"short": "short", "medium": "medium", "long": "long"}
                cfg = BacktestConfig(
                    term=term_map.get(bt_term, "medium"),
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=bt_end.strftime("%Y-%m-%d"),
                )
                ablation_list = []
                if ab_fund: ablation_list.append("fundamental")
                if ab_tech: ablation_list.append("technical")
                if ab_value: ablation_list.append("value")
                if ab_quality: ablation_list.append("quality_risk")
                if ab_money: ablation_list.append("moneyflow")
                cfg.ablation_agents = ablation_list

                engine = ReplayBacktestEngine(cfg)
                anchors = engine.generate_anchor_dates()
                progress_bar.progress(10, text=f"共{len(anchors)}个回测时点")

                pool = orch.pool_manager.get_pool(bt_term)
                async def _run_bt():
                    return await engine.run_full_backtest(pool, {})
                bt_result = _run_async(_run_bt())

                progress_bar.progress(90, text="汇总贡献...")
                contrib = bt_result.get("contribution_summary", {})

                status_box.update(label=f"回测完成: {len(anchors)}个时点", state="complete")
                progress_bar.progress(100, text="完成!")

                if contrib:
                    st.subheader("\U0001f4ca Agent贡献结果")
                    contrib_rows = []
                    for agent, info in sorted(contrib.items(),
                                            key=lambda x: x[1].get("mean_delta_L", 0),
                                            reverse=True):
                        delta = info.get("mean_delta_L", 0)
                        direction = info.get("direction", "neutral")
                        stars = "★★★" if delta > 0.03 else ("★★" if delta > 0.01 else ("☆" if delta > -0.01 else "↓"))
                        contrib_rows.append({
                            "Agent": agent,
                            "ΔL": f"{delta:+.4f}",
                            "方向": "\U0001f44d正贡献" if direction == "positive" else "\U0001f44e负贡献",
                            "显著性": stars,
                            "样本数": info.get("sample_size", 0),
                        })
                    st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True)

                    ref = bt_result.get("reference_line")
                    if ref and "error" not in ref:
                        with st.expander("\U0001f4c8 长持对照线 (SB-L6) — 现实性校验", expanded=False):
                            col_r1, col_r2, col_r3 = st.columns(3)
                            col_r1.metric("累计收益", f"{ref.get('cumulative_return_pct', 0)}%")
                            col_r2.metric("最大回撤", f"{ref.get('max_drawdown_pct', 0)}%")
                            col_r3.metric("Sharpe", f"{ref.get('sharpe_ratio', 0)}")
                            st.caption("SB-L6使用成熟短线策略（连续持仓），不参与消融ΔLoss计算。")

                for decl in bt_result.get("declarations", []):
                    st.info(f"⚠️ {decl}")

                if regime_enabled:
                    progress_bar.progress(95, text="市场环境切片分析...")
                    status_box.update(label="运行市场环境切片分析（牛/熊/震荡）...", state="running")
                    try:
                        regime_result = _run_async(
                            engine.run_regime_analysis(pool, {})
                        )
                        if regime_result:
                            with st.expander("\U0001f52c 市场环境切片 — Agent贡献在不同市场环境下的表现", expanded=True):
                                st.caption("同一优化结论在不同市场环境下是否依然成立？")
                                agg = regime_result.get("aggregate", {})
                                col_bull, col_bear, col_range = st.columns(3)
                                col_bull.metric("\U0001f402 牛市anchors", agg.get("bull_anchors", 0))
                                col_bear.metric("\U0001f43b 熊市anchors", agg.get("bear_anchors", 0))
                                col_range.metric("\U0001f4ca 震荡市anchors", agg.get("ranging_anchors", 0))

                                for regime_label, regime_emoji in [("bull", "\U0001f402 牛市"), ("bear", "\U0001f43b 熊市"), ("ranging", "\U0001f4ca 震荡市")]:
                                    rdata = regime_result.get(regime_label, {})
                                    rcontrib = rdata.get("contribution_summary", {})
                                    if rcontrib:
                                        st.caption(f"**{regime_emoji}** ({rdata.get('num_anchors', 0)}个时点):")
                                        regime_rows = []
                                        for agent, info in sorted(rcontrib.items(),
                                                                 key=lambda x: x[1].get("mean_delta_L", 0),
                                                                 reverse=True):
                                            regime_rows.append({
                                                "Agent": agent,
                                                "ΔL": f"{info.get('mean_delta_L', 0):+.4f}",
                                                "方向": info.get("direction", "neutral"),
                                            })
                                        st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"市场环境切片分析失败: {e}")
            except Exception as e:
                progress_bar.progress(100, text="失败")
                status_box.update(label=f"回测失败", state="error")
                st.error(f"回测失败: {e}")
