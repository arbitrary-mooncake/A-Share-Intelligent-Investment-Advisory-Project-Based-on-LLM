"""回测面板组件 — 参数配置 + 运行 + 结果展示 + 市场环境切片"""
import json
import os
import time
import streamlit as st
from datetime import datetime
import pandas as pd


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_STRATEGY_DEFAULTS_PATH = os.path.join(_PROJECT_ROOT, "config", "eval", "strategy_defaults.json")
_BACKTEST_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "data", "eval", "backtest_results")


def _load_strategy_defaults() -> dict:
    """加载策略参数默认值"""
    if os.path.exists(_STRATEGY_DEFAULTS_PATH):
        try:
            with open(_STRATEGY_DEFAULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


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


def _save_backtest_result(result: dict, cfg: dict) -> str:
    """保存回测结果到JSON文件，返回文件路径"""
    os.makedirs(_BACKTEST_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{cfg.get('term', 'medium')}_{timestamp}.json"
    filepath = os.path.join(_BACKTEST_RESULTS_DIR, filename)

    # 提取可序列化的结果
    serializable = {
        "config": cfg,
        "contribution_summary": result.get("contribution_summary", {}),
        "timestamp": datetime.now().isoformat(),
        "num_anchors": result.get("config", {}).get("num_anchors_executed", 0),
        "declarations": result.get("declarations", []),
    }
    # 包含参考线结果
    if result.get("reference_line") and "error" not in result.get("reference_line", {}):
        serializable["reference_line"] = {
            k: v for k, v in result["reference_line"].items()
            if k in ("cumulative_return_pct", "max_drawdown_pct", "sharpe_ratio",
                     "annualized_return_pct", "win_rate_pct", "num_anchors",
                     "line_id", "description")
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    return filepath


def _load_last_backtest_result(term: str):
    """加载上次同期限的回测结果用于对比"""
    if not os.path.isdir(_BACKTEST_RESULTS_DIR):
        return None
    matching = sorted(
        [f for f in os.listdir(_BACKTEST_RESULTS_DIR)
         if f.startswith(f"backtest_{term}_") and f.endswith(".json")],
        key=lambda f: os.path.getmtime(os.path.join(_BACKTEST_RESULTS_DIR, f)),
        reverse=True,
    )
    if matching:
        try:
            with open(os.path.join(_BACKTEST_RESULTS_DIR, matching[0]), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


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

    # ═══════════════════════════════════════════════════════════
    # Gap #11: 策略参数覆盖
    # ═══════════════════════════════════════════════════════════
    defaults = _load_strategy_defaults()
    param_overrides = {}

    with st.expander("⚙️ 策略参数覆盖 (可选)", expanded=False):
        st.caption("覆盖回测引擎中的默认策略参数。留空则使用系统默认值。")

        # 根据选择期限显示对应参数
        short_params = {}
        medium_params = {}
        long_params = {}

        # 短线参数
        if bt_term == "short":
            st.markdown("**短线策略参数**")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                short_defaults = defaults.get("strategy_short_longhold", {})
                short_params["score_buy_threshold"] = st.number_input(
                    "买入评分阈值", min_value=30, max_value=90,
                    value=short_defaults.get("score_buy_threshold_short", 60),
                    step=5, key="bt_score_buy_short",
                    help="评分高于此值才可买入",
                )
            with col_s2:
                short_params["single_weight_limit"] = st.number_input(
                    "单只权重上限", min_value=0.05, max_value=0.50,
                    value=float(short_defaults.get("single_weight_limit_short_longhold", 0.12)),
                    step=0.01, format="%.2f", key="bt_single_weight_short",
                    help="单只股票占总仓位的最大比例",
                )
            with col_s3:
                short_params["max_hold_days"] = st.number_input(
                    "最大持有时长(天)", min_value=5, max_value=60,
                    value=short_defaults.get("max_hold_days_short", 20),
                    step=5, key="bt_max_hold_short",
                )

        # 中线参数
        elif bt_term == "medium":
            st.markdown("**中线策略参数**")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                medium_defaults = defaults.get("strategy_medium_term", {})
                medium_params["score_buy_threshold"] = st.number_input(
                    "买入评分阈值", min_value=40, max_value=90,
                    value=medium_defaults.get("score_buy_threshold_medium", 65),
                    step=5, key="bt_score_buy_medium",
                )
            with col_m2:
                medium_params["single_weight_limit"] = st.number_input(
                    "单只权重上限", min_value=0.05, max_value=0.50,
                    value=float(medium_defaults.get("single_weight_limit_medium", 0.18)),
                    step=0.01, format="%.2f", key="bt_single_weight_medium",
                )
            with col_m3:
                medium_params["max_hold_days"] = st.number_input(
                    "最大持有时长(天)", min_value=10, max_value=120,
                    value=medium_defaults.get("max_hold_days_medium", 60),
                    step=5, key="bt_max_hold_medium",
                )

        # 长线参数
        elif bt_term == "long":
            st.markdown("**长线策略参数**")
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                long_defaults = defaults.get("strategy_long_term", {})
                long_params["score_buy_threshold"] = st.number_input(
                    "买入评分阈值", min_value=50, max_value=95,
                    value=long_defaults.get("score_buy_threshold_long", 70),
                    step=5, key="bt_score_buy_long",
                )
            with col_l2:
                long_params["max_drawdown_tolerance"] = st.number_input(
                    "最大回撤容忍度", min_value=0.10, max_value=0.50,
                    value=0.25, step=0.05, format="%.2f", key="bt_drawdown_long",
                    help="持仓回撤超过此值触发止损",
                )
            with col_l3:
                long_params["position_building_weeks"] = st.number_input(
                    "建仓周数", min_value=1, max_value=12,
                    value=4, step=1, key="bt_build_weeks_long",
                    help="分批次建仓的周数",
                )

        # "恢复默认" 按钮
        col_reset1, col_reset2 = st.columns([1, 4])
        with col_reset1:
            if st.button("🔄 恢复默认", key="bt_reset_params", use_container_width=True):
                st.rerun()

        # 收集选定的覆盖参数
        all_params = {**short_params, **medium_params, **long_params}
        # 只传递非默认值（用户实际修改过的）
        param_overrides = {k: v for k, v in all_params.items() if v is not None}

    # ═══════════════════════════════════════════════════════════
    # Gap #23 + #24: 带取消状态的回测运行 + ETA
    # ═══════════════════════════════════════════════════════════
    if st.button("▶️ 开始回测", type="primary"):
        if eval_ready:
            # 取消状态初始化
            if "bt_cancel_requested" not in st.session_state:
                st.session_state.bt_cancel_requested = False

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
            total_anchors = len(anchors)
            pool = orch.pool_manager.get_pool(bt_term)

            # ── Gap #24: ETA估算 ──
            # 每锚点预估耗时 ~0.5-3秒 (取决于缓存命中率)
            eta_per_anchor = 0.5 if bt_term == "long" else 1.5
            eta_total_s = total_anchors * eta_per_anchor
            eta_str = f"{int(eta_total_s // 60)}分{int(eta_total_s % 60)}秒" if eta_total_s >= 60 else f"{int(eta_total_s)}秒"

            progress_bar = st.progress(0, text=f"回测初始化... 共{total_anchors}个时点 | 预计 {eta_str}")
            status_box = st.status(f"回测中... 0/{total_anchors} 个时点 (0%) | 预计剩余 {eta_str}", expanded=True)

            st.session_state.bt_cancel_requested = False

            try:
                bt_start_time = time.time()

                async def _run_bt():
                    return await engine.run_full_backtest(pool, {}, param_overrides=param_overrides)
                bt_result = _run_async(_run_bt())

                elapsed_total = time.time() - bt_start_time
                elapsed_str = f"{int(elapsed_total // 60)}分{int(elapsed_total % 60)}秒"

                if st.session_state.bt_cancel_requested:
                    status_box.update(label="回测已取消", state="complete")
                    progress_bar.progress(100, text="已取消")
                    st.warning("回测已被用户取消。结果已丢弃。")
                    st.session_state.bt_cancel_requested = False
                    return

                status_box.update(
                    label=f"回测完成: {total_anchors}个时点 | 耗时 {elapsed_str} (预估 {eta_str})",
                    state="complete"
                )
                progress_bar.progress(100, text=f"完成! ({elapsed_str})")

                # 保存回测结果
                saved_path = _save_backtest_result(bt_result, {
                    "term": bt_term,
                    "start_date": bt_start.strftime("%Y-%m-%d"),
                    "end_date": bt_end.strftime("%Y-%m-%d"),
                    "ablation_agents": ablation_list,
                    "regime_enabled": regime_enabled,
                    "param_overrides": param_overrides,
                })

                # ============================================================
                # 存储结果到 session_state 以供持久展示（修复按钮消失 bug）
                # ============================================================

                # 市场环境切片（如果启用，在引擎可用时执行）
                regime_display = None
                if regime_enabled:
                    try:
                        regime_result = _run_async(
                            engine.run_regime_analysis(pool, {})
                        )
                        if regime_result:
                            regime_display = {
                                "aggregate": regime_result.get("aggregate", {}),
                                "bull": regime_result.get("bull", {}),
                                "bear": regime_result.get("bear", {}),
                                "ranging": regime_result.get("ranging", {}),
                            }
                    except Exception as e:
                        regime_display = {"error": str(e)}

                st.session_state["bt_display"] = {
                    "contrib": bt_result.get("contribution_summary", {}),
                    "reference_line": bt_result.get("reference_line"),
                    "declarations": bt_result.get("declarations", []),
                    "saved_path": saved_path,
                    "bt_term": bt_term,
                    "regime_enabled": regime_enabled,
                    "regime_display": regime_display,
                }
                st.session_state["_bt_shown_inline"] = True

                contrib = bt_result.get("contribution_summary", {})

                # ═══════════════════════════════════════════════════════════
                # Gap #12: Export / Compare 操作按钮
                # ═══════════════════════════════════════════════════════════
                col_export1, col_export2 = st.columns(2)

                with col_export1:
                    # 导出JSON
                    try:
                        export_json = json.dumps(bt_result.get("contribution_summary", {}),
                                                ensure_ascii=False, indent=2, default=str)
                        st.download_button(
                            label="📥 导出JSON",
                            data=export_json,
                            file_name=f"backtest_{bt_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True,
                            key="bt_export_json",
                        )
                    except Exception:
                        st.caption("导出JSON失败")

                with col_export2:
                    # 对比上次回测
                    if st.button("📊 对比上次回测", key="bt_compare", use_container_width=True):
                        last_result = _load_last_backtest_result(bt_term)
                        # Skip the current result itself
                        all_bt_files = sorted(
                            [f for f in os.listdir(_BACKTEST_RESULTS_DIR)
                             if f.startswith(f"backtest_{bt_term}_") and f.endswith(".json")],
                            key=lambda f: os.path.getmtime(os.path.join(_BACKTEST_RESULTS_DIR, f)),
                            reverse=True,
                        )
                        # Find the second most recent (skip the one just saved)
                        prev_contrib = None
                        prev_ts = None
                        for f in all_bt_files[1:]:
                            try:
                                fp = os.path.join(_BACKTEST_RESULTS_DIR, f)
                                with open(fp, "r", encoding="utf-8") as fh:
                                    prev = json.load(fh)
                                prev_contrib = prev.get("contribution_summary", {})
                                prev_ts = prev.get("timestamp", "")
                                break
                            except Exception:
                                continue

                        if prev_contrib and contrib:
                            st.markdown("---")
                            st.subheader("📊 回测对比 — 本次 vs 上次")
                            st.caption(f"上次回测: {prev_ts[:19] if prev_ts else '未知时间'}")

                            diff_rows = []
                            all_agents = sorted(set(list(contrib.keys()) + list(prev_contrib.keys())))
                            for agent in all_agents:
                                curr_info = contrib.get(agent, {}) if isinstance(contrib.get(agent, {}), dict) else {}
                                prev_info = prev_contrib.get(agent, {}) if isinstance(prev_contrib.get(agent, {}), dict) else {}
                                curr_delta = curr_info.get("mean_delta_L", 0)
                                prev_delta = prev_info.get("mean_delta_L", 0)
                                diff = curr_delta - prev_delta
                                trend = "📈 改善" if diff > 0.005 else ("📉 退化" if diff < -0.005 else "➖ 持平")
                                diff_rows.append({
                                    "Agent": agent,
                                    "本次ΔL": f"{curr_delta:+.4f}",
                                    "上次ΔL": f"{prev_delta:+.4f}",
                                    "变化": f"{diff:+.4f}",
                                    "趋势": trend,
                                })

                            if diff_rows:
                                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
                            else:
                                st.info("无法计算差异（Agent列表不一致）")
                        else:
                            st.info("上次回测无贡献数据可供对比。")

                # ── 贡献结果展示 ──
                if contrib:
                    st.subheader("\U0001f4ca Agent贡献结果")
                    contrib_rows = []
                    for agent, info in sorted(contrib.items(),
                                            key=lambda x: x[1].get("mean_delta_L", 0) if isinstance(x[1], dict) else 0,
                                            reverse=True):
                        if not isinstance(info, dict):
                            continue
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

                # ── 市场环境切片 (inline rendering from already-computed regime_display) ──
                if regime_enabled and regime_display:
                    if regime_display.get("error"):
                        st.warning(f"市场环境切片分析失败: {regime_display['error']}")
                    else:
                        with st.expander("\U0001f52c 市场环境切片 — Agent贡献在不同市场环境下的表现", expanded=True):
                            st.caption("同一优化结论在不同市场环境下是否依然成立？")
                            agg = regime_display.get("aggregate", {})
                            col_bull, col_bear, col_range = st.columns(3)
                            col_bull.metric("\U0001f402 牛市anchors", agg.get("bull_anchors", 0))
                            col_bear.metric("\U0001f43b 熊市anchors", agg.get("bear_anchors", 0))
                            col_range.metric("\U0001f4ca 震荡市anchors", agg.get("ranging_anchors", 0))

                            for regime_label, regime_emoji in [("bull", "\U0001f402 牛市"), ("bear", "\U0001f43b 熊市"), ("ranging", "\U0001f4ca 震荡市")]:
                                rdata = regime_display.get(regime_label, {})
                                rcontrib = rdata.get("contribution_summary", {})
                                if rcontrib:
                                    st.caption(f"**{regime_emoji}** ({rdata.get('num_anchors', 0)}个时点):")
                                    regime_rows = []
                                    for agent, info in sorted(rcontrib.items(),
                                                             key=lambda x: x[1].get("mean_delta_L", 0) if isinstance(x[1], dict) else 0,
                                                             reverse=True):
                                        if not isinstance(info, dict):
                                            continue
                                        regime_rows.append({
                                            "Agent": agent,
                                            "ΔL": f"{info.get('mean_delta_L', 0):+.4f}",
                                            "方向": info.get("direction", "neutral"),
                                        })
                                    st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

                # 保存路径提示
                st.caption(f"结果已保存: `{saved_path}`")

            except Exception as e:
                progress_bar.progress(100, text="失败")
                status_box.update(label=f"回测失败", state="error")
                st.error(f"回测失败: {e}")

    # ============================================================
    # 持久化展示：下载 / 对比 / 贡献 / 参考线 / 市场切片
    # 通过 session_state 驱动，不受 st.button 生命周期影响
    # ============================================================
    if "bt_display" in st.session_state and not st.session_state.pop("_bt_shown_inline", False):
        display = st.session_state["bt_display"]
        contrib = display["contrib"]
        bt_term_d = display["bt_term"]
        regime_enabled_d = display["regime_enabled"]

        # --- Gap #12: Export / Compare ---
        col_export1, col_export2 = st.columns(2)

        with col_export1:
            try:
                export_json = json.dumps(contrib, ensure_ascii=False, indent=2, default=str)
                st.download_button(
                    label="📥 导出JSON",
                    data=export_json,
                    file_name=f"backtest_{bt_term_d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="bt_export_json",
                )
            except Exception:
                st.caption("导出JSON失败")

        with col_export2:
            if st.button("📊 对比上次回测", key="bt_compare", use_container_width=True):
                all_bt_files = sorted(
                    [f for f in os.listdir(_BACKTEST_RESULTS_DIR)
                     if f.startswith(f"backtest_{bt_term_d}_") and f.endswith(".json")],
                    key=lambda f: os.path.getmtime(os.path.join(_BACKTEST_RESULTS_DIR, f)),
                    reverse=True,
                )
                prev_contrib = None
                prev_ts = None
                for f in all_bt_files[1:]:
                    try:
                        fp = os.path.join(_BACKTEST_RESULTS_DIR, f)
                        with open(fp, "r", encoding="utf-8") as fh:
                            prev = json.load(fh)
                        prev_contrib = prev.get("contribution_summary", {})
                        prev_ts = prev.get("timestamp", "")
                        break
                    except Exception:
                        continue

                if prev_contrib and contrib:
                    st.markdown("---")
                    st.subheader("📊 回测对比 — 本次 vs 上次")
                    st.caption(f"上次回测: {prev_ts[:19] if prev_ts else '未知时间'}")

                    diff_rows = []
                    all_agents = sorted(set(list(contrib.keys()) + list(prev_contrib.keys())))
                    for agent in all_agents:
                        curr_info = contrib.get(agent, {}) if isinstance(contrib.get(agent, {}), dict) else {}
                        prev_info = prev_contrib.get(agent, {}) if isinstance(prev_contrib.get(agent, {}), dict) else {}
                        curr_delta = curr_info.get("mean_delta_L", 0)
                        prev_delta = prev_info.get("mean_delta_L", 0)
                        diff = curr_delta - prev_delta
                        trend = "📈 改善" if diff > 0.005 else ("📉 退化" if diff < -0.005 else "➖ 持平")
                        diff_rows.append({
                            "Agent": agent,
                            "本次ΔL": f"{curr_delta:+.4f}",
                            "上次ΔL": f"{prev_delta:+.4f}",
                            "变化": f"{diff:+.4f}",
                            "趋势": trend,
                        })

                    if diff_rows:
                        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("无法计算差异（Agent列表不一致）")
                else:
                    st.info("上次回测无贡献数据可供对比。")

        # --- 贡献结果展示 ---
        if contrib:
            st.subheader("📊 Agent贡献结果")
            contrib_rows = []
            for agent, info in sorted(contrib.items(),
                                    key=lambda x: x[1].get("mean_delta_L", 0) if isinstance(x[1], dict) else 0,
                                    reverse=True):
                if not isinstance(info, dict):
                    continue
                delta = info.get("mean_delta_L", 0)
                direction = info.get("direction", "neutral")
                stars = "★★★" if delta > 0.03 else ("★★" if delta > 0.01 else ("☆" if delta > -0.01 else "↓"))
                contrib_rows.append({
                    "Agent": agent,
                    "ΔL": f"{delta:+.4f}",
                    "方向": "👍正贡献" if direction == "positive" else "👎负贡献",
                    "显著性": stars,
                    "样本数": info.get("sample_size", 0),
                })
            st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True)

            ref = display.get("reference_line")
            if ref and "error" not in ref:
                with st.expander("📈 长持对照线 (SB-L6) — 现实性校验", expanded=False):
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("累计收益", f"{ref.get('cumulative_return_pct', 0)}%")
                    col_r2.metric("最大回撤", f"{ref.get('max_drawdown_pct', 0)}%")
                    col_r3.metric("Sharpe", f"{ref.get('sharpe_ratio', 0)}")
                    st.caption("SB-L6使用成熟短线策略（连续持仓），不参与消融ΔLoss计算。")

        for decl in display.get("declarations", []):
            st.info(f"⚠️ {decl}")

        # --- 市场环境切片 ---
        regime_display = display.get("regime_display")
        if regime_enabled_d and regime_display:
            if regime_display.get("error"):
                st.warning(f"市场环境切片分析失败: {regime_display['error']}")
            else:
                with st.expander("🔬 市场环境切片 — Agent贡献在不同市场环境下的表现", expanded=True):
                    st.caption("同一优化结论在不同市场环境下是否依然成立？")
                    agg = regime_display.get("aggregate", {})
                    col_bull, col_bear, col_range = st.columns(3)
                    col_bull.metric("🐂 牛市anchors", agg.get("bull_anchors", 0))
                    col_bear.metric("🐻 熊市anchors", agg.get("bear_anchors", 0))
                    col_range.metric("📊 震荡市anchors", agg.get("ranging_anchors", 0))

                    for regime_label, regime_emoji in [("bull", "🐂 牛市"), ("bear", "🐻 熊市"), ("ranging", "📊 震荡市")]:
                        rdata = regime_display.get(regime_label, {})
                        rcontrib = rdata.get("contribution_summary", {})
                        if rcontrib:
                            st.caption(f"**{regime_emoji}** ({rdata.get('num_anchors', 0)}个时点):")
                            regime_rows = []
                            for agent, info in sorted(rcontrib.items(),
                                                     key=lambda x: x[1].get("mean_delta_L", 0) if isinstance(x[1], dict) else 0,
                                                     reverse=True):
                                if not isinstance(info, dict):
                                    continue
                                regime_rows.append({
                                    "Agent": agent,
                                    "ΔL": f"{info.get('mean_delta_L', 0):+.4f}",
                                    "方向": info.get("direction", "neutral"),
                                })
                            st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

        # 保存路径提示
        st.caption(f"结果已保存: `{display.get('saved_path', '')}`")
