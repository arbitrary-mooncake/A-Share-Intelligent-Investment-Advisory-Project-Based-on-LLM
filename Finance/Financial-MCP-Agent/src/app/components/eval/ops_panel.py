"""操作面板组件 — 一键检查 / 调仓 / 结算 / 池更新"""
import time

import streamlit as st
from datetime import datetime


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


def render_ops_panel(orch, eval_ready: bool) -> None:
    """渲染操作面板：一键检查 / 仅结算 / 仅调仓 / 更新精筛池"""
    st.subheader("\U0001f527 操作面板")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶️ 一键完整检查", use_container_width=True):
            if eval_ready:
                status_box = st.status("初始化...", expanded=True)
                progress_bar = st.progress(0, text="准备中...")
                try:
                    status_box.update(label="检测缺失日+补回历史+调仓...", state="running")
                    progress_bar.progress(5, text="检测缺失日+补回历史+调仓...")
                    result = _run_async(orch.run_full_check())
                    progress_bar.progress(85, text="生成报告...")

                    if result.get("reset_triggered"):
                        st.warning(f"⚠️ {result.get('reset_reason', '缺失超过上限，已重置')}")

                    progress_bar.progress(100, text="完成!")
                    status_box.update(label="检查完成", state="complete")

                    batch_info = f"批次 {result['batch_id'][:16]} 完成！"
                    catchup = result.get("catchup")
                    if catchup:
                        caught = catchup.get("caught_up", 0)
                        if caught > 0:
                            batch_info += f" 追赶了{caught}个交易日"
                    st.success(batch_info)
                    if result.get("report_path"):
                        st.caption(f"报告: {result['report_path']}")
                    st.rerun()
                except Exception as e:
                    progress_bar.progress(100, text="失败")
                    status_box.update(label=f"检查失败: {str(e)[:100]}", state="error")
                    st.error(f"检查失败: {e}")
        st.caption("\U0001f446 每个交易日收盘后点一次。自动完成: 检测缺失日→补回历史（如有）→结算历史收益→14条线独立评分调仓→收盘结算→生成报告。缺失超过7天自动重置。")

    with col2:
        if st.button("\U0001f9ee 仅收盘结算（不调仓）", use_container_width=True):
            if eval_ready:
                orch.run_daily_settlement({})
                st.success("已按最新收盘价更新所有持仓市值和收益")
        st.caption("\U0001f446 仅刷新市值和收益率，不做买卖。比如收盘后想再确认一下持仓市值，或你已通过CLI调仓过只需结算时使用。")

    col_reb1, col_reb2 = st.columns(2)
    with col_reb1:
        if st.button("\U0001f4ca 仅收盘前调仓（不结算）", use_container_width=True):
            if eval_ready:
                with st.spinner("调仓运行中: 评分→选股→生成订单→执行..."):
                    async def _rebalance_all():
                        current_date = datetime.now().strftime("%Y%m%d")
                        for term in ["short", "medium", "long"]:
                            await orch.run_daily_rebalance(term, current_date, {})
                    _run_async(_rebalance_all())
                    st.success("全部期限调仓完成！持仓已更新。")
        st.caption("\U0001f446 根据最新数据重评分→选股→生成调仓订单→执行交易，但不结算当日收益。适合盘中调整持仓时使用。")

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        pool_term = st.selectbox("选择期限", ["short", "medium", "long"],
                                  format_func=lambda x: {"short": "短线(100只)", "medium": "中线(80只)", "long": "长线(60只)"}[x],
                                  key="pool_term")
        target_sizes = {"short": 100, "medium": 80, "long": 60}
        pool_count = st.number_input("正式评分股票数",
                                      min_value=20, max_value=200,
                                      value=target_sizes.get(pool_term, 100),
                                      step=10,
                                      help="精筛候选股数（参考值，实际由四层管线按1:1.2差额机制自动决定）。")
        if st.button("\U0001f3af 更新精筛池", use_container_width=True, type="primary"):
            if eval_ready:
                status_box = st.status("四层管线初始化...", expanded=True)
                progress_bar = st.progress(0, text="Layer 0 硬筛...")
                from src.eval.job_manager import JobManager, JobStatus
                jm = JobManager()

                try:
                    existing = jm.find_running(pool_term)
                    if existing:
                        job_id = existing["job_id"]
                        st.info(
                            f"已有正在跑的更新 ({existing.get('started_at', '')}), 自动 attach"
                        )
                    else:
                        job_id = jm.start_job(pool_term)

                    st.session_state[f"pool_job_{pool_term}"] = job_id

                    # 轮询进度
                    while True:
                        job = jm.poll(job_id)
                        if not job:
                            st.error("任务信息丢失")
                            break

                        status = job.get("status")
                        prog = job.get("progress") or {}
                        pct = prog.get("overall_pct", 0) / 100.0
                        eta = prog.get("eta_str", "计算中...")
                        stall = prog.get("stall_s", 0)

                        progress_bar.progress(
                            min(pct, 1.0),
                            f"总进度 {pct*100:.0f}% | ETA: {eta}"
                            + (f" | ⚠️ 卡顿{stall:.0f}s" if stall > 60 else "")
                        )
                        status_box.update(
                            label=f"running: {prog.get('current_stage_msg', '')[:60]}",
                            state="running",
                        )

                        if status == JobStatus.COMPLETED.value:
                            progress_bar.progress(1.0, text="完成!")
                            status_box.update(label="精筛池更新完成", state="complete")
                            st.success(
                                f"精筛池[{pool_term}]更新完成！"
                                f"共{job.get('progress', {}).get('result', {}).get('final_pool_size', 0)}只"
                            )
                            st.rerun()
                        if status in (JobStatus.FAILED.value, JobStatus.ORPHANED.value):
                            status_box.update(label="更新失败", state="error")
                            st.error(job.get("error") or f"任务状态: {status}")
                            with st.expander("📋 Worker 日志", expanded=False):
                                st.code(jm.read_log(job_id, tail=200), language="log")
                            break

                        time.sleep(1.0)
                except Exception as e:
                    progress_bar.progress(100, text="失败")
                    status_box.update(label="更新失败", state="error")
                    st.error(f"精筛池更新启动失败: {e}")
        st.caption(
            "\U0001f446 四层筛选管线（总纲§4.1, V3 流式 + L2 pre-fetch 复用）:\n\n"
            "**Layer 0 硬筛**: 去ST/新股/BJ/B股/近20日日均成交额<2000万 → ~4500只\n"
            "**Layer 1 批量粗筛**: M1/M3生产模型5只/批打分, 分4档 (raw_data 同步累积)\n"
            "**Layer 2 快筛**: DSV4Pro 流式双堆 top-α, 复用 L1 raw_data (省 10-20min)\n"
            "**Layer 3 精筛**: 白名单+推荐1:1.2差额 → 正式7Agent+3Scorer (5并发) → LLM动态阈值\n\n"
            "⏱ 冷启动耗时: 短线/中线 ≈1-1.5h, 长线 ≈0.8-1h (热缓存 <30min)"
        )

    with col4:
        if st.button("\U0001f50d 刷新数据", use_container_width=True):
            st.rerun()
        st.caption("\U0001f446 页面不自动刷新。想看最新持仓/收益/趋势时点这里。")
