"""Agent贡献榜 + 优化建议组件"""
import json
import os
import streamlit as st
import pandas as pd


# ── Gap #22: Ticket accept/reject session state ──
def _init_ticket_state():
    if "ticket_decisions" not in st.session_state:
        st.session_state.ticket_decisions = {}  # {ticket_id: {"action": "accepted|rejected", "reason": "", "timestamp": ""}}
    if "show_processed_tickets" not in st.session_state:
        st.session_state.show_processed_tickets = False


def _handle_accept_ticket(ticket: dict) -> None:
    """处理接受ticket — 根据route执行不同操作。

    - auto: 已自动应用，标记为accepted
    - semi_auto: 调用optimizer应用补丁/修复
    - manual: 显示详细说明供人工实施
    """
    ticket_id = ticket.get("ticket_id", "")
    route = ticket.get("route", "")

    from datetime import datetime
    timestamp = datetime.now().isoformat()
    decision = {"action": "accepted", "timestamp": timestamp, "reason": ""}

    if route == "auto":
        # 自动路由: 已自动应用，直接标记
        try:
            from src.eval.repositories import update_ticket_status
            update_ticket_status(ticket_id, "accepted", finished_at=timestamp)
        except Exception:
            pass
        decision["reason"] = "auto路由此票已自动应用"
        st.success(f"已接受优化建议「{ticket.get('title', '')}」(auto路由 - 自动应用)")

    elif route == "semi_auto":
        # 半自动路由: 尝试调用optimizer应用
        try:
            from src.eval.repositories import update_ticket_status
            update_ticket_status(ticket_id, "accepted", finished_at=timestamp)
        except Exception:
            pass
        decision["reason"] = "半自动路由 - 系统将生成修改方案供审核后执行"
        st.success(f"已接受优化建议「{ticket.get('title', '')}」(semi_auto路由 - 请审核系统生成的修改方案)")
        # 显示建议操作
        try:
            evidence = json.loads(ticket.get("evidence_json", "{}"))
            suggestions = evidence.get("suggested_actions", evidence.get("optimization_suggestions", []))
            if suggestions:
                with st.expander("📋 系统建议操作", expanded=True):
                    if isinstance(suggestions, list):
                        for s in suggestions:
                            st.write(f"- {s}")
                    elif isinstance(suggestions, str):
                        st.write(suggestions)
        except Exception:
            pass

    elif route == "manual":
        # 人工路由: 显示详细说明
        try:
            from src.eval.repositories import update_ticket_status
            update_ticket_status(ticket_id, "accepted", finished_at=timestamp)
        except Exception:
            pass
        decision["reason"] = "人工路由 - 请根据以下指南手动实施"
        st.info(f"已接受优化建议「{ticket.get('title', '')}」(manual路由 - 需手动实施)")

        # 显示手动实施指南
        try:
            evidence = json.loads(ticket.get("evidence_json", "{}"))
            with st.expander("📖 人工实施指南", expanded=True):
                st.markdown("### 问题分析")
                st.write(evidence.get("summary", ticket.get("summary", "")))
                st.markdown("### 受影响文件")
                affected = evidence.get("affected_files", [])
                if affected:
                    for f in affected:
                        st.code(f)
                st.markdown("### 建议修改方案")
                suggestions = evidence.get("optimization_suggestions", evidence.get("suggested_actions", []))
                if isinstance(suggestions, list):
                    for i, s in enumerate(suggestions, 1):
                        st.write(f"{i}. {s}")
                elif isinstance(suggestions, str):
                    st.write(suggestions)
                st.caption("修改后请重新运行 `python -m src.eval optimize --analyze` 验证效果。")
                st.caption("完成后请在数据库中标记此 ticket 为 implemented。")
        except Exception:
            pass

    st.session_state.ticket_decisions[ticket_id] = decision


def _handle_reject_ticket(ticket: dict) -> None:
    """处理拒绝ticket — 记录拒绝状态和可选原因"""
    ticket_id = ticket.get("ticket_id", "")
    from datetime import datetime
    timestamp = datetime.now().isoformat()

    # 弹出原因输入框会破坏布局，使用expand context内记录
    reject_reason = st.session_state.get(f"reject_reason_{ticket_id}", "")
    decision = {
        "action": "rejected",
        "timestamp": timestamp,
        "reason": reject_reason,
    }

    try:
        from src.eval.repositories import update_ticket_status
        update_ticket_status(ticket_id, "rejected", finished_at=timestamp,
                            note=reject_reason or "用户拒绝")
    except Exception:
        pass

    st.session_state.ticket_decisions[ticket_id] = decision
    st.info(f"已拒绝优化建议「{ticket.get('title', '')}」")


def render_contribution_leaderboard(eval_ready: bool) -> None:
    """渲染 Agent 贡献榜 — 谁在帮忙，谁在拖后腿"""
    st.markdown("---")
    st.subheader("\U0001f3c6 Agent贡献分 — 谁在帮忙，谁在拖后腿")
    st.caption("ΔL > 0 = 正贡献（去掉它系统变差）。ΔL < 0 = 负贡献（去掉它系统反而变好）。★★★ = 统计显著。☆ = 不显著。")

    # ── Gap #21: 数据源切换 ──
    data_source = st.radio(
        "数据来源",
        ["🔄 实盘模拟结果", "🔬 回测结果"],
        horizontal=True,
        key="contrib_data_source",
        help="实盘模拟结果来自每日检查累积的贡献数据；回测结果来自历史回测引擎。",
    )

    if eval_ready:
        try:
            if "回测" in data_source:
                _render_backtest_contribution()
            else:
                _render_live_contribution()
        except Exception as e:
            st.caption(f"加载中: {e}")
    else:
        st.caption("评测系统初始化中...")


def _render_live_contribution() -> None:
    """渲染实盘模拟贡献数据"""
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
            st.caption("数据来源: 🔄 实盘模拟 (每日检查累积)")
            st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)
        else:
            st.info("暂无Agent贡献数据。运行回测或积累足够实盘数据后这里会自动填充。")
    else:
        st.info("暂无贡献记录。点'历史回测'跑一次，或每天运行'一键完整检查'积累实盘数据。")


def _render_backtest_contribution() -> None:
    """渲染回测贡献数据 — 从backtest结果和数据库读取"""
    st.caption("数据来源: 🔬 历史回测结果")

    # 尝试从数据库读取回测贡献数据
    has_data = False
    try:
        from src.eval.repositories import get_agent_contributions, get_recent_batches
        recent = get_recent_batches(limit=20)
        backtest_batches = [b for b in recent if b.get("trigger_source", "") == "backtest"]
        if backtest_batches:
            latest_bt = backtest_batches[0]
            batch_id = latest_bt["batch_id"]
            st.caption(f"显示最近回测批次: {batch_id[:20]}... | {latest_bt.get('started_at', '')[:10]}")

            for term in ["short", "medium", "long"]:
                contribs = get_agent_contributions(batch_id, term=term)
                if contribs:
                    has_data = True
                    with st.expander(f"📊 {dict(short='短线', medium='中线', long='长线').get(term, term)} 回测贡献", expanded=(term == "medium")):
                        rows = []
                        for c in sorted(contribs, key=lambda x: x.get("delta_L_total", 0), reverse=True):
                            delta = c.get("delta_L_total", 0)
                            stars = "★★★" if delta > 0.03 else ("★★" if delta > 0.01 else ("☆" if delta > -0.01 else "↓"))
                            direction = "\U0001f44d正" if delta > 0 else ("\U0001f44e负" if delta < 0 else "➖")
                            rows.append({
                                "Agent": c.get("agent_name", ""),
                                "ΔL": f"{delta:+.4f}",
                                "方向": direction,
                                "显著性": stars,
                                "CI_95%": f"[{c.get('ci_95_lower', 0):+.4f}, {c.get('ci_95_upper', 0):+.4f}]",
                                "样本": c.get("sample_size", 0),
                            })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if not has_data:
            # 尝试直接从backtest_results目录读取
            BACKTEST_RESULTS_DIR = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                "data", "eval", "backtest_results"
            )
            if os.path.isdir(BACKTEST_RESULTS_DIR):
                result_files = sorted(
                    [f for f in os.listdir(BACKTEST_RESULTS_DIR) if f.endswith(".json")],
                    key=lambda f: os.path.getmtime(os.path.join(BACKTEST_RESULTS_DIR, f)),
                    reverse=True,
                )
                if result_files:
                    latest_file = os.path.join(BACKTEST_RESULTS_DIR, result_files[0])
                    try:
                        with open(latest_file, "r", encoding="utf-8") as f:
                            bt_data = json.load(f)
                        contrib = bt_data.get("contribution_summary", {})
                        if contrib:
                            has_data = True
                            rows = []
                            contrib_filtered = {k: v for k, v in contrib.items() if isinstance(v, dict)}
                            for agent, info in sorted(contrib_filtered.items(), key=lambda x: x[1].get("mean_delta_L", 0), reverse=True):
                                delta = info.get("mean_delta_L", 0)
                                stars = "★★★" if delta > 0.03 else ("★★" if delta > 0.01 else ("☆" if delta > -0.01 else "↓"))
                                direction = "\U0001f44d正" if delta > 0 else ("\U0001f44e负" if delta < 0 else "➖")
                                rows.append({
                                    "Agent": agent,
                                    "ΔL": f"{delta:+.4f}",
                                    "方向": direction,
                                    "显著性": stars,
                                    "样本": info.get("sample_size", 0),
                                })
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass
            if not has_data:
                st.info("暂无回测贡献数据，请先运行回测。点击「回测与消融」标签页，设置参数后点击「开始回测」。")
    except Exception as e:
        st.info(f"暂无回测贡献数据，请先运行回测。({str(e)[:80]})")


def render_optimization_tickets(eval_ready: bool) -> None:
    """渲染优化建议列表 — 含Accept/Reject按钮 (Gap #22)"""
    _init_ticket_state()

    st.markdown("---")
    st.subheader("\U0001f527 优化建议 — 基于Loss和贡献数据自动生成")
    st.caption("""
    \U0001f916 自动 = 参数调整类。\U0001f464\U0001f916 半自动 = 系统生成方案，需你审核后执行。\U0001f464 人工 = 系统出分析报告，需你手动改代码。
    运行 `python -m src.eval optimize --analyze` 生成新建议。
    """)

    # ── 已处理过滤器 ──
    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        show_processed = st.checkbox(
            "显示已处理",
            value=st.session_state.show_processed_tickets,
            key="show_processed_checkbox",
            help="显示已接受/已拒绝的优化建议",
        )
        st.session_state.show_processed_tickets = show_processed

    if eval_ready:
        try:
            from src.eval.repositories import get_pending_tickets, get_tickets_by_batch, get_recent_batches
            tickets = get_pending_tickets()

            # 如果显示已处理，也加载已处理的tickets
            all_tickets = list(tickets) if tickets else []
            if show_processed:
                recent_batches = get_recent_batches(limit=10)
                for b in recent_batches:
                    bt_tickets = get_tickets_by_batch(b["batch_id"])
                    for t in bt_tickets:
                        if t.get("status") != "pending" and t.get("ticket_id") not in [at.get("ticket_id") for at in all_tickets]:
                            all_tickets.append(t)

            # 合并session_state中的决策状态
            decisions = st.session_state.ticket_decisions
            for t in all_tickets:
                tid = t.get("ticket_id", "")
                if tid in decisions:
                    t["_decision"] = decisions[tid]

            if all_tickets:
                # 统计
                pending_count = sum(1 for t in all_tickets if t.get("status", "") == "pending")
                accepted_count = sum(1 for t in all_tickets if t.get("status", "") == "accepted"
                                     or t.get("_decision", {}).get("action") == "accepted")
                rejected_count = sum(1 for t in all_tickets if t.get("status", "") == "rejected"
                                     or t.get("_decision", {}).get("action") == "rejected")
                col_p, col_a, col_r = st.columns(3)
                col_p.metric("待处理", pending_count)
                col_a.metric("已接受", accepted_count)
                col_r.metric("已拒绝", rejected_count)

                # 排序：待处理优先，然后按严重度
                severity_order = {"high": 0, "medium": 1, "low": 2}
                status_order = {"pending": 0, "accepted": 1, "rejected": 2}
                all_tickets.sort(key=lambda t: (
                    status_order.get(t.get("_decision", {}).get("action") or t.get("status", "pending"), 0),
                    severity_order.get(t.get("severity", "medium"), 1),
                ))

                for t in all_tickets[:20]:  # 最多显示20条
                    tid = t.get("ticket_id", "")
                    decision = t.get("_decision", {})
                    is_processed = (t.get("status", "") in ("accepted", "rejected")
                                    or decision.get("action") in ("accepted", "rejected"))

                    route_icon = {"auto": "\U0001f916", "semi_auto": "\U0001f464\U0001f916", "manual": "\U0001f464"}.get(t.get("route", ""), "❓")

                    # 已处理标记
                    status_badge = ""
                    if is_processed:
                        action = decision.get("action") or t.get("status", "")
                        if action == "accepted":
                            status_badge = " ✅ 已接受"
                        elif action == "rejected":
                            status_badge = " ❌ 已拒绝"

                    with st.expander(f"{route_icon} [{t.get('ticket_type', '')}] {t.get('title', '未命名')}{status_badge}"):
                        st.write(t.get("summary", "无摘要"))
                        st.caption(f"路由: {t.get('route', '')} | 严重度: {t.get('severity', '')}")

                        # 显示详细信息
                        try:
                            evidence = json.loads(t.get("evidence_json", "{}"))
                            if evidence.get("suggested_actions") or evidence.get("optimization_suggestions"):
                                with st.expander("📋 建议操作", expanded=False):
                                    suggestions = evidence.get("suggested_actions") or evidence.get("optimization_suggestions", [])
                                    if isinstance(suggestions, list):
                                        for s in suggestions:
                                            st.write(f"- {s}")
                                    elif isinstance(suggestions, str):
                                        st.write(suggestions)
                        except Exception:
                            pass

                        # 决策记录
                        if decision:
                            st.caption(f"决策时间: {decision.get('timestamp', '')[:19]} | "
                                      f"原因: {decision.get('reason', '无')}")

                        # ── 操作按钮 ──
                        if not is_processed:
                            # 检查是否处于拒绝确认模式
                            show_reject_form = st.session_state.get(f"show_reject_form_{tid}", False)

                            if not show_reject_form:
                                col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
                                with col_btn1:
                                    if st.button("✅ 接受", key=f"accept_{tid}", use_container_width=True):
                                        _handle_accept_ticket(t)
                                        st.rerun()
                                with col_btn2:
                                    if st.button("❌ 拒绝", key=f"reject_{tid}", use_container_width=True):
                                        st.session_state[f"show_reject_form_{tid}"] = True
                                        st.rerun()
                                with col_btn3:
                                    # 查看详情按钮 (使用 expander 作为 popover 降级方案)
                                    try:
                                        evidence = json.loads(t.get("evidence_json", "{}"))
                                        if evidence.get("affected_files") or evidence.get("diagnosis"):
                                            show_detail = st.button("📄 查看详情", key=f"detail_{tid}", use_container_width=True)
                                            if show_detail:
                                                with st.expander("📄 优化详情", expanded=True):
                                                    if evidence.get("diagnosis"):
                                                        st.markdown("**诊断分析**")
                                                        st.write(evidence["diagnosis"])
                                                    if evidence.get("affected_files"):
                                                        st.markdown("**受影响文件**")
                                                        for f in evidence["affected_files"]:
                                                            st.code(f, language=None)
                                    except Exception:
                                        pass
                            else:
                                # 拒绝确认表单
                                st.caption("请输入拒绝原因后确认：")
                                reject_reason = st.text_input(
                                    "拒绝原因 (可选)",
                                    key=f"reject_reason_{tid}",
                                    placeholder="例如: 不需要/优先级低/等下次迭代",
                                )
                                col_confirm, col_cancel = st.columns(2)
                                with col_confirm:
                                    if st.button("✅ 确认拒绝", key=f"confirm_reject_{tid}", use_container_width=True):
                                        _handle_reject_ticket(t)
                                        st.session_state[f"show_reject_form_{tid}"] = False
                                        st.rerun()
                                with col_cancel:
                                    if st.button("↩ 返回", key=f"cancel_reject_{tid}", use_container_width=True):
                                        st.session_state[f"show_reject_form_{tid}"] = False
                                        st.rerun()
                        else:
                            # 已处理：显示撤销按钮
                            if st.button("↩ 撤销决策", key=f"undo_{tid}", use_container_width=True):
                                try:
                                    from src.eval.repositories import update_ticket_status
                                    update_ticket_status(tid, "pending")
                                except Exception:
                                    pass
                                if tid in st.session_state.ticket_decisions:
                                    del st.session_state.ticket_decisions[tid]
                                st.rerun()
            else:
                st.info("暂无优化建议。运行 `python -m src.eval optimize --analyze` 生成。")
        except Exception as e:
            st.caption(f"加载中: {e}")
    else:
        st.caption("评测系统初始化中...")
