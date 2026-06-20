"""
模拟分析与迭代 — 第6大功能：评测控制台
"""
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="模拟分析与迭代", page_icon="📈", layout="wide")

st.title("📈 模拟分析与迭代")
st.caption("评分智能体 — 模拟盘评估、消融实验、回测、自动优化。所有结果仅供参考，使用评测模型(MiMo-V2.5)，目的是优化Agent系统。")

# ── 初始化 ──
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

# ═══════════════════════════════════════════════════
# 顶部状态栏
# ═══════════════════════════════════════════════════
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

st.markdown("---")

# ═══════════════════════════════════════════════════
# 操作面板
# ═══════════════════════════════════════════════════
st.subheader("🔧 操作面板")

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ 一键完整检查", use_container_width=True):
        if eval_ready:
            with st.spinner("运行中: 结算历史 → 评分调仓 → 收盘结算 → 生成报告..."):
                import asyncio
                result = asyncio.run(orch.run_full_check())
                st.success(f"批次 {result['batch_id'][:16]} 完成！报告: {result.get('report_path', '')}")
                st.rerun()
    st.caption("👆 每个交易日收盘后点一次。自动完成: 结算历史收益 → 14条线独立评分调仓 → 收盘结算 → 生成报告。耗时1-2分钟。")

with col2:
    if st.button("🧮 仅收盘结算（不调仓）", use_container_width=True):
        if eval_ready:
            orch.run_daily_settlement({})
            st.success("已按最新收盘价更新所有持仓市值和收益")
    st.caption("👆 仅刷新市值和收益率，不做买卖。比如收盘后想再确认一下持仓市值，或你已通过CLI调仓过只需结算时使用。")

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
                                  help="精筛候选股数。四层管线会自动按1:1.2差额组建候选，默认等于池目标容量。")
    if st.button("🎯 更新精筛池", use_container_width=True, type="primary"):
        if eval_ready:
            with st.spinner("四层管线运行中: 硬筛→M1/M3批量粗筛分档→M2快筛→1:1.2差额精筛..."):
                import asyncio
                update_result = asyncio.run(orch.run_pool_update(pool_term))
                if "error" in update_result:
                    st.error(update_result["error"])
                else:
                    st.success(f"精筛池[{pool_term}]更新完成！共{update_result.get('final_pool_size', 0)}只")
                    whitelist = update_result.get('whitelist', [])
                    blacklist = update_result.get('blacklist', [])
                    st.text(f"  白名单(强烈推荐): {len(whitelist)}只")
                    st.text(f"  候选池: {update_result.get('final_pool_size', 0) - len(whitelist)}只")
                    st.text(f"  黑名单(卖出): {len(blacklist)}只")
                    st.rerun()
    st.caption(
        "👆 四层筛选管线（总纲§4.1）:\n\n"
        "**Layer 0 硬筛**: 去ST/新股/BJ/B股/近20日日均成交额<2000万 → ~4500只\n"
        "**Layer 1 批量粗筛**: M1/M3生产模型5只/批打分, 分4档(强烈推荐→白名单 / 买入/谨慎买入/观望→初筛池 / 卖出→黑名单)\n"
        "**Layer 2 快筛**: M2(Qwen3.6-Flash)过滤初筛池, 淘汰明显不行的\n"
        "**Layer 3 精筛**: 白名单+初筛通过1:1.2差额 → 正式7Agent+3Scorer → LLM动态阈值 → 填满精筛池\n\n"
        "⏱ 预计耗时: Layer0约2分钟 + Layer1约20分钟 + Layer2约2分钟 + Layer3约40分钟 ≈ 1小时(短线100只)"
    )

with col4:
    if st.button("🔍 刷新数据", use_container_width=True):
        st.rerun()
    st.caption("👆 页面不自动刷新。想看最新持仓/收益/趋势时点这里。")

# ═══════════════════════════════════════════════════
# 精筛池概览
# ═══════════════════════════════════════════════════
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
    st.caption("精筛池 = 模拟盘的选股范围。14条线只能从这些池子里选股。池子一段时间不变（控制变量），需要更新时点上面的'更新精筛股票池'按钮。")
else:
    st.warning("评测系统初始化中...")

# ═══════════════════════════════════════════════════
# 各线持仓与收益
# ═══════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 各线路持仓与收益 — 14条模拟盘实时状态")

st.caption("""
**线路说明**: S-L0~S-L7 = 短线消融实验（每天同一起点出发，各差1个Agent，测谁有用谁拖后腿）。
S-L8 = 短线正常交易（连续持仓）。S-L9/M-L1/L-L1 = LLM自主投资对照组。M-L0/L-L0 = 中/长线。
""")

tab_s, tab_m, tab_l = st.tabs(["⚡ 短线 (10条)", "📅 中线 (2条)", "🏛️ 长线 (2条)"])

if eval_ready:
    lines = status.get("lines", [])
    for tab, term in [(tab_s, "short"), (tab_m, "medium"), (tab_l, "long")]:
        with tab:
            term_lines = [l for l in lines if l.get("term") == term]
            if term_lines:
                import pandas as pd
                df_data = []
                for l in term_lines:
                    df_data.append({
                        "线路": l["line_id"],
                        "类型": l.get("type", ""),
                        "持仓数": l.get("holdings_count", 0),
                        "总市值(万)": round(l.get("total_value", 0) / 10000, 1),
                        "累计收益%": l.get("cumulative_return_pct", 0),
                        "最大回撤%": l.get("max_drawdown_pct", 0),
                    })
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)
            else:
                st.info("👆 还没数据。先更新精筛池，再点'一键完整检查'，这里就会显示各条线的实时持仓和收益率。")
else:
    st.caption("评测系统初始化中...")

# ═══════════════════════════════════════════════════
# 回测
# ═══════════════════════════════════════════════════
with st.expander("⚙️ 历史回测（点击展开）", expanded=False):
    st.caption("回测 = 在历史数据上重放系统，快速知道哪个Agent帮忙、哪个拖后腿。不用等几个月实盘验证。每次选好参数点按钮即可。")
    st.caption("⚠️ 回测用当前精筛池，不含历史上退市/暴雷股。news/event agent在回测中不可用（历史新闻无法回溯）。结果带幸存者偏差，绝对收益仅供参考。")

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

    if st.button("▶️ 开始回测", type="primary"):
        if eval_ready:
            with st.spinner("回测运行中..."):
                from src.eval.replay_backtest_engine import ReplayBacktestEngine, BacktestConfig
                term_map = {"short": "short", "medium": "medium", "long": "long"}
                cfg = BacktestConfig(
                    term=term_map.get(bt_term, "medium"),
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=bt_end.strftime("%Y-%m-%d"),
                )
                engine = ReplayBacktestEngine(cfg)
                anchors = engine.generate_anchor_dates()
                st.success(f"回测完成: 共{len(anchors)}个回测时点")
                st.info("结果解读: 查看下方 Agent贡献榜，ΔL > 0 = 该Agent在帮忙，ΔL < 0 = 该Agent在拖后腿。显着性★★★ = 统计可信。")

# ═══════════════════════════════════════════════════
# Agent贡献榜
# ═══════════════════════════════════════════════════
st.markdown("---")
st.subheader("🏆 Agent贡献分 — 谁在帮忙，谁在拖后腿")

st.caption("ΔL > 0 = 正贡献（去掉它系统变差）。ΔL < 0 = 负贡献（去掉它系统反而变好）。★★★ = 统计显著（95%置信区间不跨0）。☆ = 不显著（可能是噪声）。")

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

            import pandas as pd
            contrib_data = []
            for name, stats in sorted(agent_stats.items(),
                                       key=lambda x: sum(x[1]["deltas"])/max(len(x[1]["deltas"]),1),
                                       reverse=True):
                avg_delta = sum(stats["deltas"]) / max(len(stats["deltas"]), 1)
                last_stars = stats["stars"][-1] if stats["stars"] else "☆"
                contrib_data.append({
                    "Agent": name, "平均ΔL": round(avg_delta, 4),
                    "样本数": len(stats["deltas"]), "最新显著性": last_stars,
                    "评价": "👍 正贡献" if avg_delta > 0.01 else ("👎 负贡献" if avg_delta < -0.01 else "➖ 中性"),
                })
            if contrib_data:
                st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)
            else:
                st.info("暂无Agent贡献数据。运行回测或积累足够实盘数据后这里会自动填充。")
        else:
            st.info("暂无贡献记录。点上方'历史回测'跑一次，或每天运行'一键完整检查'积累实盘数据。")
    except Exception as e:
        st.caption(f"加载中: {e}")
else:
    st.caption("评测系统初始化中...")

# ═══════════════════════════════════════════════════
# 趋势图
# ═══════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 趋势图 — 系统表现随时间的变化")

st.caption("每天运行'一键完整检查'会积累数据。数据越多，趋势越有意义。至少积累7-10天后才能看到明显趋势。")

if eval_ready:
    try:
        from src.eval.chart_service import ChartService
        cs = ChartService()

        tab_t1, tab_t2, tab_t3 = st.tabs(["Score趋势", "Loss趋势", "线路收益对比"])
        with tab_t1:
            score_data = cs.get_score_trend_data(90)
            if score_data["data"]:
                import pandas as pd
                df = pd.DataFrame(score_data["data"])
                if not df.empty:
                    st.line_chart(df.set_index("date")["value"], use_container_width=True)
                    st.caption("Score越高越好 = 系统评分质量在提升。曲线向上 = 优化方向对了。")
                else:
                    st.info("暂无数据，运行几次检查后出现。")
        with tab_t2:
            loss_data = cs.get_loss_trend_data(90)
            if loss_data["data"]:
                import pandas as pd
                df = pd.DataFrame(loss_data["data"])
                if not df.empty:
                    st.line_chart(df.set_index("date")["value"], use_container_width=True)
                    st.caption("Loss越低越好 = 预测与实际的差距在缩小。曲线向下 = 优化有效果。")
                else:
                    st.info("暂无数据，运行几次检查后出现。")
        with tab_t3:
            if status.get("lines"):
                line_data = cs.get_line_comparison_data(status["lines"])
                if line_data["data"]:
                    import pandas as pd
                    df = pd.DataFrame(line_data["data"])
                    if not df.empty:
                        st.bar_chart(df.set_index("line_id")["return"], use_container_width=True)
                        st.caption("对比各条线的累计收益。消融线之间差异 = Agent贡献的直观体现。")
    except Exception:
        st.caption("趋势数据加载中...")
else:
    st.caption("评测系统初始化中...")

# ═══════════════════════════════════════════════════
# 优化建议
# ═══════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔧 优化建议 — 基于Loss和贡献数据自动生成")

st.caption("""
🤖 自动 = 参数调整类，系统可自动执行。👤🤖 半自动 = 系统生成方案，需你审核后执行。👤 人工 = 复杂问题，系统出分析报告，需你手动改代码。
运行 `python -m src.eval optimize --analyze` 生成新建议。
""")

if eval_ready:
    try:
        from src.eval.repositories import get_pending_tickets
        tickets = get_pending_tickets()
        if tickets:
            for t in tickets[:5]:
                route_icon = {"auto": "🤖", "semi_auto": "👤🤖", "manual": "👤"}.get(t.get("route", ""), "❓")
                with st.expander(f"{route_icon} [{t.get('ticket_type', '')}] {t.get('title', '未命名')}"):
                    st.write(t.get("summary", "无摘要"))
                    st.caption(f"路由: {t.get('route', '')} | 严重度: {t.get('severity', '')}")
        else:
            st.info("暂无优化建议。运行 `python -m src.eval optimize --analyze` 生成。")
    except Exception as e:
        st.caption(f"加载中: {e}")
else:
    st.caption("评测系统初始化中...")

# ═══════════════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════════════
st.markdown("---")
st.warning("""
⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考，目的是优化Agent系统而非追求极限投资收益。不构成任何投资建议。市场有风险，投资需谨慎。
""")
