"""
智能投顾 — AI 投顾助手: AI顾问对话、股票推荐、持仓管理、策略市场、回测&模拟盘、收益报告
"""
import os
import sys
import json
import requests

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(page_title="智能投顾", page_icon="🤖", layout="wide")

from theme import inject_global_styles
from components.shared_sidebar import render_sidebar
inject_global_styles()
render_sidebar()

# ──────────────────────────────────────────────
# API 配置
# ──────────────────────────────────────────────
API_BASE = os.getenv("ADVISORY_API_BASE", "http://localhost:8000")

def _api(path: str, method: str = "GET", body: dict = None) -> dict:
    """调用后端 API，返回 JSON 或错误字典。"""
    url = f"{API_BASE}{path}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=30)
        else:
            resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return {"error": f"无法连接到后端服务 ({API_BASE})，请确认 API 服务已启动"}
    except requests.Timeout:
        return {"error": "请求超时，请稍后重试"}
    except Exception as e:
        return {"error": str(e)}

# ──────────────────────────────────────────────
# 状态初始化
# ──────────────────────────────────────────────
if "advisory_page" not in st.session_state:
    st.session_state["advisory_page"] = "ai_chat"
if "advisory_chat" not in st.session_state:
    st.session_state["advisory_chat"] = []
if "advisory_recommend_result" not in st.session_state:
    st.session_state["advisory_recommend_result"] = None
if "advisory_recommend_question" not in st.session_state:
    st.session_state["advisory_recommend_question"] = ""
if "advisory_portfolio_list" not in st.session_state:
    st.session_state["advisory_portfolio_list"] = []
if "advisory_strategy_list" not in st.session_state:
    st.session_state["advisory_strategy_list"] = []
if "advisory_backtest_result" not in st.session_state:
    st.session_state["advisory_backtest_result"] = None
if "advisory_sim_result" not in st.session_state:
    st.session_state["advisory_sim_result"] = None
if "advisory_free_result" not in st.session_state:
    st.session_state["advisory_free_result"] = None
if "advisory_report_result" not in st.session_state:
    st.session_state["advisory_report_result"] = None
if "advisory_chat_loading" not in st.session_state:
    st.session_state["advisory_chat_loading"] = False
if "advisory_selected_pf" not in st.session_state:
    st.session_state["advisory_selected_pf"] = None
if "advisory_health_result" not in st.session_state:
    st.session_state["advisory_health_result"] = None
if "advisory_match_result" not in st.session_state:
    st.session_state["advisory_match_result"] = None
if "advisory_auto_catchup_done" not in st.session_state:
    st.session_state["advisory_auto_catchup_done"] = False

# ── 子页面定义 ────────────────────────────────
PAGE_CONFIG = {
    "ai_chat":    {"icon": "💬", "label": "AI顾问对话", "short": "AI对话"},
    "recommend":  {"icon": "📊", "label": "股票推荐",   "short": "推荐"},
    "portfolio":  {"icon": "💰", "label": "我的持仓",   "short": "持仓"},
    "strategies": {"icon": "📈", "label": "策略市场",   "short": "策略"},
    "backtest":   {"icon": "📉", "label": "回测&模拟盘","short": "回测"},
    "report":     {"icon": "📄", "label": "收益报告",   "short": "报告"},
}


# ──────────────────────────────────────────────
# 左侧子页面渲染函数
# ──────────────────────────────────────────────

def _render_ai_chat():
    """💬 AI顾问对话 — 真实 AI 聊天界面"""
    st.markdown("### 💬 AI顾问对话")
    st.caption("与智能投顾直接对话，获取投资建议与市场分析")

    # 聊天消息展示
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["advisory_chat"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 底部输入
    prompt = st.chat_input("输入您的问题，例如：当前市场热点是什么？", key="advisory_chat_input")
    if prompt and prompt.strip():
        st.session_state["advisory_chat"].append({"role": "user", "content": prompt.strip()})
        st.session_state["advisory_chat_loading"] = True
        st.rerun()

    # 处理待发送的消息
    if st.session_state.get("advisory_chat_loading"):
        with st.spinner("AI 顾问思考中..."):
            user_msg = st.session_state["advisory_chat"][-1]["content"]
            page_ctx = json.dumps({
                "page": "AI顾问对话",
                "data": {},
            }, ensure_ascii=False)
            result = _api("/api/advisory/chat", "POST", {
                "message": user_msg,
                "session_id": "default",
                "page_context": page_ctx,
            })
            reply = result.get("reply", result.get("error", "抱歉，服务暂时不可用"))
            st.session_state["advisory_chat"].append({"role": "assistant", "content": reply})
        st.session_state["advisory_chat_loading"] = False
        st.rerun()

    # 空状态提示
    if not st.session_state["advisory_chat"]:
        st.info("👋 您好！我是您的智能投顾助手，由 M1 MiMo-V2.5-Pro 模型驱动，可以为您提供:\n\n"
                "- 📊 **个股分析** — 输入股票代码或名称\n"
                "- 📈 **市场热点** — 当前板块轮动与资金流向\n"
                "- 💰 **持仓诊断** — 组合风险与收益评估\n"
                "- 🎯 **策略建议** — 根据您的风险偏好推荐策略\n\n"
                "请在下方输入框开始对话。")


def _render_recommend():
    """📊 股票推荐 — 真实推荐结果"""
    st.markdown("### 📊 股票推荐")
    st.caption("基于精筛池 + n/x/5 规则，AI 语义筛选优质标的")

    # 搜索输入
    col_q, col_btn = st.columns([3, 1])
    with col_q:
        question = st.text_input(
            "描述您的需求", placeholder="例如：推荐几只科技股 / 适合长期持有的蓝筹 / 低估值高分红股票",
            key="recommend_question", value=st.session_state["advisory_recommend_question"]
        )
    with col_btn:
        do_search = st.button("🔍 智能推荐", type="primary", use_container_width=True)

    st.divider()

    if do_search:
        with st.spinner("AI 正在分析您的需求并筛选股票..."):
            result = _api("/api/advisory/recommend", "POST", {
                "question": question.strip(),
                "session_id": "default",
            })
            st.session_state["advisory_recommend_result"] = result
            st.session_state["advisory_recommend_question"] = question

    result = st.session_state["advisory_recommend_result"]

    if result is None:
        st.info("在上方输入您的投资需求，点击「智能推荐」获取 AI 精选的股票列表")
        return

    if "error" in result:
        st.error(f"推荐失败: {result['error']}")
        return

    recs = result.get("recommendations", [])
    total = result.get("total_candidates", 0)
    filtered = result.get("filtered_count", total)
    llm_pick = result.get("needs_llm_pick", False)
    semantic = result.get("semantic_filtered", False)

    # 状态标签
    tags = []
    if semantic:
        tags.append(f"✨ AI语义筛选: {total} → {filtered}")
    if llm_pick:
        tags.append("🤖 LLM智能挑选")
    if tags:
        st.caption(" · ".join(tags))

    if not recs:
        st.warning("未找到匹配的股票推荐，请尝试更宽泛的描述")
        return

    st.success(f"为您找到 {len(recs)} 只推荐股票")

    # 推荐卡片
    cols = st.columns(min(len(recs), 3))
    color_map = {"short": "#3b82f6", "medium": "#8b5cf6", "long": "#059669"}
    term_label = {"short": "短线", "medium": "中线", "long": "长线"}

    for i, rec in enumerate(recs[:9]):
        with cols[i % 3]:
            code = rec.get("stock_code", "--")
            name = rec.get("company_name", "--")
            industry = rec.get("industry", "")
            term = rec.get("pool_term", "")
            cache_score = rec.get("cache_score")
            pool_score = rec.get("pool_score")
            score_display = f"{cache_score:.0f}" if cache_score else f"{pool_score:.0f}" if pool_score else "--"
            term_color = color_map.get(term, "#64748b")

            st.markdown(
                f'<div class="theme-card" style="text-align:center;padding:1rem;border-top:3px solid {term_color};">'
                f'<div style="font-size:0.75em;color:{term_color};margin-bottom:4px;">'
                f'{term_label.get(term, term)}</div>'
                f'<div style="font-size:1.3em;font-weight:800;color:#2563eb;">{code}</div>'
                f'<div style="font-size:1em;color:#0f172a;margin:4px 0;">{name}</div>'
                f'<div style="font-size:0.8em;color:#64748b;">{industry}</div>'
                f'<div style="font-size:1.5em;font-weight:700;color:#059669;margin:8px 0;">'
                f'{score_display}<span style="font-size:0.6em;">分</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # 加入持仓 — 导航到持仓页手动添加（需要指定买入价格）
            portfolios = st.session_state.get("advisory_portfolio_list", [])
            if portfolios:
                add_col1, add_col2 = st.columns([1, 1])
                pf_names = [p.get("name", p.get("portfolio_id", "")) for p in portfolios]
                pf_ids = [p.get("portfolio_id", "") for p in portfolios]
                with add_col1:
                    selected_pf_name = st.selectbox(
                        "目标组合", pf_names, key=f"addpf_{code}", label_visibility="collapsed"
                    )
                with add_col2:
                    price_input = st.number_input(
                        "买入价", min_value=0.01, value=10.0, step=0.01,
                        key=f"price_{code}", label_visibility="collapsed"
                    )
                if st.button(f"📥 加入持仓", key=f"add_{code}", use_container_width=True):
                    pf_idx = pf_names.index(selected_pf_name) if selected_pf_name in pf_names else 0
                    target_pf_id = pf_ids[pf_idx]
                    add_result = _api("/api/advisory/portfolio/holding", "POST", {
                        "portfolio_id": target_pf_id,
                        "stock_code": code,
                        "company_name": name,
                        "quantity": 100,
                        "price": float(price_input),
                        "action": "buy",
                    })
                    if "error" in add_result:
                        st.error(add_result["error"][:100])
                    else:
                        st.success(f"已加入 {selected_pf_name}")
                        st.session_state["advisory_portfolio_list"] = []
            else:
                st.caption("创建组合后可加入持仓")

    st.caption("以上推荐基于 AI 初筛，仅供参考。若需深入研究，请使用「股票查询」中的打分及报告功能。")


def _render_portfolio():
    """💰 我的持仓 — 真实持仓管理"""
    st.markdown("### 💰 我的持仓")
    st.caption("管理您的投资组合，查看持仓收益与风险评估")

    # 加载组合列表
    if not st.session_state["advisory_portfolio_list"]:
        with st.spinner("加载组合列表..."):
            result = _api("/api/advisory/portfolio/list")
            if "error" not in result:
                st.session_state["advisory_portfolio_list"] = result.get("portfolios", [])

    portfolios = st.session_state["advisory_portfolio_list"]

    col_add, col_search, col_refresh = st.columns([1, 1, 3])
    with col_add:
        if st.button("＋ 新建组合", use_container_width=True):
            with st.spinner("创建中..."):
                create_result = _api("/api/advisory/portfolio/create", "POST", {
                    "name": f"我的组合{len(portfolios) + 1}",
                    "initial_capital": 100000.0,
                })
                if "error" in create_result:
                    st.error(create_result["error"])
                else:
                    st.session_state["advisory_portfolio_list"] = []
                    st.rerun()
    with col_search:
        with st.popover("🔍 搜索加股票", use_container_width=True):
            kw = st.text_input("输入股票代码或名称", key="pf_search_kw")
            if st.button("搜索", key="pf_search_btn", use_container_width=True) and kw.strip():
                with st.spinner("搜索中..."):
                    sr = _api("/api/advisory/stock/search", "POST", {"keyword": kw.strip()})
                    st.session_state["pf_search_result"] = sr
            sr = st.session_state.get("pf_search_result", {})
            if sr and "error" in sr:
                st.error(sr["error"][:80])
            elif sr and "results" in sr:
                if not sr["results"]:
                    st.caption("未找到匹配的股票")
                for item in sr["results"][:10]:
                    st.write(f"**{item['company_name']}** ({item['stock_code']}) - {item.get('industry','')}")
                    pf_names = [p.get("name", p.get("portfolio_id", "")) for p in portfolios]
                    pf_ids = [p.get("portfolio_id", "") for p in portfolios]
                    c_pf, c_qty, c_price, c_btn = st.columns([2, 1, 1, 1])
                    with c_pf:
                        sel_pf = st.selectbox("组合", pf_names, key=f"spf_{item['stock_code']}",
                                              label_visibility="collapsed")
                    with c_qty:
                        qty = st.number_input("数量", min_value=100, value=100, step=100,
                                              key=f"sqty_{item['stock_code']}", label_visibility="collapsed")
                    with c_price:
                        price = st.number_input("价格", min_value=0.01, value=10.0, step=0.01,
                                                key=f"sprice_{item['stock_code']}", label_visibility="collapsed")
                    with c_btn:
                        if st.button("买入", key=f"sbuy_{item['stock_code']}"):
                            pf_idx = pf_names.index(sel_pf) if sel_pf in pf_names else 0
                            add_r = _api("/api/advisory/portfolio/holding", "POST", {
                                "portfolio_id": pf_ids[pf_idx],
                                "stock_code": item["stock_code"],
                                "company_name": item["company_name"],
                                "quantity": int(qty),
                                "price": float(price),
                                "action": "buy",
                            })
                            if "error" in add_r:
                                st.error(add_r["error"][:80])
                            else:
                                st.success(f"已买入 {item['company_name']}")
                                st.session_state["advisory_portfolio_list"] = []
                    st.divider()
    with col_refresh:
        if st.button("🔄 刷新列表", use_container_width=True):
            st.session_state["advisory_portfolio_list"] = []
            st.rerun()

    st.divider()

    if not portfolios:
        st.info("您还没有创建任何投资组合，点击「新建组合」开始")
        return

    # 显示每个组合
    for i, pf in enumerate(portfolios):
        pf_id = pf.get("portfolio_id", "")
        with st.expander(
            f"📁 {pf.get('name', '未命名')} "
            f"（总市值: {pf.get('total_market_value', 0):,.0f} | "
            f"现金: {pf.get('cash', 0):,.0f} | "
            f"盈亏: {pf.get('total_pnl_pct', 0):+.2f}%）",
            expanded=(i == 0)
        ):
            col_info, col_health, col_match = st.columns([2, 1, 1])

            with col_info:
                st.metric("总市值", f"{pf.get('total_market_value', 0):,.0f}")
                st.metric("盈亏", f"{pf.get('total_pnl_pct', 0):+.2f}%")

            with col_health:
                if st.button("🏥 健康度", key=f"health_{pf_id}", use_container_width=True):
                    with st.spinner("评估中..."):
                        h = _api("/api/advisory/portfolio/health", "POST", {
                            "portfolio_id": pf_id, "session_id": "default"
                        })
                        st.session_state["advisory_health_result"] = h

                hresult = st.session_state.get("advisory_health_result", {})
                if hresult and hresult.get("portfolio_id") == pf_id:
                    color = hresult.get("color", "yellow")
                    color_emoji = {"deep_green": "🟢", "light_green": "🟢",
                                   "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(color, "⚪")
                    st.markdown(f"{color_emoji} {hresult.get('color_label', '')}")
                    st.caption(hresult.get("summary", ""))
                    if hresult.get("term_bias"):
                        st.markdown(f"📅 **期限偏向**: {hresult['term_bias']}")
                    for sa in (hresult.get("stock_advice") or []):
                        st.caption(f"💡 {sa.get('company_name','')}: {sa.get('advice','')}")

            with col_match:
                if st.button("🎯 偏好匹配", key=f"match_{pf_id}", use_container_width=True):
                    with st.spinner("评估中..."):
                        m = _api("/api/advisory/portfolio/preference-match", "POST", {
                            "portfolio_id": pf_id, "session_id": "default"
                        })
                        st.session_state["advisory_match_result"] = m

                mresult = st.session_state.get("advisory_match_result", {})
                if mresult and mresult.get("portfolio_id") == pf_id:
                    st.markdown(mresult.get("level_label", ""))
                    st.caption(mresult.get("reason", ""))


def _render_strategies():
    """📈 策略市场 — 真实策略列表"""
    st.markdown("### 📈 策略市场")
    st.caption("浏览并绑定投资策略，22 个内置策略供您选择")

    # 加载策略列表
    if not st.session_state["advisory_strategy_list"]:
        with st.spinner("加载策略列表..."):
            result = _api("/api/advisory/strategies")
            if "error" not in result:
                st.session_state["advisory_strategy_list"] = result.get("strategies", [])

    strategies = st.session_state["advisory_strategy_list"]

    # 自然语言创建自定义策略（设计 §6.5 路径2）
    with st.expander("🤖 AI 定制策略（自然语言描述）"):
        cs_desc = st.text_area(
            "描述你想要的策略",
            placeholder="例如：双均线金叉买入死叉卖出，但加入10%止损",
            key="cs_desc", height=80,
        )
        cs_name = st.text_input("策略名称（英文，如 my_ma_stop）", key="cs_name")
        cs_base = st.text_input("参考策略（可选，如 ma_cross）", key="cs_base", value="")
        if st.button("🚀 生成策略", key="cs_gen", type="primary"):
            if not cs_desc.strip() or not cs_name.strip():
                st.error("请填写描述和名称")
            else:
                with st.spinner("AI 正在生成策略代码（约1-2分钟）..."):
                    cs_r = _api("/api/advisory/strategy/custom", "POST", {
                        "description": cs_desc.strip(),
                        "strategy_name": cs_name.strip(),
                        "base_strategy": cs_base.strip() or None,
                    })
                    if "error" in cs_r:
                        st.error(cs_r["error"][:120])
                    else:
                        if cs_r.get("registered"):
                            st.success(f"✅ 策略 '{cs_r['strategy_name']}' 已生成并注册！")
                            st.session_state["advisory_strategy_list"] = []  # 刷新
                        else:
                            st.warning("策略已保存但注册失败，请检查代码")
                        with st.expander("查看生成的代码"):
                            st.code(cs_r.get("code", ""), language="python")

    if not strategies:
        st.info("策略列表加载中... 请确认后端服务已启动")
        return

    st.success(f"共 {len(strategies)} 个策略可用")

    # 加载组合列表用于绑定
    portfolios = st.session_state.get("advisory_portfolio_list", [])
    if not portfolios:
        pf_result = _api("/api/advisory/portfolio/list")
        if "error" not in pf_result:
            st.session_state["advisory_portfolio_list"] = pf_result.get("portfolios", [])
            portfolios = st.session_state["advisory_portfolio_list"]

    # 策略卡片
    category_colors = {
        "均线": "#3b82f6", "指标": "#8b5cf6", "布林": "#06b6d4",
        "通道": "#10b981", "量价": "#f59e0b", "复合": "#ef4444",
        "风险": "#ec4899", "动量": "#6366f1", "震荡": "#14b8a6",
        "其他": "#64748b",
    }

    for s in strategies:
        name = s.get("name", "")
        desc = s.get("description", "")
        category = s.get("category", "其他")
        cat_color = category_colors.get(category, "#64748b")

        col_info, col_bind = st.columns([3, 1])
        with col_info:
            st.markdown(
                f'<div style="padding:8px 0;">'
                f'<span style="font-weight:700;">{name}</span> '
                f'<span style="font-size:0.75em;color:{cat_color};padding:2px 8px;'
                f'border:1px solid {cat_color};border-radius:6px;">{category}</span>'
                f'<div style="font-size:0.85em;color:#64748b;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_bind:
            if portfolios:
                pf_options = {p.get("name", p.get("portfolio_id", "")): p.get("portfolio_id", "")
                              for p in portfolios}
                selected = st.selectbox(
                    "绑定到", list(pf_options.keys()),
                    key=f"bind_{name}", label_visibility="collapsed"
                )
                if st.button("绑定", key=f"btn_{name}"):
                    with st.spinner("绑定中..."):
                        bind_result = _api("/api/advisory/strategy/bind", "POST", {
                            "portfolio_id": pf_options[selected],
                            "strategy_name": name,
                        })
                        if "error" in bind_result:
                            st.error(bind_result["error"])
                        else:
                            st.success(f"已绑定到 {selected}")
                            st.session_state["advisory_portfolio_list"] = []  # 刷新
            else:
                st.caption("先创建组合")


def _render_backtest():
    """📉 回测&模拟盘 — 真实功能"""
    st.markdown("### 📉 回测&模拟盘")
    st.caption("运行历史回测或控制模拟盘")

    # 加载组合
    portfolios = st.session_state.get("advisory_portfolio_list", [])
    if not portfolios:
        pf_result = _api("/api/advisory/portfolio/list")
        if "error" not in pf_result:
            st.session_state["advisory_portfolio_list"] = pf_result.get("portfolios", [])
            portfolios = st.session_state["advisory_portfolio_list"]

    if not portfolios:
        st.info("请先在「我的持仓」中创建组合")
        return

    pf_options = {p.get("name", p.get("portfolio_id", "")): p for p in portfolios}

    tab_backtest, tab_sim, tab_free = st.tabs(["📊 历史回测", "🖥️ 模拟盘", "🤖 自由线"])

    with tab_backtest:
        selected_name = st.selectbox("选择组合", list(pf_options.keys()), key="bt_pf")
        selected_pf = pf_options[selected_name]
        pf_id = selected_pf.get("portfolio_id", "")

        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=365), key="bt_start")
        with col_end:
            end_date = st.date_input("结束日期", value=datetime.now(), key="bt_end")

        strategy_name = selected_pf.get("bound_strategy", "")
        if strategy_name:
            st.info(f"组合已绑定策略: **{strategy_name}**")
        else:
            strategy_name = st.text_input("策略名称（如 ma_cross）", placeholder="输入策略注册名")

        if st.button("🚀 运行回测", type="primary", use_container_width=True):
            if not strategy_name:
                st.error("请先绑定或输入策略名称")
            else:
                with st.spinner("回测运行中..."):
                    result = _api("/api/advisory/backtest", "POST", {
                        "portfolio_id": pf_id,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        "end_date": end_date.strftime("%Y-%m-%d"),
                        "strategy_name": strategy_name,
                    })
                    st.session_state["advisory_backtest_result"] = result

        bt_result = st.session_state.get("advisory_backtest_result")
        if bt_result:
            if "error" in bt_result:
                st.error(bt_result["error"])
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("总收益率", f"{bt_result.get('total_return_pct', 0):.2f}%")
                with col2:
                    st.metric("最大回撤", f"{bt_result.get('max_drawdown_pct', 0):.2f}%")
                with col3:
                    st.metric("夏普比率", f"{bt_result.get('sharpe_ratio', 0):.2f}")
                with col4:
                    st.metric("交易次数", bt_result.get("trade_count", 0))

                # 权益曲线图
                equity = bt_result.get("equity_curve", [])
                if equity:
                    st.line_chart(
                        {"权益曲线": [float(v) for v in equity]},
                        height=250
                    )

    with tab_sim:
        selected_sim = st.selectbox("选择组合", list(pf_options.keys()), key="sim_pf")
        sim_pf = pf_options[selected_sim]
        sim_id = sim_pf.get("portfolio_id", "")

        col_start, col_catch, col_status = st.columns(3)
        with col_start:
            if st.button("▶ 执行今日结算", use_container_width=True):
                with st.spinner("结算中..."):
                    r = _api("/api/advisory/simulation", "POST", {
                        "portfolio_id": sim_id, "action": "start"
                    })
                    st.session_state["advisory_sim_result"] = r
        with col_catch:
            if st.button("🔄 自动追赶", use_container_width=True):
                with st.spinner("追赶中..."):
                    r = _api("/api/advisory/simulation", "POST", {
                        "portfolio_id": sim_id, "action": "catch_up"
                    })
                    st.session_state["advisory_sim_result"] = r
        with col_status:
            if st.button("📋 查询状态", use_container_width=True):
                with st.spinner("查询中..."):
                    r = _api("/api/advisory/simulation", "POST", {
                        "portfolio_id": sim_id, "action": "status"
                    })
                    st.session_state["advisory_sim_result"] = r

        sim_result = st.session_state.get("advisory_sim_result")
        if sim_result:
            if "error" in sim_result:
                st.error(sim_result["error"])
            elif sim_result.get("action") == "status":
                st.metric("总市值", f"{sim_result.get('total_market_value', 0):,.0f}")
                st.metric("盈亏", f"{sim_result.get('total_pnl_pct', 0):+.2f}%")
                st.caption(f"最后结算: {sim_result.get('last_settlement_date', 'N/A')}")
            elif sim_result.get("action") == "catch_up":
                res = sim_result.get("result", {})
                st.success(res.get("message", "追赶完成"))
            else:
                res = sim_result.get("result", sim_result)
                st.json(res)

    with tab_free:
        st.caption("DeepSeek V4 Pro 自主决策对照线 — 仅使用原始 MCP 数据，不依赖 Agent 评分")
        col_run, col_status = st.columns(2)
        with col_run:
            if st.button("🤖 运行今日决策", key="fl_run", use_container_width=True, type="primary"):
                with st.spinner("DeepSeek V4 Pro 决策中（约 30-90 秒）..."):
                    r = _api("/api/advisory/free-line", "POST", {
                        "portfolio_id": "", "action": "start"
                    })
                    st.session_state["advisory_free_result"] = r
        with col_status:
            if st.button("📋 查询状态", key="fl_status", use_container_width=True):
                with st.spinner("查询中..."):
                    r = _api("/api/advisory/free-line", "POST", {
                        "portfolio_id": "", "action": "status"
                    })
                    st.session_state["advisory_free_result"] = r

        free_result = st.session_state.get("advisory_free_result")
        if free_result:
            if "error" in free_result:
                st.error(free_result["error"])
            else:
                action = free_result.get("action", "")
                ctx = free_result.get("context", {})
                if action == "status":
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("总资产", f"{ctx.get('total_value', 0):,.0f}")
                    with col_b:
                        st.metric("收益率", f"{ctx.get('return_rate', 0):+.2f}%")
                    with col_c:
                        st.metric("持仓数", ctx.get("holdings_count", 0))
                    st.caption(f"可用现金: {ctx.get('cash', 0):,.0f}")
                    positions = ctx.get("positions", {})
                    if positions:
                        st.markdown("**当前持仓:**")
                        for code, pos in positions.items():
                            st.write(f"- {code}: {pos.get('quantity',0)}股, 成本 {pos.get('cost_price',0):.2f}, 现价 {pos.get('current_price',0):.2f}")
                elif action == "start":
                    reasoning = free_result.get("reasoning", "")
                    decisions = free_result.get("decisions", [])
                    executed = free_result.get("executed", 0)
                    if reasoning:
                        st.markdown(f"**决策思路:** {reasoning}")
                    st.success(f"已执行 {executed} 笔交易")
                    if decisions:
                        with st.expander("查看决策详情", expanded=False):
                            for d in decisions:
                                st.write(f"- {d.get('action','')} {d.get('stock_code','')} "
                                         f"{d.get('quantity',0)}股 @ {d.get('price',0):.2f} — {d.get('reason','')}")
                    st.caption(f"当前总资产: {ctx.get('total_value', 0):,.0f}, 收益率: {ctx.get('return_rate', 0):+.2f}%")


def _render_report():
    """📄 收益报告 — 真实报告生成"""
    st.markdown("### 📄 收益报告")
    st.caption("生成回测/模拟盘收益分析报告（支持 Markdown / HTML / PDF）")

    portfolios = st.session_state.get("advisory_portfolio_list", [])
    if not portfolios:
        pf_result = _api("/api/advisory/portfolio/list")
        if "error" not in pf_result:
            st.session_state["advisory_portfolio_list"] = pf_result.get("portfolios", [])
            portfolios = st.session_state["advisory_portfolio_list"]

    if not portfolios:
        st.info("请先在「我的持仓」中创建组合")
        return

    pf_options = {p.get("name", p.get("portfolio_id", "")): p for p in portfolios}

    col_pf, col_type = st.columns(2)
    with col_pf:
        selected = st.selectbox("选择组合", list(pf_options.keys()), key="report_pf")
    with col_type:
        report_type = st.selectbox("报告类型", ["backtest", "simulation"],
                                   format_func=lambda x: "回测报告" if x == "backtest" else "模拟盘报告")

    pf = pf_options[selected]
    pf_id = pf.get("portfolio_id", "")

    if report_type == "backtest":
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input("开始日期", value=datetime.now() - timedelta(days=365), key="rp_start")
        with col_e:
            end_date = st.date_input("结束日期", value=datetime.now(), key="rp_end")

    include_ds = st.checkbox("包含 DeepSeek 自由线对比（如有数据）", value=False)

    if st.button("📄 生成报告", type="primary", use_container_width=True):
        body = {
            "portfolio_id": pf_id,
            "report_type": report_type,
            "include_deepseek": include_ds,
        }
        if report_type == "backtest":
            body["start_date"] = start_date.strftime("%Y-%m-%d")
            body["end_date"] = end_date.strftime("%Y-%m-%d")

        with st.spinner("正在调用 AI 生成报告（可能需要 1-2 分钟）..."):
            result = _api("/api/advisory/report", "POST", body)
            st.session_state["advisory_report_result"] = result

    report_result = st.session_state.get("advisory_report_result")
    if report_result:
        if "error" in report_result:
            st.error(report_result["error"])
        else:
            st.success("报告生成成功！")

            # 摘要指标
            summary = report_result.get("summary", {})
            if summary:
                cols = st.columns(4)
                with cols[0]:
                    st.metric("总收益率", f"{summary.get('total_return_pct', 0):.2f}%")
                with cols[1]:
                    st.metric("最大回撤", f"{summary.get('max_drawdown_pct', 0):.2f}%")
                with cols[2]:
                    st.metric("夏普比率", f"{summary.get('sharpe_ratio', 0):.2f}")
                with cols[3]:
                    st.metric("交易次数", summary.get("trade_count", 0))

            # 报告内容预览
            content = report_result.get("report_content", "")
            with st.expander("📄 报告内容预览", expanded=True):
                st.markdown(content[:5000] + ("\n\n..." if len(content) > 5000 else ""))

            # 下载链接
            st.divider()
            st.caption("📥 报告文件下载")
            col_md, col_html, col_pdf = st.columns(3)
            with col_md:
                md_path = report_result.get("report_path", "")
                if md_path and os.path.isfile(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        md_content = f.read()
                    st.download_button(
                        "📄 下载 Markdown",
                        data=md_content,
                        file_name=os.path.basename(md_path),
                        mime="text/markdown",
                        use_container_width=True,
                    )
                else:
                    st.caption("Markdown 未生成")
            with col_html:
                html_path = report_result.get("html_path", "")
                if html_path and os.path.isfile(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.download_button(
                        "🌐 下载 HTML",
                        data=html_content,
                        file_name=os.path.basename(html_path),
                        mime="text/html",
                        use_container_width=True,
                    )
                else:
                    st.caption("HTML 未生成")
            with col_pdf:
                pdf_path = report_result.get("pdf_path", "")
                if pdf_path and os.path.isfile(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_content = f.read()
                    st.download_button(
                        "📕 下载 PDF",
                        data=pdf_content,
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.caption("PDF 需要 fpdf2 库")

            chart_path = report_result.get("chart_path", "")
            if chart_path and os.path.isfile(chart_path):
                st.image(chart_path, caption="收益对比图")


# ──────────────────────────────────────────────
# 右侧 AI 顾问面板
# ──────────────────────────────────────────────

def _render_ai_panel(current_page_key: str):
    """右侧常驻 AI 顾问面板 — 上下文感知 + 真实对话"""
    page = PAGE_CONFIG.get(current_page_key, PAGE_CONFIG["ai_chat"])

    st.markdown(
        f'<div class="theme-card-l3" style="text-align:center;">'
        f'<div style="font-size:1.5em;font-weight:800;color:#1e40af;">🤖 AI 顾问</div>'
        f'<div style="font-size:0.85em;color:#64748b;margin-top:4px;">'
        f'M1 MiMo-V2.5-Pro · 当前页面：{page["icon"]} {page["label"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 上下文感知提示
    context_hints = {
        "ai_chat":    "我在关注您的对话，随时为您解答投资问题。",
        "recommend":  "我看到您在查看股票推荐。我可以解释推荐逻辑或分析某只个股。",
        "portfolio":  "您正在管理持仓组合。需要我分析持仓风险或建议调仓吗？",
        "strategies": "您正在浏览策略市场。我可以比较不同策略或推荐适合您的策略。",
        "backtest":   "您在使用回测功能。我可以帮您解读回测结果或优化参数。",
        "report":     "您在准备收益报告。需要我建议报告内容或生成草稿吗？",
    }
    hint = context_hints.get(current_page_key, "随时为您提供投资建议。")

    st.markdown(
        f'<div class="theme-card" style="font-size:0.9em;color:#0f172a;">'
        f'<div style="font-weight:600;margin-bottom:6px;">💡 当前建议</div>'
        f'{hint}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 快捷问询 — 调用真实 API
    quick_questions = {
        "ai_chat":    ["今天市场热点板块有哪些？", "帮我分析一下大盘走势"],
        "recommend":  ["推荐逻辑是什么？", "这些股票适合什么风险偏好？"],
        "portfolio":  ["如何优化我的持仓？", "当前组合风险等级如何？"],
        "strategies": ["哪种策略适合我？", "策略的历史表现如何？"],
        "backtest":   ["回测结果说明什么？", "如何改进策略参数？"],
        "report":     ["报告包含哪些内容？", "如何解读收益报告？"],
    }
    qqs = quick_questions.get(current_page_key, ["今天市场怎么样？", "有什么投资建议？"])

    st.markdown(
        f'<div style="font-size:0.85em;color:#64748b;margin-bottom:6px;">快捷提问</div>',
        unsafe_allow_html=True,
    )
    for q in qqs:
        if st.button(f"💬 {q}", key=f"qq_{q}", use_container_width=True):
            st.session_state["advisory_chat"].append({"role": "user", "content": q})
            page_ctx = json.dumps({
                "page": PAGE_CONFIG[current_page_key]["label"],
                "data": {},
            }, ensure_ascii=False)
            result = _api("/api/advisory/chat", "POST", {
                "message": q,
                "session_id": "default",
                "page_context": page_ctx,
            })
            reply = result.get("reply", result.get("error", "抱歉，服务暂时不可用"))
            st.session_state["advisory_chat"].append({"role": "assistant", "content": reply})
            st.rerun()

    st.divider()

    # 右侧面板快捷输入 — 仅在非 ai_chat 页面显示
    if current_page_key != "ai_chat":
        quick_input = st.chat_input("向 AI 顾问提问...", key="advisory_right_input")
        if quick_input and quick_input.strip():
            st.session_state["advisory_chat"].append({"role": "user", "content": quick_input.strip()})
            page_ctx = json.dumps({
                "page": PAGE_CONFIG[current_page_key]["label"],
                "data": {},
            }, ensure_ascii=False)
            result = _api("/api/advisory/chat", "POST", {
                "message": quick_input.strip(),
                "session_id": "default",
                "page_context": page_ctx,
            })
            reply = result.get("reply", result.get("error", "抱歉，服务暂时不可用"))
            st.session_state["advisory_chat"].append({"role": "assistant", "content": reply})
            st.rerun()

        st.divider()
        if st.button("💬 前往完整对话页", use_container_width=True):
            st.session_state["advisory_page"] = "ai_chat"
            st.rerun()


# ──────────────────────────────────────────────
# 页面布局
# ──────────────────────────────────────────────

st.markdown(
    '<div class="theme-page-title">🤖 智能投顾</div>',
    unsafe_allow_html=True,
)
st.caption("AI 驱动的一站式投顾服务 — 股票推荐、持仓管理、策略回测、收益分析")

# 子页面切换 radio（水平）
page_keys = list(PAGE_CONFIG.keys())
current_idx = page_keys.index(st.session_state["advisory_page"])

chosen = st.radio(
    "功能导航",
    options=page_keys,
    index=current_idx,
    format_func=lambda k: f"{PAGE_CONFIG[k]['icon']} {PAGE_CONFIG[k]['label']}",
    horizontal=True,
    label_visibility="collapsed",
)
st.session_state["advisory_page"] = chosen

st.divider()

# 进入板块时自动追赶（设计 §7.7.1）
if not st.session_state["advisory_auto_catchup_done"]:
    st.session_state["advisory_auto_catchup_done"] = True
    cu = _api("/api/advisory/auto-catch-up", "POST")
    if cu and "error" not in cu and cu.get("total_missed_days", 0) > 0:
        st.toast(cu.get('message', '已自动追赶'), icon="✅")

# 两栏布局：左(3) + 右(2)
left_col, right_col = st.columns([3, 2])

with left_col:
    {
        "ai_chat": _render_ai_chat,
        "recommend": _render_recommend,
        "portfolio": _render_portfolio,
        "strategies": _render_strategies,
        "backtest": _render_backtest,
        "report": _render_report,
    }[st.session_state["advisory_page"]]()

with right_col:
    _render_ai_panel(st.session_state["advisory_page"])
