"""
智能投顾 — AI 投顾助手: AI顾问对话、股票推荐、持仓管理、策略市场、回测&模拟盘、收益报告
"""
import os
import sys

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
from datetime import datetime

st.set_page_config(page_title="智能投顾", page_icon="🤖", layout="wide")

from theme import inject_global_styles
inject_global_styles()

# ──────────────────────────────────────────────
# 状态初始化
# ──────────────────────────────────────────────
if "advisory_page" not in st.session_state:
    st.session_state["advisory_page"] = "ai_chat"
if "advisory_chat" not in st.session_state:
    st.session_state["advisory_chat"] = []
if "advisory_recommend_loaded" not in st.session_state:
    st.session_state["advisory_recommend_loaded"] = False
if "advisory_portfolios" not in st.session_state:
    st.session_state["advisory_portfolios"] = []
if "advisory_strategies" not in st.session_state:
    st.session_state["advisory_strategies"] = []

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
    """💬 AI顾问对话 — 聊天界面"""
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
        # 模拟回复（占位）
        st.session_state["advisory_chat"].append({
            "role": "assistant",
            "content": f"收到您的问题：「{prompt.strip()}」\n\n智能投顾分析引擎正在处理，功能即将上线。敬请期待！",
        })
        st.rerun()

    # 空状态提示
    if not st.session_state["advisory_chat"]:
        st.info("👋 您好！我是您的智能投顾助手，可以为您提供:\n\n"
                "- 📊 **个股分析** — 输入股票代码或名称\n"
                "- 📈 **市场热点** — 当前板块轮动与资金流向\n"
                "- 💰 **持仓诊断** — 组合风险与收益评估\n"
                "- 🎯 **策略建议** — 根据您的风险偏好推荐策略\n\n"
                "请在下方输入框开始对话。")


def _render_recommend():
    """📊 股票推荐 — 推荐列表"""
    st.markdown("### 📊 股票推荐")
    st.caption("基于多因子模型与市场分析，为您推荐优质标的")

    col_ref, col_info = st.columns([1, 4])
    with col_ref:
        if st.button("🔄 刷新推荐", type="primary", use_container_width=True):
            st.session_state["advisory_recommend_loaded"] = True
            st.rerun()
    with col_info:
        st.caption("点击刷新获取最新推荐结果")

    st.divider()

    if not st.session_state["advisory_recommend_loaded"]:
        st.info("点击「刷新推荐」获取今日股票推荐列表")
    else:
        # 推荐结果占位
        st.markdown(
            '<div class="theme-card-l3">'
            '<div style="font-weight:700;font-size:1.1em;margin-bottom:8px;">📊 今日推荐组合</div>'
            '<div style="color:#64748b;font-size:0.9em;">推荐引擎正在计算中，推荐结果将在后端就绪后展示。</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        _recommend_placeholder()


def _recommend_placeholder():
    """推荐结果的占位展示"""
    recs = [
        ("--", "--", "等待后端推荐引擎就绪"),
        ("--", "--", "等待后端推荐引擎就绪"),
        ("--", "--", "等待后端推荐引擎就绪"),
    ]
    cols = st.columns(3)
    for i, (code, name, reason) in enumerate(recs):
        with cols[i]:
            st.markdown(
                f'<div class="theme-card" style="text-align:center;padding:1rem;">'
                f'<div style="font-size:1.3em;font-weight:800;color:#2563eb;">{code}</div>'
                f'<div style="font-size:1em;color:#0f172a;margin:4px 0;">{name}</div>'
                f'<div style="font-size:0.8em;color:#64748b;">{reason}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_portfolio():
    """💰 我的持仓 — 持仓管理"""
    st.markdown("### 💰 我的持仓")
    st.caption("管理您的投资组合，查看持仓收益与风险评估")

    # 操作按钮
    col_add, col_eval, col_sp = st.columns([1, 1, 3])
    with col_add:
        if st.button("＋ 新建组合", use_container_width=True):
            st.session_state["advisory_portfolios"].append({
                "name": f"组合{len(st.session_state['advisory_portfolios'])+1}",
                "created": datetime.now().strftime("%Y-%m-%d"),
                "holdings": [],
            })
            st.rerun()
    with col_eval:
        if st.button("📊 组合评估", use_container_width=True):
            pass

    st.divider()

    if not st.session_state["advisory_portfolios"]:
        st.info("您还没有创建任何投资组合，点击「新建组合」开始")
    else:
        for i, pf in enumerate(st.session_state["advisory_portfolios"]):
            with st.expander(f"📁 {pf['name']}（创建于 {pf['created']}）", expanded=(i == 0)):
                st.markdown(
                    f'<div style="color:#64748b;font-size:0.85em;">'
                    f'持仓 {len(pf["holdings"])} 只标的 · 组合评估功能即将上线'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if pf["holdings"]:
                    for h in pf["holdings"]:
                        st.markdown(f"- {h}")
                else:
                    st.caption("暂无持仓，即将支持添加股票")


def _render_strategies():
    """📈 策略市场 — 策略目录"""
    st.markdown("### 📈 策略市场")
    st.caption("浏览并绑定投资策略，为您的组合赋能")

    strategies = [
        {"name": "稳健增长策略", "risk": "低风险", "desc": "精选低波动蓝筹，长期持有", "term": "长期"},
        {"name": "价值发现策略", "risk": "中风险", "desc": "估值修复 + 股息收益", "term": "中长线"},
        {"name": "趋势追踪策略", "risk": "中高风险", "desc": "技术指标驱动，顺势而为", "term": "中短线"},
        {"name": "事件驱动策略", "risk": "高风险", "desc": "公告事件 + 资金流博弈", "term": "短线"},
    ]

    for s in strategies:
        st.markdown(
            f'<div class="theme-card" style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div><div style="font-weight:700;">{s["name"]}</div>'
            f'<div style="font-size:0.85em;color:#64748b;">{s["desc"]}</div>'
            f'<span style="font-size:0.75em;padding:2px 8px;border-radius:8px;'
            f'background:{"#d1fae5" if s["risk"]=="低风险" else "#fef3c7" if s["risk"]=="中风险" else "#fee2e2"};'
            f'color:{"#059669" if s["risk"]=="低风险" else "#d97706" if s["risk"]=="中风险" else "#dc2626"};'
            f'">{s["risk"]}</span>'
            f'<span style="font-size:0.75em;color:#94a3b8;margin-left:8px;">{s["term"]}</span>'
            f'</div>'
            f'<div><button disabled style="padding:4px 16px;border-radius:8px;border:1px solid #dbeafe;'
            f'background:#f0f7ff;color:#2563eb;font-size:0.85em;">即将上线</button></div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_backtest():
    """📉 回测&模拟盘 — 回测结果"""
    st.markdown("### 📉 回测&模拟盘")
    st.caption("查看策略历史回测结果与模拟盘运行情况")

    tab_backtest, tab_sim = st.tabs(["📊 历史回测", "🖥️ 模拟盘"])

    with tab_backtest:
        st.markdown(
            '<div class="theme-card-l1">'
            '<div style="font-weight:700;margin-bottom:6px;">回测任务</div>'
            '<div style="color:#64748b;font-size:0.9em;">回测引擎即将上线。届时可选择策略、时间区间与初始资金进行回测。</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("＋ 新建回测任务", disabled=True):
            pass

    with tab_sim:
        st.markdown(
            '<div class="theme-card-l1">'
            '<div style="font-weight:700;margin-bottom:6px;">模拟盘</div>'
            '<div style="color:#64748b;font-size:0.9em;">模拟盘运行状态与收益曲线即将展示。</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.metric(label="模拟盘收益", value="--", delta="--")


def _render_report():
    """📄 收益报告 — 报告生成"""
    st.markdown("### 📄 收益报告")
    st.caption("生成投资组合收益分析报告与策略绩效评估")

    col_type, col_range = st.columns(2)
    with col_type:
        st.selectbox("报告类型", ["持仓收益报告", "策略绩效报告", "市场分析报告"], index=0)
    with col_range:
        st.date_input("时间范围", value=[])

    st.divider()

    if st.button("📄 生成报告", type="primary", disabled=True):
        pass

    st.info("报告生成功能即将上线。届时将支持 PDF/HTML 格式报告下载。")

    # 报告预览占位
    st.markdown(
        '<div class="theme-card" style="min-height:200px;display:flex;'
        'align-items:center;justify-content:center;color:#94a3b8;">'
        '📄 报告预览区域'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 右侧 AI 顾问面板
# ──────────────────────────────────────────────

def _render_ai_panel(current_page_key: str):
    """右侧常驻 AI 顾问面板"""
    page = PAGE_CONFIG.get(current_page_key, PAGE_CONFIG["ai_chat"])

    # 面板标题
    st.markdown(
        f'<div class="theme-card-l3" style="text-align:center;">'
        f'<div style="font-size:1.5em;font-weight:800;color:#1e40af;">🤖 AI 顾问</div>'
        f'<div style="font-size:0.85em;color:#64748b;margin-top:4px;">'
        f'当前页面：{page["icon"]} {page["label"]}</div>'
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

    # 快捷问询
    quick_questions = {
        "ai_chat":    ["今天热点板块有哪些？", "帮我分析一下大盘走势"],
        "recommend":  ["推荐逻辑是什么？", "适合什么风险偏好？"],
        "portfolio":  ["如何优化我的持仓？", "当前风险等级是多少？"],
        "strategies": ["哪种策略适合我？", "策略的历史收益如何？"],
        "backtest":   ["回测结果说明什么？", "如何改进策略参数？"],
        "report":     ["报告包含哪些内容？", "多久生成一次？"],
    }
    qqs = quick_questions.get(current_page_key, ["今天市场怎么样？", "有什么投资建议？"])

    st.markdown(
        f'<div style="font-size:0.85em;color:#64748b;margin-bottom:6px;">快捷提问</div>',
        unsafe_allow_html=True,
    )
    for q in qqs:
        if st.button(f"💬 {q}", key=f"qq_{q}", use_container_width=True):
            st.session_state["advisory_chat"].append({"role": "user", "content": q})
            st.session_state["advisory_chat"].append({
                "role": "assistant",
                "content": f"收到您的问题：「{q}」\n\n智能投顾分析引擎正在处理，功能即将上线。敬请期待！",
            })
            st.session_state["advisory_page"] = "ai_chat"
            st.rerun()

    st.divider()

    # 聊天输入（右侧面板快捷问询）
    quick_input = st.chat_input("向 AI 顾问提问...", key="advisory_right_input")
    if quick_input and quick_input.strip():
        st.session_state["advisory_chat"].append({"role": "user", "content": quick_input.strip()})
        st.session_state["advisory_chat"].append({
            "role": "assistant",
            "content": f"收到您的问题：「{quick_input.strip()}」\n\n智能投顾分析引擎正在处理，功能即将上线。敬请期待！",
        })
        st.session_state["advisory_page"] = "ai_chat"
        st.rerun()

    # 跳转到对话页面的链接
    if current_page_key != "ai_chat":
        st.divider()
        if st.button("💬 前往完整对话页", use_container_width=True):
            st.session_state["advisory_page"] = "ai_chat"
            st.rerun()


# ──────────────────────────────────────────────
# 页面布局
# ──────────────────────────────────────────────

# 标题
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

# 两栏布局：左(3) + 右(2)
left_col, right_col = st.columns([3, 2])

with left_col:
    # 渲染当前子页面
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
