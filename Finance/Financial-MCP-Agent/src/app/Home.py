"""
Streamlit 入口页 — 七大功能导航中心
"""
import os
import sys

_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
from components.shared_sidebar import render_sidebar

st.set_page_config(
    page_title="AI 投资研究助手",
    page_icon="📈",
    layout="wide",
)

# ── 共享侧边栏（替代 Streamlit 自动生成的多页导航）──
render_sidebar()

# ── 注入首页专属样式 ──
st.markdown("""
<style>
/* ── 首页 hero 区域 ── */
.hero-block {
    background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #3b82f6 100%);
    border-radius: 18px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
    color: #ffffff;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(37,99,235,0.18);
}
.hero-block::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -10%;
    width: 380px;
    height: 380px;
    background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-block::after {
    content: "";
    position: absolute;
    bottom: -50%;
    left: -5%;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(147,197,253,0.15) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2.1em;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}
.hero-subtitle {
    font-size: 0.95em;
    color: rgba(255,255,255,0.82);
    font-weight: 400;
    position: relative;
    z-index: 1;
}
.hero-tags {
    margin-top: 14px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.hero-tag {
    font-size: 0.78em;
    padding: 4px 12px;
    border-radius: 20px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.22);
    color: #ffffff;
    backdrop-filter: blur(4px);
}

/* ── 导航卡片网格 ── */
.nav-card {
    display: block;
    text-decoration: none;
    background: #ffffff;
    border-radius: 14px;
    padding: 1.3rem 1.3rem 1.1rem 1.3rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    transition: all 0.22s cubic-bezier(0.4,0,0.2,1);
    height: 100%;
    position: relative;
    overflow: hidden;
}
.nav-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--accent, #2563eb);
    border-radius: 2px;
    opacity: 0;
    transition: opacity 0.22s;
}
.nav-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(37,99,235,0.12);
    border-color: #bfdbfe;
}
.nav-card:hover::before {
    opacity: 1;
}
.nav-card-icon {
    font-size: 2.0em;
    margin-bottom: 8px;
    display: block;
    line-height: 1;
}
.nav-card-title {
    font-size: 1.12em;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.nav-card-desc {
    font-size: 0.82em;
    color: #64748b;
    line-height: 1.5;
}
.nav-card-badge {
    position: absolute;
    top: 12px;
    right: 12px;
    font-size: 0.65em;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-hot {
    background: #fee2e2;
    color: #dc2626;
}
.badge-new {
    background: #d1fae5;
    color: #059669;
}

/* ── 模拟分析区 ── */
.feature-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    border: 1.5px solid #bfdbfe;
    box-shadow: 0 6px 20px rgba(37,99,235,0.08);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.feature-card-icon {
    font-size: 2.6em;
    flex-shrink: 0;
    width: 70px;
    height: 70px;
    border-radius: 16px;
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    display: flex;
    align-items: center;
    justify-content: center;
}
.feature-card-body {
    flex: 1;
}
.feature-card-title {
    font-size: 1.2em;
    font-weight: 700;
    color: #1e40af;
    margin-bottom: 4px;
}
.feature-card-desc {
    font-size: 0.88em;
    color: #475569;
    line-height: 1.5;
}

/* ── 分割区域标题 ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 1.4rem 0 1rem 0;
}
.section-header-bar {
    width: 4px;
    height: 20px;
    background: linear-gradient(180deg, #2563eb, #1e40af);
    border-radius: 2px;
}
.section-header-text {
    font-size: 1.15em;
    font-weight: 700;
    color: #0f172a;
}
.section-header-hint {
    font-size: 0.8em;
    color: #94a3b8;
    margin-left: auto;
}

/* ── 页脚 ── */
.home-footer {
    margin-top: 2rem;
    padding-top: 1.2rem;
    border-top: 1px solid #e2e8f0;
    text-align: center;
    color: #94a3b8;
    font-size: 0.8em;
}
</style>
""", unsafe_allow_html=True)

# ── Hero 区域 ──
st.markdown("""
<div class="hero-block">
    <div class="hero-title">📈 AI 投资研究助手</div>
    <div class="hero-subtitle">A股智能分析平台 — AI 驱动，数据支撑，多维度立体化研究</div>
    <div class="hero-tags">
        <span class="hero-tag">🔬 7 Agent 并行分析</span>
        <span class="hero-tag">📊 短/中/长三期限评分</span>
        <span class="hero-tag">🛡️ 风险门控</span>
        <span class="hero-tag">🤖 智能投顾</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 核心功能区（6 个入口，2 行 × 3 列）──
st.markdown("""
<div class="section-header">
    <div class="section-header-bar"></div>
    <div class="section-header-text">核心功能</div>
    <div class="section-header-hint">点击卡片进入对应功能</div>
</div>
""", unsafe_allow_html=True)

# Row 1: 股票查询 / 智能投顾 / 智能问答
r1c1, r1c2, r1c3 = st.columns(3, gap="medium")

with r1c1:
    st.markdown("""
    <div class="nav-card" style="--accent:#2563eb;">
        <div class="nav-card-icon">🔍</div>
        <div class="nav-card-title">股票查询</div>
        <div class="nav-card-desc">输入代码或名称，快速获取个股分析或生成深度报告</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/01_股票查询.py", label="进入股票查询 →",
                 use_container_width=True, help="单只股票快速查询 + 深度报告生成")

with r1c2:
    st.markdown("""
    <div class="nav-card" style="--accent:#059669;">
        <span class="nav-card-badge badge-hot">HOT</span>
        <div class="nav-card-icon">🤖</div>
        <div class="nav-card-title">智能投顾</div>
        <div class="nav-card-desc">AI 顾问对话、股票推荐、持仓管理、策略回测一站式服务</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/07_智能投顾.py", label="进入智能投顾 →",
                 use_container_width=True, help="AI驱动的一站式投顾服务")

with r1c3:
    st.markdown("""
    <div class="nav-card" style="--accent:#0891b2;">
        <div class="nav-card-icon">💬</div>
        <div class="nav-card-title">智能问答</div>
        <div class="nav-card-desc">与资深投研分析师直接对话，支持自然语言开放式交流</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/04_智能问答.py", label="进入智能问答 →",
                 use_container_width=True, help="自然语言开放式A股分析问答 + 多轮对话")

# Row 2: 股票池 / 批量打分 / 基金专区（最后一个）
r2c1, r2c2, r2c3 = st.columns(3, gap="medium")

with r2c1:
    st.markdown("""
    <div class="nav-card" style="--accent:#7c3aed;">
        <div class="nav-card-icon">📊</div>
        <div class="nav-card-title">股票池</div>
        <div class="nav-card-desc">管理短线/中线/长线投资池，打分筛选，持续跟踪</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/02_股票池.py", label="进入股票池 →",
                 use_container_width=True, help="多期限股票池管理 + 打分排序")

with r2c2:
    st.markdown("""
    <div class="nav-card" style="--accent:#d97706;">
        <div class="nav-card-icon">📋</div>
        <div class="nav-card-title">批量打分</div>
        <div class="nav-card-desc">上传 Excel 批量打分，快速筛选标的池，大规模初筛</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/03_批量打分.py", label="进入批量打分 →",
                 use_container_width=True, help="Excel批量上传 + 大规模初筛")

with r2c3:
    st.markdown("""
    <div class="nav-card" style="--accent:#db2777;">
        <div class="nav-card-icon">🏦</div>
        <div class="nav-card-title">基金专区</div>
        <div class="nav-card-desc">基金/ETF 深度分析 + 7维度综合打分 + 持仓池管理</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/05_基金专区.py", label="进入基金专区 →",
                 use_container_width=True, help="公募基金 & ETF多维分析 + 评分 + 报告")

# ── 系统迭代区 ──
st.markdown("""
<div class="section-header">
    <div class="section-header-bar"></div>
    <div class="section-header-text">系统迭代</div>
    <div class="section-header-hint">系统自我进化引擎</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="feature-card">
    <div class="feature-card-icon">📈</div>
    <div class="feature-card-body">
        <div class="feature-card-title">模拟分析与迭代</div>
        <div class="feature-card-desc">14条实盘模拟 + 回测消融 + Agent贡献归因 + 自动优化 — 系统自我迭代引擎</div>
    </div>
</div>
""", unsafe_allow_html=True)
st.page_link("pages/06_模拟分析与迭代.py", label="进入模拟分析与迭代 →",
             use_container_width=True, help="模拟盘评估、消融实验、回测、自动优化")

# ── 页脚 ──
st.markdown("""
<div class="home-footer">
    💡 提示：也可以使用左侧边栏直接切换页面 · 本平台仅供研究参考，不构成投资建议
</div>
""", unsafe_allow_html=True)
