"""
Streamlit 入口页 — 七大功能导航中心
"""
import streamlit as st

st.set_page_config(
    page_title="AI 投资研究助手",
    page_icon="📈",
    layout="wide",
)

# ── 侧边栏导航 ─────────────────────────────────
with st.sidebar:
    st.page_link("Home.py", label="🏠 首页", use_container_width=True)
    st.markdown("---")
    st.page_link("pages/01_股票查询.py", label="🔍 股票查询", use_container_width=True)
    st.page_link("pages/02_股票池.py", label="📊 股票池", use_container_width=True)
    st.page_link("pages/03_批量打分.py", label="📋 批量打分", use_container_width=True)
    st.page_link("pages/04_智能问答.py", label="💬 智能问答", use_container_width=True)
    st.page_link("pages/05_基金专区.py", label="🏦 基金专区", use_container_width=True)
    st.page_link("pages/06_模拟分析与迭代.py", label="📈 模拟分析与迭代", use_container_width=True)
    st.page_link("pages/07_智能投顾.py", label="🤖 智能投顾", use_container_width=True)

st.title("📈 AI 投资研究助手")
st.caption("A股智能分析平台 — AI驱动，数据支撑，仅供参考")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/01_股票查询.py", label="🔍 股票查询", icon="🔍",
                 help="输入股票代码或名称，快速获取分析或生成深度报告")
    st.caption("单只股票快速查询 + 深度报告生成")

    st.page_link("pages/02_股票池.py", label="📊 股票池", icon="📊",
                 help="管理短线/中线/长线投资池，打分筛选")
    st.caption("多期限股票池管理 + 打分排序")

with col2:
    st.page_link("pages/04_智能问答.py", label="💬 智能问答", icon="💬",
                 help="与资深投研分析师直接对话，支持自然语言开放式交流")
    st.caption("自然语言开放式A股分析问答 + 多轮对话")

    st.page_link("pages/03_批量打分.py", label="📋 批量打分", icon="📋",
                 help="上传Excel批量打分，快速筛选标的池")
    st.caption("Excel批量上传 + 大规模初筛")

with col3:
    st.page_link("pages/05_基金专区.py", label="🏦 基金专区", icon="🏦",
                 help="基金/ETF深度分析 + 7维度综合打分 + 持仓池管理")
    st.caption("公募基金 & ETF多维分析 + 评分 + 报告")

st.markdown("---")

col1_r2, col2_r2, col3_r2 = st.columns(3)

with col1_r2:
    st.page_link("pages/06_模拟分析与迭代.py", label="📈 模拟分析与迭代", icon="📈",
                 help="模拟盘评估、消融实验、回测、自动优化 — 系统自我迭代引擎")
    st.caption("14条实盘模拟 + 回测消融 + Agent贡献归因 + 自动优化")

st.markdown("---")
st.caption("提示：也可以使用左侧边栏 `〉` 直接切换页面")
