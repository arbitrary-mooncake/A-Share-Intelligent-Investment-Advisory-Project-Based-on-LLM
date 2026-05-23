"""
Streamlit 入口页 — 四大功能导航中心
"""
import streamlit as st

st.set_page_config(
    page_title="股票投资顾问 Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 股票投资顾问 Agent")
st.caption("A股智能分析平台 — 选择您需要的功能")

st.markdown("---")

col1, col2 = st.columns(2)

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

st.markdown("---")
st.caption("提示：也可以使用左侧边栏 `〉` 直接切换页面")
