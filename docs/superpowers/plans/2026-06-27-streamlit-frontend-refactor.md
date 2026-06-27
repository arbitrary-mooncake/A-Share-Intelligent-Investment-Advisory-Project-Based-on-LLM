# Streamlit Frontend Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the 6-page Streamlit frontend: unify theme/CSS, eliminate code duplication, split the 650-line Page 06 into 6 focused sub-components organized under 3 Tabs, with zero feature loss.

**Architecture:** Create a shared theme layer (`.streamlit/config.toml` + `theme.py` + `common.py`), then update all existing pages and components to use it. For Page 06, extract 6 render functions into `src/app/components/eval/` and reduce the main page to ~60 lines dispatching to 3 Tabs.

**Tech Stack:** Python 3.13, Streamlit, Pandas, httpx

## Global Constraints

- Zero feature removal — every metric, button, chart, and interaction must be preserved
- Pure refactor + reskin — no business logic changes
- Must not break any other page or function
- Keep all `st.rerun()` flows intact
- Keep all async wrapper patterns intact
- Keep all session state keys identical
- 5 rounds of post-development code review (2 syntax/bug, 2 hidden logic, 1 final syntax)

---

### Task 1: Create .streamlit/config.toml — Global Theme

**Files:**
- Create: `Finance/Financial-MCP-Agent/.streamlit/config.toml`

**Interfaces:**
- Produces: Theme config consumed by Streamlit at startup

- [ ] **Step 1: Create the config file**

```toml
[theme]
primaryColor = "#2563eb"
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#0f172a"
font = "sans serif"

[server]
maxUploadSize = 50
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/.streamlit/config.toml
git commit -m "feat: add Streamlit global theme config"
```

---

### Task 2: Create src/app/theme.py — Unified CSS/Style System

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/app/theme.py`

**Interfaces:**
- Produces: `inject_global_styles()`, `COLORS` dict, `card_container()`, `score_badge_html()`, `metric_row_html()`, `divider_html()`, `section_title_html()`

- [ ] **Step 1: Write theme.py**

```python
"""
统一主题/样式系统 — 全局 CSS 注入 + 共享样式工具函数

配色体系:
  主色（品牌蓝）: #1e40af / #2563eb / #dbeafe
  成功（绿）:     #059669 / #d1fae5
  警告（黄/橙）:  #d97706 / #fef3c7
  危险（红）:     #dc2626 / #fee2e2
  信息（青）:     #0891b2 / #ccfbf1
  中性灰:         #0f172a(文字) / #64748b(次要) / #94a3b8(辅助) / #e2e8f0(分割)
"""

import streamlit as st

# ── 颜色常量 ──────────────────────────────────
COLORS = {
    "primary": "#2563eb",
    "primary_dark": "#1e40af",
    "primary_light": "#dbeafe",
    "success": "#059669",
    "success_bg": "#d1fae5",
    "warning": "#d97706",
    "warning_bg": "#fef3c7",
    "danger": "#dc2626",
    "danger_bg": "#fee2e2",
    "info": "#0891b2",
    "info_bg": "#ccfbf1",
    "text": "#0f172a",
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "border": "#e2e8f0",
    "bg": "#f8fafc",
    "white": "#ffffff",
}

# ── 卡片阴影 ──────────────────────────────────
SHADOWS = {
    1: "0 1px 3px rgba(0,0,0,0.06)",
    2: "0 4px 12px rgba(0,0,0,0.08)",
    3: "0 8px 24px rgba(37,99,235,0.10)",
}


def inject_global_styles():
    """注入全局自定义 CSS（在页面 st.set_page_config 之后调用）"""
    st.markdown(f"""
    <style>
    /* ── 全局字体优化 ── */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    /* ── 卡片容器（L2 浮起卡片，默认样式） ── */
    .theme-card {{
        background: {COLORS["white"]};
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1px solid {COLORS["border"]};
        box-shadow: {SHADOWS[2]};
    }}

    /* ── L1 基础卡片 ── */
    .theme-card-l1 {{
        background: {COLORS["white"]};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        border: 1px solid {COLORS["border"]};
        box-shadow: {SHADOWS[1]};
    }}

    /* ── L3 重点卡片（蓝色渐变） ── */
    .theme-card-l3 {{
        background: linear-gradient(145deg, #ffffff 0%, #f0f7ff 100%);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border: 1.5px solid #bfdbfe;
        box-shadow: {SHADOWS[3]};
    }}

    /* ── 优雅分割线 ── */
    .theme-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #dbeafe, transparent);
        margin: 12px 0 10px 0;
    }}

    /* ── 细分割线 ── */
    .theme-divider-thin {{
        height: 1px;
        background: {COLORS["border"]};
        margin: 2px 0;
    }}

    /* ── 区块标题 ── */
    .theme-section-title {{
        font-size: 1.15em;
        font-weight: 700;
        color: {COLORS["primary_dark"]};
        border-left: 4px solid {COLORS["primary"]};
        padding-left: 10px;
        margin-bottom: 8px;
    }}

    /* ── 页面大标题 ── */
    .theme-page-title {{
        font-size: 1.7em;
        font-weight: 800;
        color: {COLORS["text"]};
        margin-bottom: 4px;
    }}

    /* ── Metric 美化 ── */
    .stMetric {{
        background: {COLORS["bg"]};
        border-radius: 8px;
        padding: 0.5rem;
    }}

    /* ── Button 主色覆盖 ── */
    .stButton > button[kind="primary"] {{
        background-color: {COLORS["primary"]} !important;
        border-color: {COLORS["primary"]} !important;
    }}

    /* ── 选中行高亮 ── */
    .theme-row-selected {{
        background: #f0f7ff;
        border-left: 3px solid {COLORS["primary"]};
        border-radius: 6px;
        padding: 6px 10px;
    }}

    /* ── 普通行 ── */
    .theme-row-normal {{
        border-left: 3px solid transparent;
        padding: 6px 10px;
    }}

    /* ── Expander 内列表间距 ── */
    .stExpander ul ul, .stExpander ol ul, .stExpander ul ol {{
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
        padding-left: 1.5rem;
    }}
    .stExpander li {{
        margin-bottom: 0.2rem;
    }}
    .stExpander li > p {{
        margin-bottom: 0.1rem;
    }}

    /* ── 删除确认框 ── */
    .theme-confirm-box {{
        border: 1.5px solid #fca5a5;
        border-radius: 12px;
        padding: 16px 20px;
        background: linear-gradient(145deg, #fff5f5 0%, #fef2f2 100%);
        box-shadow: 0 4px 12px rgba(220,38,38,0.08);
        margin-bottom: 10px;
    }}

    /* ── 选中操作行背景 ── */
    .theme-action-row {{
        background: #f0f7ff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #b8daff;
    }}
    </style>
    """, unsafe_allow_html=True)


def card_container(shadow: int = 2):
    """返回卡片容器的开始 div HTML"""
    if shadow == 1:
        return f'<div class="theme-card-l1">'
    elif shadow == 3:
        return f'<div class="theme-card-l3">'
    return f'<div class="theme-card">'


def card_end():
    """返回卡片容器的结束 div"""
    return '</div>'


def divider():
    """返回优雅分割线 HTML"""
    return '<div class="theme-divider"></div>'


def section_title(text: str):
    """返回区块标题 HTML"""
    return f'<div class="theme-section-title">{text}</div>'


def score_badge_html(score, label: str = "") -> str:
    """统一分数徽章 HTML。
    分数 >= 80: 绿色, >= 60: 黄色, < 60: 红色, None: 灰色。
    """
    if score is None:
        color = COLORS["text_muted"]
        bg = "#f1f5f9"
        text = "N/A"
    else:
        try:
            s = float(score)
        except (ValueError, TypeError):
            color = COLORS["text_muted"]
            bg = "#f1f5f9"
            text = str(score)
            return f'<span style="display:inline-block;padding:3px 12px;background:{bg};color:{color};border-radius:10px;font-weight:700;font-size:0.88em;border:1px solid {color}30;">{text}</span>'

        if s >= 80:
            color = COLORS["success"]
            bg = COLORS["success_bg"]
        elif s >= 60:
            color = COLORS["warning"]
            bg = COLORS["warning_bg"]
        else:
            color = COLORS["danger"]
            bg = COLORS["danger_bg"]
        text = f"{label} {s:.0f}" if label else f"{s:.0f}"

    return (
        f'<span style="display:inline-block;padding:3px 12px;'
        f'background:{bg};color:{color};border-radius:10px;'
        f'font-weight:700;font-size:0.88em;border:1px solid {color}25;">'
        f'{text}</span>'
    )


def score_color(score) -> str:
    """返回分数对应的颜色值"""
    if score is None:
        return COLORS["text_muted"]
    try:
        s = float(score)
    except (ValueError, TypeError):
        return COLORS["text_muted"]
    if s >= 80:
        return COLORS["success"]
    elif s >= 60:
        return COLORS["warning"]
    return COLORS["danger"]


def score_bg(score) -> str:
    """返回分数对应的背景色"""
    if score is None:
        return "#f3f4f6"
    try:
        s = float(score)
    except (ValueError, TypeError):
        return "#f3f4f6"
    if s >= 80:
        return COLORS["success_bg"]
    elif s >= 60:
        return COLORS["warning_bg"]
    return COLORS["danger_bg"]


def page_title(title: str):
    """渲染统一的页面标题"""
    st.markdown(
        f'<div class="theme-page-title">{title}</div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/theme.py
git commit -m "feat: add unified theme/style system (theme.py)"
```

---

### Task 3: Create src/app/components/common.py — Shared Utilities

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/app/components/common.py`

**Interfaces:**
- Produces: `safe_str()`, `safe_float()`, `fmt_time()`, `strip_exchange_prefix()`, `format_price_change()`, `format_turnover()`

- [ ] **Step 1: Write common.py**

```python
"""
共享工具函数 — 安全类型转换、时间格式化、字符串处理
所有组件和页面统一使用这些函数，消除各处重复定义。
"""


def safe_str(val) -> str:
    """安全转换：None/空/占位文本 → 'N/A'，否则返回清理后的值"""
    if val is None:
        return "N/A"
    s = str(val).strip()
    if s == "" or s == "None" or s == "数据待查询":
        return "N/A"
    s = s.replace("…", "").replace("...", "").strip()
    return s if s else "N/A"


def safe_float(val):
    """安全转换为 float，失败返回 None"""
    if val is None:
        return None
    try:
        return float(str(val).replace("%", "").replace("倍", "").strip())
    except (ValueError, TypeError):
        return None


def fmt_time(t: str) -> str:
    """把时间格式化为 yyyy-mm-dd hh:mm，去掉 T 和微秒"""
    if not t:
        return "—"
    t = t.replace("T", " ")
    if "." in t:
        t = t.split(".")[0]
    if " " in t:
        date_part, time_part = t.split(" ", 1)
        time_parts = time_part.split(":")
        if len(time_parts) >= 2:
            return f"{date_part} {time_parts[0]}:{time_parts[1]}"
    return t


def strip_exchange_prefix(code: str) -> str:
    """去除交易所前缀，如 sh.688256 → 688256"""
    if code.startswith("sh.") or code.startswith("sz."):
        return code[3:]
    return code


def format_price_change(val) -> str:
    """格式化涨跌幅，统一保留一位小数"""
    if val is None or val == "N/A" or val == "" or val == "数据待查询":
        return "N/A"
    s = str(val).replace("…", "").replace("...", "").strip()
    if s == "数据待查询":
        return "N/A"
    try:
        num = float(s.replace("%", ""))
        return f"{num:.1f}%"
    except (ValueError, TypeError):
        return s if s else "N/A"


def format_turnover(val) -> str:
    """格式化换手率"""
    if val is None or val == "N/A" or val == "" or val == "数据待查询":
        return "N/A"
    s = str(val).replace("…", "").replace("...", "").strip()
    return s if s else "N/A"


def price_color(val) -> str:
    """涨跌幅正负颜色"""
    if val == "N/A":
        return "#888"
    try:
        num = float(val.replace("%", ""))
        return "#dc3545" if num < 0 else "#28a745" if num > 0 else "#888"
    except (ValueError, TypeError):
        return "#888"
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/components/common.py
git commit -m "feat: add shared utility functions (common.py)"
```

---

### Task 4: Create eval Sub-Components (6 files)

**Files:**
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/__init__.py`
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/pool_health.py`
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/ops_panel.py`
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/holdings_table.py`
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/backtest_panel.py`
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/contribution.py`
- Create: `Finance/Financial-MCP-Agent/src/app/components/eval/trends_charts.py`

**Interfaces:**
- Each component receives `st` from the caller (standard Streamlit pattern)
- Each component is a pure render function: takes data + callbacks, returns nothing
- No component owns session state keys directly

- [ ] **Step 1: Create __init__.py**

```python
"""评测系统前端子组件包"""
```

- [ ] **Step 2: Create pool_health.py — 精筛池健康卡片**

Extract lines 74-146 from current `06_模拟分析与迭代.py` into this render function.

```python
"""精筛池健康卡片组件"""
import streamlit as st
from datetime import datetime


def render_pool_health(health: dict) -> None:
    """渲染精筛池健康状态卡片。
    
    Args:
        health: orch.pool_manager.get_pool_health() 返回的 dict
    """
    emoji_map = {"red": "\U0001f534", "yellow": "\U0001f7e1", "green": "\U0001f7e2"}
    status_label = {"red": "需更新", "yellow": "注意", "green": "健康"}
    pool_configs = [
        ("short", "短线", 100),
        ("medium", "中线", 80),
        ("long", "长线", 60),
    ]
    
    st.subheader("\U0001f4ca 精筛池健康")
    col_h1, col_h2, col_h3 = st.columns(3)
    cols_by_idx = {0: col_h1, 1: col_h2, 2: col_h3}

    for i, (term, label, target_size) in enumerate(pool_configs):
        h = health.get(term, {})
        emoji = emoji_map.get(h.get("status", "green"), "\U0001f7e2")
        s_label = status_label.get(h.get("status", "green"), "健康")
        with cols_by_idx[i]:
            st.markdown(f"### {emoji} {label}({target_size}只)")
            last_upd = h.get("last_update", "从未更新")
            if isinstance(last_upd, str) and len(last_upd) >= 10:
                last_upd = last_upd[:10]
            st.caption(
                f"状态: **{s_label}**  |  "
                f"更新于: {last_upd}  |  "
                f"平均分: {h.get('details', {}).get('avg_score', 'N/A')}"
            )
            if h.get("triggers"):
                for t in h["triggers"]:
                    if h["status"] == "red":
                        st.error(t)
                    else:
                        st.warning(t)
            else:
                st.success("状态健康，无需操作")
```

- [ ] **Step 3: Create ops_panel.py — 操作面板**

Extract lines 150-270 from current `06_模拟分析与迭代.py`.

```python
"""操作面板组件 — 一键检查 / 调仓 / 结算 / 池更新"""
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
                try:
                    def on_stage(stage_name: str, message: str):
                        progress_map = {
                            "0_hard_screen": (5, "Layer 0: 硬筛 (去ST/新股/BJ/B股/低流动)"),
                            "1_batch_score": (20, "Layer 1: M1/M3批量粗筛分档"),
                            "2_quick_screen": (60, "Layer 2: M2快筛过滤初筛池"),
                            "3_formal_score": (75, "Layer 3: 7Agent+3Scorer精筛"),
                            "done": (100, "完成!"),
                        }
                        pct, text = progress_map.get(stage_name, (50, message))
                        progress_bar.progress(pct, text=text[:80])
                        status_box.update(label=f"{stage_name}: {message[:60]}", state="running")

                    update_result = _run_async(
                        orch.run_pool_update(pool_term, on_stage=on_stage)
                    )
                    progress_bar.progress(100, text="完成!")
                    status_box.update(label="精筛池更新完成", state="complete")

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
                except Exception as e:
                    progress_bar.progress(100, text="失败")
                    status_box.update(label=f"更新失败", state="error")
                    st.error(f"精筛池更新失败: {e}")
        st.caption(
            "\U0001f446 四层筛选管线（总纲§4.1）:\n\n"
            "**Layer 0 硬筛**: 去ST/新股/BJ/B股/近20日日均成交额<2000万 → ~4500只\n"
            "**Layer 1 批量粗筛**: M1/M3生产模型5只/批打分, 分4档\n"
            "**Layer 2 快筛**: M2(Qwen3.6-Flash)过滤初筛池\n"
            "**Layer 3 精筛**: 白名单+初筛通过1:1.2差额 → 正式7Agent+3Scorer → LLM动态阈值\n\n"
            "⏱ 预计耗时: ≈1小时(短线100只)"
        )

    with col4:
        if st.button("\U0001f50d 刷新数据", use_container_width=True):
            st.rerun()
        st.caption("\U0001f446 页面不自动刷新。想看最新持仓/收益/趋势时点这里。")
```

- [ ] **Step 4: Create holdings_table.py — 各线路持仓与收益**

Extract lines 313-345 from current `06_模拟分析与迭代.py`.

```python
"""各线路持仓与收益表格组件"""
import streamlit as st
import pandas as pd


def render_holdings_table(lines: list) -> None:
    """渲染14条模拟盘持仓与收益表格。
    
    Args:
        lines: status.get("lines", []) 返回的线路数据列表
    """
    st.markdown("---")
    st.subheader("\U0001f4ca 各线路持仓与收益 — 14条模拟盘实时状态")

    st.caption("""
    **线路说明**: S-L0~S-L7 = 短线消融实验（每天同一起点出发，各差1个Agent）。
    S-L8 = 短线正常交易（连续持仓）。S-L9/M-L1/L-L1 = LLM自主投资对照组。M-L0/L-L0 = 中/长线。
    """)

    tab_s, tab_m, tab_l = st.tabs(["⚡ 短线 (10条)", "\U0001f4c5 中线 (2条)", "\U0001f3db️ 长线 (2条)"])

    for tab, term in [(tab_s, "short"), (tab_m, "medium"), (tab_l, "long")]:
        with tab:
            term_lines = [l for l in lines if l.get("term") == term]
            if term_lines:
                df_data = []
                for l in term_lines:
                    df_data.append({
                        "线路": l["line_id"],
                        "类型": l.get("type", ""),
                        "持仓数": l.get("holdings_count", 0),
                        "总市值(万)": round(l.get("total_value", 0) / 10000, 1),
                        "累计收益%": l.get("cumulative_return_pct", 0),
                        "最大回撤%": l.get("max_drawdown_pct", 0),
                        "今日收益%": l.get("daily_return_pct", 0),
                    })
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)
            else:
                st.info("\U0001f446 还没数据。先更新精筛池，再点'一键完整检查'，这里就会显示各条线的实时持仓和收益率。")
```

- [ ] **Step 5: Create backtest_panel.py — 回测面板**

Extract lines 350-485 from current `06_模拟分析与迭代.py`.

```python
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
```

- [ ] **Step 6: Create contribution.py — Agent贡献榜 + 优化建议**

Extract lines 487-531 and 620-645 from current `06_模拟分析与迭代.py`.

```python
"""Agent贡献榜 + 优化建议组件"""
import streamlit as st
import pandas as pd


def render_contribution_leaderboard(eval_ready: bool) -> None:
    """渲染 Agent 贡献榜 — 谁在帮忙，谁在拖后腿"""
    st.markdown("---")
    st.subheader("\U0001f3c6 Agent贡献分 — 谁在帮忙，谁在拖后腿")
    st.caption("ΔL > 0 = 正贡献（去掉它系统变差）。ΔL < 0 = 负贡献（去掉它系统反而变好）。★★★ = 统计显著。☆ = 不显著。")

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
                    st.dataframe(pd.DataFrame(contrib_data), use_container_width=True)
                else:
                    st.info("暂无Agent贡献数据。运行回测或积累足够实盘数据后这里会自动填充。")
            else:
                st.info("暂无贡献记录。点'历史回测'跑一次，或每天运行'一键完整检查'积累实盘数据。")
        except Exception as e:
            st.caption(f"加载中: {e}")
    else:
        st.caption("评测系统初始化中...")


def render_optimization_tickets(eval_ready: bool) -> None:
    """渲染优化建议列表"""
    st.markdown("---")
    st.subheader("\U0001f527 优化建议 — 基于Loss和贡献数据自动生成")
    st.caption("""
    \U0001f916 自动 = 参数调整类。\U0001f464\U0001f916 半自动 = 系统生成方案，需你审核后执行。\U0001f464 人工 = 系统出分析报告，需你手动改代码。
    运行 `python -m src.eval optimize --analyze` 生成新建议。
    """)

    if eval_ready:
        try:
            from src.eval.repositories import get_pending_tickets
            tickets = get_pending_tickets()
            if tickets:
                for t in tickets[:5]:
                    route_icon = {"auto": "\U0001f916", "semi_auto": "\U0001f464\U0001f916", "manual": "\U0001f464"}.get(t.get("route", ""), "❓")
                    with st.expander(f"{route_icon} [{t.get('ticket_type', '')}] {t.get('title', '未命名')}"):
                        st.write(t.get("summary", "无摘要"))
                        st.caption(f"路由: {t.get('route', '')} | 严重度: {t.get('severity', '')}")
            else:
                st.info("暂无优化建议。运行 `python -m src.eval optimize --analyze` 生成。")
        except Exception as e:
            st.caption(f"加载中: {e}")
    else:
        st.caption("评测系统初始化中...")
```

- [ ] **Step 7: Create trends_charts.py — 趋势图区**

Extract lines 532-617 from current `06_模拟分析与迭代.py`.

```python
"""趋势图组件 — Score / Loss / 收益对比 / 保真度 / 耗时"""
import streamlit as st
import pandas as pd


def render_trends_tabs(eval_ready: bool, status: dict) -> None:
    """渲染趋势图区域（5个子Tabs）"""
    st.markdown("---")
    st.subheader("\U0001f4c8 趋势图 — 系统表现随时间的变化")
    st.caption("每天运行'一键完整检查'会积累数据。至少积累10个批次后趋势图才会展示。")

    if eval_ready:
        try:
            from src.eval.chart_service import ChartService
            from src.eval.memory_manager import MemoryManager
            cs = ChartService()
            mm = MemoryManager()

            batch_count = len(mm.trends.get("score_history", []))
            min_batches = 10

            if batch_count < min_batches:
                st.info(
                    f"\U0001f4ca 数据积累中: 当前{batch_count}/{min_batches}批次 "
                    f"（至少需要{min_batches}批次才能展示趋势曲线）。"
                    f"每天运行'一键完整检查'会积累1个批次。"
                )
                st.progress(batch_count / min_batches, text=f"数据积累进度: {batch_count}/{min_batches}")
            else:
                tab_t1, tab_t2, tab_t3, tab_t4, tab_t5 = st.tabs([
                    "Score趋势", "Loss趋势", "线路收益对比", "保真度趋势", "运行耗时/Token"
                ])
                with tab_t1:
                    score_data = cs.get_score_trend_data(90)
                    if score_data["data"]:
                        df = pd.DataFrame(score_data["data"])
                        if not df.empty:
                            st.line_chart(df.set_index("date")["value"], use_container_width=True)
                            st.caption("Score越高越好 = 系统评分质量在提升。曲线向上 = 优化方向对了。")
                with tab_t2:
                    loss_data = cs.get_loss_trend_data(90)
                    if loss_data["data"]:
                        df = pd.DataFrame(loss_data["data"])
                        if not df.empty:
                            st.line_chart(df.set_index("date")["value"], use_container_width=True)
                            st.caption("Loss越低越好 = 预测与实际的差距在缩小。曲线向下 = 优化有效果。")
                with tab_t3:
                    if status.get("lines"):
                        line_data = cs.get_line_comparison_data(status["lines"])
                        if line_data["data"]:
                            df = pd.DataFrame(line_data["data"])
                            if not df.empty:
                                st.bar_chart(df.set_index("line_id")["return"], use_container_width=True)
                                st.caption("对比各条线的累计收益。消融线之间差异 = Agent贡献的直观体现。")
                with tab_t4:
                    fidelity_data = cs.get_fidelity_trend_data(90)
                    if fidelity_data["data"]:
                        df = pd.DataFrame(fidelity_data["data"])
                        if not df.empty and len(df) >= 2:
                            st.line_chart(df.set_index("date")["fidelity_loss"], use_container_width=True)
                            st.caption("保真度Loss越低 = 评测模型与生产模型输出越一致。")
                            with st.expander("\U0001f4cb 详细保真度指标", expanded=False):
                                st.dataframe(df.set_index("date"), use_container_width=True)
                        else:
                            st.info("保真度数据积累中（需要至少2个批次）。")
                with tab_t5:
                    runtime_data = cs.get_runtime_trend_data(90)
                    if runtime_data["data"]:
                        df = pd.DataFrame(runtime_data["data"])
                        if not df.empty and len(df) >= 2:
                            col_r1, col_r2 = st.columns(2)
                            with col_r1:
                                st.line_chart(df.set_index("date")["duration_minutes"], use_container_width=True)
                                st.caption("运行耗时（分钟）。持续上升 → 池子变大或Agent调用量增加。")
                            with col_r2:
                                st.line_chart(df.set_index("date")["cache_hit_rate"], use_container_width=True)
                                st.caption("缓存命中率%。越高越好 → 分析Agent结果被有效复用。")
                        else:
                            st.info("运行耗时数据积累中（需要至少2个批次）。")
        except Exception:
            st.caption("趋势数据加载中...")
    else:
        st.caption("评测系统初始化中...")
```

- [ ] **Step 8: Commit all eval components**

```bash
git add Finance/Financial-MCP-Agent/src/app/components/eval/
git commit -m "feat: extract eval sub-components (6 files) from Page 06"
```

---

### Task 5: Rewrite Page 06 — 模拟分析与迭代

**Files:**
- Modify: `Finance/Financial-MCP-Agent/src/app/pages/06_模拟分析与迭代.py`

**Interfaces:**
- Consumes: `theme.inject_global_styles`, `eval/*` components
- Produces: Same user-facing page with all functionality intact

- [ ] **Step 1: Rewrite 06_模拟分析与迭代.py to ~100 lines**

Replace the entire 650-line file with the Tabs-based dispatch version below. Every original line of functionality is preserved in the eval components.

```python
"""
模拟分析与迭代 — 第6大功能：评测控制台
Tab 1: 模拟盘运营（日常高频）
Tab 2: 回测与消融（研究分析）
Tab 3: 趋势与优化（长期追踪）
"""
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="模拟分析与迭代", page_icon="📈", layout="wide")

import sys, os
_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

from theme import inject_global_styles, page_title
from components.eval.pool_health import render_pool_health
from components.eval.ops_panel import render_ops_panel
from components.eval.holdings_table import render_holdings_table
from components.eval.backtest_panel import render_backtest_panel
from components.eval.contribution import render_contribution_leaderboard, render_optimization_tickets
from components.eval.trends_charts import render_trends_tabs

inject_global_styles()
page_title("📈 模拟分析与迭代")
st.caption("评分智能体 — 模拟盘评估、消融实验、回测、自动优化。所有结果仅供参考，使用评测模型(MiMo-V2.5)，目的是优化Agent系统。")


def _run_async(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


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

# ── 顶部状态栏 ──
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

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.metric("总Score", status.get("total_score", "N/A"))
with col_s2:
    st.metric("短线Score", status.get("short_score", "N/A"))
with col_s3:
    st.metric("中线Score", status.get("medium_score", "N/A"))
with col_s4:
    st.metric("长线Score", status.get("long_score", "N/A"))

# ── 3 个顶层 Tabs ──
tab1, tab2, tab3 = st.tabs(["📊 模拟盘运营", "🔬 回测与消融", "📈 趋势与优化"])

with tab1:
    # 精筛池健康
    if eval_ready:
        health = orch.pool_manager.get_pool_health(
            lines_status=status.get("lines", [])
        )
        render_pool_health(health)

        # 对有问题的池显示快速更新入口
        problem_terms = [
            (term, label) for term, label, target_size in
            [("short", "短线", 100), ("medium", "中线", 80), ("long", "长线", 60)]
            if health.get(term, {}).get("status") in ("red", "yellow")
        ]
        if problem_terms:
            st.markdown("---")
            st.caption("以下精筛池需要关注，可快速更新：")
            action_cols = st.columns(len(problem_terms))
            emoji_map = {"red": "🔴", "yellow": "🟡", "green": "🟢"}
            for j, (term, label) in enumerate(problem_terms):
                h = health[term]
                with action_cols[j]:
                    st.caption(
                        f"{emoji_map[h['status']]} **{label}池**: "
                        f"{h.get('suggested_action', '')}"
                    )
                    if st.button(
                        f"🔄 全量更新{label}池", key=f"health_full_{term}",
                        use_container_width=True,
                    ):
                        with st.spinner(f"全量更新{label}池中..."):
                            update_result = _run_async(orch.run_pool_update(term))
                            if "error" in update_result:
                                st.error(update_result["error"])
                            else:
                                st.success(
                                    f"更新完成！共"
                                    f"{update_result.get('final_pool_size', 0)}只"
                                )
                                st.rerun()
    else:
        st.info("精筛池未初始化，请先运行评测系统。")

    # 操作面板
    st.markdown("---")
    render_ops_panel(orch, eval_ready)

    # 精筛池概览
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
        st.caption("精筛池 = 模拟盘的选股范围。14条线只能从这些池子里选股。")

        with st.expander("📋 精筛池 Top5 成分股", expanded=False):
            for term, label in [("short", "短线"), ("medium", "中线"), ("long", "长线")]:
                st.caption(f"**{label}精筛池 Top5:**")
                pool_data = orch.pool_manager.get_pool_with_scores(term)[:5]
                if pool_data:
                    for s in pool_data:
                        code = s.get("code", "?") if isinstance(s, dict) else s
                        name = s.get("name", "?") if isinstance(s, dict) else "?"
                        score = s.get("score", "?") if isinstance(s, dict) else "?"
                        st.text(f"  {code} ({name}) | {score}分")
                else:
                    st.text("  暂无数据")
    else:
        st.warning("评测系统初始化中...")

    # 持仓收益
    if eval_ready:
        lines = status.get("lines", [])
        render_holdings_table(lines)
    else:
        st.caption("评测系统初始化中...")

    # Tab 1 底部声明
    st.markdown("---")
    st.warning("⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考，目的是优化Agent系统而非追求极限投资收益。不构成任何投资建议。")

with tab2:
    if eval_ready:
        render_backtest_panel(orch, eval_ready)
    else:
        st.warning("评测系统初始化中...")

    render_contribution_leaderboard(eval_ready)

    # Tab 2 底部声明
    st.markdown("---")
    st.warning("⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考。不构成任何投资建议。")

with tab3:
    if eval_ready:
        render_trends_tabs(eval_ready, status)
    else:
        st.caption("评测系统初始化中...")

    render_optimization_tickets(eval_ready)

    # Tab 3 底部声明
    st.markdown("---")
    st.warning("⚠️ **重要提示**：本页面所有数据基于评测模型（MiMo-V2.5），仅供参考。不构成任何投资建议。")
```

- [ ] **Step 2: Commit**

```bash
git add Finance/Financial-MCP-Agent/src/app/pages/06_模拟分析与迭代.py
git commit -m "refactor: rewrite Page 06 as 3-Tab dispatch with eval sub-components"
```

---

### Task 6: Update All Pages and Components to Use theme + common

**Files:**
- Modify: `Home.py`, `01_股票查询.py`, `02_股票池.py`, `03_批量打分.py`, `04_智能问答.py`, `05_基金专区.py`
- Modify: `components/query_form.py`, `components/result_card.py`, `components/pool_table.py`, `components/quick_screen_table.py`, `components/batch_progress.py`

This task updates every existing file to use the new `theme.py` and `common.py` imports, replacing all inline CSS and duplicate utility functions.

- [ ] **Step 1: Update Home.py**

Replace inline `st.set_page_config` with theme-injected version:

```python
# After st.set_page_config(...), add:
from theme import inject_global_styles
inject_global_styles()
```

- [ ] **Step 2: Update 01_股票查询.py**

Replace inline CSS block with `inject_global_styles()`. Replace local `st.set_page_config` bug (shouldn't be called in a page if Home.py already sets it) → remove page_config call from this page (Streamlit allows only one per app, but pages can have their own). Keep the page_config for page title, add `inject_global_styles()`.

Replace result_card imports to use `common.py` functions.

- [ ] **Step 3: Update result_card.py**

Replace functions `_safe_str`, `_safe_float`, `_strip_exchange_prefix`, `_format_price_change`, `_format_turnover`, `_price_color` with imports from `common.py`.

- [ ] **Step 4: Update pool_table.py and quick_screen_table.py**

Replace `_score_color`, `_score_bg`, `_fmt_time` with imports from `theme.py` (`score_color`, `score_bg`) and `common.py` (`fmt_time`).

- [ ] **Step 5: Update batch_progress.py**

Replace LEVEL_COLORS, LEVEL_BG, LEVEL_EMOJI, LEVEL_ORDER — these are unique to batch scoring, keep them but add `from theme import COLORS` for consistency.

- [ ] **Step 6: Update query_form.py**

Replace inline CSS with theme class.

- [ ] **Step 7: Update 02_股票池.py, 03_批量打分.py, 04_智能问答.py, 05_基金专区.py**

Add `inject_global_styles()` call, remove inline CSS blocks that are now covered by theme.

- [ ] **Step 8: Commit all updates**

```bash
git add Finance/Financial-MCP-Agent/src/app/
git commit -m "refactor: migrate all pages/components to unified theme + common utilities"
```

---

### Task 7: 5-Round Deep Code Review

- [ ] **Round 1: Syntax check** — Run `python -m py_compile` on every modified file
- [ ] **Round 2: Import/API consistency check** — Verify every import resolves, every function signature matches its callers
- [ ] **Round 3: Feature parity check** — Compare old vs new code line-by-line for each page, verify no button/metric/flow was lost
- [ ] **Round 4: Session state consistency check** — Verify all `st.session_state` keys are identical between old and new code
- [ ] **Round 5: Final syntax re-check** — Re-run `python -m py_compile` on all files to ensure no regressions

- [ ] **Final commit**

```bash
git add -A
git commit -m "chore: 5-round code review complete — frontend refactor verified"
```
