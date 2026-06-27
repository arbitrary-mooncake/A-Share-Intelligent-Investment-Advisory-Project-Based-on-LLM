"""
批量打分页面 — Excel 上传 + 实时进度 + 结果展示（排序/筛选/选中删除）+ 导出
"""
import asyncio
import io
import json
import os
import sys
import time

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st
import pandas as pd

from components.batch_progress import (
    render_progress_bar,
    render_level_badge,
    render_level_count_bar,
    build_results_dataframe,
    LEVEL_COLORS,
    LEVEL_BG,
    LEVEL_EMOJI,
    LEVEL_ORDER,
)
from config import (
    BATCH_POLL_INTERVAL,
)
from theme import inject_global_styles
from api_client import (
    APIError,
    upload_excel,
    get_batch_progress,
    get_batch_results,
    poll_batch_results,
)

# ──────────────────────────────────────────────
# 持久化存储 — 结果存磁盘，刷新不丢失
# ──────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # .../src/app/pages/
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))  # .../ (project root)
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_BATCH_RESULTS_FILE = os.path.join(_DATA_DIR, "batch_results.json")
_BATCH_TASK_FILE = os.path.join(_DATA_DIR, "batch_current_task.json")


def _load_persisted_stocks() -> list:
    """从磁盘加载持久化的批量打分结果"""
    try:
        if os.path.exists(_BATCH_RESULTS_FILE):
            with open(_BATCH_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_persisted_stocks(stocks: list):
    """保存批量打分结果到磁盘"""
    os.makedirs(_DATA_DIR, exist_ok=True)
    try:
        with open(_BATCH_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(stocks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"保存结果失败: {e}")


def _save_current_task(batch_id: str, horizon: str):
    """持久化当前运行中的任务 ID，刷新页面后可恢复"""
    try:
        with open(_BATCH_TASK_FILE, "w", encoding="utf-8") as f:
            json.dump({"batch_id": batch_id, "horizon": horizon}, f)
    except Exception:
        pass


def _clear_current_task():
    """清除持久化的任务 ID"""
    try:
        if os.path.exists(_BATCH_TASK_FILE):
            os.remove(_BATCH_TASK_FILE)
    except Exception:
        pass


def _load_current_task() -> dict:
    """加载持久化的任务 ID"""
    try:
        if os.path.exists(_BATCH_TASK_FILE):
            with open(_BATCH_TASK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _merge_stocks(existing: list, new: list) -> list:
    """合并两批股票结果：同 code 覆盖（新数据优先），不同 code 追加"""
    merged = {_stock_key(s): s for s in existing}
    for s in new:
        key = _stock_key(s)
        merged[key] = s  # 新数据覆盖旧数据
    return list(merged.values())


def _stock_key(stock: dict) -> str:
    """获取股票的唯一键（纯数字代码）"""
    code = stock.get("code", "")
    return code.replace("sh.", "").replace("sz.", "").strip()


# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(page_title="批量打分", page_icon="📋", layout="wide")

inject_global_styles()

# ──────────────────────────────────────────────
# 自定义样式：DataFrame 多选框勾选颜色改为红色
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* 覆盖 Streamlit dataframe 多选框的勾选颜色为红色 */
[data-testid="stDataFrame"] [role="checkbox"][aria-checked="true"] svg,
[data-testid="stDataFrame"] [data-checked="true"] svg,
[data-testid="stDataFrame"] .glide-cell [role="checkbox"] svg[data-checked="true"] {
    fill: #dc2626 !important;
    color: #dc2626 !important;
}
/* 勾选框外框也改为红色 */
[data-testid="stDataFrame"] [role="checkbox"][aria-checked="true"] {
    border-color: #dc2626 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 同步包装器
# ──────────────────────────────────────────────
def _ar(coro):
    return asyncio.run(coro)

# ──────────────────────────────────────────────
# Session State 初始化
# ──────────────────────────────────────────────

_defaults = {
    "batch_task_id": None,
    "batch_status": "idle",          # idle / uploading / running / done
    "batch_stocks": [],
    "batch_stocks_loaded": False,    # 是否已从磁盘加载持久化结果
    "batch_progress": None,
    "batch_error": None,
    "batch_horizon": "medium",
    "batch_selected_codes": set(),
    "batch_filter_levels": list(LEVEL_ORDER.keys()),
    "batch_search": "",
    "batch_sort_col": "评级",
    "batch_sort_asc": True,
    "batch_del_confirm": False,
    "batch_df_version": 0,            # dataframe key 版本号，全选/取消时递增以重置客户端 selection
    "batch_last_df_key": "",          # 上一次渲染使用的 dataframe key，用于判断是否需要跳过同步
}
for k, v in _defaults.items():
    if k not in st.session_state:
        if k == "batch_stocks" and not st.session_state.get("batch_stocks_loaded"):
            # 首次加载：从磁盘恢复持久化结果
            persisted = _load_persisted_stocks()
            if persisted:
                st.session_state[k] = persisted
                st.session_state["batch_stocks_loaded"] = True
                st.session_state["batch_status"] = "done"
            else:
                st.session_state[k] = v
        elif k == "batch_status" and not st.session_state.get("batch_task_id"):
            # 检查是否有未完成的任务（刷新恢复）
            saved_task = _load_current_task()
            if saved_task.get("batch_id"):
                # 查询后端是否还在跑
                try:
                    prog = _ar(get_batch_progress(saved_task["batch_id"]))
                    status = prog.get("status", "")
                    if status in ("parsed", "fetching", "fetched", "scoring"):
                        # 任务还在跑，恢复状态
                        st.session_state["batch_task_id"] = saved_task["batch_id"]
                        st.session_state["batch_status"] = "running"
                        st.session_state["batch_horizon"] = saved_task.get("horizon", "medium")
                    elif status == "completed":
                        # 已经跑完了，直接拉结果
                        result = _ar(get_batch_results(saved_task["batch_id"]))
                        st.session_state["batch_stocks"] = result.get("stocks", [])
                        _save_persisted_stocks(st.session_state["batch_stocks"])
                        st.session_state["batch_status"] = "done"
                        st.session_state["batch_stocks_loaded"] = True
                        _clear_current_task()
                    else:
                        _clear_current_task()
                        st.session_state[k] = v
                except Exception:
                    st.session_state[k] = v
            else:
                st.session_state[k] = v
        else:
            st.session_state[k] = v

# ──────────────────────────────────────────────
# 页面标题
# ──────────────────────────────────────────────
st.title("📋 批量打分")
st.caption("上传 Excel 文件，快速筛选数百只股票。支持最多 1000 只，给出 5 级推荐分类。"
           "结果自动保存，刷新不丢失。")

# 显示持久化结果加载状态
_persisted_count = len(st.session_state.batch_stocks)
if _persisted_count > 0:
    st.info(f"💾 当前已保存 {_persisted_count} 只股票的批量打分结果。"
            f"新批次结果会自动合并到现有列表中。")

# ──────────────────────────────────────────────
# 上传区域
# ──────────────────────────────────────────────
with st.container(border=True):
    col1, col2, col3 = st.columns([3, 1.5, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "选择 Excel 文件",
            type=["xlsx"],
            help="第一行为表头，需包含「股票代码」列（可选「股票名称」列）",
            disabled=st.session_state.batch_status == "running",
        )
    with col2:
        # 用 session_state 定位 index，防止页面刷新时回退到默认值
        horizon_options = ["medium", "short", "long", "all"]
        current_horizon = st.session_state.get("batch_horizon", "medium")
        try:
            default_index = horizon_options.index(current_horizon)
        except ValueError:
            default_index = 0
        horizon = st.selectbox(
            "打分维度",
            options=horizon_options,
            index=default_index,
            format_func=lambda h: {"short": "短线 (1-5天)",
                                     "medium": "中线 (1-3月)",
                                     "long": "长线 (1-3年)",
                                     "all": "全部三维度"}[h],
            disabled=st.session_state.batch_status == "running",
        )
        st.session_state.batch_horizon = horizon
    with col3:
        st.write("")
        start_btn = st.button(
            "🚀 开始批量打分",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None or st.session_state.batch_status == "running",
        )

# ──────────────────────────────────────────────
# 启动批量打分
# ──────────────────────────────────────────────
if start_btn and uploaded_file is not None:
    st.session_state.batch_status = "uploading"
    st.session_state.batch_error = None
    st.session_state.batch_selected_codes = set()
    st.session_state.batch_df_version = 0
    st.session_state.batch_last_df_key = ""

    try:
        file_content = uploaded_file.read()
        result = _ar(upload_excel(file_content, uploaded_file.name, horizon))

        st.session_state.batch_task_id = result["batch_id"]
        _save_current_task(result["batch_id"], horizon)
        st.session_state.batch_status = "running"
        # 暂存本次解析的股票列表（仅代码+名称），完成后合并到持久化结果
        st.session_state.batch_pending_stocks = result.get("stocks", [])

        st.success(f"任务已启动: {result.get('total_stocks', 0)} 只股票 ({horizon})")
        st.rerun()
    except APIError as e:
        st.session_state.batch_status = "idle"
        st.session_state.batch_error = str(e)
        st.error(f"启动失败: {e}")
    except Exception as e:
        st.session_state.batch_status = "idle"
        st.session_state.batch_error = str(e)
        st.error(f"未知错误: {e}")

# ──────────────────────────────────────────────
# 实时进度（每轮查一次，st.rerun() 驱动刷新，浏览器可见丝滑进度）
# ──────────────────────────────────────────────
if st.session_state.batch_status == "running" and st.session_state.batch_task_id:
    st.divider()
    st.subheader("⏳ 实时进度")

    try:
        prog = _ar(get_batch_progress(st.session_state.batch_task_id))
    except APIError as e:
        st.warning(f"进度查询失败，重试中... ({e})")
        time.sleep(BATCH_POLL_INTERVAL)
        st.rerun()

    status = prog.get("status", "unknown")

    render_progress_bar(
        prog.get("progress_pct", 0),
        status,
        prog.get("fetched_count", 0),
        prog.get("scored_count", 0),
        prog.get("total_stocks", 0),
        prog.get("elapsed_seconds", 0),
    )

    if status == "failed":
        st.session_state.batch_error = prog.get("error", "任务失败")
        st.error(f"任务失败: {prog.get('error', '未知错误')}")
        st.session_state.batch_status = "idle"
        st.session_state.batch_df_version = 0
        st.session_state.batch_last_df_key = ""
        st.rerun()

    elif status == "completed":
        try:
            final_result = _ar(get_batch_results(st.session_state.batch_task_id))
            new_stocks = final_result.get("stocks", [])
            existing = st.session_state.batch_stocks
            st.session_state.batch_stocks = _merge_stocks(existing, new_stocks)
            _save_persisted_stocks(st.session_state.batch_stocks)
            _clear_current_task()
            st.session_state.batch_status = "done"
            st.session_state.batch_df_version = 0
            st.session_state.batch_last_df_key = ""
            st.success(
                f"打分完成！本批 {final_result.get('total_stocks', 0)} 只, "
                f"合并后共 {len(st.session_state.batch_stocks)} 只"
            )
        except APIError as e:
            st.error(f"获取结果失败: {e}")
            st.session_state.batch_status = "idle"
            st.session_state.batch_df_version = 0
            st.session_state.batch_last_df_key = ""
        st.rerun()

    else:
        # 还在跑，短暂等待后 rerun 触发下一轮刷新
        time.sleep(BATCH_POLL_INTERVAL)
        st.rerun()

# ──────────────────────────────────────────────
# 结果展示（仅在完成状态）
# ──────────────────────────────────────────────
# 只要有股票结果就显示表格（无论是否在跑新批次）
if st.session_state.batch_stocks:
    st.divider()
    st.subheader("📊 打分结果")

    stocks = st.session_state.batch_stocks

    # ── 统计概览 ──
    cols = st.columns(5)
    with cols[0]:
        st.metric("股票总数", len(stocks))
    with cols[1]:
        scored = sum(1 for s in stocks if s.get("level"))
        st.metric("已打分", scored)
    with cols[2]:
        recommend = sum(1 for s in stocks
                       if s.get("level") in ("强烈推荐", "推荐"))
        st.metric("推荐", recommend)
    with cols[3]:
        neutral = sum(1 for s in stocks if s.get("level") == "中性")
        st.metric("中性", neutral)
    with cols[4]:
        avoid = sum(1 for s in stocks
                   if s.get("level") in ("回避", "卖出"))
        st.metric("回避/卖出", avoid)

    render_level_count_bar(stocks)

    # ── 操作栏：搜索 + 筛选 + 排序 ──
    st.divider()
    ctl_col1, ctl_col2, ctl_col3, ctl_col4, ctl_col5 = st.columns([2, 3, 1.5, 1, 1])

    with ctl_col1:
        search = st.text_input(
            "🔍 搜索代码/名称",
            value=st.session_state.batch_search,
            placeholder="输入代码或名称...",
            key="batch_search_input",
        )
        st.session_state.batch_search = search.strip()

    with ctl_col2:
        active_levels = st.session_state.batch_filter_levels
        # 级别筛选 pill 按钮
        level_options = list(LEVEL_ORDER.keys())
        cols_pills = st.columns(len(level_options))
        for i, lv in enumerate(level_options):
            with cols_pills[i]:
                emoji = LEVEL_EMOJI.get(lv, "⚪")
                is_active = lv in active_levels
                label = f"{emoji} {lv}" if is_active else f"⚫ {lv}"
                if st.button(label, key=f"flt_{lv}", use_container_width=True,
                            type="primary" if is_active else "secondary"):
                    if lv in active_levels:
                        active_levels.remove(lv)
                    else:
                        active_levels.append(lv)
                    st.session_state.batch_filter_levels = active_levels
                    st.rerun()

    with ctl_col3:
        sort_col = st.selectbox(
            "排序字段",
            options=["评级", "PE", "PB", "ROE(%)", "代码", "名称", "行业"],
            index=["评级", "PE", "PB", "ROE(%)", "代码", "名称", "行业"].index(
                st.session_state.batch_sort_col
            ) if st.session_state.batch_sort_col in ["评级", "PE", "PB", "ROE(%)", "代码", "名称", "行业"] else 0,
            key="batch_sort_col_select",
        )
        st.session_state.batch_sort_col = sort_col

    with ctl_col4:
        asc = st.checkbox("升序", value=st.session_state.batch_sort_asc,
                         key="batch_sort_asc_check")
        st.session_state.batch_sort_asc = asc

    # ── 构建并展示 DataFrame ──
    df = build_results_dataframe(stocks)

    # 搜索过滤
    if st.session_state.batch_search:
        keyword = st.session_state.batch_search.lower()
        mask = (
            df["代码"].astype(str).str.lower().str.contains(keyword) |
            df["名称"].astype(str).str.lower().str.contains(keyword) |
            df["行业"].astype(str).str.lower().str.contains(keyword)
        )
        df = df[mask]

    # 级别过滤
    active_levels = st.session_state.batch_filter_levels
    if active_levels and len(active_levels) < 5:
        df = df[df["_level_raw"].isin(active_levels)]

    # 排序
    sort_col = st.session_state.batch_sort_col
    asc = st.session_state.batch_sort_asc
    if sort_col == "评级":
        df["_sort_tmp"] = df["_level_raw"].map(lambda x: LEVEL_ORDER.get(x, 2))
        df = df.sort_values("_sort_tmp", ascending=asc)
        df = df.drop(columns=["_sort_tmp"])
    elif sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=asc, na_position="last")

    # 显示结果数
    st.caption(f"显示 {len(df)} / {len(stocks)} 只股票")

    # 自定义列配置
    column_config = {
        "_level_raw": None,
        "代码": st.column_config.TextColumn("代码", width="small"),
        "名称": st.column_config.TextColumn("名称", width="medium"),
        "PE": st.column_config.NumberColumn("PE", format="%.1f", width="small"),
        "PB": st.column_config.NumberColumn("PB", format="%.2f", width="small"),
        "ROE(%)": st.column_config.NumberColumn("ROE(%)", format="%.1f", width="small"),
        "行业": st.column_config.TextColumn("行业", width="small"),
        "评级": st.column_config.TextColumn("评级", width="medium"),
        "置信度": st.column_config.TextColumn("置信度", width="small"),
        "理由": st.column_config.TextColumn("理由", width="medium"),
        "风险": st.column_config.TextColumn("风险", width="medium"),
        "市值": st.column_config.TextColumn("市值", width="small"),
        "涨跌(1月)": st.column_config.TextColumn("涨跌(1月)", width="small"),
        "涨跌(1年)": st.column_config.TextColumn("涨跌(1年)", width="small"),
    }

    # ── 结果表格（原生多选，流畅 + 红色勾选框） ──
    table_height = min(550, max(200, len(df) * 38 + 38))
    df_key = f"batch_results_df_v{st.session_state.batch_df_version}"
    selection = st.dataframe(
        df,
        column_config=column_config,
        column_order=[
            "代码", "名称", "PE", "PB", "ROE(%)", "行业",
            "评级", "置信度", "理由", "风险", "市值",
            "涨跌(1月)", "涨跌(1年)",
        ],
        hide_index=True,
        height=table_height,
        use_container_width=True,
        selection_mode="multi-row",
        key=df_key,
        on_select="rerun",
    )

    # 将原生勾选状态同步到 session state
    # 策略：若 dataframe key 刚发生变化（全选/取消全选导致），说明客户端 selection 已被重置，
    #       此时跳过同步，避免空 selection 覆盖逻辑选中状态。
    #       只有当 key 未变时（用户手动点击行触发的 rerun），才将客户端 selection 同步到 session。
    if df_key != st.session_state.batch_last_df_key:
        # key 变了 → 跳过同步，记录本次 key
        st.session_state.batch_last_df_key = df_key
    else:
        # key 未变 → 用户手动操作，正常同步（空的 selection 表示用户取消了所有勾选）
        if selection is not None and hasattr(selection, 'get'):
            try:
                sel_rows = selection.get("selection", {}).get("rows", [])
                sel_codes = set()
                for idx in sel_rows:
                    if isinstance(idx, int) and 0 <= idx < len(df):
                        sel_codes.add(str(df.iloc[idx]["代码"]))
                st.session_state.batch_selected_codes = sel_codes
            except Exception:
                pass

    # ── 操作按钮行 ──
    st.divider()
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 1.5])

    with btn_col1:
        if st.button("✖ 取消全选", use_container_width=True,
                     key="deselect_all_btn"):
            st.session_state.batch_selected_codes = set()
            st.session_state.batch_df_version += 1
            st.rerun()

    with btn_col2:
        # 删除选中按钮（需要确认）
        sel_count = len(st.session_state.batch_selected_codes)
        if st.button(f"🗑️ 删除选中 ({sel_count})", use_container_width=True,
                     type="secondary", key="del_selected_btn",
                     disabled=sel_count == 0):
            st.session_state.batch_del_confirm = True
            st.rerun()

    with btn_col3:
        if st.button("🧹 清空全部", use_container_width=True,
                     type="secondary", key="clear_all_btn"):
            st.session_state.batch_stocks = []
            _save_persisted_stocks([])
            st.session_state.batch_status = "idle"
            st.session_state.batch_task_id = None
            st.session_state.batch_selected_codes = set()
            st.session_state.batch_df_version = 0
            st.session_state.batch_last_df_key = ""
            st.session_state.batch_search = ""
            st.session_state.batch_filter_levels = list(LEVEL_ORDER.keys())
            st.rerun()

    # 删除确认
    if st.session_state.batch_del_confirm:
        sel_codes = st.session_state.batch_selected_codes
        st.warning(f"确认删除 {len(sel_codes)} 只选中的股票？此操作不可撤销。")
        cc1, cc2, cc3 = st.columns([1, 1, 3])
        with cc1:
            if st.button("✅ 确认删除", type="primary", key="confirm_del"):
                st.session_state.batch_stocks = [
                    s for s in stocks
                    if s.get("code", "").replace("sh.", "").replace("sz.", "") not in sel_codes
                ]
                _save_persisted_stocks(st.session_state.batch_stocks)
                st.session_state.batch_selected_codes = set()
                st.session_state.batch_df_version += 1
                st.session_state.batch_del_confirm = False
                st.rerun()
        with cc2:
            if st.button("取消", key="cancel_del"):
                st.session_state.batch_del_confirm = False
                st.rerun()

    # ── 导出 Excel ──
    with btn_col4:
        export_data = []
        for s in stocks:
            pc = s.get("price_changes", {}) or {}
            code = s.get("code", "").replace("sh.", "").replace("sz.", "")
            export_data.append({
                "股票代码": code,
                "股票名称": s.get("name") or "未知",
                "评级": s.get("level", "中性"),
                "置信度": s.get("confidence", ""),
                "打分理由": s.get("reason", ""),
                "风险提示": s.get("risk", ""),
                "PE": s.get("pe", "-"),
                "PB": s.get("pb", "-"),
                "ROE": s.get("roe", "-"),
                "行业": s.get("industry", "-"),
                "市值": s.get("market_cap", "-"),
                "近1日涨跌": pc.get("1d", "-"),
                "近1月涨跌": pc.get("1m", "-"),
                "近3月涨跌": pc.get("3m", "-"),
                "近1年涨跌": pc.get("1y", "-"),
            })
        df_export = pd.DataFrame(export_data)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="批量打分结果")
        buf.seek(0)
        st.download_button(
            label="📥 导出 Excel",
            data=buf,
            file_name=f"批量打分_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="export_excel_btn",
            use_container_width=True,
        )

# ──────────────────────────────────────────────
# 重置按钮（idle 状态且有历史任务时显示）
# ──────────────────────────────────────────────
if st.session_state.batch_status in ("done", "idle") and st.session_state.batch_task_id and not st.session_state.batch_stocks:
    st.divider()
    if st.button("🔄 开始新的批量打分", type="secondary", use_container_width=True):
        for k in _defaults:
            st.session_state[k] = _defaults[k]
        st.rerun()

# ──────────────────────────────────────────────
# 提示信息
# ──────────────────────────────────────────────
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    ### Excel 格式要求
    - 第一行为表头，需包含 **股票代码** 列（如 `603871`）
    - 可选包含 **股票名称** 列
    - 最多支持 **1000 只** 股票

    ### 5 级分类说明
    | 级别 | 含义 | 操作建议 |
    |------|------|---------|
    | 🟢 强烈推荐 | 基本面优秀+估值合理 | 可进入股票池做深度分析 |
    | 🔵 推荐 | 基本面良好+有投资价值 | 关注，纳入候选 |
    | 🟡 中性 | 多空交织/数据不足 | 需更多信息判断 |
    | 🟠 回避 | 基本面偏弱/高估 | 暂时观望 |
    | 🔴 卖出 | 基本面恶化/严重高估 | 建议回避 |

    ### 结果操作
    - 点击列头 **排序**（正序/倒序切换）
    - 勾选行左侧复选框 **多选** → 点击「删除选中」
    - 使用级别 pill 按钮 **筛选** 只看特定级别
    - 搜索框支持代码/名称/行业 **模糊搜索**
    - 「清空全部」一键清除所有结果
    - 「导出 Excel」下载完整结果文件

    ### 打分维度
    - **短线 (1-5天)**: 偏重量价关系、技术信号、资金情绪
    - **中线 (1-3月)**: 偏重基本面质量、估值、技术趋势
    - **长线 (1-3年)**: 偏重商业护城河、行业景气、安全边际
    - **全部**: 按短→中→长线依次打分，结果包含三个维度

    ### 性能参考
    | 股票数 | 预计耗时 |
    |--------|---------|
    | 50 只 | ~2 分钟 |
    | 200 只 | ~5 分钟 |
    | 500 只 | ~10 分钟 |
    | 1000 只 | ~20 分钟 |
    """)
