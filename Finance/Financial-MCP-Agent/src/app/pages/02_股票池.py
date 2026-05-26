"""
股票池页面 — 快筛/短线/中线/长线四个独立池
布局：上方显示行（点击选中后出现） + 下方密集列表（点击选中） + 分页
"""

import asyncio
import os
import sys
from datetime import datetime

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import streamlit as st

from components.pool_table import (
    render_add_stock_form,
    render_display_row,
    render_pool_list,
    render_fine_display_row,
    render_fine_pool_list,
    render_pagination,
)
from components.quick_screen_table import (
    render_quick_display_row,
    render_quick_list,
)
from api_client import (
    APIError,
    get_pool,
    add_to_pool,
    remove_from_pool,
    trigger_score_all,
    poll_score_all_result,
    trigger_quick_score,
    poll_quick_score_result,
    score_quick_screen,
    trigger_report,
    poll_report_status,
)
from config import POOL_TERMS


# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────
st.set_page_config(page_title="股票池", page_icon="📊", layout="wide")


# ──────────────────────────────────────────────
# 同步包装器
# ──────────────────────────────────────────────
def _ar(coro):
    return asyncio.run(coro)

def handle_get_pool(term: str) -> list:
    return _ar(get_pool(term))

def handle_add_to_pool(term: str, code: str, name: str) -> dict:
    return _ar(add_to_pool(term, code, name))

def handle_remove_from_pool(term: str, code: str) -> dict:
    return _ar(remove_from_pool(term, code))

def handle_trigger_score_all(code: str) -> str:
    return _ar(trigger_score_all(code))

def handle_poll_score_all(task_id: str) -> dict:
    return _ar(poll_score_all_result(task_id))

def handle_trigger_quick_score(term: str, code: str) -> str:
    return _ar(trigger_quick_score(term, code))

def handle_poll_quick_score(task_id: str) -> dict:
    return _ar(poll_quick_score_result(task_id))

def handle_trigger_report(stock_input: str) -> str:
    return _ar(trigger_report(stock_input))

def handle_poll_report(task_id: str, on_progress=None) -> dict:
    return _ar(poll_report_status(task_id, progress_callback=on_progress))

def handle_quick_screen_score(term: str, code: str) -> dict:
    return _ar(score_quick_screen(term, code))


# ──────────────────────────────────────────────
# 状态初始化
# ──────────────────────────────────────────────
if "_pool_busy" not in st.session_state:
    st.session_state["_pool_busy"] = False
if "_pool_scores_fine" not in st.session_state:
    st.session_state["_pool_scores_fine"] = {}
if "_pool_scores_quick" not in st.session_state:
    st.session_state["_pool_scores_quick"] = {}
if "_report_task_id" not in st.session_state:
    st.session_state["_report_task_id"] = None
if "_report_result" not in st.session_state:
    st.session_state["_report_result"] = None
if "_report_error" not in st.session_state:
    st.session_state["_report_error"] = None

TERM_KEYS = ["quick_screen", "fine"]
for tk in TERM_KEYS:
    for s, d in [
        ("_action_result", None), ("_action_error", None),
        ("_selected", None), ("_page", 0), ("_page_size", 20),
        ("_confirm_delete", False),
    ]:
        k = f"_pool{s}_{tk}"
        if k not in st.session_state:
            st.session_state[k] = d


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────
def _rk(tk, s):
    return f"_pool{s}_{tk}"

def _refresh_pool(tk):
    try:
        st.session_state[f"_pool_data_{tk}"] = handle_get_pool(tk)
    except APIError as e:
        st.session_state[_rk(tk, "_action_error")] = f"获取股票池失败: {e}"
        st.session_state[f"_pool_data_{tk}"] = []
    except Exception as e:
        st.session_state[_rk(tk, "_action_error")] = f"获取股票池异常: {e}"
        st.session_state[f"_pool_data_{tk}"] = []


# ──────────────────────────────────────────────
# 回调函数
# ──────────────────────────────────────────────
def _on_select(tk, code):
    """点击列表行选中股票"""
    st.session_state[_rk(tk, "_selected")] = code
    st.session_state[_rk(tk, "_confirm_delete")] = False
    st.rerun()

def _on_add(tk, code, name):
    st.session_state["_pool_busy"] = True
    st.session_state[_rk(tk, "_action_result")] = None
    st.session_state[_rk(tk, "_action_error")] = None
    try:
        handle_add_to_pool(tk, code, name)
        st.session_state[_rk(tk, "_action_result")] = f"已添加 {name}({code})"
    except APIError as e:
        st.session_state[_rk(tk, "_action_error")] = f"添加失败: {e}"
    except Exception as e:
        st.session_state[_rk(tk, "_action_error")] = f"添加异常: {e}"
    finally:
        st.session_state["_pool_busy"] = False
        st.rerun()

def _on_delete(tk, code):
    st.session_state[_rk(tk, "_confirm_delete")] = code

def _on_confirm_delete(tk):
    code = st.session_state[_rk(tk, "_confirm_delete")]
    if not code: return
    st.session_state["_pool_busy"] = True
    st.session_state[_rk(tk, "_action_result")] = None
    st.session_state[_rk(tk, "_action_error")] = None
    try:
        handle_remove_from_pool(tk, code)
        st.session_state[_rk(tk, "_action_result")] = f"已删除 {code}"
        if tk == "quick_screen":
            st.session_state["_pool_scores_quick"].pop(code, None)
        elif tk == "fine":
            st.session_state["_pool_scores_fine"].pop(code, None)
        st.session_state[_rk(tk, "_selected")] = None
        st.session_state[_rk(tk, "_confirm_delete")] = False
    except APIError as e:
        st.session_state[_rk(tk, "_action_error")] = f"删除失败: {e}"
    except Exception as e:
        st.session_state[_rk(tk, "_action_error")] = f"删除异常: {e}"
    finally:
        st.session_state["_pool_busy"] = False
        st.rerun()

def _on_cancel_delete(tk):
    st.session_state[_rk(tk, "_confirm_delete")] = False
    st.rerun()

def _on_fine_score(code):
    """精筛池触发三期限并行打分"""
    st.session_state[_rk("fine", "_action_result")] = None
    st.session_state[_rk("fine", "_action_error")] = None
    try:
        task_id = handle_trigger_score_all(code)
        st.session_state[f"_fine_score_task_{code}"] = task_id
    except APIError as e:
        st.session_state[_rk("fine", "_action_error")] = f"触发打分失败: {e}"
    except Exception as e:
        st.session_state[_rk("fine", "_action_error")] = f"触发打分异常: {e}"
    st.rerun()

def _on_fine_report(code, name):
    """精筛池触发深度报告"""
    st.session_state["_pool_busy"] = True
    st.session_state["_report_task_id"] = None
    st.session_state["_report_result"] = None
    st.session_state["_report_error"] = None
    st.session_state["_report_input_label"] = f"{name}({code})"
    try:
        st.session_state["_report_task_id"] = handle_trigger_report(name)
    except APIError as e:
        st.session_state["_report_error"] = str(e)
        st.session_state["_pool_busy"] = False
    except Exception as e:
        st.session_state["_report_error"] = f"触发报告异常: {e}"
        st.session_state["_pool_busy"] = False

def _on_qs_score(tk, term, code):
    """触发快筛打分（后台异步），立即返回 task_id"""
    st.session_state[_rk(tk, "_action_result")] = None
    st.session_state[_rk(tk, "_action_error")] = None
    try:
        task_id = handle_trigger_quick_score(term, code)
        st.session_state[f"_qs_task_{tk}_{term}_{code}"] = task_id
        st.session_state[f"_qs_label_{tk}_{term}_{code}"] = f"{code} {term}"
    except APIError as e:
        st.session_state[_rk(tk, "_action_error")] = f"触发快筛打分失败: {e}"
    except Exception as e:
        st.session_state[_rk(tk, "_action_error")] = f"触发快筛打分异常: {e}"
    st.rerun()

def _on_report(code, name):
    st.session_state["_pool_busy"] = True
    st.session_state["_report_task_id"] = None
    st.session_state["_report_result"] = None
    st.session_state["_report_error"] = None
    st.session_state["_report_input_label"] = f"{name}({code})"
    try:
        st.session_state["_report_task_id"] = handle_trigger_report(name)
    except APIError as e:
        st.session_state["_report_error"] = str(e)
        st.session_state["_pool_busy"] = False
    except Exception as e:
        st.session_state["_report_error"] = f"触发报告异常: {e}"
        st.session_state["_pool_busy"] = False

def _on_deselect(tk):
    st.session_state[_rk(tk, "_selected")] = None
    st.session_state[_rk(tk, "_confirm_delete")] = False
    st.rerun()


# ──────────────────────────────────────────────
# 页面标题
# ──────────────────────────────────────────────
st.markdown(
    '<div style="margin-bottom:4px;">'
    '<span style="font-size:1.7em;font-weight:800;color:#0f172a;">📊 股票池</span>'
    '</div>',
    unsafe_allow_html=True,
)
st.caption("点击列表行选中股票 → 显示行进行打分/报告/删除。✕ 关闭显示行，股票仍在列表中。")

# ──────────────────────────────────────────────
# Tab 渲染
# ──────────────────────────────────────────────
tabs = st.tabs([POOL_TERMS[k] for k in TERM_KEYS])

for tab, tk in zip(tabs, TERM_KEYS):
    with tab:
        # 刷新数据
        if f"_pool_data_{tk}" not in st.session_state:
            _refresh_pool(tk)

        stocks = st.session_state.get(f"_pool_data_{tk}", [])
        is_busy = st.session_state.get("_pool_busy", False)

        # 操作结果/错误
        rk = _rk(tk, "_action_result")
        ek = _rk(tk, "_action_error")
        if st.session_state.get(rk):
            st.success(st.session_state[rk])
            st.session_state[rk] = None
            _refresh_pool(tk)
        if st.session_state.get(ek):
            st.error(st.session_state[ek])
            st.session_state[ek] = None
            _refresh_pool(tk)

        # 添加股票表单
        render_add_stock_form(
            on_add=lambda c, n: _on_add(tk, c, n),
            disabled=is_busy, key_prefix=tk,
        )
        st.markdown("---")

        # 加载分数
        if tk == "quick_screen":
            qs = st.session_state["_pool_scores_quick"]
            for s in stocks:
                code = s.get("stock_code", "")
                persisted = s.get("quick_scores", {})
                if persisted and code not in qs:
                    qs[code] = {}
                    for t in ("short", "medium", "long"):
                        if t in persisted and persisted[t]:
                            qs[code][t] = persisted[t]

        # 构建展示数据
        if tk == "quick_screen":
            qs = st.session_state["_pool_scores_quick"]
            d_stocks = [{"stock_code": s.get("stock_code", ""),
                          "company_name": s.get("company_name", ""),
                          "scores": qs.get(s.get("stock_code", ""), {})}
                        for s in stocks]
        else:
            # 精筛池：直接从后端数据读取三期限评分
            d_stocks = []
            for s in stocks:
                code = s.get("stock_code", "")
                name = s.get("company_name", "")
                ss = s.get("scores", {})
                d_stocks.append({
                    "stock_code": code,
                    "company_name": name,
                    "scores": {
                        "short": ss.get("short", {}),
                        "medium": ss.get("medium", {}),
                        "long": ss.get("long", {}),
                    },
                    "last_updated": s.get("last_updated", ""),
                    "status": s.get("status", "pending"),
                })

        # 选中状态
        sk = _rk(tk, "_selected")
        sel_code = st.session_state.get(sk)

        # 删除确认弹窗
        ck = _rk(tk, "_confirm_delete")
        if st.session_state.get(ck):
            dc = st.session_state[ck]
            dn = dc
            for ds in d_stocks:
                if ds.get("stock_code") == dc:
                    dn = ds.get("company_name", dc)
                    break
            st.markdown(
                '<div style="'
                'border:1.5px solid #fca5a5;'
                'border-radius:12px;'
                'padding:16px 20px;'
                'background: linear-gradient(145deg, #fff5f5 0%, #fef2f2 100%);'
                'box-shadow: 0 4px 12px rgba(220,38,38,0.08);'
                'margin-bottom:10px;'
                '">',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="font-size:1.1em;font-weight:700;color:#b91c1c;margin-bottom:10px;">'
                f'⚠️ 确定要删除 <span style="color:#7f1d1d;">{dn}</span>（{dc}）吗？此操作不可撤销。'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns([1, 1, 3])
            with c1:
                if st.button("确认删除", type="primary", key=f"cfm_yes_{tk}"):
                    _on_confirm_delete(tk)
            with c2:
                if st.button("取消", key=f"cfm_no_{tk}"):
                    _on_cancel_delete(tk)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

        # ── 上方：显示行 ──
        sel_stock = None
        if sel_code:
            for ds in d_stocks:
                if ds.get("stock_code") == sel_code:
                    sel_stock = ds
                    break

        if sel_stock and not st.session_state.get(ck):
            if tk == "quick_screen":
                render_quick_display_row(
                    stock=sel_stock, term_key=tk,
                    on_score=lambda t, c: _on_qs_score(tk, t, c),
                    on_delete=lambda c: _on_delete(tk, c),
                    on_deselect=lambda: _on_deselect(tk),
                    disabled=is_busy,
                )
            else:
                render_fine_display_row(
                    stock=sel_stock, term_key=tk,
                    on_score=lambda c: _on_fine_score(c),
                    on_report=lambda c, n: _on_fine_report(c, n),
                    on_delete=lambda c: _on_delete(tk, c),
                    on_deselect=lambda: _on_deselect(tk),
                    disabled=is_busy,
                )

        # ── 下方：密集列表 ──
        pk = _rk(tk, "_page")
        psk = _rk(tk, "_page_size")
        page = st.session_state.get(pk, 0)
        page_size = st.session_state.get(psk, 20)

        if tk == "quick_screen":
            st.caption("点击列表行选中股票，在上方显示行中进行交互。使用 qwen3.6-flash 高速模型。")
            render_quick_list(d_stocks, tk, sel_code, lambda c: _on_select(tk, c), page, page_size)
        else:
            st.caption("点击列表行选中股票。三期限并行打分：短线/中线/长线同步产出。")
            render_fine_pool_list(d_stocks, tk, sel_code, lambda c: _on_select(tk, c), page, page_size)

        if not d_stocks:
            st.info("当前池为空，请添加股票")

        # ── 分页 ──
        np_, nps = render_pagination(tk, len(d_stocks), page, page_size)
        if np_ != page:
            st.session_state[pk] = np_
            st.rerun()
        if nps != page_size:
            st.session_state[psk] = nps
            st.session_state[pk] = 0
            st.rerun()


# ──────────────────────────────────────────────
# 精筛打分轮询（Tab 外，刷新安全）
# ──────────────────────────────────────────────
for key in list(st.session_state.keys()):
    if not key.startswith("_fine_score_task_"):
        continue
    code = key[len("_fine_score_task_"):]
    task_id = st.session_state[key]
    if not task_id:
        continue
    with st.status(f"正在对 **{code}** 进行精筛三期限打分...", expanded=True, state="running") as status:
        try:
            r = handle_poll_score_all(task_id)
            if r.get("status") == "completed":
                res = r["result"]
                short_ts = res.get("short_term_score", {})
                medium_ts = res.get("medium_term_score", {})
                long_ts = res.get("long_term_score", {})
                ss = short_ts.get("score", "-")
                ms = medium_ts.get("score", "-")
                ls = long_ts.get("score", "-")
                st.session_state["_pool_scores_fine"][code] = {
                    "short": short_ts, "medium": medium_ts, "long": long_ts,
                    "score_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state[_rk("fine", "_action_result")] = (
                    f"打分完成: {code} → 短线{ss} / 中线{ms} / 长线{ls}"
                )
                status.update(label=f"打分完成: {code}", state="complete", expanded=False)
            else:
                st.session_state[_rk("fine", "_action_error")] = f"打分失败: {r.get('error', '未知错误')}"
                status.update(label="打分失败", state="error", expanded=False)
        except APIError as e:
            st.session_state[_rk("fine", "_action_error")] = f"打分轮询失败: {e}"
            status.update(label="打分失败", state="error", expanded=False)
        except Exception as e:
            st.session_state[_rk("fine", "_action_error")] = f"打分轮询异常: {e}"
            status.update(label="打分失败", state="error", expanded=False)
        finally:
            st.session_state.pop(key, None)
            _refresh_pool("fine")

# 快筛打分轮询
for code_key in list(st.session_state.keys()):
    if not code_key.startswith("_qs_task_"):
        continue
    # key format: _qs_task_{tk}_{term}_{code}
    parts = code_key[len("_qs_task_"):].split("_", 2)
    if len(parts) < 3:
        continue
    tk, term, code = parts[0], parts[1], parts[2]
    task_id = st.session_state[code_key]
    if not task_id:
        continue
    lbl = {"short": "短线", "medium": "中线", "long": "长线"}.get(term, term)
    pool_label = POOL_TERMS.get(tk, tk)
    with st.status(f"正在对 **{code}** 进行快筛{lbl}打分...", expanded=True, state="running") as status:
        try:
            r = handle_poll_quick_score(task_id)
            if r.get("status") == "completed":
                res = r["result"]
                raw = res.get("score")
                try:
                    sc = float(raw) if raw is not None else None
                except (ValueError, TypeError):
                    sc = None
                qs = st.session_state["_pool_scores_quick"]
                if code not in qs: qs[code] = {}
                qs[code][term] = {
                    "score": sc, "score_time": res.get("score_time", ""),
                    "recommendation": res.get("recommendation", ""),
                    "suggested_action": res.get("suggested_action", ""),
                    "reasoning": res.get("reasoning", ""),
                    "risk_warning": res.get("risk_warning", ""),
                }
                st.session_state[_rk(tk, "_action_result")] = (
                    f"快筛{lbl}打分完成: {code} → {sc:.1f} 分" if sc is not None
                    else f"快筛{lbl}打分完成: {code} → 评分失败"
                )
                status.update(label=f"快筛{lbl}打分完成: {code}", state="complete", expanded=False)
            else:
                st.session_state[_rk(tk, "_action_error")] = f"快筛打分失败: {r.get('error', '未知错误')}"
                status.update(label="快筛打分失败", state="error", expanded=False)
        except APIError as e:
            st.session_state[_rk(tk, "_action_error")] = f"快筛打分轮询失败: {e}"
            status.update(label="快筛打分失败", state="error", expanded=False)
        except Exception as e:
            st.session_state[_rk(tk, "_action_error")] = f"快筛打分轮询异常: {e}"
            status.update(label="快筛打分失败", state="error", expanded=False)
        finally:
            st.session_state.pop(code_key, None)
            _refresh_pool(tk)

# ──────────────────────────────────────────────
# 报告生成状态（Tab 外，避免切换丢失）
# ──────────────────────────────────────────────
task_id = st.session_state.get("_report_task_id")

if task_id and not st.session_state.get("_report_result") and not st.session_state.get("_report_error"):
    label = st.session_state.get("_report_input_label", "该股票")
    with st.status(f"正在为 **{label}** 生成深度分析报告...", expanded=True, state="running") as status:
        progress = st.progress(0, text="初始化分析任务...")

        def _on_progress(rp):
            p = min(rp, 1.0)
            stage = "正在执行多维分析..." if p < 0.6 else "正在生成报告..."
            progress.progress(p, text=f"{stage} ({p:.0%})")

        try:
            r = handle_poll_report(task_id, on_progress=_on_progress)
            if r.get("status") == "completed":
                progress.progress(1.0, text="报告生成完成！")
                status.update(label="报告已生成", state="complete", expanded=False)
                st.session_state["_report_result"] = r
            else:
                status.update(label="报告生成失败", state="error", expanded=False)
                st.session_state["_report_error"] = r.get("error", "未知错误")
        except APIError as e:
            status.update(label="报告生成失败", state="error", expanded=False)
            st.session_state["_report_error"] = str(e)
        except Exception as e:
            status.update(label="报告生成失败", state="error", expanded=False)
            st.session_state["_report_error"] = f"轮询异常: {e}"
        finally:
            st.session_state["_pool_busy"] = False

if st.session_state.get("_report_result"):
    r = st.session_state["_report_result"]
    st.success("深度分析报告已生成！")
    content = r.get("report_content", "")
    sn = r.get("company_name", "股票")
    from datetime import datetime
    date_str = datetime.now().strftime('%Y%m%d')

    pdf_path = r.get("report_pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        pdf_fn = f"{sn}_分析报告_{date_str}.pdf"
        st.download_button("📄 下载 PDF 报告", data=pdf_data, file_name=pdf_fn,
                          mime="application/pdf", type="primary")
    else:
        md_fn = f"{sn}_分析报告_{date_str}.md"
        st.download_button("下载报告 (Markdown)", data=content, file_name=md_fn,
                          mime="text/markdown", type="primary")
    with st.expander("预览报告内容"):
        st.markdown(content)
    if st.button("清除结果，返回股票池"):
        for k in ["_report_task_id", "_report_result", "_report_error", "_report_input_label"]:
            st.session_state.pop(k, None)
        st.rerun()

if st.session_state.get("_report_error"):
    st.error(st.session_state["_report_error"])
    if st.button("清除错误"):
        for k in ["_report_task_id", "_report_result", "_report_error", "_report_input_label"]:
            st.session_state.pop(k, None)
        st.rerun()
