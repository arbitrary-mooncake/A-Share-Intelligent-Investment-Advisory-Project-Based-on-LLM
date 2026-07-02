# 精筛池更新跨重启存活 — 方案设计

- 日期：2026-07-02
- 作者：智能投顾 Agent 助手
- 范围：`Finance/Financial-MCP-Agent/src/eval/`、`Finance/Financial-MCP-Agent/src/app/`
- 关联文档：
  - `20260628流式精筛启动开发计划.md`（V3 流式管线设计）
  - `CLAUDE.md` 中 "Pool Screening Performance" 章节

## 1. 背景与目标

### 1.1 问题陈述

Streamlit 页面通过 `WebAdapter.run_pool_update_streaming()` 在 `threading.Thread(daemon=False)` 中运行精筛池全量更新（V3 四层管线）。该线程在以下场景会**被杀死或丢失进度**：

- 用户刷新浏览器 / 关闭 tab
- Streamlit 进程重启（`run.ps1 restart`、代码热重载）
- 用户在 Streamlit 终端 Ctrl+C

结果：跑了 45 分钟的冷启动进度全部清零，用户只能从头再来。

### 1.2 现有"半成品"保障（无需新增）

- **Per-agent 缓存**（`data/intermediate_cache/`）：每个 agent 完成立刻落盘，TTL 1-15 天
- **L3 `full_pool_analysis` 缓存**：每只股票 3-term 评分结果落盘，TTL 1 天

这两层保证：**数据不丢**——重启后新进程对已分析过的股票会命中缓存，跳过 LLM 调用。

### 1.3 缺失的一层（本设计补上）

**任务可见性**——UI 不知道"上次跑到哪了、现在还在不在跑、跑到多少了"。

### 1.4 设计目标

| 目标 | 达成方式 |
|---|---|
| Streamlit 重启后，正在跑的更新**继续跑** | 子进程以 detached 方式 spawn，与 Streamlit 进程生命周期解耦 |
| 重启后 UI **自动 attach** 到现有任务 | JobManager 扫 `running` 状态 job 文件，UI 轮询同一文件 |
| 子进程自身崩溃时**过程数据不丢** | Job 文件 2Hz 原子写入，记录已完成股票 |
| 用户可**诊断 worker 异常** | Worker stdout/stderr 重定向到 per-job log 文件，UI 提供日志查看面板 |
| 不引入外部依赖（Celery/RQ/systemd） | 纯 stdlib：`subprocess.Popen` + `os.replace` + `json` |

### 1.5 非目标（明确不做）

- 不覆盖 `run_daily_rebalance` / `run_daily_settlement` / 单股票评分 / 回测
- 不做"断点续跑"逻辑——既有 per-agent 缓存天然支持
- 不做多任务队列——同一 term 同时只允许 1 个 running job（幂等约束）
- 不做 Windows Service / systemd unit 形式的常驻 worker（那是方案 D，留作后续）

## 2. 架构

```
┌─────────────────────────────────────────────────────────────────┐
│  Streamlit (PID=A)                                               │
│  ├─ UI: st.button("🎯 更新精筛池")                                │
│  ├─ JobManager.start_job(term) ────────── 启动 ─────────┐        │
│  └─ JobManager.poll(job_id) ◄──────── 读文件 ──────────│──┐     │
└─────────────────────────────────────────────────────────│──│─────┘
                                                          │  │
                  ┌───────────────────────────────────────┘  │
                  ▼                                          │
    ┌─────────────────────────────────────────┐              │
    │  python -m src.eval.pool_update_worker  │ (PID=B,独立)│
    │  ├─ 加载 EvalOrchestrator                │              │
    │  ├─ run_pool_update(term, on_progress)   │              │
    │  │    └─ on_progress → 写 job JSON       │ ────────────┘
    │  │         (原子 rename, 2Hz)             │
    │  └─ 完成/失败 → 写 final state → exit     │
    └─────────────────────────────────────────┘
                  │
                  ▼
    data/pool_update_jobs/{job_id}.json  (IPC, 双方共享)
    data/pool_update_jobs/{job_id}.log   (worker stdout/stderr)
    data/stock_pool.json                  (最终池, 仅成功时改写)
```

## 3. Job 文件 Schema

路径：`Finance/Financial-MCP-Agent/data/pool_update_jobs/{job_id}.json`

`job_id` 命名：`pool_update_{term}_{YYYYMMDD_HHMMSS}_{rand4}`（示例：`pool_update_medium_20260702_143052_7a3f`）

```json
{
  "job_id": "pool_update_medium_20260702_143052_7a3f",
  "term": "medium",
  "status": "running",
  "pid": 12488,
  "parent_pid": 5212,
  "started_at": "2026-07-02T14:30:52.123456",
  "finished_at": null,
  "error": null,
  "progress": {
    "overall_pct": 42.3,
    "elapsed_s": 1842,
    "eta_s": 2500,
    "eta_str": "42min",
    "queue_depth": 3,
    "stall_s": 0,
    "stages": {
      "0_hard_screen":    {"label": "硬筛",     "pct": 100.0, "done": 1,    "total": 1},
      "1_batch_score":    {"label": "批量粗筛", "pct": 100.0, "done": 4953, "total": 4953},
      "2_stream_heap":    {"label": "流式排序", "pct": 100.0, "done": 1500, "total": 1500},
      "3_formal_score":   {"label": "精筛打分", "pct": 56.2,  "done": 45,   "total": 80}
    },
    "completed_stocks": [
      {"code": "sz.000333", "name": "美的集团", "final_score": 78, "recommendation": "推荐"},
      ...
    ]
  }
}
```

### 3.1 字段约定

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | enum | `pending` / `running` / `completed` / `failed` / `orphaned` |
| `pid` | int | worker 子进程 PID |
| `parent_pid` | int | 启动 worker 的 Streamlit PID（用于 orphan 检测） |
| `progress` | dict | 与 `PipelineProgress.emit_progress()` 输出结构完全一致 |
| `progress.completed_stocks` | list | L3 每完成 1 只追加 1 条；用于 UI 显示 + 续跑参考 |
| `error` | str\|null | `failed` 状态时含完整 traceback |

### 3.2 写入策略

- 写入频率：**2Hz**（每 500ms 最多 1 次），避免 I/O 抢占 LLM 调用
- 写入方式：`write` 到 `{job_id}.tmp` → `os.replace()` **原子重命名**，防止读到半写文件
- `completed_stocks` 增长：仅在 L3 完成一只时 append（低频，~1 次/200s）

### 3.3 Orphan 检测

如果 job 文件 `status=running` 但 `pid` 对应的进程已不存在 → 自动标为 `orphaned`，UI 提示"上次被系统杀死，可点继续"。

检测方式：`os.kill(pid, 0)`（POSIX）或 `GetExitCodeProcess`（Windows）；失败则视为 dead。

## 4. 子进程 detach 策略

### 4.1 跨平台 spawn

```python
def _spawn_worker(job_id: str, term: str, log_path: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "src.eval.pool_update_worker",
           "--job-id", job_id, "--term", term]

    kwargs = dict(
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        env={**os.environ, "POOL_UPDATE_JOB_ID": job_id},
    )

    if sys.platform == "win32":
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(cmd, **kwargs)
```

### 4.2 验证点

| 平台 | 保护手段 | 验证方式 |
|---|---|---|
| Windows | `CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS` | 启 Streamlit → 点更新 → `run.ps1 stop` → 看 worker PID 是否仍存活 |
| POSIX | `start_new_session=True` | 启 Streamlit → 点更新 → 杀 Streamlit 进程 → 看 worker PID 是否仍存活 |
| 共用 | `stdin=DEVNULL` + `close_fds` | 父进程文件描述符关闭不影响子进程 |

### 4.3 不依赖

- 不用 `multiprocessing`（避免 fork/pickle 限制）
- 不引入 systemd / supervisor / schtasks
- 不要求用户预先启动 worker 进程（区别于方案 D）

## 5. Worker 入口（`src/eval/pool_update_worker.py`）

新增独立可执行模块，不被其他代码 import。

```python
"""精筛池更新 worker — detached subprocess 入口。

由 JobManager 以 subprocess.Popen 启动, 与 Streamlit 进程生命周期解耦。
通过 AtomicJobWriter 把进度 2Hz 写入 job 文件, 供 UI 轮询读取。
"""
import asyncio
import os
import sys
import time
import traceback

def main():
    from src.eval.job_manager import AtomicJobWriter, parse_worker_args
    args = parse_worker_args()
    writer = AtomicJobWriter(args.job_id)
    writer.update(status="running", pid=os.getpid(), parent_pid=os.getppid())

    from src.eval.orchestrator import EvalOrchestrator
    orch = EvalOrchestrator()

    last_write = 0.0
    throttle = 0.5  # 2Hz

    def on_progress(data):
        nonlocal last_write
        now = time.time()
        if now - last_write < throttle:
            return
        last_write = now
        writer.update(progress=data)

    result = asyncio.run(orch.run_pool_update(args.term, on_progress=on_progress))
    writer.update(status="completed", progress={"result": result})

if __name__ == "__main__":
    from src.eval.job_manager import AtomicJobWriter, parse_worker_args, record_failure
    try:
        main()
    except Exception as e:
        record_failure(os.environ.get("POOL_UPDATE_JOB_ID", "unknown"), e)
        sys.exit(1)
```

## 6. JobManager（`src/eval/job_manager.py`）

### 6.1 主要 API

```python
class JobManager:
    """精筛池更新任务管理器。

    - start_job(term): 启动新 worker, 返回 job_id (同 term 已有 running job 则返回现有)
    - poll(job_id): 读取最新 job 状态 (自动 orphan 检测)
    - find_running(term): 找同 term 当前 running 的 job (用于 attach)
    - find_latest(term): 找同 term 最近的 job (任何状态)
    - stop(job_id): 向 worker 发送 SIGTERM / CTRL_BREAK_EVENT (用于"取消")
    - cleanup(): 按保留策略删除过期 job 文件
    """
```

### 6.2 AtomicJobWriter

```python
class AtomicJobWriter:
    """worker 子进程用的原子写入器。

    write → .tmp → os.replace() → .json, 防止 UI 读到半写文件。
    """
    def __init__(self, job_id: str): ...
    def update(self, **fields): ...  # 合并更新; progress 字段 deep-merge
```

### 6.3 保留策略

每 term 保留：
- 最近 **5 个 completed** job
- 最近 **5 个 failed** job
- 所有 running / pending job（不动）

超过数量上限的最旧文件直接删除。不按时间窗口，按数量管理更可控。

上限估算：3 term × 10 job × 50KB = 1.5MB，永远不超。

清理时机：每次 `start_job()` 成功后调用一次 `cleanup()`。

## 7. UI 改造

### 7.1 `src/app/components/eval/ops_panel.py`

替换 `adapter.run_pool_update_streaming(...)` 调用为：

```python
from src.eval.job_manager import JobManager
jm = JobManager()

# 启动 (或 attach)
existing = jm.find_running(pool_term)
if existing:
    job_id = existing["job_id"]
    st.info(f"已有正在跑的更新 ({existing['started_at']}), 自动 attach")
else:
    job_id = jm.start_job(pool_term)
    jm.cleanup()

st.session_state[f"pool_job_{pool_term}"] = job_id

# 轮询 (保留原有 while 循环结构, 只换数据源)
while True:
    job = jm.poll(job_id)
    pct = job["progress"]["overall_pct"]
    progress_bar.progress(min(pct / 100.0, 1.0), f"总进度 {pct:.0f}% | ETA: {job['progress']['eta_str']}")
    ...
    if job["status"] == "completed":
        st.success(...)
        break
    if job["status"] in ("failed", "orphaned"):
        st.error(job.get("error") or "任务异常终止")
        break
    time.sleep(1.0)
```

### 7.2 `src/app/pages/06_模拟分析与迭代.py` 健康卡片

"🔄 全量更新{label}池" 按钮走同一路径（已有）。

### 7.3 Worker 日志查看面板

UI 新增折叠面板：

```python
with st.expander("📋 Worker 日志", expanded=False):
    log_text = jm.read_log(job_id, tail=200)
    st.code(log_text, language="log")
```

### 7.4 删除旧接口

- 删除 `WebAdapter.run_pool_update_streaming()` 方法
- 删除其 import（如果有）

## 8. 错误处理

| 异常场景 | JobManager 行为 | UI 表现 |
|---|---|---|
| Worker spawn 失败（权限/路径问题） | `start_job` 抛异常，UI try/except 显示 | "更新启动失败：..." |
| Worker 中途崩溃（Python 异常） | 顶层 try/except 写 `status=failed` + traceback | 轮询看到 `failed` → `st.error` + expander 显示日志 |
| Worker 被 OS kill（OOM / 注销） | 无法写 final state | `poll` 做 orphan 检测 → `status=orphaned` → UI 提示 |
| Job 文件被用户手动删除 | `poll` 返回 `None` | UI 显示"任务信息丢失"，允许用户重新触发 |
| 同 term 重复点击 | `find_running` 命中 → attach，不启新进程 | `st.info` 提示 "已在跑" |

## 9. 测试与验证

### 9.1 单元测试

- `tests/test_job_manager.py`：
  - `test_start_and_poll`
  - `test_find_running_returns_existing`
  - `test_atomic_write_no_partial_reads`（并发读写压测）
  - `test_orphan_detection`
  - `test_cleanup_keeps_5_completed_per_term`

### 9.2 集成测试

- `tests/test_pool_update_worker.py`：
  - `test_worker_detaches_after_parent_exit`（关键：父进程退出后子进程继续写文件）
  - `test_worker_writes_completed_stocks_incrementally`
  - `test_worker_records_failure_on_exception`

### 9.3 手动验收

1. **Detach 验证**：启 Streamlit → 点更新 → 看到进度 10% → `run.ps1 stop` → 用 `tasklist` / `ps` 确认 worker PID 仍存活 → job 文件 `status` 仍为 `running` → 重新 `run.ps1 start` → 浏览器打开 → UI 自动 attach → 进度条继续推进
2. **Crash 验证**：跑一半用 `taskkill /PID <worker> /F` 强杀 → 等 1 秒 → UI 看到 `status=orphaned` + 已完成的 N 只股票列表
3. **恢复验证**：强杀后点"继续" → 新 worker 启动 → per-agent 缓存命中 → 跳过已完成股票 → 总耗时明显短于冷启动
4. **并发验证**：同一 term 快速点两次更新 → 第二次按钮显示 "已在跑"，不启新进程
5. **跨 term 验证**：同时跑 short + medium + long 三个更新 → 互不干扰，各自有独立 job 文件

### 9.4 5 轮深度代码检查（实施阶段执行）

1. Round 1：py_compile + pyflakes（两个新模块 + UI 改动文件）
2. Round 2：asyncio 路径 review（worker 内 event loop 隔离、信号处理）
3. Round 3：功能正确性（attach 路径、orphan 检测、cleanup 保留策略）
4. Round 4：副作用 review（不动 orchestrator/pool_screening/scoring_engine/agents）
5. Round 5：`pytest tests/` 全绿回归

## 10. 影响面与回滚

### 10.1 改动文件清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `src/eval/job_manager.py` | 新增 | ~250 |
| `src/eval/pool_update_worker.py` | 新增 | ~100 |
| `src/app/components/eval/ops_panel.py` | 修改 | ~40 |
| `src/app/pages/06_模拟分析与迭代.py` | 修改 | ~20 |
| `src/eval/web_adapter.py` | 修改（删除 `run_pool_update_streaming`） | -40 |
| `data/pool_update_jobs/.gitignore` | 新增 | ~2 |
| `tests/test_job_manager.py` | 新增 | ~150 |
| `tests/test_pool_update_worker.py` | 新增 | ~100 |

### 10.2 不动的文件

- `src/eval/orchestrator.py`
- `src/eval/pool_screening.py`
- `src/stock_pool/scoring_engine.py`
- 所有 `src/agents/*`
- `src/tools/mcp_client.py`（上一轮已修复单例竞态，本轮不碰）
- `src/utils/cache_utils.py`
- `src/utils/tushare_client.py`

### 10.3 回滚预案

如果新 UI 有问题，临时回滚方案：
1. `git revert` 本轮 commit
2. 或手动把 `ops_panel.py` 的 `JobManager` 调用改回 `adapter.run_pool_update_streaming(...)`（需要把删除的方法加回）

更稳妥的做法：保留一个 commit 边界，UI 改动单独一个 commit，便于 cherry-pick revert。

## 11. 风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Windows `DETACHED_PROCESS` 在某些 Python 版本下行为不一致 | 低 | worker 跟着父进程死 | 集成测试覆盖；兜底用 `creationflags=CREATE_NEW_CONSOLE` |
| Job 文件被外部工具锁定（杀毒/OneDrive 同步） | 中 | `os.replace` 失败 | 失败时 fallback 到 rename-retry-3 次；仍失败则 log 警告但不阻塞 worker |
| Orphan 检测误判（PID 被复用） | 极低 | 把活 worker 判成死 | 同时检查 `started_at`，PID 复用通常跨数天，不会冲突 |
| Streamlit session state 丢失导致 UI 不知道 attach 哪个 job | 中 | 用户看不到之前跑的任务 | `find_running(term)` 不依赖 session state，扫目录 |

## 12. 未来可扩展（不在本设计范围）

- 扩展到 rebalance / backtest 的 Job 化
- 多任务队列（方案 D 的 worker 模式）
- Job 历史查看 UI（表格 + 筛选）
- Prometheus / Grafana 接入（每 job 的耗时、成功率指标）
