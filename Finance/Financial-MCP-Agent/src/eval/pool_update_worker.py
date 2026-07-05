"""精筛池更新 worker — detached subprocess 入口。

由 JobManager 以 subprocess.Popen 启动, 与 Streamlit 进程生命周期解耦。
通过 AtomicJobWriter 把进度 2Hz 写入 job 文件, 供 UI 轮询读取。

运行方式:
    python -m src.eval.pool_update_worker --job-id <id> --term short|medium|long
"""
import asyncio
import datetime
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pool_update_worker")


class _CancellationError(Exception):
    """Raised when job cancellation is requested."""
    pass


def _make_fake_orchestrator():
    """测试用假 orchestrator: 模拟 run_pool_update 触发 3 次进度回调然后返回。"""
    class _Fake:
        async def run_pool_update(self, term, mode="full", on_progress=None, on_stage=None):
            for pct in (10.0, 50.0, 100.0):
                if on_progress:
                    on_progress({
                        "overall_pct": pct, "elapsed_s": 1, "eta_s": 1,
                        "eta_str": "1s", "queue_depth": 0, "stall_s": 0,
                        "stages": {},
                    })
                await asyncio.sleep(0.05)
            return {"term": term, "pool": [], "final_pool_size": 0, "stats": {}}
    return _Fake()


def main() -> None:
    from src.eval.job_manager import (
        AtomicJobWriter, parse_worker_args, JobStatus,
    )

    args = parse_worker_args()
    job_id = args.job_id
    term = args.term
    mode = args.mode

    logger.info("pool_update_worker starting: job_id=%s term=%s mode=%s", job_id, term, mode)
    writer = AtomicJobWriter(job_id)
    writer.update(
        status=JobStatus.RUNNING.value,
        pid=os.getpid(),
        parent_pid=os.getppid(),
    )

    if os.environ.get("_FAKE_ORCH") == "1":
        orch = _make_fake_orchestrator()
        if os.environ.get("_FAKE_ORCH_RAISE") == "1":
            async def _raise(*args, **kw):
                raise RuntimeError("simulated orchestrator failure")
            orch.run_pool_update = _raise
    else:
        from src.eval.orchestrator import EvalOrchestrator
        orch = EvalOrchestrator()

    last_write = 0.0
    throttle = 0.5  # 2Hz
    _cancelled = False

    def _check_cancelled() -> bool:
        """Check if cancellation was requested via job file flag."""
        nonlocal _cancelled
        if _cancelled:
            return True
        try:
            data = writer.read()
            if data.get("cancellation_requested"):
                _cancelled = True
                return True
        except Exception:
            pass
        return False

    def on_progress(data):
        nonlocal last_write
        now = time.time()
        if now - last_write < throttle and not data.get("force"):
            return
        last_write = now
        if _check_cancelled():
            raise _CancellationError("job cancelled during execution")
        writer.merge_progress(data)
        # 如果 data 里带了 completed_stocks 增量, 也追加 (V3 管线目前不带, 留扩展)
        for s in data.get("new_completed_stocks", []) or []:
            writer.append_completed_stock(s)

    def on_stage(stage_name, message):
        # 阶段切换也触发一次进度刷新 (即使未到 500ms)
        writer.merge_progress({"current_stage": stage_name, "current_stage_msg": message})

    try:
        # 执行前检查取消
        if _check_cancelled():
            raise _CancellationError("job cancelled before execution")

        result = asyncio.run(
            orch.run_pool_update(term, mode=mode, on_progress=on_progress, on_stage=on_stage)
        )

        # 执行后再次检查取消
        if _check_cancelled():
            raise _CancellationError("job cancelled after execution")

        writer.merge_progress({"result": _json_safe(result)})

        # 检测管线返回的 error 字段 (如 reserve_empty, 池为空等)
        if isinstance(result, dict) and result.get("error"):
            err = result["error"]
            msg = result.get("message", str(err))
            writer.update(
                status=JobStatus.FAILED.value,
                error=err if isinstance(err, str) else str(err),
                finished_at=datetime.datetime.now().isoformat(),
            )
            logger.warning("pool_update_worker finished with error: job_id=%s error=%s", job_id, msg)
        else:
            writer.update(
                status=JobStatus.COMPLETED.value,
                finished_at=datetime.datetime.now().isoformat(),
            )
            logger.info("pool_update_worker completed: job_id=%s", job_id)

    except _CancellationError:
        logger.info("pool_update_worker cancelled: job_id=%s", job_id)
        writer.update(
            status=JobStatus.CANCELLED.value,
            error="cancelled by user request",
            finished_at=datetime.datetime.now().isoformat(),
        )


def _json_safe(obj):
    """确保 result 可 JSON 序列化: 非序列化对象转 str。"""
    import json
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {"repr": repr(obj)[:500]}


if __name__ == "__main__":
    from src.eval.job_manager import record_failure, AtomicJobWriter, JobStatus

    # Extract job_id from args or env for safety net
    _job_id = os.environ.get("POOL_UPDATE_JOB_ID", "unknown")
    for i, arg in enumerate(sys.argv):
        if arg == "--job-id" and i + 1 < len(sys.argv):
            _job_id = sys.argv[i + 1]
            break

    try:
        main()
    except BaseException as e:
        logger.error("worker failed: %s", e)
        try:
            record_failure(_job_id, e)
        except Exception:
            pass
        sys.exit(1 if not isinstance(e, SystemExit) else (e.code or 1))
    finally:
        # Safety net: ensure status is never left as "running" zombie
        try:
            writer = AtomicJobWriter(_job_id)
            data = writer.read()
            if data.get("status") == JobStatus.RUNNING.value:
                writer.update(
                    status=JobStatus.FAILED.value,
                    error="worker exited without explicit status (final state cleanup)",
                    finished_at=datetime.datetime.now().isoformat(),
                )
        except Exception:
            pass
