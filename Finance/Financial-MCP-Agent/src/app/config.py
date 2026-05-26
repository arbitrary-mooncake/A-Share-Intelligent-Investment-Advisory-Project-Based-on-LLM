"""
前端配置模块 — API 地址、超时、模型选择等
"""

import os

# ──────────────────────────────────────────────
# FastAPI 后端地址
# ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# ──────────────────────────────────────────────
# 超时设置（秒）
# ──────────────────────────────────────────────
QUERY_TIMEOUT = 60        # 快速查询超时
REPORT_TIMEOUT = 2400     # 深度报告超时（40分钟，4分析Agent+总结Agent+MCP）
SCORE_TRIGGER_TIMEOUT = 15   # 打分触发超时（秒，仅触发调用）
SCORE_POLL_INTERVAL = 2      # 打分任务轮询间隔（秒）
SCORE_POLL_MAX_ATTEMPTS = 450  # 打分任务最大轮询次数 ≈ 15 分钟
POLL_INTERVAL = 3            # 报告任务轮询间隔（秒）
POLL_MAX_ATTEMPTS = 800      # 报告任务最大轮询次数 ≈ 40 分钟

# ──────────────────────────────────────────────
# 模型选择（供前端展示用，实际由后端控制）
# ──────────────────────────────────────────────
QUICK_QUERY_MODEL = os.getenv("QUICK_QUERY_MODEL", "Haiku-level (low-latency)")
DEEP_REPORT_MODEL = os.getenv("DEEP_REPORT_MODEL", "Sonnet/Opus-level (strong-reasoning)")

# ──────────────────────────────────────────────
# 股票池期限
# ──────────────────────────────────────────────
POOL_TERMS = {
    "quick_screen": "⚡ 快筛股票池",
    "short": "短线投资池",
    "medium": "中线投资池",
    "long": "长线投资池",
}

QUICK_SCREEN_TIMEOUT = 600  # 快筛单期打分超时（秒），需覆盖4分析Agent+1打分Agent

# ──────────────────────────────────────────────
# 批量打分超时设置
# ──────────────────────────────────────────────
BATCH_UPLOAD_TIMEOUT = 900     # 批量打分上传+解析超时（最长15分钟，覆盖500只）
BATCH_POLL_INTERVAL = 3        # 批量任务轮询间隔（秒）
BATCH_POLL_MAX_ATTEMPTS = 600  # 最大轮询次数 ≈ 30 分钟（覆盖1000只）

# ──────────────────────────────────────────────
# 中间产物缓存目录（与后端一致）
# ──────────────────────────────────────────────
INTERMEDIATE_CACHE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "intermediate_cache"
)

# ──────────────────────────────────────────────
# 智能问答超时设置（秒）
# ──────────────────────────────────────────────
QA_TIMEOUT = 120        # QA HTTP 请求超时
QA_STREAM_TIMEOUT = 180  # QA 流式总超时
