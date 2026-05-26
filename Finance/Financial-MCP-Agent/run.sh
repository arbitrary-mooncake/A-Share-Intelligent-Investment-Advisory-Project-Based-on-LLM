#!/usr/bin/env bash
# ============================================================
# run.sh — 一键启动 FastAPI 后端 + Streamlit 前端
#
# 用法:
#   ./run.sh          # 启动两个服务
#   ./run.sh stop     # 停止两个服务
#   ./run.sh status   # 查看服务状态
# ============================================================

set -e

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
FASTAPI_ENTRY="$PROJECT_DIR/src/api/app.py"
STREAMLIT_ENTRY="$PROJECT_DIR/src/app/Home.py"
FASTAPI_HOST="127.0.0.1"
FASTAPI_PORT=8000
STREAMLIT_PORT=8501

# PID 文件
PID_DIR="$PROJECT_DIR/.run"
FASTAPI_PID="$PID_DIR/fastapi.pid"
STREAMLIT_PID="$PID_DIR/streamlit.pid"

mkdir -p "$PID_DIR"

# ── 颜色输出 ──────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── 检查依赖 ──────────────────────────────────────────────
check_deps() {
    for cmd in python3 uvicorn streamlit; do
        if ! command -v "$cmd" &>/dev/null; then
            error "缺少依赖: $cmd 未找到，请先安装"
            exit 1
        fi
    done
    info "依赖检查通过"
}

# ── 启动 ──────────────────────────────────────────────────
start() {
    check_deps

    # 检查是否已经在运行
    if [ -f "$FASTAPI_PID" ] && kill -0 "$(cat "$FASTAPI_PID")" 2>/dev/null; then
        warn "FastAPI 已在运行 (PID $(cat "$FASTAPI_PID"))"
    else
        info "启动 FastAPI 后端 → http://$FASTAPI_HOST:$FASTAPI_PORT"
        cd "$PROJECT_DIR"
        nohup uvicorn src.api.app:app \
            --host "$FASTAPI_HOST" \
            --port "$FASTAPI_PORT" \
            --log-level info \
            >> "$PID_DIR/fastapi.log" 2>&1 &
        disown
        echo $! > "$FASTAPI_PID"
        info "FastAPI PID: $(cat "$FASTAPI_PID")"
    fi

    if [ -f "$STREAMLIT_PID" ] && kill -0 "$(cat "$STREAMLIT_PID")" 2>/dev/null; then
        warn "Streamlit 已在运行 (PID $(cat "$STREAMLIT_PID"))"
    else
        info "启动 Streamlit 前端 → http://localhost:$STREAMLIT_PORT"
        cd "$PROJECT_DIR"
        nohup streamlit run "$STREAMLIT_ENTRY" \
            --server.port "$STREAMLIT_PORT" \
            --server.headless true \
            --browser.gatherUsageStats false \
            >> "$PID_DIR/streamlit.log" 2>&1 &
        disown
        echo $! > "$STREAMLIT_PID"
        info "Streamlit PID: $(cat "$STREAMLIT_PID")"
    fi

    echo ""
    info "服务启动中，请稍候..."
    sleep 2

    # 检查启动结果
    if kill -0 "$(cat "$FASTAPI_PID")" 2>/dev/null; then
        info "FastAPI 运行中  → http://$FASTAPI_HOST:$FASTAPI_PORT/docs"
    else
        error "FastAPI 启动失败，请查看日志"
    fi

    if kill -0 "$(cat "$STREAMLIT_PID")" 2>/dev/null; then
        info "Streamlit 运行中 → http://localhost:$STREAMLIT_PORT"
    else
        error "Streamlit 启动失败，请查看日志"
    fi
}

# ── 停止 ──────────────────────────────────────────────────
stop() {
    stopped_any=false

    if [ -f "$FASTAPI_PID" ]; then
        pid=$(cat "$FASTAPI_PID")
        if kill -0 "$pid" 2>/dev/null; then
            info "停止 FastAPI (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            # 等待进程退出
            for i in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            # 如果还在运行，强制终止
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            stopped_any=true
        fi
        rm -f "$FASTAPI_PID"
    fi

    if [ -f "$STREAMLIT_PID" ]; then
        pid=$(cat "$STREAMLIT_PID")
        if kill -0 "$pid" 2>/dev/null; then
            info "停止 Streamlit (PID $pid)..."
            kill "$pid" 2>/dev/null || true
            for i in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 0.5
            done
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            stopped_any=true
        fi
        rm -f "$STREAMLIT_PID"
    fi

    # 清理残留的 uvicorn / streamlit 子进程
    pkill -f "uvicorn src.api.app:app" 2>/dev/null || true
    pkill -f "streamlit run.*Home.py" 2>/dev/null || true

    if $stopped_any; then
        info "服务已停止"
    else
        warn "未发现运行中的服务"
    fi
}

# ── 状态 ──────────────────────────────────────────────────
status() {
    if [ -f "$FASTAPI_PID" ] && kill -0 "$(cat "$FASTAPI_PID")" 2>/dev/null; then
        info "FastAPI 运行中  (PID $(cat "$FASTAPI_PID")) → http://$FASTAPI_HOST:$FASTAPI_PORT/docs"
    else
        error "FastAPI 未运行"
    fi

    if [ -f "$STREAMLIT_PID" ] && kill -0 "$(cat "$STREAMLIT_PID")" 2>/dev/null; then
        info "Streamlit 运行中 (PID $(cat "$STREAMLIT_PID")) → http://localhost:$STREAMLIT_PORT"
    else
        error "Streamlit 未运行"
    fi
}

# ── 主入口 ────────────────────────────────────────────────
case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
