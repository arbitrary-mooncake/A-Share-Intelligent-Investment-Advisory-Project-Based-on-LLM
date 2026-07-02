from langchain_mcp_adapters.client import MultiServerMCPClient
from src.utils.logging_config import setup_logger, SUCCESS_ICON, ERROR_ICON, WAIT_ICON
from src.tools.mcp_config import SERVER_CONFIGS
import asyncio
import json
import time
from typing import List, Optional

logger = setup_logger(__name__)

_mcp_client_instance = None
_mcp_tools = None
_mcp_connection_failures = 0
_mcp_last_failure_time = 0.0
# 单例初始化锁: 防止并发首次调用 get_mcp_tools() 各自创建 MultiServerMCPClient 导致
# stdio 子进程泄漏 (2026-07-02 修复: 冷启动 7 agent 同时进入 → 泄漏 6 套 stdio 子进程)。
_mcp_init_lock: Optional[asyncio.Lock] = None
# Circuit breaker: 连续3次失败后进入冷却期30秒
_MCP_CIRCUIT_THRESHOLD = 3
_MCP_COOLDOWN_SECONDS = 30


def print_tool_details(tools):
    """打印工具的详细信息，用于调试"""
    logger.info(f"{SUCCESS_ICON} 工具详细信息:")
    for i, tool in enumerate(tools, 1):
        logger.info(f"  {i}. 工具名称: {tool.name}")
        logger.info(f"     描述: {tool.description}")

        # 打印其他可能的属性
        for attr in ['input_schema', 'parameters', 'schema']:
            if hasattr(tool, attr):
                attr_value = getattr(tool, attr)
                if attr_value:
                    logger.info(f"     {attr}: {attr_value}")

        logger.info(f"     工具类型: {type(tool)}")
        # logger.info(f"     所有属性: {dir(tool)}")
        logger.info("     " + "-" * 50)


async def get_mcp_tools(tool_filter: Optional[List[str]] = None):
    """
    使用定义的服务器配置初始化MultiServerMCPClient，
    并从a-share-mcp-v2服务器获取可用工具。

    全量工具列表只加载一次并全局缓存。不同 Agent 通过 tool_filter
    从缓存中过滤所需工具，不会重复创建 MCP 客户端或子进程。
    集成熔断器：连续3次连接失败后进入30s冷却期。

    Args:
        tool_filter: 可选工具名白名单，只返回列表中指定的工具

    返回:
        list: 从MCP服务器加载的LangChain兼容工具列表。
    """
    global _mcp_client_instance, _mcp_tools, _mcp_init_lock

    # 延迟创建锁 (必须在 async 函数内, 因为模块导入时 event loop 未必存在)
    if _mcp_init_lock is None:
        _mcp_init_lock = asyncio.Lock()

    # 熔断器检查 (快速路径, 无锁)
    if _mcp_tools is not None and not _check_circuit_breaker():
        # 已缓存工具且熔断器开启：返回缓存工具（仍可尝试调用）
        pass
    elif _mcp_tools is None and not _check_circuit_breaker():
        logger.error(f"{ERROR_ICON} MCP 熔断器开启，拒绝连接请求")
        return []

    # 首次加载：在锁内 double-check, 防止 7 agent 并发进入各自创建 MCP 客户端
    # (每个 MultiServerMCPClient 启动 4 个 stdio 子进程, 泄漏会长期占用 Tushare/MCP 通道)
    if _mcp_tools is None:
        async with _mcp_init_lock:
            if _mcp_tools is None:
                for attempt in range(3):
                    logger.info(
                        f"{WAIT_ICON} Initializing MultiServerMCPClient (attempt {attempt+1}/3)")
                    try:
                        _mcp_client_instance = MultiServerMCPClient(SERVER_CONFIGS)
                        loaded_tools = await _mcp_client_instance.get_tools()
                        if loaded_tools:
                            _mcp_tools = loaded_tools
                            _record_mcp_success()
                            logger.info(
                                f"{SUCCESS_ICON} Successfully loaded {len(_mcp_tools)} tools (cached).")
                            break
                        logger.warning(
                            f"{ERROR_ICON} MCP 返回空工具列表 (attempt {attempt+1}/3)")
                    except Exception as e:
                        logger.error(
                            f"{ERROR_ICON} MCP 初始化失败 (attempt {attempt+1}/3): {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 * (attempt + 1))

                if _mcp_tools is None:
                    _record_mcp_failure()
                    _mcp_tools = None  # 保持 None 以便下次调用触发重试
                    return []

    # 从缓存过滤
    if tool_filter is not None:
        filtered = [t for t in _mcp_tools if t.name in tool_filter]
        logger.info(
            f"{SUCCESS_ICON} Returning {len(filtered)}/{len(_mcp_tools)} tools (filtered).")
        return filtered

    logger.info(f"{SUCCESS_ICON} Returning all {len(_mcp_tools)} cached MCP tools.")
    return _mcp_tools


async def reconnect_mcp() -> bool:
    """重置并重新建立 MCP 连接。返回是否成功。

    必须在 _mcp_init_lock 内执行, 防止与 get_mcp_tools() 首次初始化竞态。
    """
    global _mcp_client_instance, _mcp_tools, _mcp_connection_failures, _mcp_last_failure_time, _mcp_init_lock
    if _mcp_init_lock is None:
        _mcp_init_lock = asyncio.Lock()
    async with _mcp_init_lock:
        logger.info(f"{WAIT_ICON} 正在重置 MCP 连接...")
        # 关闭旧连接
        if _mcp_client_instance:
            try:
                await _mcp_client_instance.__aexit__(None, None, None)
            except Exception:
                pass
        _mcp_client_instance = None
        _mcp_tools = None
        _mcp_connection_failures = 0
        _mcp_last_failure_time = 0.0
    # 重新加载 (get_mcp_tools 会自己获取锁)
    tools = await get_mcp_tools()
    return len(tools) > 0 if tools else False


def _check_circuit_breaker() -> bool:
    """检查熔断器状态。返回 True 表示电路关闭（可正常请求）。"""
    global _mcp_connection_failures, _mcp_last_failure_time
    if _mcp_connection_failures < _MCP_CIRCUIT_THRESHOLD:
        return True
    since_last = time.time() - _mcp_last_failure_time
    if since_last > _MCP_COOLDOWN_SECONDS:
        # 冷却期结束，半开状态：允许一次尝试
        logger.info(f"{WAIT_ICON} MCP 熔断器冷却期结束 ({since_last:.0f}s)，进入半开状态")
        _mcp_connection_failures = 0
        return True
    logger.warning(f"⚠ MCP 熔断器开启（{_mcp_connection_failures}次连续失败，{_MCP_COOLDOWN_SECONDS - since_last:.0f}s后重试）")
    return False


def _record_mcp_failure():
    """记录一次 MCP 连接失败。"""
    global _mcp_connection_failures, _mcp_last_failure_time
    _mcp_connection_failures += 1
    _mcp_last_failure_time = time.time()


def _record_mcp_success():
    """重置 MCP 失败计数器。"""
    global _mcp_connection_failures
    _mcp_connection_failures = 0


async def close_mcp_client_sessions():
    """
    关闭MultiServerMCPClient管理的任何开放会话。
    如果必要，应在应用程序关闭时调用此函数。
    """
    global _mcp_client_instance
    if _mcp_client_instance:
        logger.info(f"{WAIT_ICON} Closing MCP client sessions...")
        try:
            logger.info(
                f"{SUCCESS_ICON} MCP client sessions (if any were persistently open) assumed closed or managed by library.")
            _mcp_client_instance = None   # 允许重新初始化
            global _mcp_tools
            _mcp_tools = None
        except Exception as e:
            logger.error(
                f"{ERROR_ICON} Error during MCP client session cleanup: {e}", exc_info=True)
    else:
        logger.info("MCP client was not initialized, no sessions to close.")


# 测试此模块的示例（可选，用于直接执行）
async def _main_test_mcp_client():
    logger.info("--- Testing MCP Client Tool Loading ---")
    tools = await get_mcp_tools()
    if tools:
        print(f"Successfully loaded {len(tools)} tools:")
        for tool in tools:
            print(
                f"- Name: {tool.name}")

        # 测试一个简单的工具调用（如果有合适的工具）
        if tools:
            logger.info("--- Testing Tool Call ---")
            # 尝试调用第一个工具（需要根据实际工具调整参数）
            first_tool = tools[0]
            logger.info(f"尝试调用工具: {first_tool.name}")

            # 这里需要根据实际的工具参数schema来构造测试参数
            # 暂时跳过实际调用，只是展示结构
            logger.info("工具调用测试跳过（需要实际参数）")
    else:
        print("Failed to load tools or no tools found.")

    # 测试关闭（如果适用）
    await close_mcp_client_sessions()
    logger.info("--- MCP Client Test Complete ---")

if __name__ == '__main__':
    # 这允许直接运行测试，例如：python -m src.tools.mcp_client
    # 确保您的环境已设置（例如，'uv'命令可用）。
    # E:\github\a_share_mcp的a_share_mcp服务器应该准备好运行。

    # 如果尚未配置，为测试运行设置基本日志记录
    if not logger.hasHandlers():
        import logging
        logging.basicConfig(level=logging.INFO)
        logger.info("Basic logging configured for test run.")

    asyncio.run(_main_test_mcp_client())
