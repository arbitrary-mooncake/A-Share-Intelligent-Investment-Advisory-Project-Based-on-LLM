"""
MCP服务器配置模块 - 包含连接A股MCP服务器和Tushare MCP服务器的配置信息
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "a-share-mcp-is-just-i-need")
MCP_SERVER_DIR = os.path.normpath(MCP_SERVER_DIR)
# Financial-MCP-Agent 项目根目录 (用于 tushare_mcp_server 导入 src.utils.tushare_client)
AGENT_ROOT = os.path.normpath(os.path.join(CURRENT_DIR, "..", ".."))

PYTHON_EXE = sys.executable

_ENV = os.environ.copy()
_existing_pythonpath = _ENV.get("PYTHONPATH", "")
if _existing_pythonpath:
    _ENV["PYTHONPATH"] = MCP_SERVER_DIR + os.pathsep + AGENT_ROOT + os.pathsep + _existing_pythonpath
else:
    _ENV["PYTHONPATH"] = MCP_SERVER_DIR + os.pathsep + AGENT_ROOT

SERVER_CONFIGS = {
    "a_share_mcp_v2": {
        "command": PYTHON_EXE,
        "args": [
            "-u",
            os.path.join(MCP_SERVER_DIR, "mcp_server.py")
        ],
        "transport": "stdio",
        "env": _ENV
    },
    "tushare_mcp": {
        "command": PYTHON_EXE,
        "args": [
            "-u",
            os.path.join(MCP_SERVER_DIR, "tushare_mcp_server.py")
        ],
        "transport": "stdio",
        "env": _ENV
    }
}
