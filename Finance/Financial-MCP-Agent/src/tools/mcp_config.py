"""
MCP服务器配置模块 - 包含连接A股MCP服务器的配置信息
"""

import os

# 获取当前文件的绝对路径，并计算出MCP服务器的路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "a-share-mcp-is-just-i-need")
MCP_SERVER_DIR = os.path.normpath(MCP_SERVER_DIR)

SERVER_CONFIGS = {
    "a_share_mcp_v2": {
        "command": "python",
        "args": [
            "-u",
            os.path.join(MCP_SERVER_DIR, "mcp_server.py")
        ],
        "transport": "stdio",
        "env": {
            "PYTHONPATH": MCP_SERVER_DIR
        }
    }
}