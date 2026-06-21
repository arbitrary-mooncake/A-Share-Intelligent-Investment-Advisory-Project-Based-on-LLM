"""
安全补丁执行器 — 在candidate分支上应用修复，运行测试，通过则保留，失败则回滚。
"""
import os
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any, Optional


PATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "eval", "patches"
)


class SafePatchExecutor:
    """安全补丁执行器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.backup_dir = os.path.join(PATCH_DIR, "backups")
        os.makedirs(PATCH_DIR, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def apply_patch(self, file_path: str, new_content: str,
                    description: str = "") -> Dict[str, Any]:
        """
        安全应用补丁：
        1. 备份原文件
        2. 写入新内容
        3. 运行测试
        4. 失败则回滚

        Returns:
            {"success": bool, "backup_path": str, "test_results": str}
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}"}

        # 1. 备份
        backup_path = self._backup_file(file_path)

        try:
            # 2. 写入新内容
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # 3. 运行测试
            test_result = self._run_tests()

            if test_result["passed"]:
                return {
                    "success": True,
                    "file": file_path,
                    "backup_path": backup_path,
                    "test_results": test_result,
                    "description": description,
                    "applied_at": datetime.now().isoformat(),
                }
            else:
                # 4. 回滚
                self._restore_backup(file_path, backup_path)
                return {
                    "success": False,
                    "error": "测试未通过，已自动回滚",
                    "test_results": test_result,
                }
        except Exception as e:
            # 异常时回滚
            self._restore_backup(file_path, backup_path)
            return {"success": False, "error": str(e)}

    def _backup_file(self, file_path: str) -> str:
        """备份文件"""
        backup_name = f"{os.path.basename(file_path)}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        backup_path = os.path.join(self.backup_dir, backup_name)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _restore_backup(self, file_path: str, backup_path: str):
        """恢复备份"""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)

    def _run_tests(self) -> Dict[str, Any]:
        """运行测试套件"""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-q", "--tb=short"],
                capture_output=True, text=True, timeout=120,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-200:],
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
