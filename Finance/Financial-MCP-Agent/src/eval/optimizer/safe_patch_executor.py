"""
安全补丁执行器 — 在candidate分支上应用修复，运行测试，通过则保留，失败则回滚。

安全层级（belt + suspenders）：
  1. 主防护：Git分支隔离（stash → 创建candidate分支 → 应用 → 测试 → 合并/丢弃）
  2. 备用防护：文件级备份（shutil.copy2），在Git操作失败时启用
  3. 所有Git操作均通过 subprocess.run() 执行，带完整错误处理
"""

import os
import subprocess
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple


PATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "eval", "patches"
)

# 项目根目录（git仓库根）
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)


def _run_git(args: list, cwd: str = REPO_ROOT, timeout: int = 30) -> Tuple[bool, str, str]:
    """执行Git命令并返回 (success, stdout, stderr)。

    Args:
        args: git命令参数列表，不含 'git' 前缀
        cwd: 工作目录
        timeout: 超时秒数

    Returns:
        (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.returncode == 0, result.stdout.strip(), result.stderr.strip())
    except subprocess.TimeoutExpired:
        return (False, "", f"Git命令超时 ({timeout}s): git {' '.join(args)}")
    except FileNotFoundError:
        return (False, "", "git 命令不可用（未安装或不在PATH中）")
    except Exception as e:
        return (False, "", f"Git命令异常: {e}")


def _get_current_branch(cwd: str = REPO_ROOT) -> Optional[str]:
    """获取当前Git分支名。"""
    ok, stdout, stderr = _run_git(["branch", "--show-current"], cwd=cwd)
    if ok and stdout:
        return stdout
    return None


def _is_git_repo(cwd: str = REPO_ROOT) -> bool:
    """检查是否在Git仓库中。"""
    ok, stdout, stderr = _run_git(["rev-parse", "--git-dir"], cwd=cwd)
    return ok


def _has_uncommitted_changes(cwd: str = REPO_ROOT) -> bool:
    """检查是否有未提交的更改。"""
    ok, stdout, stderr = _run_git(["status", "--porcelain"], cwd=cwd)
    return ok and bool(stdout)


class SafePatchExecutor:
    """安全补丁执行器 — Git分支隔离 + 文件级备份双重防护。

    工作流：
      1. 文件级备份（shutil.copy2）— 安全兜底
      2. Git stash（保存当前未提交工作）
      3. 创建 candidate/{ticket_id}_{timestamp} 分支
      4. 在candidate分支上应用补丁
      5. 运行测试
      6a. 测试通过 → 合并到原分支 → 删除candidate分支 → 还原stash
      6b. 测试失败 → 切回原分支 → 强制删除candidate分支 → 还原备份 → 还原stash
    """

    def __init__(self, config: Dict[str, Any] = None):
        """初始化补丁执行器。

        Args:
            config: 可选配置字典，支持键：
                - branch_prefix (str): candidate分支前缀，默认 "candidate"
                - use_git_isolation (bool): 是否启用Git隔离，默认 True
                - keep_candidate_on_failure (bool): 失败时是否保留candidate分支（调试用），默认 False
                - test_command (list): 测试命令，默认 ["python", "-m", "pytest", "tests/", "-q", "--tb=short"]
                - test_timeout (int): 测试超时秒数，默认 120
                - git_timeout (int): Git命令超时秒数，默认 30
        """
        self.config = config or {}
        self.branch_prefix = self.config.get("branch_prefix", "candidate")
        self.use_git_isolation = self.config.get("use_git_isolation", True)
        self.keep_candidate_on_failure = self.config.get("keep_candidate_on_failure", False)
        self.test_command = self.config.get(
            "test_command",
            ["python", "-m", "pytest", "tests/", "-q", "--tb=short"]
        )
        self.test_timeout = self.config.get("test_timeout", 120)
        self.git_timeout = self.config.get("git_timeout", 30)

        self.backup_dir = os.path.join(PATCH_DIR, "backups")
        os.makedirs(PATCH_DIR, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

        # 检查Git可用性
        self._git_available = _is_git_repo()
        if not self._git_available:
            print("[SafePatchExecutor] 警告: Git仓库不可用，将只使用文件级备份")

    # ── Git 分支隔离方法 ───────────────────────────────────────────

    def _git_stash_and_branch(self, candidate_branch_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Git stash + 创建candidate分支。

        流程：
          1. 获取当前分支名
          2. 如果有未提交更改 → git stash push
          3. git checkout -b {candidate_branch_name}

        Args:
            candidate_branch_name: 候选分支名

        Returns:
            (success, original_branch_name, stash_ref)
            - stash_ref: 如果创建了stash则返回其引用，否则为 None
        """
        original_branch = _get_current_branch()
        if not original_branch:
            return (False, None, None)

        stash_ref = None

        # 步骤1: 检查并保存未提交更改
        if _has_uncommitted_changes():
            ok, stdout, stderr = _run_git(
                ["stash", "push", "-m", f"auto-stash before candidate patch {candidate_branch_name}"],
                timeout=self.git_timeout,
            )
            if not ok:
                print(f"[SafePatchExecutor] Git stash 失败: {stderr}")
                # stash失败不算致命，继续
            elif stdout and "No local changes to save" not in stdout and "Saved working directory" in stdout:
                # 仅当确认stash实际保存了更改时，才记录stash_ref
                # git stash push可能返回exit 0但实际未保存（如仅有untracked文件且未加-u）
                stash_ref = "stash@{0}"  # 最近一个stash

        # 步骤2: 创建candidate分支
        ok, stdout, stderr = _run_git(
            ["checkout", "-b", candidate_branch_name],
            timeout=self.git_timeout,
        )
        if not ok:
            # 如果分支已存在，尝试切换
            if "already exists" in stderr:
                ok2, _, err2 = _run_git(
                    ["checkout", candidate_branch_name],
                    timeout=self.git_timeout,
                )
                if not ok2:
                    # 创建失败，还原stash
                    self._pop_stash(stash_ref)
                    return (False, original_branch, stash_ref)
            else:
                # 创建失败，还原stash
                self._pop_stash(stash_ref)
                return (False, original_branch, stash_ref)

        return (True, original_branch, stash_ref)

    def _git_merge_and_cleanup(
        self,
        candidate_branch_name: str,
        original_branch: str,
        stash_ref: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """测试通过后的合并和清理。

        流程：
          1. git checkout {original_branch}
          2. git merge --no-ff {candidate_branch_name}
          3. git branch -d {candidate_branch_name}
          4. 如有stash → git stash pop {stash_ref}

        Args:
            candidate_branch_name: 候选分支名
            original_branch: 原始分支名
            stash_ref: stash引用

        Returns:
            (success, message)
        """
        # 步骤1: 切回原分支
        ok, stdout, stderr = _run_git(
            ["checkout", original_branch],
            timeout=self.git_timeout,
        )
        if not ok:
            # 切回原分支失败 → 不可在candidate分支上继续merge操作
            if stash_ref:
                self._pop_stash(stash_ref)
            return (False, f"切换回 {original_branch} 失败: {stderr}")

        errors = []

        # 步骤2: 合并candidate分支
        ok, stdout, stderr = _run_git(
            ["merge", "--no-ff", candidate_branch_name, "-m",
             f"auto-merge candidate patch: {candidate_branch_name}"],
            timeout=self.git_timeout,
        )
        if not ok:
            errors.append(f"合并 {candidate_branch_name} 失败: {stderr}")
            # 合并失败 → 中止合并
            _run_git(["merge", "--abort"], timeout=self.git_timeout)
        else:
            # 步骤3: 删除candidate分支（合并成功后）
            ok_del, _, err_del = _run_git(
                ["branch", "-d", candidate_branch_name],
                timeout=self.git_timeout,
            )
            if not ok_del:
                print(f"[SafePatchExecutor] 删除candidate分支警告: {err_del}")

        # 步骤4: 还原stash
        if stash_ref:
            self._pop_stash(stash_ref)

        if errors:
            return (False, "; ".join(errors))
        return (True, f"成功合并 {candidate_branch_name} → {original_branch}")

    def _git_abort_candidate(
        self,
        candidate_branch_name: str,
        original_branch: str,
        stash_ref: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """测试失败后的中止和清理。

        流程：
          1. git checkout {original_branch}
          2. git branch -D {candidate_branch_name}（强制删除）
          3. 如有stash → git stash pop {stash_ref}

        Args:
            candidate_branch_name: 候选分支名
            original_branch: 原始分支名
            stash_ref: stash引用

        Returns:
            (success, message)
        """
        # 步骤1: 切回原分支
        ok, stdout, stderr = _run_git(
            ["checkout", original_branch],
            timeout=self.git_timeout,
        )
        if not ok:
            # 切回原分支失败 → 不可在candidate分支上继续删除分支操作
            if stash_ref:
                self._pop_stash(stash_ref)
            return (False, f"切换回 {original_branch} 失败: {stderr}")

        errors = []

        # 步骤2: 强制删除candidate分支
        if not self.keep_candidate_on_failure:
            ok_del, _, err_del = _run_git(
                ["branch", "-D", candidate_branch_name],
                timeout=self.git_timeout,
            )
            if not ok_del:
                print(f"[SafePatchExecutor] 强制删除candidate分支警告: {err_del}")
        else:
            print(f"[SafePatchExecutor] 保留candidate分支（调试模式）: {candidate_branch_name}")

        # 步骤3: 还原stash
        if stash_ref:
            self._pop_stash(stash_ref)

        if errors:
            return (False, "; ".join(errors))
        return (True, f"已中止并清理 {candidate_branch_name}")

    def _pop_stash(self, stash_ref: Optional[str]) -> None:
        """安全地还原stash。"""
        if not stash_ref:
            return
        ok, stdout, stderr = _run_git(
            ["stash", "pop", stash_ref],
            timeout=self.git_timeout,
        )
        if not ok:
            print(f"[SafePatchExecutor] Git stash pop 失败: {stderr}")

    def _generate_branch_name(self, ticket_id: str = "") -> str:
        """生成candidate分支名: candidate/{ticket_id}_{timestamp}。

        Args:
            ticket_id: 可选的问题/变更标识

        Returns:
            分支名，如 candidate/PARAM_TUNE_20260705_143025
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if ticket_id:
            # 清理ticket_id中的特殊字符
            safe_id = "".join(c for c in ticket_id if c.isalnum() or c in "_-")
            if not safe_id:
                safe_id = "auto"
            return f"{self.branch_prefix}/{safe_id}_{ts}"
        return f"{self.branch_prefix}/auto_{ts}"

    # ── 公共API ──────────────────────────────────────────────────

    def apply_patch(
        self,
        file_path: str,
        new_content: str,
        description: str = "",
        ticket_id: str = "",
    ) -> Dict[str, Any]:
        """安全应用补丁（Git分支隔离 + 文件备份双重防护）。

        流程：
          1. 文件级备份（兜底安全网）
          2. Git stash + 创建candidate分支
          3. 在candidate分支上应用补丁
          4. 运行测试
          5a. 通过 → 合并 + 清理 → 成功
          5b. 失败 → 中止 + 还原备份 + 清理 → 失败
          6. 如果Git任何步骤失败 → 回退到纯文件备份模式

        Args:
            file_path: 要修改的文件路径
            new_content: 新文件内容
            description: 变更描述
            ticket_id: 关联的ticket标识（用于分支命名）

        Returns:
            {"success": bool, "backup_path": str, "test_results": dict, "isolation_mode": str, ...}
        """
        if not os.path.exists(file_path):
            return {"success": False, "error": f"文件不存在: {file_path}", "isolation_mode": "none"}

        # 步骤1: 文件级备份（始终执行 — 安全兜底）
        backup_path = self._backup_file(file_path)
        applied_at = datetime.now().isoformat()

        # 如果Git隔离不可用，直接回退到纯文件备份模式
        if not self._git_available or not self.use_git_isolation:
            return self._apply_with_file_backup_only(file_path, new_content, backup_path, description, applied_at)

        # 步骤2: Git分支隔离
        branch_name = self._generate_branch_name(ticket_id)
        git_success, original_branch, stash_ref = self._git_stash_and_branch(branch_name)

        if not git_success:
            # Git操作失败 → 回退到纯文件备份模式
            print(f"[SafePatchExecutor] Git分支创建失败，回退到文件备份模式")
            self._restore_backup(file_path, backup_path)
            return self._apply_with_file_backup_only(file_path, new_content, backup_path, description, applied_at)

        try:
            # 步骤3: 在candidate分支上应用补丁
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            # 步骤4: 运行测试
            test_result = self._run_tests()

            if test_result["passed"]:
                # 步骤5a: 测试通过 → 合并 + 清理
                # 先提交更改到candidate分支
                commit_ok, _, commit_err = _run_git(
                    ["add", file_path],
                    timeout=self.git_timeout,
                )
                if commit_ok:
                    _run_git(
                        ["commit", "-m", f"auto-patch: {description or branch_name}"],
                        timeout=self.git_timeout,
                    )

                merge_ok, merge_msg = self._git_merge_and_cleanup(
                    branch_name, original_branch, stash_ref
                )

                return {
                    "success": True,
                    "file": file_path,
                    "backup_path": backup_path,
                    "test_results": test_result,
                    "description": description,
                    "applied_at": applied_at,
                    "isolation_mode": "git_branch",
                    "branch_name": branch_name,
                    "merge_result": merge_msg,
                }
            else:
                # 步骤5b: 测试失败 → 中止 + 还原备份 + 清理
                # 还原文件到原始状态
                self._restore_backup(file_path, backup_path)

                # 中止candidate分支
                abort_ok, abort_msg = self._git_abort_candidate(
                    branch_name, original_branch, stash_ref
                )

                return {
                    "success": False,
                    "error": "测试未通过，已自动回滚",
                    "file": file_path,
                    "backup_path": backup_path,
                    "test_results": test_result,
                    "isolation_mode": "git_branch",
                    "branch_name": branch_name,
                    "abort_result": abort_msg,
                }

        except Exception as e:
            # 异常时还原备份
            self._restore_backup(file_path, backup_path)
            # 尝试清理Git分支
            try:
                if original_branch:
                    self._git_abort_candidate(branch_name, original_branch, stash_ref)
            except Exception:
                pass
            return {"success": False, "error": str(e), "isolation_mode": "git_branch_fallback"}

    def _apply_with_file_backup_only(
        self,
        file_path: str,
        new_content: str,
        backup_path: str,
        description: str,
        applied_at: str,
    ) -> Dict[str, Any]:
        """纯文件备份模式的补丁应用（Git不可用时的回退）。"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            test_result = self._run_tests()

            if test_result["passed"]:
                return {
                    "success": True,
                    "file": file_path,
                    "backup_path": backup_path,
                    "test_results": test_result,
                    "description": description,
                    "applied_at": applied_at,
                    "isolation_mode": "file_backup_only",
                }
            else:
                self._restore_backup(file_path, backup_path)
                return {
                    "success": False,
                    "error": "测试未通过，已自动回滚",
                    "test_results": test_result,
                    "isolation_mode": "file_backup_only",
                }
        except Exception as e:
            self._restore_backup(file_path, backup_path)
            return {"success": False, "error": str(e), "isolation_mode": "file_backup_only"}

    # ── 文件备份（兜底安全网）─────────────────────────────────

    def _backup_file(self, file_path: str) -> str:
        """备份文件（文件级）。"""
        backup_name = f"{os.path.basename(file_path)}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        backup_path = os.path.join(self.backup_dir, backup_name)
        shutil.copy2(file_path, backup_path)
        return backup_path

    def _restore_backup(self, file_path: str, backup_path: str):
        """恢复备份（文件级）。"""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)

    # ── 测试运行 ────────────────────────────────────────────────

    def _run_tests(self) -> Dict[str, Any]:
        """运行测试套件。"""
        try:
            result = subprocess.run(
                self.test_command,
                capture_output=True, text=True, timeout=self.test_timeout,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-200:],
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"测试超时 ({self.test_timeout}s)"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
