"""记忆工具（精简移植版）—— 持久化精选记忆。

对应原版 hermes-agent 的 tools/memory_tool.py（1248 行）。两个存储：
  - MEMORY.md：agent 的个人笔记与观察（环境事实、项目约定、工具怪癖、学到的经验）
  - USER.md：agent 对用户的了解（偏好、沟通风格、期望、工作习惯）

两者在会话开始时以「冻结快照」注入系统提示。会话中的写入会立即落盘
（持久），但**不**改变系统提示——这保住了整个会话的前缀缓存（prefix
cache）。快照在下一个会话开始时刷新。

条目分隔符：§（小节符）。条目可以多行。
字符上限（不是 token 上限），因为字符数不依赖模型。

精简版改动（相对原版）：
- 砍掉审批门（_apply_write_gate / _apply_batch_write_gate，依赖原版
  tools.write_approval）与 apply_memory_pending（审批回放用）——
  my-hermes 无审批系统，写入直接放行（与原版"门加载失败则放行"一致）；
- 其余（MemoryStore 全量、memory_tool handler、MEMORY_SCHEMA、
  load_on_disk_store）照原版移植，函数签名逐字一致。
"""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from utils import atomic_write_text

# fcntl 仅 Unix；Windows 用 msvcrt 做文件锁
msvcrt = None
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        pass

logger = logging.getLogger(__name__)


def get_memory_dir() -> Path:
    """返回按 profile 作用域的记忆目录。"""
    return get_hermes_home() / "memories"


# 系统提示记忆块渲染使用的稳定头（与 _render_block 保持一致）
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)",
    "user": "USER PROFILE (who the user is)",
}

ENTRY_DELIMITER = "\n§\n"


from tools.threat_patterns import first_threat_message as _first_threat_message


def _scan_memory_content(content: str) -> Optional[str]:
    """扫描记忆内容中的注入/外泄模式。命中返回错误串，否则返回 None。"""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """检测到外部漂移时返回的错误字典。

    磁盘上的记忆文件包含无法通过工具解析/序列化往返的内容（可能是
    patch 工具、shell 追加、手动编辑或并发会话写入的）——直接冲刷会
    丢弃这些内容。拒绝本次变更，把 .bak.<ts> 快照路径指给操作者。
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). A snapshot was saved to {bak_path}. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


# _reload_target 在目标文件「存在但读不出来」时返回的哨兵。
# 与漂移备份路径（str）和干净重载（None）都不同：调用方必须中止变更，
# 而不是覆盖一个不可读的文件。
_READ_FAILED = object()


def _read_failed_error(path: "Path") -> Dict[str, Any]:
    """磁盘记忆文件不可读时的错误字典。

    存在但读不出来的文件**不是**空存储。把它当 [] 再持久化，会把整个
    文件从空条目列表重写——抹掉用户的记忆。拒绝写入以免丢失。
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: the file exists on disk but could "
            f"not be read right now (temporarily locked by another program, a "
            f"permission change, invalid/corrupt text encoding, or a filesystem "
            f"error). Treating an unreadable file as empty and saving would wipe "
            f"existing memory, so the write is refused. Nothing was changed — "
            f"retry in a moment."
        ),
    }


class MemoryStore:
    """
    有界的精选记忆，文件持久化。每个 AIAgent 一个实例。

    维护两份并行状态：
      - _system_prompt_snapshot：加载时冻结，用于系统提示注入。会话中
        绝不修改。保持前缀缓存稳定。
      - memory_entries / user_entries：实时状态，由工具调用修改并落盘。
        工具响应始终反映这份实时状态。
    """

    # 单回合内连续整合失败（溢出 / 零匹配）达到该次数后，停止指导模型
    # "本回合内重试"，返回终态 "save skipped" 结果，防止脆弱的
    # replace/add 把回合循环到预算耗尽、压掉用户回复（issue #42405）。
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # 系统提示用的冻结快照——在 load_from_disk() 时设置一次
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        # 每回合容量整合失败计数；回合边界由 reset_consolidation_failures() 清零
        self._consolidation_failures = 0

    def reset_consolidation_failures(self) -> None:
        """重置本回合的整合失败计数（回合开始时调用）。"""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """记录一次容量整合失败并优雅降级。

        未超上限时原样返回 response（其中已含"如何自纠正 + 本回合重试"的
        指引）。超上限后去掉重试指引，返回**终态**结果，让模型停止循环
        调用记忆并继续回复用户——失败的记忆副作用绝不能阻塞回合回复。
        """
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {self._consolidation_failures} times "
                "this turn. Stop retrying memory calls — leave memory unchanged for "
                "now and continue with your reply to the user. The fact can be saved "
                "in a later turn."
            ),
        }

    def load_from_disk(self):
        """从 MEMORY.md 和 USER.md 加载条目，并捕获系统提示快照。

        冻结快照才是进入系统提示的内容。构建快照时逐条扫描注入/提示词
        攻击模式——**任何**命中都把该条在快照中替换为占位符
        （如 ``[BLOCKED: …]``），使磁盘上被投毒的记忆文件（供应链、被
        入侵工具、姊妹会话写入）无法注入系统提示。

        实时 memory_entries / user_entries 保留原文，用户仍能通过检查
        源文件看到被投毒的条目并删除它们——静默丢弃反而会向用户隐藏
        攻击。扫描由磁盘字节确定性决定，快照在整个会话内保持稳定
        （前缀缓存不变式成立）。
        """
        mem_dir = get_memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # 去重（保留顺序，保留首次出现）
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # 只对系统提示快照做净化。实时状态保留原文，方便用户通过
        # memory 工具查看并删除被投毒的条目。
        sanitized_memory = self._sanitize_entries_for_snapshot(
            self.memory_entries, "MEMORY.md"
        )
        sanitized_user = self._sanitize_entries_for_snapshot(
            self.user_entries, "USER.md"
        )

        # 捕获冻结快照供系统提示注入
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """返回把命中威胁的条目替换为占位符后的 entries。

        每条用共享威胁模式库的 ``"strict"`` 作用域扫描（与记忆写入一致）。
        命中时在返回列表里替换为 ``"[BLOCKED: <filename> entry contained
        threat pattern: <ids>. Removed from system prompt.]"``——占位符进入
        快照，原文留在实时状态供用户查看删除。

        空条目或已是 [BLOCKED: 标记的条目原样通过。
        """
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry or entry.startswith("[BLOCKED:"):
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename,
                    ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    f"use memory(action=remove) "
                    f"to delete the original.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """获取独占文件锁，保证读-改-写安全。

        使用单独的 .lock 文件，这样记忆文件本身仍可经 os.replace() 原子替换。
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        mem_dir = get_memory_dir()
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str, *, skip_drift: bool = False):
        """在文件锁下重新从磁盘读条目到内存状态。

        变更前调用以拿到最新状态。若检测到外部漂移（磁盘文件包含无法
        往返的内容，或某条超过整个存储的字符上限）则返回备份路径——
        此时调用方必须中止变更，冲刷会丢弃无法往返的内容。干净重载返回
        None。

        文件「存在但读不出来」时返回 _READ_FAILED 哨兵。调用方必须中止：
        磁盘条目未知，从"视为空"的视图覆盖会抹掉它们。这是 add 的真实
        暴露面——它跳过漂移守卫是因为追加安全，但这个推理只在重载确实
        看到文件时才成立。读取失败被当成 [] 会把 add 变成整文件重写。

        skip_drift=True 时绕过往返/条目大小检查。add 用（追加不重写，
        已有内容永不被覆盖）。
        """
        path = self._path_for(target)
        raw, read_ok = self._read_raw_checked(path)
        if not read_ok:
            # 保持内存条目不动，让调用方中止；覆盖不可读文件会毁掉它
            return _READ_FAILED
        # 漂移检查与条目解析都基于同一份原始快照。漂移守卫曾自己重读
        # 文件并把第二次读取失败当作"无漂移"——这会让 checked 重载与
        # 漂移检查之间的读取失败被 replace/remove/apply_batch 用陈旧
        # 视图重写文件，静默丢弃外部写入者刚加的内容。一次读取、一份
        # 快照、无窗口。
        bak = None if skip_drift else self._detect_external_drift(target, raw)
        fresh = self._parse_entries(raw)
        fresh = list(dict.fromkeys(fresh))  # 去重
        self._set_entries(target, fresh)
        return bak

    def save_to_disk(self, target: str):
        """把条目持久化到对应文件。每次变更后调用。"""
        get_memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """追加新条目。会超限时返回错误。"""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # 接受前扫描注入/外泄
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # 锁下重读磁盘，拾取其他会话的写入。add（仅追加）跳过漂移
            # 守卫——追加从不覆盖已有内容，所以同会话内先前工具写入的
            # 往返不匹配无害。replace/remove 保持漂移守卫（整文件重写
            # 会丢弃无法往返的内容，issue #26045）。
            #
            # 但"追加不覆盖"只在重载确实读到文件时成立。add 会从解析
            # 出的条目重写**整个**文件，所以存在但读成空（瞬时锁、权限
            # 波动、I/O 错误）会把文件重写成只剩新条目——抹掉全部既有
            # 记忆。拒绝。
            if self._reload_target(target, skip_drift=True) is _READ_FAILED:
                return _read_failed_error(self._path_for(target))

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # 拒绝完全重复
            if content in entries:
                return self._success_response(
                    target, "Entry already exists (no duplicate added)."
                )

            # 计算新总量
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Memory at {current:,}/{limit:,} chars. "
                        f"Adding this entry ({len(content)} chars) would exceed the limit. "
                        f"Consolidate now: use 'replace' to merge overlapping entries into "
                        f"shorter ones or 'remove' stale or less important entries (see "
                        f"current_entries below), then retry this add — all in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """找到包含 old_text 子串的条目，替换为 new_content。"""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {
                "success": False,
                "error": "new_content cannot be empty. Use 'remove' to delete entries.",
            }

        # 扫描替换内容中的注入/外泄
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to replace.",
                    "current_entries": entries,
                })

            if len(matches) > 1:
                # 若所有匹配完全相同（精确重复），只操作第一个
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # 全部相同——安全地只替换第一个

            idx = matches[0][0]
            limit = self._char_limit(target)

            # 检查替换是否超预算
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                        f"Shorten the new content, or 'remove' other stale or less important "
                        f"entries to make room (see current_entries below), then retry — all "
                        f"in this turn."
                    ),
                    "current_entries": entries,
                    "usage": f"{current:,}/{limit:,}",
                })

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """删除包含 old_text 子串的条目。"""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if bak:
                return _drift_error(self._path_for(target), bak)

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to remove.",
                    "current_entries": entries,
                })

            if len(matches) > 1:
                # 若所有匹配完全相同（精确重复），删除第一个
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # 全部相同——安全地只删除第一个

            idx = matches[0][0]
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def apply_batch(
        self, target: str, operations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """对同一目标原子地应用一串 add/replace/remove 操作。

        所有操作都按**最终**预算校验并应用——中间溢出无关紧要。这让模型
        在**一次**工具调用里释放空间（remove/replace）并添加新条目，而不
        是多次往返的"先整合再重试"舞步（那会反复重发整个对话上下文）。

        语义：全有或全无。任一操作畸形、不匹配、或净结果超限，**什么都不
        写**，返回描述第一个失败 + 实时状态的错误。
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # 触碰磁盘前先扫描每个 add/replace 内容的注入/外泄——
        # 一条被投毒的操作拒绝整个批次
        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content")
            if act in {"add", "replace"} and new_content:
                scan_error = _scan_memory_content(new_content)
                if scan_error:
                    return {
                        "success": False,
                        "error": f"Operation {i + 1}: {scan_error}",
                    }

        with self._file_lock(self._path_for(target)):
            bak = self._reload_target(target)
            if bak is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if bak:
                return _drift_error(self._path_for(target), bak)

            # 在副本上工作；整批校验通过才提交
            working: List[str] = list(self._entries_for(target))
            limit = self._char_limit(target)

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    if content in working:
                        continue  # 幂等——跳过重复，不使批次失败
                    working.append(content)

                elif act == "replace":
                    if not old_text:
                        return self._batch_error(
                            target, f"{pos}: old_text is required."
                        )
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(
                            target, f"{pos}: no entry matched '{old_text}'."
                        )
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working[matches[0]] = content

                elif act == "remove":
                    if not old_text:
                        return self._batch_error(
                            target, f"{pos}: old_text is required."
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(
                            target, f"{pos}: no entry matched '{old_text}'."
                        )
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working.pop(matches[0])

                else:
                    return self._batch_error(
                        target,
                        f"{pos}: unknown action. Use add, replace, or remove.",
                    )

            # 只按**最终**状态检查预算
            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if new_total > limit:
                current = self._char_count(target)
                return self._consolidation_failure({
                    "success": False,
                    "error": (
                        f"After applying all {len(operations)} operations, memory would be at "
                        f"{new_total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                        f"entries in the same batch (see current_entries below), then retry."
                    ),
                    "current_entries": self._entries_for(target),
                    "usage": f"{current:,}/{limit:,}",
                })

            # 提交
            self._set_entries(target, working)
            self.save_to_disk(target)

        return self._success_response(
            target, f"Applied {len(operations)} operation(s)."
        )

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """构造报告实时（未提交）状态的批次中止错误。"""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return self._consolidation_failure({
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
        })

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """返回系统提示注入用的冻结快照。

        返回 load_from_disk() 时捕获的状态，**不是**实时状态。会话中的
        写入不影响它。这让系统提示在所有回合间保持稳定，保住前缀缓存。

        快照为空（加载时无条目）时返回 None。
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- 内部助手 --

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """错误反馈用的截断单行预览。"""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # 成功写入意味着整合循环取得进展，所以本回合失败预算清零
        # （上限计连续失败，不计回合内累计，issue #42405）。
        self._consolidation_failures = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # 成功响应刻意是**终态**：确认写入已落地并让模型停止。不在这里
        # 回显完整条目列表——倾倒会引诱模型"找更多要修的"并重复发出同样
        # 的操作（观察到的抖动：第 1 次正确的批次，随后 5 次冗余重复）。
        # 条目只在错误/超预算路径展示，那时模型确实需要它们来决定整合。
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """渲染带标题与用量指示的系统提示块。"""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = (
                f"{MEMORY_BLOCK_HEADERS['user']} [{pct}% — {current:,}/{limit:,} chars]"
            )
        else:
            header = f"{MEMORY_BLOCK_HEADERS['memory']} [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_raw_checked(path: Path) -> Tuple[str, bool]:
        """读取记忆文件原文，区分「不可读」与「空」。

        返回 ``(raw, read_ok)``。仅当文件**存在但读不出来**时 read_ok 为
        False——文件不存在是干净的 ``("", True)``。无效 UTF-8 也算不可读：
        磁盘字节包含无法忠实往返的内容，重写会像读取失败一样破坏/丢弃它。
        读-改-写调用方必须把 read_ok=False 当作"中止"而非"空存储"，否则
        瞬时读取失败会让他们覆盖并抹掉磁盘记忆（issue #26045 同类：绝不
        从非真实视图重写文件）。

        无需文件锁：_write_file 用原子重命名，读者永远看到旧完整文件或
        新完整文件。
        """
        if not path.exists():
            return "", True
        try:
            # utf-8-sig 剥掉开头 UTF-8 BOM（Windows 记事本编辑的记忆文件），
            # 其他情况与 utf-8 字节一致。纯 utf-8 会把 U+FEFF 粘在第一条
            # 条目上，永久破坏该条目的匹配/去重（issue #10878）。
            # 解码错误刻意保持严格：errors="replace" 会给读-改-写调用方一
            # 个有损视图，后续保存会把它覆盖到真实字节上——即上述抹掉类。
            return path.read_text(encoding="utf-8-sig"), True
        except (OSError, IOError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _parse_entries(raw: str) -> List[str]:
        """把记忆文件原文拆成去除空白、非空的条目列表。"""
        if not raw.strip():
            return []
        # 与 _write_file 一致用 ENTRY_DELIMITER。只按 "§" 拆会错误拆分
        # 内容本身包含 "§" 的条目。
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _read_entries_checked(path: Path) -> Tuple[List[str], bool]:
        """读取 + 解析记忆文件，区分不可读与空。

        返回 ``(entries, read_ok)``——read_ok 契约见 _read_raw_checked。
        """
        raw, read_ok = MemoryStore._read_raw_checked(path)
        if not read_ok:
            return [], False
        return MemoryStore._parse_entries(raw), True

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """读取记忆文件并拆成条目（任何错误都返回空列表）。

        供只读调用方（load_from_disk）使用——它们构建内存状态而不持久化，
        读取失败降级为 [] 无害，因为不会写回。读-改-写路径用
        _read_raw_checked，以便拒绝覆盖不可读文件（见 _reload_target）。
        """
        return MemoryStore._read_entries_checked(path)[0]

    def _detect_external_drift(self, target: str, raw: str) -> Optional[str]:
        """磁盘内容显示外部漂移时返回备份路径字符串。

        *raw* 是调用方 checked 读取（_read_raw_checked）已读到的文件内容。
        漂移检测**必须**基于同一份快照——早期版本在这里重读文件并把第
        二次读取失败当"无漂移"，让变更从陈旧的第一份快照继续，重写掉外
        部写入者在两次读取之间加的内容。

        记忆文件本应是工具写入的小条目列表，用 § 连接。通过两个信号检测
        漂移：
        1. 往返不匹配——重新解析再序列化不产生相同字节（罕见；可捕获
           编码异常的定界符）。
        2. 条目大小溢出——任何单条解析条目超过整个文件的字符上限。工具
           按该上限给**整个**存储做预算；没有任何工具写入的单条能超过它。
           看到单条超限，说明外部写入者（patch 工具、shell 追加、手动
           编辑、姊妹会话）把自由文本追加进了会被工具当成一条的条目。
           冲刷会截断该条到模型的新内容，丢弃追加的字节（issue #26045）。

        发现漂移并备份时返回 .bak 文件绝对路径；文件形态正常返回 None。

        注意：这是实例方法（非 static），因为信号 2 需要按目标的
        char_limit。
        """
        path = self._path_for(target)
        if not raw.strip():
            return None

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        if not drift_detected:
            return None

        # 确认漂移——快照文件供操作者恢复外部写入者加的内容，然后返回
        # .bak 路径让调用方拒绝变更
        ts = int(time.time())
        bak_path = path.with_suffix(path.suffix + f".bak.{ts}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except (OSError, IOError):
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """用临时文件 + 原子重命名把条目写入记忆文件。

        旧实现用 open("w") + flock，但 "w" 在拿到锁**之前**就截断文件，
        制造并发读者看到空文件的竞态窗口。原子重命名避免它：读者永远
        看到旧完整文件或新完整文件。
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            atomic_write_text(path, content, tmp_prefix=".mem_")
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def load_on_disk_store() -> "MemoryStore":
    """构建全新的磁盘 MemoryStore，遵守配置的字符上限。

    供任何没有活动 agent 的上下文使用（消息网关、桌面 GUI、裸 CLI
    /memory 处理器）——它们仍需要读取或应用已批准的记忆写入。与活动
    agent 构建 store 的方式一致（含 memory.memory_char_limit /
    memory.user_char_limit 覆盖），保证批准的写入与活动 agent 施加
    **相同**的上限。

    配置加载失败时回退内置默认值，所以缺失/不可读配置也绝不会抛异常。
    """
    memory_char_limit = 2200
    user_char_limit = 1375
    try:
        from hermes_cli.config import load_config

        mem_cfg = (load_config() or {}).get("memory", {}) or {}
        memory_char_limit = int(mem_cfg.get("memory_char_limit", memory_char_limit))
        user_char_limit = int(mem_cfg.get("user_char_limit", user_char_limit))
    except Exception:
        pass  # config optional — fall back to defaults rather than break /memory

    store = MemoryStore(
        memory_char_limit=memory_char_limit,
        user_char_limit=user_char_limit,
    )
    store.load_from_disk()
    return store


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """构造 replace/remove 调用缺少 old_text 时的可恢复错误。

    replace/remove 天生有目标——没有 old_text 就没有可操作的条目，无法
    完成调用。但返回裸 "old_text is required" 是死胡同：一些结构化输出
    客户端会省略可选 old_text 字段。所以改为返回当前条目清单 + 明确重试
    指引，让模型用条目唯一子串重发调用（issues #43412, #49466）。
    """
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- a short unique substring of the entry "
                f"to {action}. None was provided. Reissue the {action} with old_text "
                f"set to part of one of the current_entries below."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )


def memory_tool(
    action: str = None,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    store: Optional[MemoryStore] = None,
) -> str:
    """
    memory 工具的唯一入口。派发到 MemoryStore 方法。

    两种形态：
      - 单操作：action + (content / old_text)。
      - 批量：operations=[{action, content?, old_text?}, ...]，在一次调用
        里按最终字符预算原子应用。

    返回 JSON 字符串结果。
    """
    if store is None:
        return tool_error(
            "Memory is not available. It may be disabled in config or this environment.",
            success=False,
        )

    # 一些严格 provider 会把可选 schema 字段填成 JSON null 而非省略。
    # 把 target: null 当省略处理，让记忆写入仍用文档默认存储。
    if target is None:
        target = "memory"

    if target not in {"memory", "user"}:
        return tool_error(
            f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False
        )

    # --- 批量路径 ---
    if operations:
        if not isinstance(operations, list):
            return tool_error(
                "operations must be a list of {action, content?, old_text?} objects.",
                success=False,
            )
        result = store.apply_batch(target, operations)
        return json.dumps(result, ensure_ascii=False)

    # --- 单操作路径 ---
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
    if action == "replace" and (not old_text or not content):
        missing = "old_text" if not old_text else "content"
        if not old_text:
            # 客户端/模型省略了 old_text。replace 天生有目标——无法猜测
            # 是哪条。返回当前清单 + 重试指引，让模型带 old_text 重发，
            # 而不是撞死胡同错误（issues #43412, #49466）。
            return _missing_old_text_error(store, target, "replace")
        return tool_error(f"{missing} is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    if action == "add":
        result = store.add(target, content)

    elif action == "replace":
        result = store.replace(target, old_text, content)

    elif action == "remove":
        result = store.remove(target, old_text)

    else:
        return tool_error(
            f"Unknown action '{action}'. Use: add, replace, remove", success=False
        )

    return json.dumps(result, ensure_ascii=False)


def check_memory_requirements() -> bool:
    """memory 工具无外部依赖——始终可用。"""
    return True


# OpenAI Function-Calling Schema
# =============================================================================

MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Save durable facts to persistent memory that survive across sessions. Memory is "
        "injected into every future turn, so keep entries compact and high-signal.\n\n"
        "HOW: make ALL your changes in ONE call via an 'operations' array (each item: "
        "{action, content?, old_text?}). The batch applies atomically and the char limit is "
        "checked only on the FINAL result — so a single call can remove/replace stale entries "
        "to free room AND add new ones, even when an add alone would overflow. The response "
        "reports current/limit chars and confirms completion; one batch call finishes the "
        "update, so don't repeat it. Use the bare action/content/old_text fields only for a "
        "single lone change.\n\n"
        "WHEN: save proactively when the user states a preference, correction, or personal "
        "detail, or you learn a stable fact about their environment, conventions, or workflow. "
        "Priority: user preferences & corrections > environment facts > procedures. The best "
        "memory stops the user repeating themselves.\n\n"
        "IF FULL: an add is rejected with the current entries shown. Reissue as ONE batch that "
        "removes or shortens enough stale entries and adds the new one together.\n\n"
        "TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your "
        "notes (environment, conventions, tool quirks, lessons).\n\n"
        "SKIP: trivial/obvious info, easily re-discovered facts, raw data dumps, task progress, "
        "completed-work logs, temporary TODO state (use session_search for those). Reusable "
        "procedures belong in a skill, not memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove"],
                "description": "The action to perform (single-op shape). Omit when using 'operations'.",
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile.",
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape).",
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'.",
            },
            "operations": {
                "type": "array",
                "description": (
                    "Batch shape: a list of operations applied atomically in one call "
                    "against the final char budget. Preferred when making multiple changes "
                    "or consolidating to make room. Each item is {action, content?, old_text?}."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "replace", "remove"],
                        },
                        "content": {
                            "type": "string",
                            "description": "Entry content for add/replace.",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Substring identifying the entry for replace/remove.",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}


# --- 模块级自注册（对应原版 memory_tool.py:1231 registry.register）---
from tools.registry import registry, tool_error

registry.register(
    name="memory",
    toolset="memory",
    schema=MEMORY_SCHEMA,
    handler=memory_tool,
    check_fn=check_memory_requirements,
    emoji="🧠",
)
