"""文件工具（精简移植版）。

对应原版 hermes-agent 的 tools/file_tools.py（2319 行）与
tools/file_operations.py（2805 行）的核心。精简版全部用 Python
标准库实现（不依赖 shell / ripgrep / fuzzy_match / lint / LSP /
file_state / redact / file_safety），砍掉了：
- 文件安全 / 审批：get_read_block_error、sensitive / cross-profile 路径
  检查、binary_extensions、设备路径守卫；
- 上下文压缩：dedup / read_tracker / negative cache / staleness 跟踪；
- 结构化文档提取（.docx/.xlsx）与 anydoc 转换；
- V4A patch 的 fuzzy 匹配与多文件支持（patch 仅支持 replace 模式）；
- lint / LSP 语法检查。

schema 照抄原版（READ_FILE_SCHEMA 等）。handler 签名与 schema 参数一致，
额外用 **kwargs 吸收模型可能多传的参数，保证 _execute_tool_calls 契约
（impl(**args)）不抛 TypeError。
"""

import difflib
import fnmatch
import json
import os
import re
from typing import Any, Dict, List, Optional

from tools.registry import tool_error


# ---------------------------------------------------------------------------
# 读取护栏（对应原版 file_tools.py:55-118；砍掉 config.yaml 动态配置）
# ---------------------------------------------------------------------------

# 单次读取返回给模型的字符上限：100K 字符 ≈ 25-35K token，超出是上下文
# 窗口风险，模型应改用 offset+limit 分段读。
_DEFAULT_MAX_READ_CHARS = 100_000
# 单行长度上限：超长行截断（对应原版 tool_output_limits.get_max_line_length）
_MAX_LINE_LENGTH = 2000
# 读文件分页上限
_MAX_READ_LIMIT = 2000
# 二进制判定：文件头多少字节内出现 NUL 即视为二进制
_BINARY_PROBE_BYTES = 8000


def _expand_tilde(path: str) -> str:
    """展开 ``~``（对应原版 file_tools.py:29；砍掉 profile home 逻辑）。"""
    if not path or "~" not in path:
        return path
    return os.path.expanduser(path)


def _resolve_path(path: str) -> str:
    """把工具入参解析为绝对路径（相对路径以进程 cwd 为基准）。

    对应原版 file_tools.py:151 _resolve_path；砍掉了 task_id 维度
    （terminal cwd / 容器路径 / workspace root 覆盖）。
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path is required")
    expanded = _expand_tilde(path.strip())
    return os.path.abspath(expanded)


def _normalize_read_pagination(offset: Any, limit: Any) -> tuple[int, int]:
    """规范化 read_file 分页参数（对应原版 file_operations.py:737）。"""
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        offset = 1
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 2000
    offset = max(1, offset)
    limit = max(1, min(limit, _MAX_READ_LIMIT))
    return offset, limit


def _is_likely_binary(path: str) -> bool:
    """探测文件是否二进制（前 8000 字节含 NUL）。

    对应原版 file_operations.py:885 _is_likely_binary 的简化版，
    只查 NUL 字节，不做扩展名/比例启发式。
    """
    try:
        with open(path, "rb") as fh:
            sample = fh.read(_BINARY_PROBE_BYTES)
    except OSError:
        return False
    return b"\x00" in sample


def _truncate_content_to_budget(
    content: str,
    max_chars: int,
) -> tuple[str, int, bool]:
    """把带行号内容截断到字符预算内（保留完整行，返回截断后行数）。

    对应原版 file_tools.py:87 _truncate_to_char_budget；返回
    (截断后内容, 保留行数, 是否发生截断)。
    """
    if len(content) <= max_chars:
        return content, 0, False
    lines = content.split("\n")
    kept: List[str] = []
    total = 0
    for line in lines:
        if total + len(line) + (1 if kept else 0) > max_chars:
            break
        kept.append(line)
        total += len(line) + (1 if kept else 0)
    return "\n".join(kept), len(kept), True


def _add_line_numbers(content: str, start_line: int = 1) -> str:
    """给内容加行号，格式 ``LINE_NUM|CONTENT``（对应原版 :919）。

    长行截断到 _MAX_LINE_LENGTH，避免单行撑爆上下文。
    """
    numbered = []
    for i, line in enumerate(content.split("\n"), start=start_line):
        if len(line) > _MAX_LINE_LENGTH:
            line = line[:_MAX_LINE_LENGTH] + "... [truncated]"
        numbered.append(f"{i}|{line}")
    return "\n".join(numbered)


def read_file(path: str, offset: int = 1, limit: int = 2000, **kwargs) -> str:
    """读取文本文件，带行号与分页（对应原版 file_tools.py:1264）。

    schema 参数：path / offset / limit；**kwargs 吸收多余参数。
    """
    del kwargs
    try:
        resolved = _resolve_path(path)
        offset, limit = _normalize_read_pagination(offset, limit)

        # 目录 / 不存在
        if not os.path.exists(resolved):
            return tool_error(f"File not found: {path}")
        if os.path.isdir(resolved):
            return tool_error(f"Cannot read '{path}': it is a directory")

        # 二进制守卫
        if _is_likely_binary(resolved):
            return tool_error(
                f"Cannot read binary file '{path}'. Use terminal to inspect binary files."
            )

        # 读全文件（utf-8，替换非法字节避免崩溃）
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
                all_lines = fh.read().splitlines()
        except OSError as exc:
            return tool_error(f"Failed to read file: {exc}")

        total_lines = len(all_lines)
        end_line = min(offset + limit - 1, total_lines)
        page_lines = all_lines[offset - 1:end_line]
        page_text = "\n".join(page_lines)
        content = _add_line_numbers(page_text, offset)
        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = (
                f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{end_line} of {total_lines} lines)"
            )

        # 字符预算截断：超 100K 字符时截到完整行并给出 next_offset
        if len(content) > _DEFAULT_MAX_READ_CHARS:
            trimmed, lines_kept, _ = _truncate_content_to_budget(
                content, _DEFAULT_MAX_READ_CHARS
            )
            next_offset = offset + lines_kept
            shown_end = offset + lines_kept - 1
            content = trimmed
            truncated = True
            hint = (
                f"Output truncated at the {_DEFAULT_MAX_READ_CHARS:,}-char read "
                f"budget after {lines_kept} line(s) (showing lines {offset}-"
                f"{shown_end} of {total_lines}). Use offset={next_offset} to continue."
            )

        result = {
            "content": content,
            "total_lines": total_lines,
            "file_size": os.path.getsize(resolved),
            "truncated": truncated,
        }
        if hint:
            result["hint"] = hint
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


def write_file(path: str, content: str, **kwargs) -> str:
    """写文件，整体覆盖，自动创建父目录（对应原版 file_tools.py:1757）。

    schema 参数：path / content / cross_profile（cross_profile 是原版
    profile 隔离概念，精简版忽略）；**kwargs 吸收多余参数。
    """
    del kwargs
    if not isinstance(path, str) or not path.strip():
        return tool_error("write_file: missing required field 'path'")
    if not isinstance(content, str):
        return tool_error(
            f"write_file: 'content' must be a string, got {type(content).__name__}"
        )
    try:
        resolved = _resolve_path(path)
        parent = os.path.dirname(resolved)
        dirs_created = False
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
            dirs_created = True
        with open(resolved, "w", encoding="utf-8", newline="") as fh:
            bytes_written = fh.write(content)
        return json.dumps({
            "bytes_written": bytes_written,
            "dirs_created": dirs_created,
        }, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


def _unified_diff(old_content: str, new_content: str, filename: str) -> str:
    """生成 unified diff（对应原版 file_operations.py:1128 的简化）。"""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=filename, tofile=filename, lineterm="",
    )
    return "\n".join(diff)


def patch(
    mode: str = "replace",
    path: Optional[str] = None,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    replace_all: bool = False,
    patch: Optional[str] = None,
    **kwargs,
) -> str:
    """定点替换编辑文件（对应原版 file_tools.py:1840 patch_tool）。

    精简版只支持 mode='replace'（精确匹配，不做 fuzzy）；V4A patch 模式
    返回明确错误提示。schema 参数：mode / path / old_string / new_string /
    replace_all / patch / cross_profile。
    """
    del kwargs
    if mode != "replace":
        return tool_error(
            "patch: V4A patch mode is not supported in this lite version; "
            "use mode='replace' with path + old_string + new_string."
        )
    if not path:
        return tool_error("patch: path required")
    if old_string is None or new_string is None:
        return tool_error("patch: old_string and new_string required")
    try:
        resolved = _resolve_path(path)
        if not os.path.exists(resolved):
            return tool_error(f"File not found: {path}")
        if os.path.isdir(resolved):
            return tool_error(f"Cannot patch '{path}': it is a directory")
        with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
            old_content = fh.read()

        count = old_content.count(old_string)
        if count == 0:
            return tool_error(
                f"Could not find match for old_string in {path}. "
                "Use read_file to verify the current content, or search_files "
                "to locate the text."
            )
        if count > 1 and not replace_all:
            return tool_error(
                f"old_string is not unique in {path} ({count} matches). "
                "Use replace_all=true or include more surrounding context."
            )
        new_content = (
            old_content.replace(old_string, new_string)
            if replace_all
            else old_content.replace(old_string, new_string, 1)
        )

        # 已是目标文本（old==new）→ 无需写入，返回 no_change
        if old_content == new_content:
            return json.dumps({
                "success": True,
                "no_change": True,
                "note": (
                    f"File already contains the target text — the edit appears "
                    f"to be already applied to {path}. No write performed."
                ),
            }, ensure_ascii=False)

        with open(resolved, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_content)

        diff = _unified_diff(old_content, new_content, path)
        return json.dumps({
            "success": True,
            "diff": diff,
            "files_modified": [resolved],
        }, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


# 搜索时跳过这些目录（对应原版 ripgrep 的 .gitignore 行为；精简版硬编码）
_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", ".tox",
}


def _normalize_search_pagination(offset: Any, limit: Any) -> tuple[int, int]:
    """规范化 search 分页参数（对应原版 file_operations.py:756）。"""
    try:
        offset = int(offset)
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    offset = max(0, offset)
    limit = max(1, limit)
    return offset, limit


def _iter_search_files(root: str, file_glob: Optional[str]):
    """遍历 root 下应搜索的文件（跳过二进制与 _SKIP_DIRS）。"""
    if os.path.isfile(root):
        if file_glob and not fnmatch.fnmatch(os.path.basename(root), file_glob):
            return
        yield root
        return
    if not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # 原地过滤跳过的目录，避免无谓深入
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for filename in filenames:
            if file_glob and not fnmatch.fnmatch(filename, file_glob):
                continue
            full = os.path.join(dirpath, filename)
            if not _is_likely_binary(full):
                yield full


def search_files(
    pattern: str,
    target: str = "content",
    path: str = ".",
    file_glob: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    output_mode: str = "content",
    context: int = 0,
    **kwargs,
) -> str:
    """搜索文件内容或按文件名找文件（对应原版 file_tools.py:2042 search_tool）。

    精简版用标准库 os.walk + re/fnmatch 实现（原版用 ripgrep/shell），
    结果结构与原版一致。schema 参数：pattern / target / path / file_glob /
    limit / offset / output_mode / context。
    """
    del kwargs
    if not isinstance(pattern, str) or not pattern:
        return tool_error("search_files: missing required field 'pattern'")
    try:
        offset, limit = _normalize_search_pagination(offset, limit)
        resolved = _resolve_path(path)
        if not os.path.exists(resolved):
            return tool_error(f"Path not found: {path}")

        # 按文件名找文件：glob 匹配 + 按 mtime 降序排序
        if target == "files":
            matches: List[str] = []
            for full in _iter_search_files(resolved, pattern or file_glob):
                matches.append(full)
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            total = len(matches)
            page = matches[offset:offset + limit]
            return json.dumps({
                "files": page,
                "total_count": total,
                "truncated": total > offset + limit,
            }, ensure_ascii=False)

        # 内容搜索：逐文件按行正则匹配
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return tool_error(f"Invalid regex pattern: {exc}")

        all_matches: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        file_list: List[str] = []
        seen_files: set = set()
        try:
            context = max(0, int(context))
        except (TypeError, ValueError):
            context = 0

        for full in _iter_search_files(resolved, file_glob):
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                continue
            file_count = 0
            for idx, line in enumerate(lines, start=1):
                if regex.search(line):
                    file_count += 1
                    if output_mode == "content":
                        if context > 0:
                            lo = max(0, idx - 1 - context)
                            hi = min(len(lines), idx + context)
                            ctx = "\n".join(
                                f"{j}|{lines[j - 1]}" for j in range(lo + 1, hi + 1)
                            )
                            all_matches.append({
                                "path": full, "line": idx, "content": line,
                                "context": ctx,
                            })
                        else:
                            all_matches.append({
                                "path": full, "line": idx, "content": line,
                            })
                    elif full not in seen_files:
                        seen_files.add(full)
                        file_list.append(full)
            if file_count:
                counts[full] = file_count

        if output_mode == "content":
            total = len(all_matches)
        elif output_mode == "count":
            total = sum(counts.values())
        else:
            total = len(file_list)
        page = all_matches[offset:offset + limit]

        result: Dict[str, Any] = {"total_count": total}
        if output_mode == "content":
            result["matches"] = page
        elif output_mode == "files_only":
            result["files"] = file_list[offset:offset + limit]
        elif output_mode == "count":
            result["counts"] = dict(
                list(counts.items())[offset:offset + limit]
            )
        else:
            result["matches"] = page
        if total > offset + limit:
            result["truncated"] = True
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


def _check_file_reqs() -> bool:
    """文件工具无外部依赖，恒可用（对应原版 file_tools.py:2154 的简化）。"""
    return True


# ---------------------------------------------------------------------------
# Schemas（照抄原版 file_tools.py:2167-2245；描述含 cross_profile 等
# 原版概念，精简版 handler 忽略对应参数）
# ---------------------------------------------------------------------------

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": (
        "Read a text file with line numbers and pagination. Use this instead "
        "of cat/head/tail in terminal. Output format: 'LINE_NUM|CONTENT'. "
        "Use offset and limit for large files. Reads exceeding ~100K "
        "characters are truncated on a line boundary and return a next_offset; "
        "continue with offset to read the rest. "
        "NOTE: Cannot read images or other binary files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (absolute, relative, or ~/path)"
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (1-indexed, default: 1)",
                "default": 1, "minimum": 1
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of lines to read (default: 2000, max: 2000). "
                    "Reads are additionally capped at a ~100K-character budget "
                    "with a next_offset continuation."
                ),
                "default": 2000, "maximum": 2000
            }
        },
        "required": ["path"]
    }
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": (
        "Write content to a file, completely replacing existing content. Use "
        "this instead of echo/cat heredoc in terminal. Creates parent "
        "directories automatically. OVERWRITES the entire file — use 'patch' "
        "for targeted edits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Path to the file to write (will be created if it doesn't "
                    "exist, overwritten if it does)"
                )
            },
            "content": {
                "type": "string",
                "description": "Complete content to write to the file"
            },
            "cross_profile": {
                "type": "boolean",
                "description": (
                    "Opt out of the cross-profile soft guard. Defaults to false. "
                    "Set true ONLY after explicit user direction to edit another "
                    "Hermes profile's skills/plugins/cron/memories."
                ),
                "default": False,
            },
        },
        "required": ["path", "content"]
    }
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": (
        "Targeted find-and-replace edits in files. Use this instead of sed/awk "
        "in terminal. Returns a unified diff.\n\n"
        "REPLACE MODE (mode='replace', default): find a unique string and "
        "replace it. REQUIRED PARAMETERS: mode, path, old_string, new_string.\n"
        "PATCH MODE (mode='patch'): apply V4A multi-file patches for bulk "
        "changes. REQUIRED PARAMETERS: mode, patch."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["replace", "patch"],
                "description": (
                    "Edit mode. 'replace' (default): requires path + old_string "
                    "+ new_string. 'patch': requires patch content only."
                ),
                "default": "replace",
            },
            "path": {
                "type": "string",
                "description": "REQUIRED when mode='replace'. File path to edit.",
            },
            "old_string": {
                "type": "string",
                "description": (
                    "REQUIRED when mode='replace'. Exact text to find and "
                    "replace. Must be unique in the file unless replace_all=true. "
                    "Include surrounding context lines to ensure uniqueness."
                ),
            },
            "new_string": {
                "type": "string",
                "description": (
                    "REQUIRED when mode='replace'. Replacement text. Pass empty "
                    "string '' to delete the matched text."
                ),
            },
            "replace_all": {
                "type": "boolean",
                "description": (
                    "Replace all occurrences instead of requiring a unique match "
                    "(default: false)"
                ),
                "default": False,
            },
            "patch": {
                "type": "string",
                "description": "REQUIRED when mode='patch'. V4A format patch content.",
            },
            "cross_profile": {
                "type": "boolean",
                "description": (
                    "Opt out of the cross-profile soft guard. Defaults to false. "
                    "Set true ONLY after explicit user direction to edit another "
                    "Hermes profile's skills/plugins/cron/memories."
                ),
                "default": False,
            },
        },
        "required": ["mode"],
    },
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": (
        "Search file contents or find files by name. Use this instead of "
        "grep/rg/find/ls in terminal.\n\n"
        "Content search (target='content'): Regex search inside files. Output "
        "modes: full matches with line numbers, file paths only, or match "
        "counts.\n\n"
        "File search (target='files'): Find files by glob pattern (e.g., "
        "'*.py', '*config*'). Also use this instead of ls — results sorted "
        "by modification time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "Regex pattern for content search, or glob pattern "
                    "(e.g., '*.py') for file search"
                )
            },
            "target": {
                "type": "string",
                "enum": ["content", "files"],
                "description": (
                    "'content' searches inside file contents, 'files' searches "
                    "for files by name"
                ),
                "default": "content"
            },
            "path": {
                "type": "string",
                "description": (
                    "Directory or file to search in (default: current working "
                    "directory)"
                ),
                "default": "."
            },
            "file_glob": {
                "type": "string",
                "description": (
                    "Filter files by pattern in grep mode (e.g., '*.py' to only "
                    "search Python files)"
                )
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 50)",
                "default": 50
            },
            "offset": {
                "type": "integer",
                "description": "Skip first N results for pagination (default: 0)",
                "default": 0
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_only", "count"],
                "description": (
                    "Output format for grep mode: 'content' shows matching "
                    "lines with line numbers, 'files_only' lists file paths, "
                    "'count' shows match counts per file"
                ),
                "default": "content"
            },
            "context": {
                "type": "integer",
                "description": (
                    "Number of context lines before and after each match "
                    "(grep mode only)"
                ),
                "default": 0
            }
        },
        "required": ["pattern"]
    }
}


# --- 模块级自注册（对应原版 file_tools.py:2316-2319）---
from tools.registry import registry

registry.register(
    name="read_file", toolset="file", schema=READ_FILE_SCHEMA,
    handler=read_file, check_fn=_check_file_reqs, emoji="📖",
    max_result_size_chars=100_000,
)
registry.register(
    name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA,
    handler=write_file, check_fn=_check_file_reqs, emoji="✍️",
    max_result_size_chars=100_000,
)
registry.register(
    name="patch", toolset="file", schema=PATCH_SCHEMA,
    handler=patch, check_fn=_check_file_reqs, emoji="🔧",
    max_result_size_chars=100_000,
)
registry.register(
    name="search_files", toolset="file", schema=SEARCH_FILES_SCHEMA,
    handler=search_files, check_fn=_check_file_reqs, emoji="🔎",
    max_result_size_chars=100_000,
)
