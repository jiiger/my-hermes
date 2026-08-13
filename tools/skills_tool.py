"""Skill 工具（核心完整移植自原版 hermes-agent）。

SKILL.md 存储规范（agentskills.io 兼容）、frontmatter 全字段解析、三工具
（skills_list / skill_view / skill_manage 六动作）、技能索引进 system
prompt、路径安全全部对齐原版。

裁剪（my-hermes 精简定位）：云端 hub/同步、使用统计（skill_usage）、
/skill CLI 命令、溯源/静态审计/捆绑包、插件命名空间（ns:skill）、
external_dirs、disabled 配置、LRU+磁盘快照缓存（skills 少，直接扫描）、
skill_preprocessing（模板渲染，二期可补）。

目录布局（两种均支持）：
  skills/<name>/SKILL.md                 category=None
  skills/<category>/<name>/SKILL.md      category=一级目录名
  子目录：references/ templates/ assets/
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# 平台映射：sys.platform → frontmatter platforms 合法值
_PLATFORM_MAP = {"linux": "linux", "darwin": "macos", "win32": "windows"}
_CURRENT_PLATFORM = _PLATFORM_MAP.get(sys.platform, "linux")

# YAML frontmatter 结尾围栏（对齐原版 skill_utils.parse_frontmatter）
_FRONTMATTER_RE = re.compile(r"\n---\s*\n")

_NAME_MAX = 64
_DESCRIPTION_MAX = 1024


# ── 基础工具 ─────────────────────────────────────────────────────────

def _skills_dir() -> Path:
    """返回 $HERMES_HOME/skills（调用时解析 HERMES_HOME，幂等建目录）。"""
    d = get_hermes_home() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """解析 markdown frontmatter，返回 (dict, body)。

    对齐原版 skill_utils.parse_frontmatter：
    - 剥 UTF-8 BOM（Windows 编辑器保存时前置，不剥会导致围栏判定失败）；
    - YAML 解析失败回退简单 key:value 解析；
    - 非法/缺失返回空 dict，不抛异常。
    """
    if not content:
        return {}, content
    if content.startswith("\ufeff"):  # UTF-8 BOM
        content = content[1:]
    body = content
    if not content.startswith("---"):
        return {}, body
    m = _FRONTMATTER_RE.search(content[3:])
    if not m:
        return {}, body
    yaml_content = content[3:m.start() + 3]
    body = content[m.end() + 3:]
    frontmatter: Dict[str, Any] = {}
    try:
        parsed = yaml.safe_load(yaml_content)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except Exception:
        # Fallback：畸形 YAML 时简单 key:value 解析
        for line in yaml_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter, body


def _normalize_list(value: Any) -> List[str]:
    """把 frontmatter 的列表/逗号串归一为字符串列表。"""
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _platform_matches(frontmatter: Dict[str, Any]) -> bool:
    """platforms 匹配：缺省全平台；不匹配则 skill 不可见。"""
    platforms = _normalize_list(frontmatter.get("platforms"))
    if not platforms:
        return True
    return _CURRENT_PLATFORM in platforms


def _tags_of(frontmatter: Dict[str, Any]) -> List[str]:
    """metadata.hermes.tags 提取。"""
    metadata = frontmatter.get("metadata") or {}
    if isinstance(metadata, dict):
        hermes = metadata.get("hermes") or {}
        if isinstance(hermes, dict):
            return _normalize_list(hermes.get("tags"))
    return []


def _related_skills_of(frontmatter: Dict[str, Any]) -> List[str]:
    """metadata.hermes.related_skills 提取。"""
    metadata = frontmatter.get("metadata") or {}
    if isinstance(metadata, dict):
        hermes = metadata.get("hermes") or {}
        if isinstance(hermes, dict):
            return _normalize_list(hermes.get("related_skills"))
    return []


def _required_env_vars(frontmatter: Dict[str, Any]) -> List[str]:
    """合并 required_environment_variables（str 或 dict.name，对齐原版）
    与旧式 prerequisites.env_vars，去重。"""
    result: List[str] = []
    seen = set()
    req_raw = frontmatter.get("required_environment_variables")
    if isinstance(req_raw, dict):
        req_raw = [req_raw]
    if isinstance(req_raw, list):
        for item in req_raw:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("env_var") or "").strip()
            else:
                continue
            if name and name not in seen:
                seen.add(name)
                result.append(name)
    prereq = frontmatter.get("prerequisites") or {}
    if isinstance(prereq, dict):
        for name in _normalize_list(prereq.get("env_vars")):
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


def _missing_env_vars(frontmatter: Dict[str, Any]) -> List[str]:
    """当前未设置的环境变量（advisory）。"""
    return [e for e in _required_env_vars(frontmatter) if not os.environ.get(e)]


def _missing_commands(frontmatter: Dict[str, Any]) -> List[str]:
    """prerequisites.commands 中 PATH 找不到的命令（advisory）。"""
    import shutil

    prereq = frontmatter.get("prerequisites") or {}
    commands = []
    if isinstance(prereq, dict):
        commands = _normalize_list(prereq.get("commands"))
    return [c for c in commands if shutil.which(c) is None]


def _missing_credential_files(frontmatter: Dict[str, Any]) -> List[str]:
    """required_credential_files 中不存在的文件（advisory）。"""
    files = _normalize_list(frontmatter.get("required_credential_files"))
    return [
        f for f in files
        if not Path(os.path.expanduser(f)).exists()
    ]


# ── 查找与安全 ───────────────────────────────────────────────────────

def _skill_lookup_path_error(name: Any) -> Optional[str]:
    """校验 skill 名/路径可安全用于查找。

    拒绝：非字符串/空、`.`/`..`、绝对路径、`..` 穿越、反斜杠与盘符注入。
    """
    if not isinstance(name, str) or not name.strip():
        return "skill name must be a non-empty string"
    name = name.strip()
    if name in (".", ".."):
        return f"invalid skill name: {name!r}"
    p = Path(name)
    if p.is_absolute():
        return "skill name must be relative to the skills directory"
    if ".." in p.parts:
        return "skill name must not contain '..' path traversal"
    if "\\" in name or re.match(r"^[a-zA-Z]:[\\/]", name):
        return "skill name must use '/' separators and no drive prefixes"
    return None


def _iter_skill_dirs() -> List[Tuple[str, Optional[str], Path]]:
    """扫描 skills 树，返回 [(name, category, skill_dir)]。"""
    root = _skills_dir()
    result: List[Tuple[str, Optional[str], Path]] = []
    if not root.exists():
        return result
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "SKILL.md").exists():
            result.append((child.name, None, child))
        else:
            for grand in sorted(child.iterdir()):
                if grand.is_dir() and (grand / "SKILL.md").exists():
                    result.append((grand.name, child.name, grand))
    return result


def _find_skill(name: str) -> Optional[Tuple[str, Optional[str], Path]]:
    """按 name 或 category/name 查找 skill 目录。"""
    for sname, cat, sdir in _iter_skill_dirs():
        if name == sname or name == f"{cat}/{sname}":
            return sname, cat, sdir
    return None


def _read_skill_metadata(sdir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """读取并解析 skill 的 SKILL.md，返回 (frontmatter, body)。"""
    md = sdir / "SKILL.md"
    if not md.exists():
        return None, ""
    try:
        content = md.read_text(encoding="utf-8")
    except Exception:
        return None, ""
    return _parse_frontmatter(content)


def _resolve_skill_subfile(sdir: Path, file_path: str) -> Optional[Path]:
    """把子文件路径限定在 skill 目录内；非法返回 None。"""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    fp = file_path.strip()
    if "\\" in fp:
        return None
    try:
        p = (sdir / fp).resolve()
        p.relative_to(sdir.resolve())
    except (ValueError, OSError):
        return None
    return p


def _canonical_skill_dir(name: str, category: Optional[str]) -> Path:
    """create 用的目标目录（name 不允许含 '/'，category 可选）。"""
    if "/" in name:
        raise ValueError("skill name must not contain '/' (use category param)")
    if len(name) > _NAME_MAX:
        raise ValueError(f"skill name too long (max {_NAME_MAX})")
    base = _skills_dir()
    if category and category.strip():
        cat = category.strip().replace("\\", "/").strip("/")
        if "/" in cat or ".." in cat.split("/"):
            raise ValueError("invalid category")
        return base / cat / name
    return base / name


# ── 工具：skills_list（tier 1 元数据）───────────────────────────────

def skills_list(category: str = None) -> str:
    """列出所有可用 skill（渐进式披露 tier 1：name/description/category/tags）。"""
    category = (category or "").strip() or None
    skills: List[Dict[str, Any]] = []
    categories = set()
    for sname, cat, sdir in _iter_skill_dirs():
        if category and cat != category:
            continue
        fm, _ = _read_skill_metadata(sdir)
        if not fm or not fm.get("name") or not fm.get("description"):
            continue
        if not _platform_matches(fm):
            continue
        skills.append({
            "name": str(fm.get("name"))[:_NAME_MAX],
            "description": str(fm.get("description"))[:_DESCRIPTION_MAX],
            "category": cat,
            "tags": _tags_of(fm),
            "version": fm.get("version"),
        })
        if cat:
            categories.add(cat)
    return json.dumps({
        "success": True,
        "skills": skills,
        "categories": sorted(categories),
        "count": len(skills),
    }, ensure_ascii=False)


# ── 工具：skill_view（tier 2-3 全文）────────────────────────────────

def skill_view(name: str, file_path: str = None) -> str:
    """查看 skill 全文；file_path 可读 references/templates 等子文件。"""
    err = _skill_lookup_path_error(name)
    if err:
        return tool_error(err)
    name = name.strip()
    found = _find_skill(name)
    if not found:
        return tool_error(f"skill not found: {name}")
    sname, cat, sdir = found
    fm, body = _read_skill_metadata(sdir)
    if not fm or not fm.get("name"):
        return tool_error(f"skill '{sname}' has invalid or missing frontmatter")

    if file_path:
        fp = _resolve_skill_subfile(sdir, file_path)
        if fp is None:
            return tool_error(f"invalid file_path: {file_path}")
        if not fp.exists() or not fp.is_file():
            return tool_error(f"file not found: {file_path}")
        try:
            content = fp.read_text(encoding="utf-8")
        except Exception as exc:
            return tool_error(f"failed to read {file_path}: {exc}")
    else:
        content = body

    result: Dict[str, Any] = {
        "success": True,
        "name": sname,
        "category": cat,
        "description": str(fm.get("description") or "")[:_DESCRIPTION_MAX],
        "version": fm.get("version"),
        "license": fm.get("license"),
        "compatibility": fm.get("compatibility"),
        "tags": _tags_of(fm),
        "related_skills": _related_skills_of(fm),
        "content": content,
    }
    for key in ("setup", "environments"):
        if fm.get(key):
            result[key] = fm[key]
    missing_env = _missing_env_vars(fm)
    missing_cmds = _missing_commands(fm)
    missing_files = _missing_credential_files(fm)
    notes = []
    if missing_env:
        notes.append("env vars not set: " + ", ".join(missing_env))
    if missing_cmds:
        notes.append("commands not found: " + ", ".join(missing_cmds))
    if missing_files:
        notes.append("credential files missing: " + ", ".join(missing_files))
    if notes:
        result["prerequisites_note"] = " (advisory); ".join(notes) + " (advisory)"
    return json.dumps(result, ensure_ascii=False)


# ── 工具：skill_manage（六动作）─────────────────────────────────────

def skill_manage(
    action: str,
    name: str,
    content: str = None,
    category: str = None,
    description: str = None,
    old_string: str = None,
    new_string: str = None,
    file_path: str = None,
    replace_all: bool = False,
) -> str:
    """管理用户创建的 skill。

    action: create / edit / patch / delete / write_file / remove_file
    """
    if not isinstance(action, str) or not action.strip():
        return tool_error("action is required")
    action = action.strip().lower()
    if action not in ("create", "edit", "patch", "delete",
                      "write_file", "remove_file"):
        return tool_error(
            f"unknown action '{action}'. "
            "Use: create, edit, patch, delete, write_file, remove_file"
        )
    err = _skill_lookup_path_error(name)
    if err:
        return tool_error(err)
    name = name.strip()

    if action == "create":
        return _manage_create(name, content, category, description)
    if action == "edit":
        return _manage_edit(name, content)
    if action == "patch":
        return _manage_patch(name, old_string, new_string, replace_all)
    if action == "delete":
        return _manage_delete(name)
    if action == "write_file":
        return _manage_write_file(name, file_path, content)
    return _manage_remove_file(name, file_path)


def _manage_create(name, content, category, description) -> str:
    if not isinstance(content, str) or not content.strip():
        return tool_error("content is required for 'create'")
    try:
        sdir = _canonical_skill_dir(name, category)
    except ValueError as exc:
        return tool_error(str(exc))
    if (sdir / "SKILL.md").exists():
        return tool_error(f"skill '{name}' already exists")
    fm, _ = _parse_frontmatter(content)
    if fm is None or not fm.get("name"):
        # 无 frontmatter → 自动补 name/description
        desc = (description or "").strip() or name
        content = (
            f"---\nname: {name}\ndescription: {desc}\n---\n\n{content}"
        )
    try:
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "SKILL.md").write_text(content, encoding="utf-8")
    except Exception as exc:
        return tool_error(f"create failed: {exc}")
    return json.dumps(
        {"success": True, "message": f"skill '{name}' created"},
        ensure_ascii=False,
    )


def _manage_edit(name, content) -> str:
    if not isinstance(content, str) or not content.strip():
        return tool_error("content is required for 'edit'")
    found = _find_skill(name)
    if not found:
        return tool_error(f"skill not found: {name}")
    _, _, sdir = found
    try:
        (sdir / "SKILL.md").write_text(content, encoding="utf-8")
    except Exception as exc:
        return tool_error(f"edit failed: {exc}")
    return json.dumps(
        {"success": True, "message": f"skill '{name}' updated"},
        ensure_ascii=False,
    )


def _manage_patch(name, old_string, new_string, replace_all) -> str:
    if not isinstance(old_string, str) or not old_string:
        return tool_error("old_string is required for 'patch'")
    if new_string is None:
        return tool_error("new_string is required for 'patch' (use '' to delete)")
    found = _find_skill(name)
    if not found:
        return tool_error(f"skill not found: {name}")
    _, _, sdir = found
    md = sdir / "SKILL.md"
    try:
        text = md.read_text(encoding="utf-8")
    except Exception as exc:
        return tool_error(f"read failed: {exc}")
    if old_string not in text:
        return tool_error(f"old_string not found in skill '{name}'")
    if replace_all:
        new_text = text.replace(old_string, new_string)
    else:
        new_text = text.replace(old_string, new_string, 1)
    try:
        md.write_text(new_text, encoding="utf-8")
    except Exception as exc:
        return tool_error(f"patch failed: {exc}")
    return json.dumps(
        {"success": True, "message": f"skill '{name}' patched"},
        ensure_ascii=False,
    )


def _manage_delete(name) -> str:
    found = _find_skill(name)
    if not found:
        return tool_error(f"skill not found: {name}")
    _, _, sdir = found
    try:
        import shutil
        shutil.rmtree(sdir)
    except Exception as exc:
        return tool_error(f"delete failed: {exc}")
    return json.dumps(
        {"success": True, "message": f"skill '{name}' deleted"},
        ensure_ascii=False,
    )


def _manage_write_file(name, file_path, content) -> str:
    if not isinstance(file_path, str) or not file_path.strip():
        return tool_error("file_path is required for 'write_file'")
    if not isinstance(content, str):
        return tool_error("content is required for 'write_file'")
    found = _find_skill(name)
    if not found:
        return tool_error(f"skill not found: {name}")
    _, _, sdir = found
    fp = _resolve_skill_subfile(sdir, file_path)
    if fp is None:
        return tool_error(f"invalid file_path: {file_path}")
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
    except Exception as exc:
        return tool_error(f"write_file failed: {exc}")
    return json.dumps(
        {"success": True, "message": f"wrote {file_path} in skill '{name}'"},
        ensure_ascii=False,
    )


def _manage_remove_file(name, file_path) -> str:
    if not isinstance(file_path, str) or not file_path.strip():
        return tool_error("file_path is required for 'remove_file'")
    found = _find_skill(name)
    if not found:
        return tool_error(f"skill not found: {name}")
    _, _, sdir = found
    fp = _resolve_skill_subfile(sdir, file_path)
    if fp is None or fp == (sdir / "SKILL.md").resolve():
        return tool_error(f"invalid file_path: {file_path}")
    if not fp.exists():
        return tool_error(f"file not found: {file_path}")
    try:
        fp.unlink()
    except Exception as exc:
        return tool_error(f"remove_file failed: {exc}")
    return json.dumps(
        {"success": True, "message": f"removed {file_path} from skill '{name}'"},
        ensure_ascii=False,
    )


# ── 技能索引（system prompt 注入）──────────────────────────────────

def build_skills_system_prompt() -> str:
    """构建技能索引（category 分组，name: description）。空目录返回 ""。"""
    entries: List[Tuple[Optional[str], str, str]] = []
    for sname, cat, sdir in _iter_skill_dirs():
        fm, _ = _read_skill_metadata(sdir)
        if not fm or not fm.get("name") or not fm.get("description"):
            continue
        if not _platform_matches(fm):
            continue
        entries.append((cat, str(fm.get("name"))[:_NAME_MAX],
                        str(fm.get("description"))[:_DESCRIPTION_MAX]))
    if not entries:
        return ""
    lines = ["### Skills"]
    for cat, n, d in entries:
        if not cat:
            lines.append(f"- {n}: {d}")
    for cat in sorted({c for c, _, _ in entries if c}):
        lines.append(f"{cat}:")
        for c, n, d in entries:
            if c == cat:
                lines.append(f"- {n}: {d}")
    return "\n".join(lines)


def check_skills_requirements() -> bool:
    """依赖本地 skills 目录；HERMES_HOME 可解析即可用。"""
    try:
        get_hermes_home()
        return True
    except Exception:
        return False


# ── OpenAI Function-Calling Schema ──────────────────────────────────

_SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": (
        "List all available skills (program memory) with name, description, "
        "category and tags. Token-efficient metadata only — load full "
        "instructions with skill_view.\n\n"
        "WHEN TO USE\n"
        "When the user asks what you can do, what workflows you know, or "
        "when you suspect a skill covers the current task. Skills capture "
        "'how to do this class of task' — they are not conversation history "
        "(use session_search) and not durable facts (use memory)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter (e.g. 'mlops').",
            },
        },
        "required": [],
    },
}

_SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": (
        "Load a skill's full instructions (progressive disclosure tier 2-3). "
        "Pass file_path (e.g. 'references/api.md') to read a support file "
        "inside the skill.\n\n"
        "WHEN TO USE\n"
        "After skills_list shows a relevant skill, or when a skill was "
        "loaded/consulted this session and you need its details again."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name (or 'category/name').",
            },
            "file_path": {
                "type": "string",
                "description": "Optional file within the skill (e.g. "
                               "'references/api.md').",
            },
        },
        "required": ["name"],
    },
}

_SKILL_MANAGE_SCHEMA = {
    "name": "skill_manage",
    "description": (
        "Manage user-created skills (program memory). Actions:\n"
        "- create: name + content (full SKILL.md, frontmatter auto-added if "
        "missing; optional category, description)\n"
        "- edit: name + content (replace whole SKILL.md)\n"
        "- patch: name + old_string + new_string (replace_all optional) — "
        "update a loaded/consulted skill when it is outdated/wrong\n"
        "- delete: name\n"
        "- write_file: name + file_path + content (e.g. "
        "'references/api.md')\n"
        "- remove_file: name + file_path\n\n"
        "WHEN TO USE\n"
        "After completing a complex task (5+ tool calls), fixing a tricky "
        "error, or discovering a non-trivial workflow — save the approach "
        "as a skill so you can reuse it next time. When using a skill and "
        "finding it outdated, incomplete, or wrong, patch it immediately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "edit", "patch", "delete",
                         "write_file", "remove_file"],
                "description": "Operation to perform.",
            },
            "name": {
                "type": "string",
                "description": "Skill name.",
            },
            "content": {
                "type": "string",
                "description": "Full SKILL.md content (create/edit/write_file).",
            },
            "category": {
                "type": "string",
                "description": "Optional category for create.",
            },
            "description": {
                "type": "string",
                "description": "Optional description when auto-adding "
                               "frontmatter on create.",
            },
            "old_string": {
                "type": "string",
                "description": "Text to find (patch).",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text (patch).",
            },
            "file_path": {
                "type": "string",
                "description": "File within the skill (write_file/remove_file).",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (patch).",
            },
        },
        "required": ["action", "name"],
    },
}


# ── 模块级自注册 ───────────────────────────────────────────────────

registry.register(
    name="skills_list",
    toolset="skills",
    schema=_SKILLS_LIST_SCHEMA,
    handler=skills_list,
    check_fn=check_skills_requirements,
    emoji="📚",
)

registry.register(
    name="skill_view",
    toolset="skills",
    schema=_SKILL_VIEW_SCHEMA,
    handler=skill_view,
    check_fn=check_skills_requirements,
    emoji="📖",
)

registry.register(
    name="skill_manage",
    toolset="skills",
    schema=_SKILL_MANAGE_SCHEMA,
    handler=skill_manage,
    check_fn=check_skills_requirements,
    emoji="🛠️",
)
