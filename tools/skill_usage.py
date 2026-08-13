"""Skill 使用遥测（精简移植自原版 tools/skill_usage.py）。

在 sidecar 文件 ``~/.hermes/skills/.usage.json`` 记录每个 skill 的活动状态：
  {skill_name: {"state": "active|stale|archived",
                "activity_count": N,
                "last_activity_at": ISO,
                "created_at": ISO,
                "created_by": "user" | "background_review"}}

``created_by`` 是 **curator-management opt-in 标志**（对齐原版语义）：
只有 background_review 自主沉淀的 skill 才允许 curator 自动归档；
用户/前台写的一律不动。
"""

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"

# review fork 的写来源标识（对齐原版 provenance）
CURATOR_ORIGIN = "background_review"
USER_ORIGIN = "user"

_lock = threading.RLock()


def _usage_file() -> Path:
    return get_hermes_home() / "skills" / ".usage.json"


def _archive_dir() -> Path:
    return get_hermes_home() / "skills" / "_archive"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def load_usage() -> Dict[str, Any]:
    with _lock:
        try:
            data = json.loads(_usage_file().read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def save_usage(data: Dict[str, Any]) -> None:
    with _lock:
        try:
            _usage_file().parent.mkdir(parents=True, exist_ok=True)
            _usage_file().write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("skill usage save failed: %s", exc)


def latest_activity_at(record: Any) -> Optional[str]:
    if isinstance(record, dict):
        return record.get("last_activity_at")
    return None


def activity_count(record: Any) -> int:
    if isinstance(record, dict):
        return int(record.get("activity_count", 0) or 0)
    return 0


def is_curator_managed_record(record: Any) -> bool:
    """created_by == background_review → 允许 curator 自动管理（原版语义）。"""
    return bool(
        isinstance(record, dict) and record.get("created_by") == CURATOR_ORIGIN
    )


def seed_record_if_missing(name: str, created_by: str = USER_ORIGIN) -> bool:
    """首次使用某 skill 时建记录（created_at 锚定 now，避免新 skill 立即归档）。"""
    with _lock:
        data = load_usage()
        if name in data and isinstance(data[name], dict):
            return False
        data[name] = {
            "state": STATE_ACTIVE,
            "activity_count": 0,
            "last_activity_at": None,
            "created_at": _now_iso(),
            "created_by": created_by,
        }
        save_usage(data)
        return True


def bump_use(name: str) -> None:
    """skill 被查看/使用时调用：计数 +1、刷新活动时间、解除 stale/archived。"""
    with _lock:
        data = load_usage()
        rec = data.get(name)
        if not isinstance(rec, dict):
            seed_record_if_missing(name)
            data = load_usage()
            rec = data.get(name)
        rec["activity_count"] = int(rec.get("activity_count", 0) or 0) + 1
        rec["last_activity_at"] = _now_iso()
        if rec.get("state") != STATE_ACTIVE:
            rec["state"] = STATE_ACTIVE  # 重新使用 → reactivate
        data[name] = rec
        save_usage(data)


def set_state(name: str, state: str) -> None:
    with _lock:
        data = load_usage()
        if isinstance(data.get(name), dict):
            data[name]["state"] = state
            save_usage(data)


def get_record(name: str) -> Optional[Dict[str, Any]]:
    rec = load_usage().get(name)
    return rec if isinstance(rec, dict) else None


def curated_report() -> List[Dict[str, Any]]:
    """返回带记录的 skill 报告（含未持久化标记由 curator 处理）。"""
    report = []
    data = load_usage()
    for name, rec in data.items():
        if isinstance(rec, dict):
            item = dict(rec)
            item["name"] = name
            report.append(item)
    return report


def archive_skill(name: str) -> "tuple[bool, str]":
    """把 skill 移到 skills/_archive/<name>/（可恢复，对齐原版"只归档不删除"）。"""
    skills_dir = get_hermes_home() / "skills"
    src = skills_dir / name
    if not (src / "SKILL.md").exists():
        return False, f"skill '{name}' not found"
    dst = _archive_dir() / name
    try:
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        set_state(name, STATE_ARCHIVED)
        return True, f"archived '{name}'"
    except Exception as exc:
        return False, f"archive failed: {exc}"
