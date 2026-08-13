"""Skill Curator 自治维护（精简移植自原版 agent/curator.py）。

确定性 inactivity prune：读取 .usage.json（skill_usage），把长期未用的
curator-managed skill 标记 stale、超时归档（移到 skills/_archive/，可恢复）。

对齐原版语义：
- 默认开启（curator.enabled 缺省 True）；
- stale_after_days 默认 30 / archive_after_days 默认 90；
- **只处理 curator-managed（created_by == background_review）**——用户写的
  skill 绝不被自动归档；
- 无 cron/pinned/hub/bundled 保护清单（my-hermes 无这些机制）；
- 从未使用的 skill 有宽限期：不早于 stale 窗口归档。
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_STALE_AFTER_DAYS = 30
DEFAULT_ARCHIVE_AFTER_DAYS = 90


def _load_config() -> Dict:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly().get("curator", None) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def is_enabled() -> bool:
    """默认开启（对齐原版 "Default ON when no config says otherwise"）。"""
    return bool(_load_config().get("enabled", True))


def get_stale_after_days() -> int:
    try:
        return int(_load_config().get("stale_after_days", DEFAULT_STALE_AFTER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_STALE_AFTER_DAYS


def get_archive_after_days() -> int:
    try:
        return int(_load_config().get("archive_after_days", DEFAULT_ARCHIVE_AFTER_DAYS))
    except (TypeError, ValueError):
        return DEFAULT_ARCHIVE_AFTER_DAYS


def apply_automatic_transitions(now: Optional[datetime] = None) -> Dict[str, int]:
    """遍历 curator-managed skill，按最近活动时间迁移 active/stale/archived。

    返回变更计数（对齐原版 apply_automatic_transitions）。
    """
    from tools import skill_usage as _u

    if now is None:
        now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(days=get_stale_after_days())
    archive_cutoff = now - timedelta(days=get_archive_after_days())

    counts = {"marked_stale": 0, "archived": 0, "reactivated": 0, "checked": 0}

    for row in _u.curated_report():
        counts["checked"] += 1
        name = row["name"]
        # 只整理 curator-managed（background_review 自主沉淀）；用户写的不动
        if not _u.is_curator_managed_record(row):
            continue

        last_activity = _u._parse_iso(row.get("last_activity_at"))
        anchor = last_activity or _u._parse_iso(row.get("created_at")) or now
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)

        current = row.get("state", _u.STATE_ACTIVE)

        # 从未使用：宽限——不早于 stale 窗口归档（absence of evidence）
        never_used = _u.activity_count(row) == 0
        if never_used and anchor > stale_cutoff:
            if current == _u.STATE_STALE:
                _u.set_state(name, _u.STATE_ACTIVE)
                counts["reactivated"] += 1
            continue

        if anchor <= archive_cutoff and current != _u.STATE_ARCHIVED:
            ok, _msg = _u.archive_skill(name)
            if ok:
                counts["archived"] += 1
        elif anchor <= stale_cutoff and current == _u.STATE_ACTIVE:
            _u.set_state(name, _u.STATE_STALE)
            counts["marked_stale"] += 1
        elif anchor > stale_cutoff and current == _u.STATE_STALE:
            # 重新使用过 → 解除 stale
            _u.set_state(name, _u.STATE_ACTIVE)
            counts["reactivated"] += 1

    return counts


def maybe_run_curator() -> Dict[str, int]:
    """Curator 入口：enabled 检查 → 自动迁移（best-effort）。"""
    if not is_enabled():
        return {"disabled": True}
    try:
        counts = apply_automatic_transitions()
        if any(counts.values()):
            logger.info("curator: %s", counts)
        return counts
    except Exception as exc:
        logger.warning("curator run failed: %s", exc, exc_info=True)
        return {"error": str(exc)}
