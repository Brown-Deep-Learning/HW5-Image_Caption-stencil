"""
Submission tracking for the CS1470 Gradescope autograder.

Reads configuration from ``submission_config.json`` (located inside the
``tests/`` directory or one directory above it) and uses Gradescope's
``submission_metadata.json`` to enforce grace-period and submission-limit
policies.
"""

import json
import os
from datetime import datetime, timedelta, time
from datetime import timezone as tz
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def _locate_config() -> str:
    """Return the path to the nearest submission_config.json."""
    # Search order:
    #   1. Same directory as this file (tests/)  – works on Gradescope where
    #      the autograder copies the whole tests/ folder.
    #   2. One level above this file (project root) – legacy / local fallback.
    #   3. Current working directory.
    candidates = [
        os.path.join(os.path.dirname(__file__), "submission_config.json"),
        os.path.join(os.path.dirname(__file__), "..", "submission_config.json"),
        os.path.join(os.getcwd(), "submission_config.json"),
    ]
    for path in candidates:
        full = os.path.abspath(path)
        if os.path.isfile(full):
            return full
    raise FileNotFoundError(
        "submission_config.json not found.  "
        "Expected it in tests/ or the project root (one level above tests/)."
    )


def load_config() -> Dict[str, Any]:
    """Load and return the parsed JSON config."""
    path = _locate_config()
    with open(path, "r") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EST = tz(timedelta(hours=-5))


def _parse_datetime(raw: str) -> datetime:
    """Parse an ISO-format string into a timezone-aware datetime."""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz.utc)
        return dt
    except ValueError:
        print(f"WARNING: Could not parse datetime '{raw}', falling back to now(UTC)")
        return datetime.now(tz.utc)


def grace_period_end(release_date: datetime, cfg: Dict[str, Any]) -> datetime:
    """Calculate the grace-period deadline from the release date and config."""
    days = cfg.get("grace_period_days", 14)
    target_date = (release_date + timedelta(days=days)).date()

    hh, mm = (cfg.get("deadline_time") or "23:59").split(":")
    offset_hours = cfg.get("timezone_offset_hours", -5)
    local_tz = tz(timedelta(hours=offset_hours))

    return datetime.combine(target_date, time(int(hh), int(mm)), tzinfo=local_tz)


def format_datetime_est(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S EST") -> str:
    """Format *dt* as an EST string."""
    return dt.astimezone(_EST).strftime(fmt)


# ---------------------------------------------------------------------------
# SubmissionInfo data container
# ---------------------------------------------------------------------------

@dataclass
class SubmissionInfo:
    """Immutable snapshot of a student's submission status."""

    total_submissions: int
    submissions_after_grace: int
    is_in_grace_period: bool
    should_hide_results: bool
    should_show_score: bool
    student_message: str
    remaining_submissions: int          # -1 means unlimited
    time_until_grace_end: str
    grace_period_end_dt: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_submissions": self.total_submissions,
            "submissions_after_grace": self.submissions_after_grace,
            "is_in_grace_period": self.is_in_grace_period,
            "should_hide_results": self.should_hide_results,
            "should_show_score": self.should_show_score,
            "student_message": self.student_message,
            "remaining_submissions": self.remaining_submissions,
            "grace_period_end": self.grace_period_end_dt.isoformat(),
        }


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class SubmissionTracker:
    """
    Analyses Gradescope submission metadata and applies the policies
    defined in submission_config.json.
    """

    TRACKING_TEST_NAME = "0.0) Submission Tracking"

    def __init__(
        self,
        metadata_path: str = "/autograder/submission_metadata.json",
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = cfg if cfg is not None else load_config()
        self.metadata = self._load_metadata(metadata_path)

    # -- metadata -----------------------------------------------------------

    def _load_metadata(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except FileNotFoundError:
            print(f"WARNING: Metadata file not found at {path}")
            return self._dummy_metadata()
        except json.JSONDecodeError as exc:
            print(f"ERROR: Malformed metadata file: {exc}")
            return self._dummy_metadata()

    @staticmethod
    def _dummy_metadata() -> Dict[str, Any]:
        now = datetime.now(tz.utc)
        return {
            "id": 999999,
            "created_at": now.isoformat(),
            "assignment": {
                "release_date": now.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ).isoformat(),
                "title": "Test Assignment",
                "total_points": "100.0",
            },
            "users": [{"email": "test@example.com", "name": "Test User"}],
            "previous_submissions": [],
        }

    # -- counting -----------------------------------------------------------

    def _count_submissions_after(self, after: datetime) -> int:
        count = 0
        for sub in self.metadata.get("previous_submissions", []):
            sub_time = _parse_datetime(sub["submission_time"])
            if (
                not self.cfg.get("count_autograder_errors", False)
                and sub.get("autograder_error", False)
            ):
                continue
            if sub_time >= after:
                count += 1

        current_time = _parse_datetime(self.metadata["created_at"])
        if current_time >= after:
            count += 1
        return count

    # -- public API ---------------------------------------------------------

    def get_submission_info(self) -> SubmissionInfo:
        release = _parse_datetime(
            self.metadata["assignment"]["release_date"]
        )
        current = _parse_datetime(self.metadata["created_at"])
        gp_end = grace_period_end(release, self.cfg)

        total = len(self.metadata.get("previous_submissions", [])) + 1
        after_grace = self._count_submissions_after(gp_end)
        in_grace = current < gp_end

        max_after = self.cfg.get("max_submissions_after_grace", 15)

        # Time remaining
        if in_grace:
            remaining_td = gp_end - datetime.now(tz.utc)
            days = remaining_td.days
            hours, rem = divmod(remaining_td.seconds, 3600)
            minutes, _ = divmod(rem, 60)
            time_str = f"{days}d {hours}h {minutes}m"
        else:
            time_str = "Grace period has ended"

        remaining = -1 if in_grace else max(0, max_after - after_grace)

        hide = self._should_hide(in_grace, after_grace)
        show_score = self._should_show_score(in_grace, after_grace)
        message = self._student_message(
            in_grace, after_grace, remaining, gp_end, hide
        )

        return SubmissionInfo(
            total_submissions=total,
            submissions_after_grace=after_grace,
            is_in_grace_period=in_grace,
            should_hide_results=hide,
            should_show_score=show_score,
            student_message=message,
            remaining_submissions=remaining,
            time_until_grace_end=time_str,
            grace_period_end_dt=gp_end,
        )

    # -- visibility ---------------------------------------------------------

    def _should_hide(self, in_grace: bool, after: int) -> bool:
        if in_grace:
            return False
        if self.cfg.get("hide_detailed_results_after_grace_period", False):
            return True
        threshold = self.cfg.get("hide_results_threshold", 15)
        if self.cfg.get("hide_detailed_results_after_threshold", True) and after > threshold:
            return True
        return False

    def _should_show_score(self, in_grace: bool, after: int) -> bool:
        if in_grace:
            return True
        max_after = self.cfg.get("max_submissions_after_grace", 15)
        return after <= max_after

    # -- messaging ----------------------------------------------------------

    def _student_message(
        self,
        in_grace: bool,
        after: int,
        remaining: int,
        gp_end: datetime,
        hide: bool,
    ) -> str:
        msgs = self.cfg.get("messages", {})
        max_after = self.cfg.get("max_submissions_after_grace", 15)

        if in_grace:
            tpl = msgs.get("grace_period", "")
            return tpl.format(
                grace_end_date=format_datetime_est(gp_end, "%B %d at %I:%M %p EST"),
                max_submissions=max_after,
            )

        if after > max_after:
            tpl = msgs.get("over_limit", "")
            return tpl.format(current_count=after, max_submissions=max_after)

        if hide:
            tpl = msgs.get("results_hidden", "")
            return tpl.format(current_count=after)

        if remaining <= 3:
            tpl = msgs.get("approaching_limit", "")
            return tpl.format(remaining=remaining)

        tpl = msgs.get("normal_tracking", "")
        return tpl.format(
            current_count=after,
            max_submissions=max_after,
            remaining=remaining,
        )

    # -- Gradescope test entry ----------------------------------------------

    def create_tracking_test_result(self) -> Dict[str, Any]:
        """Return a zero-point Gradescope test that shows submission status."""
        info = self.get_submission_info()
        name = self.cfg.get("assignment_name", "CS1470")

        lines = [
            f"**{name} Submission Tracking Report**",
            "",
            f"- Total submissions: {info.total_submissions}",
        ]

        if info.is_in_grace_period:
            lines += [
                f"- Grace period ends: {format_datetime_est(info.grace_period_end_dt, '%B %d at %I:%M %p EST')}",
                f"- Time remaining: {info.time_until_grace_end}",
                "- Submissions during grace period: Unlimited.",
            ]
        else:
            lines += [
                f"- Submissions since grace period ended: {info.submissions_after_grace}",
                f"- Remaining submissions: {'None' if info.remaining_submissions == 0 else info.remaining_submissions if info.remaining_submissions >= 0 else 'Unlimited'}",
            ]

        lines += ["", info.student_message]

        return {
            "name": self.TRACKING_TEST_NAME,
            "score": 0,
            "max_score": 0,
            "output": "\n".join(lines),
            "output_format": "md",
            "visibility": "visible",
        }