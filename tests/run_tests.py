"""
Unified Gradescope test runner with submission tracking.

Reads project-specific settings from ``../submission_config.json`` and
applies grace-period / submission-limit policies to the results written
to ``/autograder/results/results.json``.

This file is identical across all CS1470 assignments.  Per-project
behaviour (leaderboard entries, directory restructuring, etc.) is
controlled entirely through submission_config.json.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout

try:
    from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
except ImportError:
    print("\n\033[91mERROR: gradescope_utils not found.\033[0m")
    print("Activate the virtual environment first:")
    print("  \033[93msource /autograder/source/csci1470-env/bin/activate\033[0m\n")
    sys.exit(1)

# Ensure the tests/ directory is on the path so local modules resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

from submission_tracker import SubmissionTracker, load_config


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def copy_directory_structure(sol_dir: str, stu_dir: str) -> None:
    """
    Restructure student files to mirror the solution's subdirectory layout.

    When students upload a flat set of files but the solution organises them
    into subdirectories, this function moves matching files into the correct
    subdirectory under *stu_dir*.
    """
    for root, _dirs, files in os.walk(sol_dir):
        if root == sol_dir:
            continue
        for fname in files:
            src = os.path.join(stu_dir, fname)
            if os.path.exists(src):
                dest_dir = os.path.join(stu_dir, os.path.basename(root))
                os.makedirs(dest_dir, exist_ok=True)
                os.rename(src, os.path.join(dest_dir, fname))


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def extract_leaderboard(cfg: dict, results: dict) -> None:
    """
    If the config defines ``leaderboard.entries``, scan the test results
    and populate ``results["leaderboard"]`` accordingly.
    """
    lb_cfg = cfg.get("leaderboard")
    if not lb_cfg:
        return

    entries = lb_cfg.get("entries", [])
    if not entries:
        return

    leaderboard = []
    tests = results.get("tests", [])

    for entry in entries:
        name = entry["name"]
        test_number = entry["test_number"]
        pattern = entry.get("output_pattern", "Accuracy :")

        matching = [t for t in tests if t.get("number") == test_number]
        if not matching:
            leaderboard.append({"name": name, "value": 0})
            continue

        output_text = matching[0].get("output", "")
        if pattern in output_text:
            try:
                value = output_text.split(pattern)[1].strip().split()[0]
                leaderboard.append({"name": name, "value": float(value)})
            except (IndexError, ValueError):
                leaderboard.append({"name": name, "value": 0})
        else:
            leaderboard.append({"name": name, "value": 0})

    results["leaderboard"] = leaderboard


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_test_suite() -> dict:
    """Discover and run tests, returning the JSON results dict."""
    suite = unittest.defaultTestLoader.discover("tests")

    buf = io.StringIO()
    with redirect_stdout(buf):
        runner = JSONTestRunner(visibility=None, stream=buf)
        runner.run(suite)

    buf.seek(0)
    raw = buf.getvalue()
    json_start = raw.find("{")
    if json_start != -1:
        try:
            return json.loads(raw[json_start:])
        except json.JSONDecodeError:
            pass

    return {"score": 0, "execution_time": 0, "output": "Test execution completed", "tests": []}


# ---------------------------------------------------------------------------
# Result post-processing
# ---------------------------------------------------------------------------

def apply_tracking(tracker: SubmissionTracker, results: dict) -> dict:
    """Inject submission-tracking metadata and apply visibility rules."""
    info = tracker.get_submission_info()
    tracking_test = tracker.create_tracking_test_result()

    results.setdefault("tests", []).insert(0, tracking_test)

    # Hide detailed results if policy says so.
    if info.should_hide_results:
        for test in results["tests"]:
            if test.get("name") == tracker.TRACKING_TEST_NAME:
                continue
            test["output"] = "Test output hidden due to submission limit policy."
            test["visibility"] = "after_due_date"

    # Hide scores if over limit.
    if not info.should_show_score:
        results.pop("score", None)
        for test in results["tests"]:
            if test.get("name") == tracker.TRACKING_TEST_NAME:
                continue
            test.pop("score", None)
            test.pop("max_score", None)

    # Prepend the student-facing message.
    msg = info.student_message
    results["output"] = f"{msg}\n\n{results.get('output', '')}"

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()

    print("=" * 50)
    print(f"CS1470 AUTOGRADER  —  {cfg.get('assignment_name', 'Unknown')}")
    print("=" * 50)

    # Optional directory restructuring.
    if cfg.get("copy_directory_structure", False):
        copy_directory_structure("./solution", "./student")

    # Run tests.
    print("Running tests ...")
    results = run_test_suite()

    # Leaderboard extraction (config-driven).
    extract_leaderboard(cfg, results)

    # Submission tracking.
    tracker = SubmissionTracker(cfg=cfg)
    info = tracker.get_submission_info()

    print(f"  Grace period active : {info.is_in_grace_period}")
    print(f"  Submissions (total) : {info.total_submissions}")
    print(f"  After grace period  : {info.submissions_after_grace}")
    print(f"  Hide results        : {info.should_hide_results}")

    results = apply_tracking(tracker, results)

    # Suppress leaderboard when over limit.
    if not info.should_show_score:
        results.pop("leaderboard", None)

    # Write results.
    os.makedirs("/autograder/results", exist_ok=True)
    with open("/autograder/results/results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # Debug sidecar.
    with open("/autograder/results/submission_debug.json", "w") as fh:
        json.dump(
            {"submission_info": info.to_dict(), "config": cfg},
            fh,
            indent=2,
        )

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(f"FATAL ERROR in run_tests.py:\n{tb}")
        # Ensure Gradescope receives *something* so it doesn't show the
        # opaque "terminated improperly" message.
        os.makedirs("/autograder/results", exist_ok=True)
        with open("/autograder/results/results.json", "w") as fh:
            json.dump({
                "score": 0,
                "output": f"Autograder encountered an internal error:\n\n{tb}",
                "tests": [],
            }, fh, indent=2)