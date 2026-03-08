#!/usr/bin/env python3
"""
Universal local test runner for CS1470 assignments.

Temporarily symlinks solution/ -> student/ so that autograder tests
can be validated against the reference implementation.

    1. Backs up any existing student/ directory
    2. Creates a symlink: student -> solution
    3. Runs the test suite via pytest
    4. Restores the original student/ directory (even on crash)

Usage (from project root):
    python run_tests_local.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

CONDA_ENV = "csci1470"

# ── ANSI escape codes ────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"

BG_RED   = "\033[41m"
BG_GREEN = "\033[42m"
BG_CYAN  = "\033[46m"

# ── Styled helpers ───────────────────────────────────────────────────────

def _hr(char="─", width=62):
    return DIM + char * width + RESET

def _banner(title, color=CYAN):
    width = 62
    border = color + BOLD + "+" + "─" * (width - 2) + "+" + RESET
    padding = color + BOLD + "|" + RESET
    centered = title.center(width - 2)
    return f"\n{border}\n{padding}{BOLD}{centered}{RESET}{padding}\n{border}"

def _status(label, msg, color=CYAN):
    tag = f"{color}{BOLD}[{label}]{RESET}"
    return f"  {tag} {msg}"


# ── Core logic ───────────────────────────────────────────────────────────

def get_project_root():
    """Return the directory containing this script."""
    return Path(__file__).resolve().parent


def backup_student_dir(project_root):
    """
    Move any existing student/ directory to student_backup/.

    Returns the backup path if a backup was created, otherwise None.
    """
    student_dir = project_root / "student"
    backup_dir = project_root / "student_backup"

    if backup_dir.exists():
        print(_status("WARN", f"Backup directory already exists: {backup_dir}", YELLOW))
        response = input(f"  {YELLOW}Remove it and continue? (y/n):{RESET} ").strip().lower()
        if response == "y":
            shutil.rmtree(backup_dir)
        else:
            print(_status("EXIT", "Aborting. Handle the backup directory manually.", RED))
            sys.exit(1)

    if student_dir.exists():
        print(_status("INFO", f"Backing up student/ -> {backup_dir.name}", CYAN))
        if student_dir.is_symlink():
            backup_dir.symlink_to(os.readlink(student_dir))
            student_dir.unlink()
        else:
            shutil.move(str(student_dir), str(backup_dir))
        return backup_dir

    print(_status("INFO", "No existing student/ directory to back up.", CYAN))
    return None


def create_symlink(project_root):
    """Create the student -> solution symlink."""
    student_dir = project_root / "student"
    solution_dir = project_root / "solution"

    if not solution_dir.exists():
        print(_status("ERROR", f"Solution directory not found: {solution_dir}", RED))
        sys.exit(1)

    if student_dir.exists() or student_dir.is_symlink():
        if student_dir.is_symlink():
            student_dir.unlink()
        else:
            shutil.rmtree(student_dir)

    print(_status("LINK", "student/ -> solution/", CYAN))
    student_dir.symlink_to("solution", target_is_directory=True)


def restore_student_dir(project_root, backup_dir):
    """Restore student/ from its backup, removing the symlink first."""
    student_dir = project_root / "student"

    if student_dir.exists() or student_dir.is_symlink():
        if student_dir.is_symlink():
            student_dir.unlink()
        else:
            shutil.rmtree(student_dir)

    if backup_dir and backup_dir.exists():
        print(_status("INFO", "Restoring student/ from backup", CYAN))
        if backup_dir.is_symlink():
            student_dir.symlink_to(os.readlink(backup_dir))
            backup_dir.unlink()
        else:
            shutil.move(str(backup_dir), str(student_dir))
        print(_status(" OK ", "student/ directory restored.", GREEN))
    else:
        print(_status("INFO", "No backup to restore.", CYAN))


def run_tests(project_root):
    """
    Run the test suite under the csci1470 conda environment.

    Returns the subprocess exit code.
    """
    print(f"\n{_hr()}")
    print(_status("TEST", "Running tests with solution code ...", CYAN))
    print(_hr())
    print()

    cmd = [
        "conda", "run", "-n", CONDA_ENV,
        "python", "-m", "pytest", "tests/", "-v", "--tb=short",
    ]

    # Ensure both the project root (for `import student` / `import solution`)
    # and tests/ (for sibling imports like `grader_utils`) are on PYTHONPATH.
    env = os.environ.copy()
    tests_dir = str(project_root / "tests")
    root_dir = str(project_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [root_dir, tests_dir, env.get("PYTHONPATH", "")]
    )

    try:
        result = subprocess.run(cmd, cwd=project_root, env=env)
        return result.returncode
    except FileNotFoundError:
        print(_status("ERROR", "'conda' not found on PATH.", RED))
        print(f"         Make sure conda is installed and the '{CONDA_ENV}' environment exists.")
        return 1


# ── Entrypoint ───────────────────────────────────────────────────────────

HEADER_ART = rf"""
{CYAN}{BOLD}   _____ _____ __ _  _ ______ ___
  / ____/ ____|_/| || |____  / _ \        {WHITE}Local Test Runner{CYAN}
 | |   | (___   | || |_  / | | | |       {DIM}{WHITE}solution -> student{CYAN}{BOLD}
 | |    \___ \  |__   _|/ /| | | |       {DIM}{WHITE}conda env: {CONDA_ENV}{CYAN}{BOLD}
 | |___ ____) |    | | / / | |_| |
  \_____|_____/     |_|/_/  \___/{RESET}
"""


def main():
    project_root = get_project_root()
    backup_dir = None

    print(HEADER_ART)
    print(f"  {DIM}Project:{RESET} {project_root}")
    print(f"  {DIM}Assignment:{RESET} {BOLD}{project_root.name}{RESET}")
    print(_hr())

    try:
        backup_dir = backup_student_dir(project_root)
        create_symlink(project_root)
        return_code = run_tests(project_root)

        print()
        print(_hr())
        if return_code == 0:
            print(f"\n  {BG_GREEN}{BOLD}{WHITE}  ALL TESTS PASSED  {RESET}\n")
        else:
            print(f"\n  {BG_RED}{BOLD}{WHITE}  TESTS FAILED  (exit {return_code})  {RESET}\n")
        print(_hr())

    except Exception as exc:
        print(f"\n{RED}{BOLD}ERROR:{RESET} {exc}")
        import traceback
        traceback.print_exc()
        return_code = 1

    finally:
        print(f"\n{_hr()}")
        print(_status("CLEAN", "Tearing down test environment ...", YELLOW))
        print(_hr())
        restore_student_dir(project_root, backup_dir)
        print()

    return return_code


if __name__ == "__main__":
    sys.exit(main())
