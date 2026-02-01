"""Recursively delete hidden files.

This script is intentionally standalone and unrelated to the rest of the repo.

Behavior:
- Recursively walks INPUT_DIR (including subfolders).
- Deletes files whose *filename* starts with a dot (.), e.g. ".DS_Store".
- Does not delete directories.
- Prints a summary at the end.

Notes:
- On Windows, files can be hidden via filesystem attributes without starting with '.',
  but per requirement this script uses the “starts with '.'” rule.
"""

from __future__ import annotations

from pathlib import Path

# =========================
# Input (edit this value)
# =========================
INPUT_DIR = Path(r"prompt/src/lesson_plans/RAZ/C")


def main() -> int:
    root = INPUT_DIR
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"INPUT_DIR does not exist or is not a directory: {root}")

    deleted = 0
    errors = 0

    for p in root.rglob("*"):
        # rglob('*') yields both files and directories
        if not p.is_file():
            continue
        if not p.name.startswith("."):
            continue
        try:
            p.unlink()
            deleted += 1
        except OSError:
            errors += 1

    print(f"Root: {root.resolve()}")
    print(f"Deleted hidden files: {deleted}")
    print(f"Delete errors: {errors}")

    # Non-zero exit if there were delete errors.
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
