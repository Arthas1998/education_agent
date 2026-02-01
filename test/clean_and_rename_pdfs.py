"""Clean hidden files and copy/rename PDFs.

This script is intentionally standalone and unrelated to the rest of the repo.

What it does:
1) Deletes hidden files in INPUT_DIR whose filename starts with a dot (.)
2) Copies PDF files named like: "06、On the Farm.pdf" to OUTPUT_DIR
   and renames them to: "06_OnTheFarm.pdf"

Notes:
- It only targets files directly under INPUT_DIR (non-recursive).
- It does not modify non-hidden, non-pdf files.
- It creates OUTPUT_DIR if it doesn't exist.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# =========================
# Inputs (edit these values)
# =========================
INPUT_DIR = Path(r"D:\data\project\RAZ\G级\G级绘本PDF（90本）")
OUTPUT_DIR = Path(r"D:\data\project\RAZ_Tidy\G")

# If True: overwrite existing files in OUTPUT_DIR.
OVERWRITE = False


_PDF_NAME_RE = re.compile(
    r"^\s*(?P<num>\d+)\s*[、,，\.\-—_ ]+\s*(?P<title>.+?)\s*\.pdf\s*$",
    re.IGNORECASE,
)


def _to_camel_case(title: str) -> str:
    """Convert a title like 'On the Farm' -> 'OnTheFarm'.

    Keeps only alphanumeric tokens; joins them in TitleCase.
    """

    # Normalize separators to spaces, then extract alphanumeric chunks.
    tokens = re.findall(r"[A-Za-z0-9]+", title)
    return "".join(t.capitalize() for t in tokens)


def _target_name(src_name: str) -> str | None:
    """Return the renamed file name for a PDF, or None if name doesn't match pattern."""

    m = _PDF_NAME_RE.match(src_name)
    if not m:
        return None
    num = m.group("num")
    title = m.group("title")
    camel = _to_camel_case(title)
    if not camel:
        return None
    return f"{num}_{camel}.pdf"


def main() -> int:
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"INPUT_DIR does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Delete hidden files starting with '.'
    for p in input_dir.iterdir():
        if p.is_file() and p.name.startswith("."):
            p.unlink()

    # 2) Copy and rename PDFs
    copied = 0
    skipped = 0
    for p in input_dir.iterdir():
        if not p.is_file():
            continue

        if p.suffix.lower() != ".pdf":
            continue

        new_name = _target_name(p.name)
        if new_name is None:
            skipped += 1
            continue

        dst = output_dir / new_name
        if dst.exists() and not OVERWRITE:
            skipped += 1
            continue

        shutil.copy2(p, dst)
        copied += 1

    print(f"Deleted hidden files: done")
    print(f"Copied PDFs: {copied}")
    print(f"Skipped files: {skipped}")
    print(f"Output dir: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
