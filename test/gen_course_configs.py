"""Standalone bulk generator for course config YAMLs.

Inputs are the variables below (edit them as needed):
- INPUT_DIR: folder containing per-course lesson-plan YAMLs (filenames drive output)
- OUTPUT_DIR: folder to write per-course config YAMLs
- TEMPLATE_PATH: a single template config YAML to clone

Behavior:
- For every *.yaml file directly under INPUT_DIR, create OUTPUT_DIR/<same filename>
- The output file is identical to the template except:
  - course.group = last two path components of INPUT_DIR (e.g. "RAZ/C")
  - course.id = filename stem (e.g. "70_Smile")

Notes:
- This script intentionally reads ONLY TEMPLATE_PATH and does not read any other project files.
- Existing output files are overwritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import sys

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required to run this script. Install with: pip install pyyaml"
    ) from e


# =====================
# Input parameters
# =====================
INPUT_DIR = Path(r"D:\data\project\education_agent\prompt\src\lesson_plans\RAZ\B")
OUTPUT_DIR = Path(r"D:\data\project\education_agent\prompt\config\RAZ\B")
TEMPLATE_PATH = Path(r"D:\data\project\education_agent\prompt\config\RAZ\A\70_Smile.yaml")


def _compute_group(input_dir: Path) -> str:
    parts = [p for p in input_dir.parts if p not in (".", "")]
    if len(parts) < 2:
        raise ValueError(
            f"INPUT_DIR must have at least 2 path components to derive group, got: {input_dir}"
        )
    return f"{parts[-2]}/{parts[-1]}"


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Template YAML must be a mapping/object, got {type(data)}")
    return data


def _dump_yaml(data: Dict[str, Any]) -> str:
    # Keep it stable and readable. Exact formatting doesn't need to match template.
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )


def generate_configs(input_dir: Path, output_dir: Path, template_path: Path) -> int:
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    group = _compute_group(input_dir)

    template = _load_yaml(template_path)

    yaml_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".yaml", ".yml"}]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for src in yaml_files:
        course_id = src.stem

        # Clone template and modify only the two fields.
        out_obj: Dict[str, Any] = dict(template)
        course = dict(out_obj.get("course") or {})
        course["group"] = group
        course["id"] = course_id
        out_obj["course"] = course

        out_path = output_dir / src.name
        out_path.write_text(_dump_yaml(out_obj), encoding="utf-8")
        written += 1

    return written


def main() -> None:
    count = generate_configs(INPUT_DIR, OUTPUT_DIR, TEMPLATE_PATH)
    print(f"Generated {count} config YAML(s) into: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
