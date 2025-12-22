# _*_ coding: utf-8 _*_
# @File:    val_prompt_loader
# @Time:    2025/12/21 22:25
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_prompt_loader.py

A minimal standalone validation script for PromptLoader.

Example:
  python validate_prompt_loader.py \
    --config ./prompts/70_Smile.yaml \
    --template generator \
    --messages system user \
    --turn-index 3 \
    --student-answer "I brush my teeth." \
    --prev-summary "Last time we learned family words." \
    --step-ids greet cover \
    --pdf-pages 3-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---- IMPORTANT ----
# Put your PromptLoader implementation in prompt_loader.py (same dir),
# or change this import to wherever you saved the class.
try:
    from utils.prompt_loader import PromptLoader, RenderError, PromptConfigError  # type: ignore
except Exception as e:
    print(
        "ERROR: Failed to import PromptLoader.\n"
        "Make sure you saved the class code as 'prompt_loader.py' next to this script,\n"
        "or adjust the import path in validate_prompt_loader.py.\n"
        f"Import error: {e}",
        file=sys.stderr,
    )
    sys.exit(2)


def _parse_pdf_pages(value: Optional[str]) -> Optional[Union[int, str, List[int]]]:
    """
    Supported inputs:
      - "3" -> int(3)
      - "3-5" -> "3-5" (range string, parsed by loader)
      - "3,4,7" -> [3,4,7]
      - None -> None
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None

    # range form
    if "-" in s and "," not in s:
        return s  # "3-5"

    # comma list
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        nums: List[int] = []
        for p in parts:
            if not p.isdigit():
                raise ValueError(f"Invalid --pdf-pages list element: {p!r}")
            nums.append(int(p))
        return nums

    # single page number
    if s.isdigit():
        return int(s)

    raise ValueError(f"Invalid --pdf-pages format: {value!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and render a PromptLoader config.")
    ap.add_argument("--config", default=r"D:\data\project\education_agent\prompt\config\multiple\example.yaml", help="Path to the course YAML config file.")
    ap.add_argument("--template", default="generator", help="Template name under templates.* (default: generator).")
    ap.add_argument(
        "--messages",
        nargs="+",
        default=["user"],
        help="Message names inside the template to render (default: system user).",
    )

    ap.add_argument("--student-answer", default="hello", help="runtime_vars.student_answer (text).")

    # memory
    ap.add_argument("--prev-summary", default="123", help="memory.prev_summary (text).")

    # params (selectors)
    ap.add_argument(
        "--step-ids",
        nargs="*",
        default=["greet", "cover"],
        help="params.step_ids for yaml_by_id_text: e.g. --step-ids greet cover",
    )
    ap.add_argument(
        "--pdf-pages",
        default=None,
        help="params.pdf_pages for pdf_pages selector: '3' or '3-5' or '3,4,7'",
    )

    # output control
    ap.add_argument("--pretty", action="store_true", help="Pretty-print rendered messages JSON.")
    ap.add_argument("--no-render", action="store_true", help="Only validate config; skip rendering.")

    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}", file=sys.stderr)
        return 2

    # Build input dicts
    runtime_vars: Dict[str, Any] = {
        # "turn_index": args.turn_index,
        "student_answer": args.student_answer,
    }
    memory: Dict[str, Any] = {
        # "prev_summary": args.prev_summary,
    }
    params: Dict[str, Any] = {}

    if args.step_ids:
        # If only one step id, either pass string or list; both are allowed.
        # We'll pass list to exercise the multi-id path.
        params["step_ids"] = args.step_ids if len(args.step_ids) > 1 else args.step_ids[0]

    if args.pdf_pages is not None:
        try:
            params["pdf_pages"] = _parse_pdf_pages(args.pdf_pages)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    # Load
    try:
        loader = PromptLoader.from_yaml(cfg_path)
    except (PromptConfigError, Exception) as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        return 2

    # Validate
    issues = loader.validate()
    if issues:
        print("CONFIG VALIDATION ISSUES:")
        for i, err in enumerate(issues, 1):
            print(f"  {i}. {err}")
        print("")
    else:
        print("CONFIG VALIDATION: OK\n")

    if args.no_render:
        return 0 if not issues else 1

    # Render

    result = loader.render(
        template=args.template,
        message_names=args.messages,
        runtime_vars=runtime_vars,
        memory=memory,
        params=params,
    )


    # Output messages
    if args.pretty:
        print("RENDERED MESSAGES (pretty):")
        print(json.dumps(result.messages, ensure_ascii=False, indent=2))
    else:
        print("RENDERED MESSAGES:")
        print(json.dumps(result.messages, ensure_ascii=False))

    # Output warnings
    if result.warnings:
        print("\nWARNINGS:")
        for w in result.warnings:
            # RenderWarning is a dataclass; it will have .code/.message/.context
            print(f"- [{w.code}] {w.message}")
            if getattr(w, "context", None):
                print(f"  context: {json.dumps(w.context, ensure_ascii=False)}")
    else:
        print("\nWARNINGS: none")

    # Non-zero exit if config issues exist
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
