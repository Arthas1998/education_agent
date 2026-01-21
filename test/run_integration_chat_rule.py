# -*- coding: utf-8 -*-

"""run_integration_chat_rule.py

A minimal CLI REPL to verify the collaboration of:
- utils.prompt_loader.PromptLoader
- utils.chat.Chat (turn_count / params / messages)
- utils.rule_based_manager.SimpleRuleDialogueManager (rule scheduling)

Interactive multi-turn loop:
  (1) mgr.decide(total_turn=...) -> decision (external turn driven)
  (2) apply decision -> chat.params
  (3) user types input() -> chat.add_user_text() renders the next user message
  (4) optional: call LLM for assistant reply (chat.stream_reply)
  (5) mgr.next_state(total_turn=...) syncs manager state for next turn

By default, no real LLM call is required; we can run in "render-only" mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from utils.chat import Chat, apply_dialogue_decision_to_chat_params
from utils.prompt_loader import PromptLoader
from utils.rule_based_manager import SimpleRuleDialogueManager


def _short_messages(messages: Any, *, max_items: int = 6) -> Any:
    """Return a small JSON-serializable summary of messages to avoid huge dumps."""
    if not isinstance(messages, list):
        return messages

    out = []
    for msg in messages[-max_items:]:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t.strip())
            preview = " ".join(text_parts)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            out.append({"role": role, "text_preview": preview, "parts": len(content)})
        else:
            out.append({"role": role, "content": content})
    return out


def _iter_exit_words(words: Optional[List[str]]) -> Iterable[str]:
    if not words:
        return ("/quit", "/exit", "quit", "exit")
    return (w.strip() for w in words if isinstance(w, str) and w.strip())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Interactive CLI runner for Chat + PromptLoader + SimpleRuleDialogueManager"
    )

    ap.add_argument(
        "--prompt-config",
        default=r"D:\data\project\education_agent\prompt\config\multiple\70_Smile_img.yaml",
        # default=r"D:\data\PythonProject\HITProject\education_agent\prompt\config\multiple\57_HeRuns_img.yaml",
        help="PromptLoader YAML config path.",
    )
    # ap.add_argument(
    #     "--lesson-plan",
    #     default=r"D:\data\project\education_agent\prompt\src\lesson_plans\57_HeRuns.yaml",
    #     # default=r"D:\data\PythonProject\HITProject\education_agent\prompt\src\lesson_plans\57_HeRuns.yaml",
    #     help="Lesson plan YAML for SimpleRuleDialogueManager.",
    # )
    ap.add_argument(
        "--start-step-id",
        default=None,
        help="Optional: start from a given step_id (must exist in lesson plan).",
    )

    ap.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Max user turns (0 means unlimited until exit word / Ctrl+C).",
    )

    # runtime UX
    ap.add_argument(
        "--exit-words",
        nargs="*",
        default=None,
        help="Words that end the session (default: /quit /exit quit exit).",
    )

    # output control
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON payloads.")
    ap.add_argument("--print-messages", action="store_true", help="Print rendered message summaries each turn.")
    ap.add_argument("--print-params", action="store_true", help="Print chat.params/decision each turn.")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable Chat debug_mode (prints PromptLoader warnings).",
    )

    # model call control
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="Render-only mode: do not call chat.stream_reply().",
    )
    ap.add_argument(
        "--auto-first-turn",
        action="store_true",
        default=True,
        help=(
            "Auto-run the first turn without waiting for CLI input. "
            "The first user input will be treated as empty string."
        ),
    )

    args = ap.parse_args()

    prompt_path = Path(args.prompt_config)
    # lesson_path = Path(args.lesson_plan)

    if not prompt_path.exists():
        print(f"ERROR: prompt config not found: {prompt_path}", file=sys.stderr)
        return 2
    # if not lesson_path.exists():
    #     print(f"ERROR: lesson plan not found: {lesson_path}", file=sys.stderr)
    #     return 2

    loader = PromptLoader.from_yaml(prompt_path)
    lesson_path = loader.config["registry"]["lesson_plan"]["from"]["path"]
    mgr = SimpleRuleDialogueManager.from_yaml(lesson_path)
    mgr.reset(start_step_id=args.start_step_id)

    client = OpenAI(
        api_key="sk-4ab9e4105ed44934860ef17cf366a7f6", # 190新账号
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    # By default we run render-only. If you want real LLM calls, pass a real client into Chat.
    chat = Chat(
        client=client,
        prompt_loader=loader,
        model="qwen3-vl-plus",
        debug_mode=bool(args.debug))

    exit_words = {w.lower() for w in _iter_exit_words(args.exit_words)}

    # Perf timing: time from user input submitted (after add_user_text) to first streamed token.
    t_stream_start: Optional[float] = None

    def _stream_print_reply() -> None:
        nonlocal t_stream_start
        if args.no_llm:
            return
        if chat.client is None:
            print("[WARN] --no-llm not set but Chat.client is None; skipping model call.")
            return

        print("Assistant> ", end="", flush=True)

        first_token_printed = False
        for part in chat.stream_reply():
            # Record the first *non-empty* streamed text chunk.
            if (not first_token_printed) and isinstance(part, str) and part.strip():
                first_token_printed = True
                if t_stream_start is not None:
                    latency_ms = (time.perf_counter() - t_stream_start) * 1000.0
                    print(f"\n[METRIC] first_stream_token_latency_ms={latency_ms:.1f}\nAssistant> ", end="", flush=True)
            print(part, end="", flush=True)

        if not first_token_printed and t_stream_start is not None:
            latency_ms = (time.perf_counter() - t_stream_start) * 1000.0
            print(f"\n[METRIC] no_stream_token_received_ms={latency_ms:.1f}")

        print("\n")

    print("\n[CLI] Multi-turn chat started.")
    print("[CLI] Type your answer and press Enter.")
    print(f"[CLI] Exit words: {', '.join(sorted(exit_words))}. Ctrl+C also works.\n")

    user_turns = 0
    first_turn_pending = bool(args.auto_first_turn)

    try:
        while True:
            if args.max_turns and user_turns >= int(args.max_turns):
                print("\n[CLI] Reached --max-turns. Bye.")
                break


            decision = mgr.decide(total_turn=chat.turn_count)
            apply_dialogue_decision_to_chat_params(chat, decision)

            if args.print_params:
                payload: Dict[str, Any] = {
                    "turn_count_before_add": chat.turn_count,
                    "decision": {
                        "step_id": getattr(decision, "step_id", None),
                        "pages": getattr(decision, "pages", None),
                        "total_turn_index": getattr(decision, "total_turn_index", None),
                        "step_turn_index": getattr(decision, "step_turn_index", None),
                    },
                    "chat_params": dict(chat.params),
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))

            # When end_behavior=return_none, step_id can become None.
            if getattr(decision, "step_id", None) is None:
                print("\n[CLI] Dialogue manager ended (step_id=None). Bye.")
                break

            if first_turn_pending:
                # First round: user input is empty (auto-start). Still render/send the first user message.
                user_text = ""
                first_turn_pending = False
                print("[CLI] Auto first turn: sending initial user prompt (empty input).")
            else:
                user_text = input("You> ").strip()

            if user_text.lower() in exit_words:
                print("\n[CLI] Bye.")
                break

            chat.add_user_text(user_text)
            # Start timing right after user input is submitted into the chat.
            t_stream_start = time.perf_counter()
            user_turns += 1

            if args.print_messages:
                summary = {
                    "turn_count_after_add": chat.turn_count,
                    "messages_tail": _short_messages(chat.messages),
                }
                print(json.dumps(summary, ensure_ascii=False, indent=2 if args.pretty else None))

            _stream_print_reply()

            # Sync manager with the *next* external turn (after user turn_count increased)
            mgr.next_state(total_turn=chat.turn_count)

    except KeyboardInterrupt:
        print("\n[CLI] Interrupted. Bye.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

