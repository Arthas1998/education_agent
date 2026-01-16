# _*_ coding: utf-8 _*_
# @File:    rule_based_manager
# @Time:    2025/12/22 11:39
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class Step:
    id: str
    title: str = ""
    pages: List[int] = field(default_factory=list)
    plan_nl: str = ""  # 保留原字段，不在调度器里使用


@dataclass
class Policy:
    step_turns: Dict[str, int] = field(default_factory=dict)
    default_step_turns: int = 1
    # 可选：流程结束后的行为
    end_behavior: str = "stay_last"  # stay_last | return_none


@dataclass
class DialogueState:
    # 当前 step 在 steps 列表里的位置
    step_index: int = 0
    # 当前 step 内的轮次（1-based）
    step_turn_index: int = 1
    # 总轮次（1-based）——注意：从外部输入同步，不再由 manager 内部自增
    total_turn_index: int = 1
    # 是否已结束（当 end_behavior=return_none 时会用到）
    ended: bool = False


@dataclass
class StepDecision:
    step_id: Optional[str]
    step_index: int
    step_turn_index: int
    total_turn_index: int
    pages: List[int]
    warnings: List[str] = field(default_factory=list)


class SimpleRuleDialogueManager:
    """
    最简单的规则调度器：
      - 按 steps 顺序推进
      - 每个 step 执行 policies.step_turns[step_id] 轮（未配置则用 default_step_turns）

    重要：total_turn_index 必须由外部输入驱动。
      - decide(total_turn=...) / next_state(total_turn=...) 每次调用都会同步 state.total_turn_index。
      - 规则判断（当前 step / step_turn）也以外部 total_turn 为准，避免重试/回放导致漂移。

    轮次约定：
      - total_turn 使用 1-based（第一轮为 1）。
      - 若你的外部计数是 0-based（例如 Chat.turn_count），请传 total_turn = chat.turn_count + 1。

    边界行为：
      - total_turn 与当前 state.total_turn_index 相同：认为是同一轮的重复调用（幂等，不推进）。
      - total_turn 向前跳跃：快速推进到对应轮次并返回当轮决策，同时给出 warnings。
      - total_turn 回退：允许回退并重算状态（用于回放/撤销），同时给出 warnings。
    """

    def __init__(self, steps: List[Step], policy: Policy):
        if not steps:
            raise ValueError("steps is empty.")
        self.steps = steps
        self.policy = policy
        self.state = DialogueState(step_index=0, step_turn_index=1, total_turn_index=1, ended=False)

        # 预先建立 id -> index 映射，便于调试/扩展
        self._id_to_index = {s.id: i for i, s in enumerate(steps)}

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SimpleRuleDialogueManager":
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML must be a mapping/dict.")

        steps_raw = data.get("steps", [])
        if not isinstance(steps_raw, list):
            raise ValueError("'steps' must be a list.")

        steps: List[Step] = []
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                raise ValueError(f"steps[{i}] must be a dict.")
            sid = s.get("id")
            if not isinstance(sid, str) or not sid:
                raise ValueError(f"steps[{i}].id must be a non-empty string.")
            title = s.get("title", "") if isinstance(s.get("title", ""), str) else ""
            pages = s.get("pages", [])
            if pages is None:
                pages = []
            if not (isinstance(pages, list) and all(isinstance(x, int) for x in pages)):
                raise ValueError(f"steps[{i}].pages must be a list[int].")
            plan_nl = s.get("plan_nl", "") if isinstance(s.get("plan_nl", ""), str) else ""
            steps.append(Step(id=sid, title=title, pages=pages, plan_nl=plan_nl))

        policies_raw = data.get("policies", {}) or {}
        if not isinstance(policies_raw, dict):
            raise ValueError("'policies' must be a dict (or {}).")

        step_turns = policies_raw.get("step_turns", {}) or {}
        if not isinstance(step_turns, dict):
            raise ValueError("policies.step_turns must be a dict of {step_id: int}.")

        # 只保留 int 且 >=1 的
        cleaned_turns: Dict[str, int] = {}
        for k, v in step_turns.items():
            if isinstance(k, str) and isinstance(v, int) and v >= 1:
                cleaned_turns[k] = v
            else:
                # 非法配置直接忽略，避免启动就崩；你也可以改成 raise
                pass

        defaults_raw = policies_raw.get("defaults", {}) or {}
        default_step_turns = defaults_raw.get("step_turns", 2)
        if not isinstance(default_step_turns, int) or default_step_turns < 1:
            default_step_turns = 2

        end_behavior = (policies_raw.get("end", {}) or {}).get("behavior", "stay_last")
        if end_behavior not in ("stay_last", "return_none"):
            end_behavior = "stay_last"

        policy = Policy(
            step_turns=cleaned_turns,
            default_step_turns=default_step_turns,
            end_behavior=end_behavior,
        )

        return cls(steps=steps, policy=policy)

    def reset(self, *, start_step_id: Optional[str] = None) -> None:
        """重置状态。可选从指定 step_id 开始。"""
        if start_step_id is not None:
            if start_step_id not in self._id_to_index:
                raise ValueError(f"Unknown start_step_id: {start_step_id}")
            self.state = DialogueState(
                step_index=self._id_to_index[start_step_id],
                step_turn_index=1,
                total_turn_index=1,
                ended=False,
            )
        else:
            self.state = DialogueState(step_index=0, step_turn_index=1, total_turn_index=1, ended=False)

    def _build_prefix_sums(self, warnings_out: List[str]) -> List[Tuple[int, int]]:
        """Return list of (start_turn_inclusive, end_turn_inclusive) for each step.

        Turns are 1-based. Example with 2 steps of 2 turns each:
          step0: (1,2)
          step1: (3,4)
        """
        ranges: List[Tuple[int, int]] = []
        start = 1
        for s in self.steps:
            mt = self._max_turns_for(s.id, warnings_out)
            end = start + mt - 1
            ranges.append((start, end))
            start = end + 1
        return ranges

    def _resync_to_total_turn(self, total_turn: int, warnings_out: List[str]) -> None:
        """Sync internal state to an externally supplied total_turn (1-based)."""
        if not isinstance(total_turn, int) or total_turn < 1:
            raise ValueError(f"total_turn must be an int >= 1, got: {total_turn!r}")

        prev_total = self.state.total_turn_index
        if total_turn == prev_total:
            # same turn: idempotent
            return

        if total_turn > prev_total + 1:
            warnings_out.append(
                f"external total_turn jumped forward from {prev_total} to {total_turn}; fast-forwarding state"
            )
        elif total_turn < prev_total:
            warnings_out.append(
                f"external total_turn moved backward from {prev_total} to {total_turn}; rewinding state"
            )

        # Update total turn first
        self.state.total_turn_index = total_turn

        # If dialogue had ended earlier but external turn is within range again, allow rewind
        self.state.ended = False

        ranges = self._build_prefix_sums(warnings_out)
        if not ranges:
            # should be impossible due to __init__ check
            self.state.step_index = 0
            self.state.step_turn_index = 1
            return

        # Locate which step the total_turn lands in
        found = False
        for idx, (start, end) in enumerate(ranges):
            if start <= total_turn <= end:
                self.state.step_index = idx
                self.state.step_turn_index = (total_turn - start) + 1
                found = True
                break

        if found:
            return

        # total_turn is beyond the whole plan
        last_idx = len(self.steps) - 1
        last_start, last_end = ranges[last_idx]
        last_max_turns = last_end - last_start + 1

        if self.policy.end_behavior == "return_none":
            self.state.step_index = last_idx
            self.state.step_turn_index = last_max_turns
            self.state.ended = True
        else:
            # stay_last
            self.state.step_index = last_idx
            self.state.step_turn_index = last_max_turns

    def decide(self, *, total_turn: int) -> StepDecision:
        """Return current step decision for the given external total_turn (does not advance)."""
        warnings_out: List[str] = []
        self._resync_to_total_turn(total_turn, warnings_out)

        if self.state.ended:
            return StepDecision(
                step_id=None,
                step_index=self.state.step_index,
                step_turn_index=self.state.step_turn_index,
                total_turn_index=self.state.total_turn_index,
                pages=[],
                warnings=warnings_out,
            )

        step = self.get_current_step()
        _ = self._max_turns_for(step.id, warnings_out)  # trigger missing-policy warning if any
        return StepDecision(
            step_id=step.id,
            step_index=self.state.step_index,
            step_turn_index=self.state.step_turn_index,
            total_turn_index=self.state.total_turn_index,
            pages=list(step.pages),
            warnings=warnings_out,
        )

    def next_state(self, *, total_turn: int) -> StepDecision:
        """Sync to external total_turn and return the StepDecision for that turn.

        Note: because total_turn is externally driven, next_state() does not increment turns.
        It exists mainly for backward-compatible call sites that expect a "state update" API.
        """
        warnings_out: List[str] = []
        self._resync_to_total_turn(total_turn, warnings_out)

        if self.state.ended:
            return StepDecision(
                step_id=None,
                step_index=self.state.step_index,
                step_turn_index=self.state.step_turn_index,
                total_turn_index=self.state.total_turn_index,
                pages=[],
                warnings=warnings_out,
            )

        step = self.get_current_step()
        _ = self._max_turns_for(step.id, warnings_out)
        return StepDecision(
            step_id=step.id,
            step_index=self.state.step_index,
            step_turn_index=self.state.step_turn_index,
            total_turn_index=self.state.total_turn_index,
            pages=list(step.pages),
            warnings=warnings_out,
        )

    def _max_turns_for(self, step_id: str, warnings_out: List[str]) -> int:
        if step_id in self.policy.step_turns:
            return self.policy.step_turns[step_id]
        warnings_out.append(
            f"policies.step_turns missing for step_id='{step_id}', using defaults.step_turns={self.policy.default_step_turns}"
        )
        return self.policy.default_step_turns

    def get_current_step(self) -> Step:
        return self.steps[self.state.step_index]

    def get_prompt_params(self, decision: StepDecision, *, include_pdf_pages: bool = True) -> Dict[str, object]:
        """给 PromptLoader 的 params（最简版）。"""
        params: Dict[str, object] = {}
        if decision.step_id is not None:
            params["step_ids"] = decision.step_id
        if include_pdf_pages and decision.pages:
            params["pdf_pages"] = decision.pages
        return params


# -------------------------
# Minimal demo
# -------------------------
if __name__ == "__main__":
    mgr = SimpleRuleDialogueManager.from_yaml(r"D:\data\project\education_agent\prompt\src\lesson_plans\70_Smile.yaml")

    # 2) first decision (before any turn happens)
    d0 = mgr.decide(total_turn=1)
    print("DECIDE:", d0)

    # 3) simulate a few turns
    for t in range(1, 11):
        d = mgr.next_state(total_turn=t)
        print("NEXT:", d, "params:", mgr.get_prompt_params(d))
