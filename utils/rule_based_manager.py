# _*_ coding: utf-8 _*_
# @File:    rule_based_manager
# @Time:    2025/12/22 11:39
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    default_step_turns: int = 2
    # 可选：流程结束后的行为
    end_behavior: str = "stay_last"  # stay_last | return_none


@dataclass
class DialogueState:
    # 当前 step 在 steps 列表里的位置
    step_index: int = 0
    # 当前 step 内的轮次（1-based）
    step_turn_index: int = 1
    # 总轮次（1-based）
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
      - 每轮返回当前 step_id
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

    def _max_turns_for(self, step_id: str, warnings_out: List[str]) -> int:
        if step_id in self.policy.step_turns:
            return self.policy.step_turns[step_id]
        warnings_out.append(
            f"policies.step_turns missing for step_id='{step_id}', using defaults.step_turns={self.policy.default_step_turns}"
        )
        return self.policy.default_step_turns

    def get_current_step(self) -> Step:
        return self.steps[self.state.step_index]

    def decide(self) -> StepDecision:
        """
        返回当前应执行的 step 信息（不推进状态）。
        你可以在每轮对话开始时调用它，拿 step_id 去 PromptLoader 渲染。
        """
        warnings_out: List[str] = []

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
        _ = self._max_turns_for(step.id, warnings_out)  # 用于触发默认 warning（可选）
        return StepDecision(
            step_id=step.id,
            step_index=self.state.step_index,
            step_turn_index=self.state.step_turn_index,
            total_turn_index=self.state.total_turn_index,
            pages=list(step.pages),
            warnings=warnings_out,
        )

    def next_state(self) -> StepDecision:
        """
        推进“一轮对话”后的状态，并返回推进后的当前 step（即下一轮应执行的 step）。
        建议调用时机：
          - 每完成一轮（你定义的 turn）后调用一次。
        """
        warnings_out: List[str] = []

        if self.state.ended:
            # 已结束直接返回
            return StepDecision(
                step_id=None,
                step_index=self.state.step_index,
                step_turn_index=self.state.step_turn_index,
                total_turn_index=self.state.total_turn_index,
                pages=[],
                warnings=warnings_out,
            )

        current = self.get_current_step()
        max_turns = self._max_turns_for(current.id, warnings_out)

        # 先增加 total_turn
        self.state.total_turn_index += 1

        # step 内轮次推进
        if self.state.step_turn_index < max_turns:
            self.state.step_turn_index += 1
            nxt = self.get_current_step()
            return StepDecision(
                step_id=nxt.id,
                step_index=self.state.step_index,
                step_turn_index=self.state.step_turn_index,
                total_turn_index=self.state.total_turn_index,
                pages=list(nxt.pages),
                warnings=warnings_out,
            )

        # step 内轮次已达上限，切到下一个 step
        if self.state.step_index < len(self.steps) - 1:
            self.state.step_index += 1
            self.state.step_turn_index = 1
            nxt = self.get_current_step()
            return StepDecision(
                step_id=nxt.id,
                step_index=self.state.step_index,
                step_turn_index=self.state.step_turn_index,
                total_turn_index=self.state.total_turn_index,
                pages=list(nxt.pages),
                warnings=warnings_out,
            )

        # 已是最后一个 step
        if self.policy.end_behavior == "return_none":
            self.state.ended = True
            return StepDecision(
                step_id=None,
                step_index=self.state.step_index,
                step_turn_index=self.state.step_turn_index,
                total_turn_index=self.state.total_turn_index,
                pages=[],
                warnings=warnings_out,
            )

        # stay_last：保持最后一个 step
        self.state.step_turn_index = max_turns  # 保持在上限
        last = self.get_current_step()
        return StepDecision(
            step_id=last.id,
            step_index=self.state.step_index,
            step_turn_index=self.state.step_turn_index,
            total_turn_index=self.state.total_turn_index,
            pages=list(last.pages),
            warnings=warnings_out,
        )

    def get_prompt_params(self, decision: StepDecision, *, include_pdf_pages: bool = True) -> Dict[str, Any]:
        """
        给 PromptLoader 的 params（最简版）：
          - step_ids: 当前 step id
          - pdf_pages: 直接用 step.pages（如果有）
        """
        params: Dict[str, Any] = {}
        if decision.step_id is not None:
            params["step_ids"] = decision.step_id
        if include_pdf_pages and decision.pages:
            # 这里直接把 pages 作为 list[int] 传给 PromptLoader 的 pdf_pages selector（1-based）
            params["pdf_pages"] = decision.pages
        return params


# -------------------------
# Minimal demo
# -------------------------
if __name__ == "__main__":
    # 1) load from your lesson plan yaml (with policies filled)
    mgr = SimpleRuleDialogueManager.from_yaml("./70_smile.yaml")

    # 2) first decision (before any turn happens)
    d0 = mgr.decide()
    print("DECIDE:", d0)

    # 3) simulate a few turns
    for _ in range(10):
        d = mgr.next_state()
        print("NEXT:", d, "params:", mgr.get_prompt_params(d))


