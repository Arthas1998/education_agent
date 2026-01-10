# utils/chat.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Dict, Any, Optional, Iterable, cast

from utils.prompt_loader import PromptLoader, RenderWarning

model_name = "gpt-4"  # 默认模型名称，可根据需要修改


def apply_dialogue_decision_to_chat_params(chat: "Chat", decision: Any) -> None:
    """把“对话调度器的决策”写入 chat.params。

    设计目标：
    - 适配层尽量薄，只做字段映射，不引入复杂校验。
    - 规则管理器只需提供常见字段：step_id / pages（或 step_ids / pdf_pages）。

    约定写入：
    - params.step_ids
    - params.pdf_pages

    注意：需要在下一次 chat.add_user_text(...) 调用（即渲染 user 提示词）之前调用。
    """
    if decision is None:
        return

    # 兼容 StepDecision(step_id, pages)
    step_id = getattr(decision, "step_id", None)
    pages = getattr(decision, "pages", None)

    # 也兼容直接给出 step_ids/pdf_pages 的对象
    step_ids = getattr(decision, "step_ids", None)
    pdf_pages = getattr(decision, "pdf_pages", None)

    if step_ids is None and step_id is not None:
        step_ids = step_id
    if pdf_pages is None and pages is not None:
        pdf_pages = pages

    if step_ids is not None:
        chat.params["step_ids"] = step_ids
    if pdf_pages is not None:
        chat.params["pdf_pages"] = pdf_pages
    if pages is not None:
        chat.params["pages"] = pages


class Chat:
    """多轮对话封装类（适配新版 PromptLoader.render_message）。

    约定：
    - system 提示词仅渲染一次
    - user 提示词每轮都渲染（将真实用户输入渲染进模板）
    - 业务层通过直接赋值 chat.params = {...} 决定每轮渲染的教案/页码等
    - 对外接口保持不变：add_user_text/stream_reply/get_last_assistant_reply/is_started/set_started
    """

    def __init__(
        self,
        client: Any,
        prompt_loader: PromptLoader,
        model: str = None,
        debug_mode: bool = False,
        use_textbook: bool = False,
        include_first_n: Optional[int] = None,
    ):
        self.client = client
        self.model = model or model_name
        self.debug = debug_mode
        self.prompt_loader = prompt_loader
        self.include_first_n = include_first_n

        # 当前对话轮次数（每次渲染并追加 user 提示词后 +1）
        # 说明：这是“每个 Chat 会话实例”的计数，而非全局共享。
        self.turn_count: int = 1

        # 业务层可直接覆盖/替换该 dict：chat.params = {...}
        self.params: Dict[str, Any] = {}

        # 内部可选：给 prompt_loader.registry runtime/memory 提供数据
        self._memory: Dict[str, Any] = {}

        # 消息池
        self.messages: List[Dict[str, Any]] = []

        # 是否已完成首次启动（讲课）
        self._started = False

        # system 是否已渲染（每个对话只做一次）
        self._system_rendered = False

        # 旧参数保留（不再在 Chat 内使用）；避免调用方传参报错
        _ = use_textbook

    @staticmethod
    def _normalize_openai_message(msg: Dict[str, Any]) -> Dict[str, Any]:
        """把 render_message 输出统一成 OpenAI parts 结构。

        OpenAI chat.completions:
        - 文本消息可以是 content: str
        - 也可以是 content: [{type:'text',text:'...'}, {type:'image_url',...}]

        为了简化下游处理，这里统一转为 parts list。
        """
        if not isinstance(msg, dict):
            raise TypeError(f"Expected message dict, got: {type(msg)}")
        role = msg.get("role")
        if not isinstance(role, str):
            raise ValueError("Message missing 'role'.")

        content = msg.get("content")
        if isinstance(content, str):
            msg = dict(msg)
            msg["content"] = [{"type": "text", "text": content}]
            return msg

        if isinstance(content, list):
            # assume already parts
            return msg

        raise ValueError(f"Unsupported message content type: {type(content)}")

    def _ensure_system_message(self) -> None:
        if self._system_rendered:
            return

        warnings_out: List[RenderWarning] = []
        system_msg = self.prompt_loader.render_message(
            template="generator",
            message="system",
            runtime_vars={},
            memory=self._memory,
            params=self.params,
            warnings_out=warnings_out,
        )
        self.messages.append(self._normalize_openai_message(system_msg))
        self._system_rendered = True

        if self.debug and warnings_out:
            # 不改变接口：仅在 debug_mode 下打印，帮助定位缺参/selector 未命中
            for w in warnings_out:
                print(f"[PromptWarning][{w.code}] {w.message} ctx={w.context}")

    def add_user_text(self, text: str):
        """添加用户文本消息（每一轮都通过 PromptLoader 渲染 user 模板）。"""
        self._ensure_system_message()

        warnings_out: List[RenderWarning] = []
        user_msg = self.prompt_loader.render_message(
            template="generator",
            message="user",
            runtime_vars={"student_answer": text},
            memory=self._memory,
            params=self.params,
            warnings_out=warnings_out,
        )
        self.messages.append(self._normalize_openai_message(user_msg))

        # 轮次计数：每次 user 提示词渲染完成后递增
        self.turn_count += 1

        if self.debug and warnings_out:
            for w in warnings_out:
                print(f"[PromptWarning][{w.code}] {w.message} ctx={w.context}")

    def stream_reply(self) -> Iterable[str]:
        """流式向模型请求回复"""
        self._ensure_system_message()

        create_fn = cast(Any, self.client.chat.completions.create)
        response = create_fn(
            model=self.model,
            # OpenAI SDK 的类型定义比较严格，这里保持运行时兼容即可
            messages=cast(Any, self.messages),
            stream=True,
            stream_options={"include_usage": True},
        )

        full = ""
        for chunk in response:
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    text = delta.content
                    full += text
                    yield text

        # 把 assistant 回复加入 messages
        self.messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": full}],
            }
        )

    def get_last_assistant_reply(self) -> Optional[str]:
        """读取最近一条 assistant 消 Messages"""
        for msg in reversed(self.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                        texts.append(part["text"])
                return "".join(texts) if texts else None
        return None

    def is_started(self) -> bool:
        """是否已完成首次启动"""
        return self._started

    def set_started(self):
        """标记已完成首次启动"""
        self._started = True

