# -*- coding: utf-8 -*-
"""
utils/chat.py

封装一个多轮文本对话 Chat 类：
- 使用 PromptLoader 初始化 system + initial user
- 支持流式转发回复
- 支持 input() 或外部文本驱动
- 提供读取最新 assistant 完整输出接口
- 可直接运行：python -m utils.chat
"""

import os
from typing import Optional, Callable, List, Dict, Any, Iterator

# ===== 统一包式导入，禁止相对路径 =====
from utils.prompt import PromptLoader
from utils.config import (
    API_KEY,
    BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_STREAM_OPTIONS,
    DEBUG_MODE,
    DEFAULT_PROMPT_PATH,
    DEFAULT_INCLUDE_FIRST_N,
)


# utils/chat.py
# -*- coding: utf-8 -*-

from openai import OpenAI
from typing import List, Dict, Any, Optional, Iterable

from utils.config import DEFAULT_CHAT_MODEL, CHAT_STREAM_PRINT
from utils.prompt import PromptLoader


class Chat:
    """
    多轮对话封装类
    """

    def __init__(
        self,
        client: OpenAI,
        prompt_loader: PromptLoader,
        use_textbook: bool = False,
        model: str = DEFAULT_CHAT_MODEL,
        debug_mode: bool = False,
    ):
        self.client = client
        self.model = model
        self.debug = debug_mode
        self.prompt_loader = prompt_loader

        # 消息池
        self.messages: List[Dict[str, Any]] = []

        # ===== 加载系统提示词 =====
        system_msg = self.prompt_loader.load_system_prompt()
        self.messages.append(system_msg)

        # ===== 加载初始用户提示词 =====
        init_user_msg = self.prompt_loader.load_initial_user_prompt(
            use_textbook=use_textbook,
        )
        self.messages.append(init_user_msg)

        if self.debug:
            print("[DEBUG] System Prompt Loaded")
            # print(system_msg)
            print("[DEBUG] Initial User Prompt Loaded")
            # print(init_user_msg)

    # ======================
    # 基础接口
    # ======================

    def add_user_text(self, text: str):
        """
        正常对话时，用户输入转为 messages
        """
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
        })

    # ======================
    # 模型回复
    # ======================

    def stream_reply(self) -> Iterable[str]:
        """
        流式向模型请求回复，并实时输出
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True,
        )

        full = ""

        for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                text = delta.content
                full += text
                if CHAT_STREAM_PRINT:
                    print(text, end="", flush=True)
                yield text

        # 把 assistant 回复加入 messages
        self.messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full}]
        })

    def get_last_assistant_reply(self) -> Optional[str]:
        """
        读取最近一条 assistant 消息
        """
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                content = msg.get("content", [])
                if content and content[0].get("type") == "text":
                    return content[0].get("text")
        return None

    # ======================
    # 启动第一轮（讲课）
    # ======================

    def startup(self):
        """
        使用系统提示词 + 初始用户提示词
        触发首次模型讲课（只调用一次）
        """
        print("Assistant:", end=" ")
        for _ in self.stream_reply():
            pass
        print("\n------ 进入对话模式 ------\n")

    # ======================
    # Debug 交互循环
    # ======================

    def run(self):
        """
        调试模式用：input 多轮对话
        """
        self.startup()

        while True:
            user = input("\n你: ").strip()
            if not user:
                continue
            self.add_user_text(user)
            print("Assistant:", end=" ")
            for _ in self.stream_reply():
                pass
            print()


# =========================
# 命令行测试入口
# =========================
if __name__ == "__main__":
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("请先安装 openai SDK")

    # 初始化 PromptLoader
    loader = PromptLoader(DEFAULT_PROMPT_PATH)

    # 构造 OpenAI Compatible Client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    # 创建 Chat 实例
    chat = Chat(
        client=client,
        prompt_loader=loader,
        use_textbook=False,
        model=DEFAULT_CHAT_MODEL,
        debug_mode=True,
    )

    # 启动对话
    chat.run()
