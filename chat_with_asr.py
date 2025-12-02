# test_asr_chat.py
# -*- coding: utf-8 -*-

"""
启动流程：
1. 加载 system + initial user prompt
2. 让模型先讲课（startup）
3. 启动 ASR
4. 每识别一句完整话 -> Chat
"""

import threading
import queue

from utils.asr import RealtimeASRSession
from utils.chat import Chat
from utils.prompt import PromptLoader
from utils.config import (
    API_KEY,
    BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_PROMPT_PATH,
)

from openai import OpenAI


# =====================
# 句子收集器
# =====================

class SentenceCollector:
    def __init__(self):
        self.queue = queue.Queue()

    def feed(self, result: dict):
        if result.get("end_time") is not None:
            text = result.get("text", "").strip()
            if text:
                self.queue.put(text)

    def wait(self) -> str:
        return self.queue.get()


# =====================
# 主程序
# =====================

def main():
    print("=== 启动 ASR 讲课系统 ===")

    # ===== Chat 初始化 =====
    loader = PromptLoader(DEFAULT_PROMPT_PATH)

    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    chat = Chat(
        client=client,
        prompt_loader=loader,
        model=DEFAULT_CHAT_MODEL,
        debug_mode=False
    )

    # ===== 模型先讲课 =====
    print("\n[系统] 正在加载讲课内容...\n")
    chat.startup()

    # ===== ASR 初始化 =====
    session = RealtimeASRSession()
    session.start()

    collector = SentenceCollector()

    # ===== 麦克风输入 =====
    import pyaudio

    RATE = 16000
    CHANNELS = 1
    CHUNK = 3200

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    def audio_feeder():
        print("[ASR] 开始说话")
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                session.send_audio(data)
        finally:
            session.stop()
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def asr_reader():
        for result in session.stream():
            print("[ASR Raw]", result)
            collector.feed(result)

    threading.Thread(target=audio_feeder, daemon=True).start()
    threading.Thread(target=asr_reader, daemon=True).start()

    # ===== 多轮对话 =====
    print("\n系统已就绪，请开始对话...\n")

    try:
        while True:
            user_text = collector.wait()
            print(f"\n你说: {user_text}")

            chat.add_user_text(user_text)

            print("Assistant:", end=" ")
            for _ in chat.stream_reply():
                pass

            print("\n--------------------------")

    except KeyboardInterrupt:
        print("\n退出系统")
        session.stop()


if __name__ == "__main__":
    main()
