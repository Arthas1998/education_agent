# utils/asr.py
# -*- coding: utf-8 -*-

import os
import time
import queue
import threading

from utils.config import (
    ENGINE,
    API_KEY,
    ASR_MODEL_NAME,
)

# ======================
# 注入 DashScope API Key
# ======================
if API_KEY:
    import dashscope
    dashscope.api_key = API_KEY
elif os.getenv("DASHSCOPE_API_KEY"):
    import dashscope
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

try:
    from dashscope.audio.asr import (
        Recognition,
        RecognitionCallback,
        RecognitionResult,
    )
    from dashscope import api_resources
    _HAS_DASHSCOPE = True
except Exception:
    Recognition = None
    RecognitionCallback = object
    RecognitionResult = None
    api_resources = None
    _HAS_DASHSCOPE = False


class ASREngineError(Exception):
    pass


# =========================
# 实时流式识别 Callback
# =========================
class _RealtimeCallback(RecognitionCallback):

    def __init__(self, queue: queue.Queue):
        self.queue = queue
        self.text_buffer = []

    def on_open(self):
        print("[ASR] 连接建立")

    def on_close(self):
        print("[ASR] 连接关闭")
        self.queue.put(None)

    def on_event(self, result: RecognitionResult):
        try:
            sentence = result.get_sentence()
            if sentence:
                self.text_buffer.append(sentence)
                self.queue.put(sentence)
        except Exception:
            pass


# =========================
# 实时流式 ASR Session
# =========================
class RealtimeASRSession:

    def __init__(self):
        self.queue = queue.Queue()
        self.callback = _RealtimeCallback(self.queue)
        self.recognition = None

    def start(self):
        if ENGINE != "realtime":
            raise ASREngineError("ENGINE 必须为 realtime")

        if not _HAS_DASHSCOPE:
            raise ASREngineError("DashScope SDK 不可用")

        self.recognition = Recognition(
            model=ASR_MODEL_NAME,
            format="pcm",
            sample_rate=16000,
            callback=self.callback
        )
        self.recognition.start()

    def stop(self):
        if self.recognition:
            self.recognition.stop()
            self.recognition = None

    def send_audio(self, chunk: bytes):
        if not self.recognition:
            raise RuntimeError("识别未启动")
        self.recognition.send_audio_frame(chunk)

    def stream(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            yield item

    def get_text(self):
        return "".join(self.callback.text_buffer)


# =========================
# 文件 ASR（新增）
# =========================
class FileASRSession:
    """
    用于直接识别音频文件（wav/mp3/m4a）
    """

    def __init__(self, model=None):
        self.model = model or ASR_MODEL_NAME

    def recognize(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if not _HAS_DASHSCOPE:
            raise ASREngineError("DashScope SDK 不可用")

        print(f"[ASR] 开始识别文件: {audio_path}")

        response = api_resources.AudioRecognition.call(
            model=self.model,
            file_path=audio_path,
            format="auto"
        )

        if response.status_code != 200:
            raise RuntimeError(f"ASR 失败: {response.message}")

        result = response.output.get("sentences", [])
        texts = []

        for sent in result:
            text = sent.get("text", "")
            if text:
                texts.append(text)

        return "".join(texts)


# =========================
# 统一外部接口（推荐）
# =========================
def recognize_file(audio_path: str) -> str:
    """
    给外部模块使用的统一文件识别接口
    """
    engine = FileASRSession()
    return engine.recognize(audio_path)


# =========================
# 脚本测试入口
# =========================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="识别音频文件路径")
    args = parser.parse_args()

    # --- 文件识别模式 ---
    if args.file:
        print("=== 文件 ASR 模式 ===")
        text = recognize_file(args.file)
        print("\n识别结果：")
        print(text)
        exit(0)

    # --- 默认：实时麦克风模式 ---
    print("=== 启动实时麦克风 ASR ===")

    try:
        import pyaudio
    except Exception:
        raise RuntimeError("请先安装 pyaudio：pip install pyaudio")

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

    session = RealtimeASRSession()
    session.start()

    def feeder():
        print("开始说话（Ctrl+C 停止）...")
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                session.send_audio(data)
        finally:
            session.stop()
            stream.stop_stream()
            stream.close()
            audio.terminate()

    threading.Thread(target=feeder, daemon=True).start()

    try:
        for text in session.stream():
            print("[识别]", text)
    except KeyboardInterrupt:
        print("退出中...")
        session.stop()
