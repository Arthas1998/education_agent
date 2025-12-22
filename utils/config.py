# utils/config.py
# -*- coding: utf-8 -*-
"""
全局配置文件（ASR / TTS / Chat）
"""

# =========================
# 基础参数
# =========================

# 最大上传文件大小（MB）
MAX_UPLOAD_MB = 10

# 允许上传的音频格式
ALLOWED_AUDIO_EXTS = [
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac"
]

# =========================
# ASR / TTS 超时参数
# =========================

# ASR 默认超时（秒）
ASR_TIMEOUT_S = 15

# TTS 默认超时（秒）
TTS_TIMEOUT_S = 15

# =========================
# 语音参数
# =========================

# TTS 默认音色
DEFAULT_VOICE = "female"

# =========================
# 引擎选择位
# =========================

"""
ENGINE:
    mock     - 本地模拟（不调用真实服务）
    realtime - 通义千问实时模型
    cloud    - 预留，将来扩展
"""
ENGINE = "realtime"

# =========================
# DashScope API Key
# =========================

"""
优先级（从高到低）：
1. 若在代码中填写 API_KEY，则使用该值；
2. 否则尝试从环境变量读取 DASHSCOPE_API_KEY；
3. 如果仍为空，则调用时抛异常。
"""

API_KEY = 'sk-0932fa1904874a43a9a7593e8441e30b'


# =========================
# ASR / TTS 模型
# =========================

ASR_MODEL_NAME = "fun-asr-realtime-2025-11-07"

TTS_MODEL_NAME = "cosyvoice-tts"   # 占位，tts.py 中可用


# =========================
# Chat 模块配置
# =========================

# DashScope OpenAI-Compatible Base URL
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 默认对话模型
DEFAULT_CHAT_MODEL = "qwen3-max"

# PromptLoader 默认路径
DEFAULT_PROMPT_PATH = r"prompt/config/zh/70_Smile.yaml"

# 是否只加载 prompt 中前 N 条 user
# None 表示全部
DEFAULT_INCLUDE_FIRST_N = None

# 是否启用 debug input 模式（input()）
DEBUG_MODE = True

# Streaming Options
DEFAULT_STREAM_OPTIONS = {
    "include_usage": True
}

CHAT_STREAM_PRINT = True

# =========================
# 运行时校验（启动即失败）
# =========================

def _validate():
    import os

    if ENGINE == "realtime":
        if not API_KEY and not os.getenv("DASHSCOPE_API_KEY"):
            raise RuntimeError(
                "DashScope API_KEY 未配置！请在 utils/config.py 中填写 API_KEY，"
                "或设置环境变量 DASHSCOPE_API_KEY"
            )

    if DEFAULT_CHAT_MODEL is None:
        raise RuntimeError("DEFAULT_CHAT_MODEL 未配置")

    if not DEFAULT_PROMPT_PATH:
        raise RuntimeError("DEFAULT_PROMPT_PATH 未配置")

_validate()
