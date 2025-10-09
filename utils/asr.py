# _*_ coding: utf-8 _*_
# @File:    asr_qwen.py
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.3

"""
基于 silero-vad 的自动语音分段+通义千问语音识别（ASR）模块。
可直接 import 并在其他文件中调用 record_and_transcribe() 实现实时语音转写。
"""

import pyaudio
import wave
import openai
import os
import uuid
import torch
import numpy as np
from silero import vad  # pip install silero-vad

API_KEY = "你的通义千问API-KEY"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_MS = 30  # 每帧 ms
CHUNK = int(RATE * CHUNK_MS / 1000)
MAX_SILENCE_MS = 1000  # 语音结束静默阈值（ms）

def write_wave(path, audio, sample_rate):
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(audio))
    wf.close()

def qwen_asr(audio_file_path):
    """调用通义千问的API进行语音转文字"""
    client = openai.OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    with open(audio_file_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
            language="zh"
        )
    return str(response).strip()

def record_and_transcribe(prompt="请开始说话（自动检测停顿，结束自动转写）："):
    """silero-vad自动VAD分段录音并识别，返回识别文本。"""
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)
    # print(prompt)
    voiced_frames = []
    silence_ms = 0
    triggered = False

    # 初始化silero-vad模型
    model, utils = vad.load_vad_model(torch.device('cpu'))
    get_speech_ts = utils['get_speech_ts']
    vad_iterator = vad.StreamingVAD(model, sampling_rate=RATE)

    try:
        while True:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            # 转为float32（silero-vad要求归一化到-1~1）
            audio_np = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
            is_speech = vad_iterator(audio_np, return_seconds=False)
            if is_speech:
                voiced_frames.append(frame)
                silence_ms = 0
                triggered = True
            else:
                if triggered:
                    silence_ms += CHUNK_MS
                    if silence_ms > MAX_SILENCE_MS:
                        break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if not voiced_frames:
        print("未检测到语音，请重试。")
        return ""
    # 保存临时音频文件
    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    write_wave(temp_wav, voiced_frames, RATE)
    print("正在识别...")
    text = qwen_asr(temp_wav)
    os.remove(temp_wav)
    print("识别结果：", text)
    return text

# 测试用
if __name__ == "__main__":
    while True:
        record_and_transcribe()