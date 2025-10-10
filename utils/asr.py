# _*_ coding: utf-8 _*_
# @File:    asr.py
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.5

"""
基于 silero-vad 的自动语音分段+通义千问语音识别（ASR）模块。
record_and_transcribe()：检测到连续静音（如1秒）则自动结束录音并转写。
无任何打印输出。静音结束是一次语音输入的结束，不影响对话主循环。
"""

import pyaudio
import wave
import openai
import os
import uuid
import torch
import numpy as np
import silero_vad as vad  # pip install silero-vad

API_KEY = "sk-21a49acda5994dadad615d4c7e549bc5"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_MS = 30  # 每帧 ms
CHUNK = int(RATE * CHUNK_MS / 1000)
MAX_SILENCE_MS = 100000  # 连续静音多长(ms)判定为一次输入结束

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

def record_and_transcribe():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK)
    voiced_frames, silence_ms, triggered = [], 0, False

    model = vad.load_silero_vad()
    vad_iterator = vad.VoiceActivityDetector(model, sampling_rate=RATE)

    try:
        while True:
            frame = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
            if vad_iterator(audio_np):
                voiced_frames.append(frame)
                silence_ms = 0
                triggered = True
            elif triggered:
                silence_ms += CHUNK_MS
                if silence_ms >= MAX_SILENCE_MS:
                    break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if not voiced_frames:
        return ""
    temp_wav = f"temp_{uuid.uuid4().hex}.wav"
    write_wave(temp_wav, voiced_frames, RATE)
    try:
        text = qwen_asr(temp_wav)
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    return text
