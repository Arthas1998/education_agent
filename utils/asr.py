import os
import sys
import time
import queue

import dashscope
import pyaudio
import webrtcvad
from dashscope.audio.asr import *

# -------------------------------
# Config
# -------------------------------
DASHSCOPE_API_KEY = 'sk-21a49acda5994dadad615d4c7e549bc5'
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
VAD_MODE = 1
SILENCE_DURATION = 1.0  # seconds

FRAME_DURATION_MS = 30  # 10/20/30ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
FRAME_BYTES = FRAME_SIZE * 2  # 16-bit PCM → 2 bytes per sample

# -------------------------------
# DashScope
# -------------------------------
dashscope.api_key = DASHSCOPE_API_KEY

# -------------------------------
class Callback(RecognitionCallback):
    def __init__(self):
        super().__init__()
        self.transcribed_text = ""
        self.sentence_end = False

    def on_open(self):
        print("RecognitionCallback open.")

    def on_close(self):
        print("RecognitionCallback close.")

    def on_event(self, result):
        sentence = result.get_sentence()
        text = sentence.get('text')
        if text:
            self.transcribed_text += text + " "
            if RecognitionResult.is_sentence_end(sentence):
                self.sentence_end = True

    def on_error(self, message):
        print("RecognitionCallback error:", message.message)

# -------------------------------
def record_and_transcribe():
    mic = pyaudio.PyAudio()
    stream = mic.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=SAMPLE_RATE,
                      input=True,
                      frames_per_buffer=FRAME_SIZE)

    vad = webrtcvad.Vad(VAD_MODE)
    callback = Callback()
    recognition = Recognition(
        model='fun-asr-realtime',
        format='pcm',
        sample_rate=SAMPLE_RATE,
        semantic_punctuation_enabled=False,
        callback=callback
    )
    recognition.start()

    print("Start recording... Speak now.")

    silent_chunks = 0
    max_silent_chunks = int(SILENCE_DURATION * 1000 / FRAME_DURATION_MS)

    try:
        while True:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            if len(data) != FRAME_BYTES:
                continue  # 跳过异常长度帧

            # VAD 判断
            try:
                is_speech = vad.is_speech(data, SAMPLE_RATE)
            except webrtcvad.Error:
                continue  # 跳过异常帧

            # 发送给模型
            recognition.send_audio_frame(data)

            if not is_speech:
                silent_chunks += 1
            else:
                silent_chunks = 0

            # 静音或句子结束
            if silent_chunks > max_silent_chunks or callback.sentence_end:
                print("Silence detected or sentence end, stopping...")
                break

    finally:
        stream.stop_stream()
        stream.close()
        mic.terminate()
        recognition.stop()

    return callback.transcribed_text.strip()


if __name__ == "__main__":
    text = record_and_transcribe()
    print("Transcribed text:", text)
