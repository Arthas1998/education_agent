# dashscope SDK 版本需不低于 1.23.9
import os
import base64
import signal
import sys
import time
import pyaudio
from dashscope.audio.qwen_omni import *
import dashscope

# 新增：导入PromptComposer
from utils.prompt import PromptComposer

DASHSCOPE_API_KEY = 'sk-21a49acda5994dadad615d4c7e549bc5'
# 如果没有设置环境变量，请用您的 API Key 将下行替换为dashscope.api_key = "sk-xxx"
dashscope.api_key = DASHSCOPE_API_KEY
conversation = None

class MyCallback(OmniRealtimeCallback):
    def __init__(self):
        super().__init__()
        self.bot_started = False
        self.last_user_input = ""
        self.first_bot_done = False  # 新增：标志首轮Bot回答是否完成

    def on_open(self) -> None:
        global pya
        global mic_stream
        print('connection opened, init microphone')
        pya = pyaudio.PyAudio()
        mic_stream = pya.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=16000,
                              input=True)

    def on_close(self, close_status_code, close_msg) -> None:
        print('connection closed with code: {}, msg: {}, destroy microphone'.format(close_status_code, close_msg))
        sys.exit(0)

    def on_event(self, response: str) -> None:
        try:
            type = response['type']
            if type == 'session.created':
                print('start session: {}'.format(response['session']['id']))

            # 用户语音转写完成，打印语音输入内容
            if type == 'conversation.item.input_audio_transcription.completed':
                user_text = response.get('transcript', '').strip()
                if user_text:
                    self.last_user_input = user_text
                    print(f"\nUser: {user_text}")

            # Bot流式输出
            if type == 'response.audio_transcript.delta':
                text = response['delta']
                if not self.bot_started:
                    print("Bot:", end='', flush=True)
                    self.bot_started = True
                print(text, end='', flush=True)

            # Bot输出完成，换行并重置标志
            if type == 'response.done':
                if self.bot_started:
                    print()
                self.bot_started = False

            if type == 'input_audio_buffer.speech_started':
                print('======VAD Speech Start======')
        except Exception as e:
            print('[Error] {}'.format(e))
            return

if __name__  == '__main__':
    print('Initializing ...')

    # 新增：读取prompt
    TEMPLATE_FILE = r"D:\data\project\qwen\templates\teaching.yaml"
    TEXTBOOK_FILE = r"D:\data\project\textbook\57_HeRuns.json"
    PAGE_NUMBER = 3  # 可根据实际需求调整
    composer = PromptComposer(template_file=TEMPLATE_FILE)
    textbook_data = composer.load_textbook(TEXTBOOK_FILE)
    page_info = composer.extract_page_info(textbook_data, page_number=PAGE_NUMBER)
    prompt_text = composer.compose_prompt(page_info)

    callback = MyCallback()
    conversation = OmniRealtimeConversation(
        model='qwen-omni-turbo-realtime-latest',
        callback=callback,
    )
    conversation.connect()
    conversation.update_session(
        output_modalities=[MultiModality.TEXT],
        voice="Cherry",
        input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
        enable_input_audio_transcription=True,
        input_audio_transcription_model='gummy-realtime-v1',
        enable_turn_detection=True,
        turn_detection_type='server_vad',
        system_prompt=prompt_text,  # 你的提示词
    )

    # 新增：先发送prompt文本
    # conversation.send_text(prompt_text)  # 修正方法名为 send_text

    def signal_handler(sig, frame):
        print('Ctrl+C pressed, stop recognition ...')
        conversation.close()
        print('omni realtime stopped.')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    print("Press 'Ctrl+C' to stop conversation...")

    # 修改：不再限制语音输入，始终允许麦克风输入
    while True:
        if mic_stream:
            audio_data = mic_stream.read(3200, exception_on_overflow=False)
            audio_b64 = base64.b64encode(audio_data).decode('ascii')
            conversation.append_audio(audio_b64)
        else:
            break