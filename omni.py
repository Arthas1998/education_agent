# _*_ coding: utf-8 _*_
# @File:    omni
# @Time:    2025/10/9 12:21
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
import os
import base64
from openai import OpenAI
import fitz  # PyMuPDF

# 新增：导入语音转写工具
from utils.asr import record_and_transcribe

def pdf_to_images(pdf_path, output_folder="pdf_images", dpi=144):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for idx, page in enumerate(doc):
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"page_{idx+1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    return image_paths

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def make_image_contents(image_paths):
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
        }
        for path in image_paths
    ]

if __name__ == "__main__":
    client = OpenAI(
        api_key='sk-21a49acda5994dadad615d4c7e549bc5',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # pdf_path = r"D:\data\PythonProject\HITProject\education_agent\textbook\57_HeRuns.pdf"
    # image_paths = pdf_to_images(pdf_path)
    image_paths = [r"pdf_images\page_1.png", r"pdf_images\page_2.png", r"pdf_images\page_3.png", r"pdf_images\page_4.png", r"pdf_images\page_5.png"]
    image_contents = make_image_contents(image_paths[0:4])  # 假设只用第一页

    system_prompt = """
你是专注于57_HeRuns.pdf（Level A英语分级读物）的低幼英语启蒙教师，输出内容仅为数字人可直接语音的“讲解/提问/反馈”，无动作描述、无括号标注、无多余文字，每轮仅推进单一步骤，需等待学生回答后再下一轮。

### 教材背景
57_HeRuns.pdf共8页图片，第1页对应“He runs to the bus”（插画：主角跑向带“BUS”标识的公交车），第2页对应“He runs to the train”…第8页对应“He runs home”，教学仅围绕当前页图片，不提前透露其他页面。

### 教学指令
1. 第1页分3轮对话，每轮仅输出“单一步骤”：
   - 第1轮：仅输出“引导观察插画的提问”（不讲解句子）；
   - 学生回答后，第2轮：仅输出“插画关联+句子带读+词汇解释”（不带巩固提问）；
   - 学生回答第2轮内容后，第3轮：仅输出“巩固提问”，学生回答后再输出“正向反馈”。
2. 所有输出为短句，无长段落，无“等待回答”“指向插画”等标注，不提前说后续步骤。

### 教学风格
语气亲切，用“呀”“哦”等语气词，仅含语音可用内容，无任何多余文字。
        """

    # 多轮对话历史，仅加入 system prompt
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    # 初始 user 输入内容
    initial_user_content = "请输出57_HeRuns.pdf第1页的第1轮教学内容——仅引导学生观察插画的提问。"
    round_num = 1
    # DEBUG_MODE = True  # True=命令行输入，False=语音输入
    DEBUG_MODE = False  # True=命令行输入，False=语音输入

    first_round = True  # 新增：首轮标志

    while True:
        # 首轮时，先加入初始 user 输入
        if first_round:
            messages.append({
                "role": "user",
                "content": initial_user_content,
            })
            first_round = False

        # 1. 流式打印模型回复（不自动换行）
        print(f"模型：", end="")
        completion = client.chat.completions.create(
            model="qwen3-omni-flash",
            messages=messages,
            modalities=["text"],
            audio={"voice": "Cherry", "format": "wav"},
            stream=True,
            stream_options={"include_usage": True},
        )

        reply_text = ""
        for chunk in completion:
            if chunk.choices and hasattr(chunk.choices[0].delta, "content"):
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
                    reply_text += content
            # if chunk.choices and hasattr(chunk.choices[0], "delta"):
            #     delta = chunk.choices[0].delta
            #     if isinstance(delta, dict) and "content" in delta:
            #         print(delta["content"], end="", flush=True)
            #         reply_text += delta["content"]
            #     elif isinstance(delta, str):
            #         print(delta, end="", flush=True)
            #         reply_text += delta
            # usage信息不打印
        print()  # 本轮模型回复后换行

        # 2. 将模型回复加入 messages
        messages.append({
            "role": "assistant",
            "content": reply_text,
        })

        # 3. 用户输入（根据调试模式选择）
        if DEBUG_MODE:
            user_input = input("你：")
        else:
            user_input = record_and_transcribe()
        if user_input.strip().lower() == "exit":
            print("对话结束。")
            break
        print()  # 用户回复后换行

        # 4. 用户回复加入 messages
        messages.append({
            "role": "user",
            "content": user_input,
        })

        round_num += 1