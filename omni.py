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
    output = []
    for path in image_paths:
        item = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
        }
        output.append(item)
    return output

if __name__ == "__main__":
    client = OpenAI(
        api_key='sk-21a49acda5994dadad615d4c7e549bc5',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # pdf_path = r"D:\data\PythonProject\HITProject\education_agent\textbook\57_HeRuns.pdf"
    # image_paths = pdf_to_images(pdf_path)
    image_paths = [
        r"pdf_images\page_1.png", r"pdf_images\page_2.png", r"pdf_images\page_3.png", r"pdf_images\page_4.png", r"pdf_images\page_5.png",
        r"pdf_images\page_6.png", r"pdf_images\page_7.png", r"pdf_images\page_8.png", r"pdf_images\page_9.png", r"pdf_images\page_10.png",
        r"pdf_images\page_11.png", r"pdf_images\page_12.png"
    ]
    image_contents = make_image_contents(image_paths[0:4])  # 假设只用第一页

    system_prompt = """
你是低幼英语启蒙教师，能针对任意低幼英语绘本开展多轮对话教学，每轮可输出多句话详细讲解（如词汇含义、词性、用法），但每轮仅聚焦1个知识点，需等待学生回应后再推进，并基于前课总结衔接内容，课后生成通用总结。

### 核心信息
1. 前一节课总结：无；
2. 本节课目标：学习当前绘本的核心词汇[如：run]及核心句型[如：He runs to the + 地点]，巩固前课薄弱点。
3. 步骤规则：
   - 复习：1-2轮，每轮聚焦1个前课知识点（提问→详细反馈+拓展），无前一课总结时直接开始新内容讲解；
   - 新内容：逐页进行教学，首先根据每页的内容进行讲解，基于这一课核心内容详细讲解新知识点（词汇含义→用法→例句/句型拆解→巩固）；
   - 总结：2轮回顾→下课，课后生成含“前课巩固、新内容掌握、建议”的总结。
4. 输出要求：每轮3-6句话，通俗讲解，不跨步骤，适配低幼理解水平。所有输出为短句，无长段落，无“等待回答”“指向插画”等标注，不提前说后续步骤和后一页的内容。讲解语法、句型时可以适当增加段落长度。注意，你返回的内容将直接作为数字人的语音输出，因此你不需要任何非讲解性质的文字说明。
        """

    # 多轮对话历史，仅加入 system prompt
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        }
    ]

    # 初始 user 输入内容
    initial_user_content = "开始今天的英语课吧，教材是这本新绘本"

    image_contents.append(
        {
            "type": "text",
            "text": initial_user_content
        },
    )

    round_num = 1
    DEBUG_MODE = True  # True=命令行输入，False=语音输入
    # DEBUG_MODE = False  # True=命令行输入，False=语音输入

    first_round = True  # 首轮标志

    while True:
        # 首轮时，加入初始 user 输入 + 图片内容
        if first_round:
            messages.append({
                "role": "user",
                "content": image_contents,
            })
            first_round = False
        else:
            # 之后用户的输入只包含文字或语音识别结果
            messages.append({
                "role": "user",
                "content": user_input,
            })

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
        print()  # 模型回复后换行

        # 2. 记录模型回复
        messages.append({
            "role": "assistant",
            "content": reply_text,
        })

        # 3. 用户输入（根据调试模式）
        if DEBUG_MODE:
            user_input = input("你：")
        else:
            user_input = record_and_transcribe()
        if user_input.strip().lower() == "exit":
            print("对话结束。")
            break
        print()