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

    pdf_path = r"D:\data\PythonProject\HITProject\education_agent\textbook\57_HeRuns.pdf"
    image_paths = pdf_to_images(pdf_path)
    image_contents = make_image_contents([image_paths[0]])  # 假设只用第一页

    system_prompt = """
你是一名专注于57_HeRuns.pdf（Level A英语分级读物）的低幼英语启蒙教师，熟悉该教材每一页的插画细节（由Darcy Tom绘制）与课文句子，擅长通过“插画观察→句子讲解→互动提问”的逻辑引导5-6岁学生，能精准结合已输入的教材图片序列开展教学，不涉及任何其他教材或无关内容。

### 教材背景
1. 57_HeRuns.pdf是Level A分级读物，作者Ned Jensen，核心句型“ He runs to the + 交通工具/地点 ”，共8个核心句子（对应8页图片序列）：He runs to the bus（）、He runs to the train（）、He runs to the boat（）、He runs to the plane（）、He runs to the school（）、He runs to the store（）、He runs to the pool（）、He runs home（）。
2. 已将教材逐页转为图片输入模型，每一页图片含“对应课文句子+具象化插画”（如“bus页”含“He runs to the bus”文字与“主角跑向带‘BUS’标识的公交车”插画），教学需完全依托当前页图片，不提前透露后续页面。
3. 学生仅掌握10-15个基础英语词汇，需用简单中文辅助，避免复杂术语。

### 教学指令
1. 逐页教学：从第1页“bus”开始，每一页分3轮对话：第1轮引导观察插画并提问；第2轮结合插画讲句子、带读并解词汇；第3轮设计“插画+句子”巩固提问，学生回答后正向反馈再过渡。
2. 上下文维护：每轮关联上一轮内容，逻辑连贯。
3. 内容限定：仅基于57_HeRuns.pdf，不添加教材外内容。

### 教学风格
- 语气亲切耐心，多用“呀”“哦”等语气词；
- 短句为主，中文简单，英语带读缓慢；
- 每轮含提问，反馈以鼓励为主。

### 第一轮输出要求
针对第1页“bus”图片，先引导观察插画并提问，再引入句子带读解词，最后设计1个巩固问题。
    """

    # 多轮对话历史，初始加入 system prompt 和第一页图片
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": "现在请基于已输入的57_HeRuns.pdf第1页图片（含“He runs to the bus”文字与主角跑向公交车的插画），按照设定的教学逻辑，开始第一轮教学吧～",
        }
    ]
    # 可按需将 image_contents 加入 messages，这里假设不用 image_contents（如需添加可加进 user.content）

    print("多轮对话开始，输入 exit 结束。\n")

    round_num = 1
    while True:
        # 1. 流式打印模型回复（不自动换行）
        print(f"模型：", end="")  # 保持模型同一行输出
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
            if chunk.choices and hasattr(chunk.choices[0], "delta"):
                delta = chunk.choices[0].delta
                if isinstance(delta, dict) and "content" in delta:
                    print(delta["content"], end="", flush=True)
                    reply_text += delta["content"]
                elif isinstance(delta, str):
                    print(delta, end="", flush=True)
                    reply_text += delta
            # usage信息不打印
        print()  # 本轮模型回复后换行

        # 2. 将模型回复加入 messages
        messages.append({
            "role": "assistant",
            "content": reply_text,
        })

        # 3. 用户输入
        user_input = input("用户：")
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