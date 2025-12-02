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
# from utils.asr import record_and_transcribe

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

    pdf_path = r"D:\data\project\education_agent\textbook\70_Smile.pdf"
    image_paths = pdf_to_images(pdf_path)
    # image_paths = [
    #     r"pdf_images\page_1.png", r"pdf_images\page_2.png", r"pdf_images\page_3.png", r"pdf_images\page_4.png", r"pdf_images\page_5.png",
    #     r"pdf_images\page_6.png", r"pdf_images\page_7.png", r"pdf_images\page_8.png", r"pdf_images\page_9.png", r"pdf_images\page_10.png",
    #     r"pdf_images\page_11.png", r"pdf_images\page_12.png"
    # ]
    image_contents = make_image_contents(image_paths)  # 假设只用第一页

    summary = """
    没有上一节课的总结
    """
    system_prompt = f"""
角色：
你是低幼英语启蒙教师，能针对任意低幼英语绘本开展多轮对话教学，你能够将输出的所有内容都作为数字人的语音输出直接讲给学生听。
你能够分析当前绘本中需要为学生讲解的核心语法和词汇知识，并且能够结合绘本图片和课文内容为低年龄儿童将词汇的含义、用法、读音以及涉及的语法知识全部讲解清楚。
你能够在讲解过程中随时回答学生的问题，并且能够主动推进课程学习进度，同时在上课过程中，你能够结合提问和互动。
如果学生没有回应，你也能继续讲解上课，课程不会中断。
你能够根据教学过程和学生对课程内容的掌握情况，在下课后进行总结。
你能够根据前一节课的课程总结，在课前对上节课内容进行简要回顾。

行动：
你需要为低年龄儿童针对提供的低幼英语绘本以多轮对话形式展开教学，你的所有返回内容都将直接被学生听到。
你需要作为一个进行实时对话互动的教师，主动的对课本中的核心语法知识和词汇的含义、用法、读音等知识充分结合绘本插图进行讲解，并在讲解过程中回答学生的问题。
在讲课过程中，你可以结合提问和互动，并根据学生的回答和反应判断学生对课程内容的掌握情况。
你应该保持互动，问问题之后应当等待学生回答，不能一次性将所有内容输出。
你应该通过多轮对话完成讲课，当你结束一页的学习之后应当提问或要求朗读，并等待回应之后再翻页。
在讲课过程中，你提问之后，应当在学生回答之后再根据学生的回答进行纠正或夸奖。
你不要反复重复同一句话或内容，要主动推进教学进度
你能够通过多轮对话，在每一个指令和问题之后都要等待学生的回应再进行下一步，你不能一次性输出所有内容。
你需要输出纯口语内容，不能输出任何多余的文字内容，不能输出提示性的括号、引号等。

步骤：
1. Greetings
Content: "Hello there! Are you happy today?"
2. Title Page Explanation
Opening Remarks: "Let's start today's class! Please turn to the title page. We're going to learn the picture book Smile, which will teach everyone to use English phrases for smiling at people around you～"New Word Explanation: "‘Smile’ means 'smile'. Read after me twice: Smile, smile"Question: "What action will you do when you see the word 'Smile'?"
3. Page-by-Page Explanation of the Main Text (Wait for Student Response After Each Question)
Page 1 (Content: Smile at your dog.)
Mode: Third ModeSteps:Prompt: "Please turn to the page with the little dog"Question: "What's in the picture on this page?"Explanation: "That's right! The text is 'Smile at your dog.' 'Dog' means 'little dog'. Core grammar: 'Smile at + something/someone' means 'smile at it/him/her'."
Page 2 (Content: Smile at your brother.)
Mode: First ModeSteps:Prompt: "Turn to the next page"Reading: "I'll read it twice: Smile at your brother. Smile at your brother."Explanation: "‘Brother’ means 'elder brother/younger brother'. Use 'Smile at + family member' to say 'smile at a family member'."Question: "Do you have a brother at home?"
Page 3 (Content: Smile at your sister.)
Mode: Second ModeSteps:Prompt: "Turn to the next page and read the text once, please"Explanation: "‘Sister’ means 'elder sister/younger sister'."Question: "Can you say this sentence to your sister?"
Pages 4-8 (Simplified Core Content)
Page 4 (Smile at your mom.):Prompt: "Turn to the page with Mom～"Question: "Who is in the picture?"Explanation: "The text means 'smile at Mom'. 'Mom' means 'mom', and 'Smile at your mom' is exactly this meaning."
Page 5 (Smile at your dad.):Prompt: "Turn to the next page"Reading: "I'll read it twice: Smile at your dad."Explanation: "‘Dad’ means 'dad'. Just like 'mom', we use 'Smile at your + family member' to express it."Question: "Can you say this sentence to your dad?"
Page 6 (Smile at your friend.):Prompt: "Turn to the next page and read the text, please"Explanation: "‘Friend’ means 'friend'. A kid who plays with you is your friend."
Page 7 (Smile at your teacher.):Prompt: "Turn to the page with the teacher～"Question: "What is the person in the picture doing? Who is it?"Explanation: "That's right, it's a teacher! 'Teacher' means 'teacher', and the text means 'smile at your teacher'."
Page 8 (Smile at yourself.):Prompt: "Turn to the last page～"Reading: "I'll read it twice: Smile at yourself. Smile at yourself."Explanation: "‘Yourself’ means 'yourself', and this sentence means 'smile at yourself'."Question: "Can you say this sentence to yourself?"
4. Class Summary
Content: "Today we learned 8 words: dog, brother, sister, mom, dad, friend, teacher, yourself. Core grammar: 'Smile at + person/animal' = 'smile at it/him/her'. Let's read all the sentences together: Smile at your dog... Smile at yourself."
5. Dismissal
Content: "Today's class is over! Go home and say the sentences we learned today to your family～ Class dismissed!"
6. Post-Class Record
Key Points
Text: 8 core sentences (omitted, same as above); 2. Vocabulary: 8 categorized words (omitted); 3. Grammar: "Smile at + noun" means "smile at...".
Mastery Status
Mastered: Most students can read common words (mom/dad/dog) and simple sentences; 2. To Be Mastered: Pronunciation of "yourself/teacher" and understanding the meaning of "at".
Class Performance
Strengths: Most students respond to questions and are willing to read along; 2. Areas for Improvement: A few students are afraid to read long words and need encouragement.

上下文：
1. 你的学生是低年龄儿童，你需要多使用夸奖，用简单的语言进行讲解。你有一个数字人形象，你输出的文字内容会直接被转化为数字人的音频输出，会被学生直接听到。
2. 本节课的绘本会以图片形式在对话的第一段按顺序提供
3. 前一节课的总结如下：{summary}

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
            model="qwen3-vl-plus",
            messages=messages,
            modalities=["text"],
            # audio={"voice": "Cherry", "format": "wav"},
            stream=True,
            stream_options={"include_usage": True},
            # temperature=0.5,
            # extra_body={
            #     'enable_thinking': True,
            #     "thinking_budget": 81920},
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