# _*_ coding: utf-8 _*_
# @File:    prompt
# @Time:    2025/9/28 23:01
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1

# prompt.py
import json
import yaml
from typing import List, Dict, Any

class PromptComposer:
    def __init__(self, template_file: str):
        """
        初始化提示词合成器
        :param template_file: YAML 模板文件路径
        """
        with open(template_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.template = data["template"]
        self.name = data.get("name", "")
        self.description = data.get("description", "")

    def load_textbook(self, json_path: str) -> List[Dict[str, Any]]:
        """
        读取教材 JSON 文件
        :param json_path: 教材 json 文件路径
        :return: 页面数据列表
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def extract_page_info(self, page_data: List[Dict[str, Any]], page_number: int) -> Dict[str, Any]:
        """
        提取某一页的句子和核心词汇
        :param page_data: 全部教材数据
        :param page_number: 当前页码
        :return: dict 包含 current_teaching_page, page_core_sentences, page_key_vocabularies
        """
        sentences = [item["sentence"] for item in page_data if item["page"] == page_number]
        words = []
        for item in page_data:
            if item["page"] == page_number:
                for match in item.get("matches", []):
                    words.append(match["word"])

        return {
            "current_teaching_page": page_number,
            "page_core_sentences": sentences,
            "page_key_vocabularies": sorted(set(words))  # 去重并排序
        }

    def compose_prompt(self, variables: Dict[str, Any]) -> str:
        """
        根据模板和变量生成提示词
        """
        return self.template.format(
            current_teaching_page=variables["current_teaching_page"],
            page_core_sentences="；".join(variables["page_core_sentences"]),
            page_key_vocabularies="、".join(variables["page_key_vocabularies"])
        )


if __name__ == "__main__":
    # 调试配置
    TEMPLATE_FILE = "templates/teaching.yaml"
    TEXTBOOK_FILE = "57_HeRuns.json"

    composer = PromptComposer(template_file=TEMPLATE_FILE)

    # 加载教材
    data = composer.load_textbook(TEXTBOOK_FILE)

    # 测试第 3 页
    page_number = 3
    page_info = composer.extract_page_info(data, page_number=page_number)
    prompt = composer.compose_prompt(page_info)

    print("===== 模板名称 =====")
    print(composer.name)
    print("===== 模板说明 =====")
    print(composer.description)
    print("===== 第", page_number, "页合成提示词 =====")
    print(prompt)

