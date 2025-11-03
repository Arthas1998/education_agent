# _*_ coding: utf-8 _*_
# @File:    prompt
# @Time:    2025/9/28 23:01
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1

import os
import base64
import fitz  # PyMuPDF
import yaml


class PromptLoader:
    def __init__(self, yaml_path: str):
        """
        初始化PromptLoader，加载YAML配置文件。
        """
        self.cfg = self._load_yaml(yaml_path)

    # ============ 内部方法 ============

    def _load_yaml(self, yaml_path: str) -> dict:
        """
        从YAML文件中读取配置并返回为字典。
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data

    def _load_txt_as_template(self, txt_path: str) -> str:
        """
        读取txt文件内容，并返回一个可被f-string格式化的字符串模板。
        用法示例:
            template = loader._load_txt_as_template("prompt.txt")
            text = eval(f"f'''{template}'''", {}, {'var1': 'value'})
        """
        with open(txt_path, "r", encoding="utf-8") as f:
            template = f.read()
        return template

    def _pdf_to_images(self, output_folder="pdf_images", dpi=144):
        """
        将PDF文件转为图像序列，路径来自self.cfg['pdf_path']。
        """
        pdf_path = self.cfg.get("pdf_path", None)
        if pdf_path is None:
            raise ValueError("YAML配置中未找到 'pdf_path' 字段。")

        os.makedirs(output_folder, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_paths = []
        for idx, page in enumerate(doc):
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(output_folder, f"page_{idx + 1}.png")
            pix.save(image_path)
            image_paths.append(image_path)
        return image_paths

    # ============ 静态内部方法 ============

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """
        将图像编码为base64字符串。
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _make_image_contents(image_paths: list) -> list:
        """
        将图像路径列表转换为OpenAI兼容的图像内容列表。
        """
        output = []
        for path in image_paths:
            item = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{PromptLoader._encode_image(path)}"},
            }
            output.append(item)
        return output



