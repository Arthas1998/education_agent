# _*_ coding: utf-8 _*_
# @File:    prompt
# @Time:    2025/9/28 23:01
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1

import os
import re
import yaml
import base64
import fitz  # PyMuPDF
from typing import List, Optional, Dict, Any


class PromptLoader:
    """Load system and initial user prompts based on a YAML config.

    Behavior:
    - Reads YAML config (default: prompt/config/zh_temp_1.yaml relative to repo root)
    - Loads template files (txt) and replaces {placeholders} using components mapping
    - Converts PDF pages (textbook/) to in-memory PNG data URLs (no files written)
    - Provides two public methods:
        * load_system_prompt() -> dict (can be appended to messages)
        * load_initial_user_prompt(include_first_n=None) -> dict (user message with images + text)
    """

    def __init__(self, config_path: str = "prompt/config/zh_temp_1.yaml"):
        # project root (one level above utils/)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.config_path = os.path.join(self.project_root, config_path)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            # allow YAML files that may contain // comments (strip them) - basic cleanup
            raw = f.read()
            # remove lines starting with // (simple support for the repo's style)
            cleaned = "\n".join([ln for ln in raw.splitlines() if not ln.strip().startswith("//")])
            self.config = yaml.safe_load(cleaned)

        # normalize locations in config for convenience
        self.current_course_pdf = self.config.get("current_course")
        # user may provide just filename like "70_Smile.pdf". Build textbook path
        if self.current_course_pdf and not os.path.isabs(self.current_course_pdf):
            self.current_course_pdf = os.path.join(self.project_root, "textbook", self.current_course_pdf)

        self.system_conf = self.config.get("system_prompts", {})
        self.initial_conf = self.config.get("initial_user_prompts", {})

    # --------------------- Text template helpers ---------------------
    def _read_text(self, path_or_paths: Any) -> str:
        """Read a single text file or a list of files and concatenate with newlines.
        Paths may be absolute or relative to project root.
        """
        if not path_or_paths:
            return ""
        if isinstance(path_or_paths, list):
            parts = [self._read_text(p) for p in path_or_paths]
            return "\n".join([p for p in parts if p])
        path = path_or_paths
        # allow inline strings if file not found -> treat as literal
        candidate = path
        if not os.path.isabs(candidate):
            candidate = os.path.join(self.project_root, path)
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return f.read()
        # fallback: return the original string (allow small literals)
        return str(path_or_paths)

    def _apply_placeholders(self, template_text: str, components: Dict[str, Any]) -> str:
        """Replace {placeholder} occurrences in template_text using components mapping.

        If a component value is a filepath, it will be read; if it's a list, files will be concatenated.
        Missing placeholders are replaced with empty string.
        """
        if not template_text:
            return ""
        components = components or {}

        def repl(match: re.Match) -> str:
            key = match.group(1)
            if key in components:
                value = components[key]
                # if value looks like a path or list of paths, read them
                return self._read_text(value)
            # not found -> empty
            return ""

        return re.sub(r"{([a-zA-Z0-9_]+)}", repl, template_text)

    # --------------------- PDF -> data URLs (in-memory) ---------------------
    def _pdf_pages_to_dataurls(self, pdf_path: str, max_pages: Optional[int] = None, dpi: int = 144) -> List[str]:
        """Render PDF pages to PNG bytes and return data URLs (no files written).

        Returns list of dataurls in page order.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        doc = fitz.open(pdf_path)
        urls: List[str] = []
        total = len(doc)
        to_render = total if max_pages is None else min(max_pages, total)
        for idx in range(to_render):
            page = doc[idx]
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            try:
                png_bytes = pix.tobytes(output="png")
            except TypeError:
                # older/newer PyMuPDF variants might use different signature
                png_bytes = pix.tobytes()
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            dataurl = f"data:image/png;base64,{b64}"
            urls.append(dataurl)
        doc.close()
        return urls

    def _make_image_contents_from_dataurls(self, dataurls: List[str]) -> List[Dict[str, Any]]:
        out = []
        for url in dataurls:
            out.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        return out

    # --------------------- Public API ---------------------
    def load_system_prompt(self, assemble_mode: str = "join_with_newline") -> Dict[str, Any]:
        """Load system prompt, apply placeholders, and return a dict suitable for messages.append().

        assemble_mode: 'join_with_newline' (default) or 'list_items' - controls whether multiple template files
        are combined into a single text item or each becomes its own text item.
        """
        template = self.system_conf.get("template")
        components = self.system_conf.get("components", {})

        # template may be a path string or list; read it
        raw_template = self._read_text(template)
        final_text = self._apply_placeholders(raw_template, components)

        content = []
        if assemble_mode == "list_items":
            # split by double-newline or single newline into items
            parts = [p for p in re.split(r"\n\n+|\n", final_text) if p.strip()]
            for p in parts:
                content.append({"type": "text", "text": p})
        else:
            content.append({"type": "text", "text": final_text})

        return {"role": "system", "content": content}

    def load_initial_user_prompt(self, use_textbook: Optional[bool] = False, include_first_n: Optional[int] = None) -> Dict[str, Any]:
        """Load initial user prompt: include course images (from configured course pdf) and initial template text.

        include_first_n: if provided, only render the first N pages; otherwise render all pages.
        Returns a dict suitable for messages.append().
        """
        # load initial user template and apply placeholders
        template = self.initial_conf.get("template")
        components = self.initial_conf.get("components", {})
        raw_template = self._read_text(template)
        final_text = self._apply_placeholders(raw_template, components)

        # assemble content: images followed by text
        content = []
        if use_textbook:
            # Build image items from current course pdf
            if not self.current_course_pdf or not os.path.exists(self.current_course_pdf):
                raise FileNotFoundError(f"Configured current course PDF not found: {self.current_course_pdf}")
            dataurls = self._pdf_pages_to_dataurls(self.current_course_pdf, max_pages=include_first_n)
            image_items = self._make_image_contents_from_dataurls(dataurls)
            content.extend(image_items)
        content.append({"type": "text", "text": final_text})

        return {"role": "user", "content": content}



# --------------------- Script test runner ---------------------
if __name__ == "__main__":
    try:
        loader = PromptLoader("prompt/config/zh_temp_1.yaml")
        sys_prompt = loader.load_system_prompt()
        user_prompt = loader.load_initial_user_prompt(
            use_textbook=True,
            include_first_n=13
        )

        print("System prompt dict:")
        print(sys_prompt)
        print("\nInitial user prompt dict (first page + text):")
        # print only keys and text lengths to avoid huge base64 dumps
        # but show that image_url exists
        simplified = {
            "role": user_prompt.get("role"),
            "content": []
        }
        for item in user_prompt.get("content", []):
            if item.get("type") == "image_url":
                simplified["content"].append({"type": "image_url", "has_url": True, "url_preview_len": len(item.get("image_url", {}).get("url", ""))})
            else:
                simplified["content"].append(item)
        print(simplified)
    except Exception as e:
        print("Error during prompt loader test:", e)
