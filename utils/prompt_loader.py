# _*_ coding: utf-8 _*_
# @File:    prompt_loader
# @Time:    2025/12/21 22:19
# @Author:  ArthasMenethil/wuweihang
# @Contact: wuweihang1998@gmail.com
# @Version: V 0.1
from __future__ import annotations

import base64
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


# =========================
# Types
# =========================

OpenAITextMessage = Dict[str, Any]
OpenAIPartsMessage = Dict[str, Any]
OpenAIMessage = Dict[str, Any]


@dataclass
class RenderWarning:
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    messages: List[OpenAIMessage]
    warnings: List[RenderWarning]


# =========================
# Exceptions
# =========================

class PromptConfigError(Exception):
    """Raised when configuration is invalid or inconsistent."""


class RenderError(Exception):
    """Raised when rendering fails."""


# =========================
# Utility: interpolation
# =========================

_INTERP_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_\.]*)\}")


def _get_by_dotted_path(root: Dict[str, Any], dotted: str) -> Any:
    cur: Any = root
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise PromptConfigError(f"Interpolation reference not found: {{{dotted}}}")
    return cur


def interpolate_string(
    s: str,
    config_root: Dict[str, Any],
    allowed_roots: Tuple[str, ...] = ("course", "paths"),
) -> str:
    """
    Interpolate {course.id} / {paths.template_dir} style references.
    Only allows dotted-paths starting with allowed_roots.
    """

    def repl(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        if "." not in expr:
            # Enforce explicit roots: {paths.base_dir} not {base_dir}
            raise PromptConfigError(
                f"Invalid interpolation '{{{expr}}}': must be dotted (e.g., {{paths.base_dir}})."
            )
        root_key = expr.split(".", 1)[0]
        if root_key not in allowed_roots:
            raise PromptConfigError(
                f"Interpolation root '{root_key}' is not allowed. Allowed: {allowed_roots}"
            )
        val = _get_by_dotted_path(config_root, expr)
        if not isinstance(val, (str, int, float)):
            raise PromptConfigError(f"Interpolation '{{{expr}}}' must resolve to a scalar, got: {type(val)}")
        return str(val)

    return _INTERP_PATTERN.sub(repl, s)


def interpolate_paths_and_registry(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Two-phase interpolation:
      1) paths.* strings can reference {paths.base_dir} and {course.*}
      2) registry.*.from.path strings can reference {paths.*} and {course.*}
    """
    cfg = dict(config)  # shallow copy; we will mutate nested dicts carefully

    if "course" not in cfg or "paths" not in cfg:
        raise PromptConfigError("Config must contain 'course' and 'paths'.")

    # Phase B: interpolate paths
    paths = dict(cfg.get("paths", {}))
    cfg["paths"] = paths
    # Interpolate all path values (strings)
    for k, v in list(paths.items()):
        if isinstance(v, str):
            paths[k] = interpolate_string(v, cfg, allowed_roots=("course", "paths"))
        else:
            raise PromptConfigError(f"paths.{k} must be a string.")

    # Phase C: interpolate registry file paths
    registry = dict(cfg.get("registry", {}))
    cfg["registry"] = registry
    for reg_key, reg_item in list(registry.items()):
        if not isinstance(reg_item, dict):
            raise PromptConfigError(f"registry.{reg_key} must be a dict.")
        from_def = reg_item.get("from")
        if isinstance(from_def, dict) and from_def.get("kind") == "file":
            path_val = from_def.get("path")
            if isinstance(path_val, str):
                from_def = dict(from_def)
                from_def["path"] = interpolate_string(path_val, cfg, allowed_roots=("course", "paths"))
                reg_item = dict(reg_item)
                reg_item["from"] = from_def
                registry[reg_key] = reg_item

    # Also interpolate any file path-like strings under templates if you add later (not needed now)

    return cfg


# =========================
# Utility: template rendering
# =========================

_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def render_text_template(template_text: str, slot_values: Dict[str, Any]) -> str:
    """
    Replace {name} placeholders using slot_values.
    Any missing placeholder => error (config/template mismatch).
    """

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in slot_values:
            raise RenderError(f"Missing slot value for placeholder: {{{key}}}")
        v = slot_values[key]
        # For safety, coerce to string
        return "" if v is None else str(v)

    return _PLACEHOLDER_PATTERN.sub(repl, template_text)


# =========================
# Registry resolving with cache
# =========================

@dataclass
class ResourceCache:
    text_files: Dict[str, str] = field(default_factory=dict)
    yaml_objects: Dict[str, Any] = field(default_factory=dict)
    pdf_docs: Dict[str, Any] = field(default_factory=dict)  # fitz.Document cache


class RegistryResolver:
    def __init__(self, config: Dict[str, Any], base_dir: Path, cache: ResourceCache):
        self.config = config
        self.base_dir = base_dir
        self.cache = cache

    def _resolve_file_path(self, p: str) -> Path:
        # Treat as relative to current working dir unless absolute.
        path = Path(p)
        return path if path.is_absolute() else path

    def resolve(self, key: str, runtime_vars: Dict[str, Any], memory: Dict[str, Any]) -> Any:
        registry = self.config.get("registry", {})
        if key not in registry:
            raise RenderError(f"Registry key not found: {key}")

        item = registry[key]
        item_type = item.get("type")
        default = item.get("default", None)
        from_def = item.get("from", {})
        kind = from_def.get("kind")

        if kind == "runtime":
            k = from_def.get("key")
            return runtime_vars.get(k, default)

        if kind == "memory":
            k = from_def.get("key")
            return memory.get(k, default)

        if kind == "file":
            path = from_def.get("path")
            if not isinstance(path, str) or not path:
                raise PromptConfigError(f"registry.{key}.from.path must be a non-empty string.")
            fs_path = self._resolve_file_path(path)

            if item_type == "text":
                if key in self.cache.text_files:
                    return self.cache.text_files[key]
                encoding = from_def.get("encoding", "utf-8")
                text = fs_path.read_text(encoding=encoding)
                self.cache.text_files[key] = text
                return text

            if item_type in ("yaml_object", "yaml"):
                if key in self.cache.yaml_objects:
                    return self.cache.yaml_objects[key]
                encoding = from_def.get("encoding", "utf-8")
                raw = fs_path.read_text(encoding=encoding)
                obj = yaml.safe_load(raw) if raw.strip() else {}
                self.cache.yaml_objects[key] = obj
                return obj

            if item_type == "pdf":
                # Keep as a path; open lazily in PDF renderer
                return fs_path

            raise PromptConfigError(f"Unsupported registry file item type: {item_type}")

        raise PromptConfigError(f"Unsupported registry.from.kind: {kind}")

    def resolve_text_template(self, key: str, runtime_vars: Dict[str, Any], memory: Dict[str, Any]) -> str:
        val = self.resolve(key, runtime_vars, memory)
        if not isinstance(val, str):
            raise RenderError(f"Registry '{key}' is not text.")
        return val


# =========================
# Selectors
# =========================

class SelectorEngine:
    def __init__(self, resolver: RegistryResolver):
        self.resolver = resolver

    def eval_select(
        self,
        select_def: Dict[str, Any],
        *,
        runtime_vars: Dict[str, Any],
        memory: Dict[str, Any],
        params: Dict[str, Any],
        warnings_out: List[RenderWarning],
        allow_pdf_parts: bool,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Returns:
          - slot_text_value: str (used for placeholder substitution in text template)
          - extra_parts: list of OpenAI parts to append (only used in parts mode)
        """
        kind = select_def.get("kind")
        if kind == "yaml_by_id_text":
            return self._yaml_by_id_text(select_def, runtime_vars, memory, params, warnings_out), []

        if kind == "pdf_pages":
            if not allow_pdf_parts:
                raise RenderError("pdf_pages selector can only be used when content.kind == 'parts'.")
            placeholder_text = select_def.get("placeholder_text", "")
            parts = self._pdf_pages_to_parts(select_def, runtime_vars, memory, params)
            return str(placeholder_text), parts

        raise PromptConfigError(f"Unsupported select.kind: {kind}")

    def _yaml_by_id_text(
        self,
        select_def: Dict[str, Any],
        runtime_vars: Dict[str, Any],
        memory: Dict[str, Any],
        params: Dict[str, Any],
        warnings_out: List[RenderWarning],
    ) -> str:
        from_key = select_def.get("from")
        list_path = select_def.get("list_path")
        match = select_def.get("match", {})
        take = select_def.get("take", {})
        join = select_def.get("join", {})

        if not isinstance(from_key, str):
            raise PromptConfigError("yaml_by_id_text.select.from must be a registry key string.")
        if not isinstance(list_path, str):
            raise PromptConfigError("yaml_by_id_text.select.list_path must be a string.")
        match_field = match.get("field")
        input_param = match.get("input_param")
        take_field = take.get("field")
        sep = join.get("sep", "\n")

        if not isinstance(match_field, str) or not isinstance(input_param, str) or not isinstance(take_field, str):
            raise PromptConfigError("yaml_by_id_text.match.field / match.input_param / take.field must be strings.")

        data = self.resolver.resolve(from_key, runtime_vars, memory)
        if not isinstance(data, dict):
            raise RenderError(f"Registry '{from_key}' did not resolve to a YAML object (dict).")

        # Only support one-level list_path for simplicity; can expand later
        if list_path not in data or not isinstance(data[list_path], list):
            raise RenderError(f"YAML object '{from_key}' does not contain list '{list_path}'.")

        lst: List[Any] = data[list_path]
        want_ids = params.get(input_param)

        if want_ids is None:
            # No ids provided means empty result (not an error).
            warnings_out.append(RenderWarning(
                code="YAML_SELECT_NO_INPUT",
                message=f"No ids provided in params['{input_param}']; selector output will be empty.",
                context={"input_param": input_param, "selector": "yaml_by_id_text"},
            ))
            return ""

        if isinstance(want_ids, str):
            ids: List[str] = [want_ids]
        elif isinstance(want_ids, list) and all(isinstance(x, str) for x in want_ids):
            ids = want_ids
        else:
            raise RenderError(
                f"params['{input_param}'] must be a string or list[str], got: {type(want_ids)}"
            )

        # Build index by id (first occurrence wins)
        index: Dict[str, Dict[str, Any]] = {}
        for item in lst:
            if isinstance(item, dict) and match_field in item and isinstance(item[match_field], str):
                if item[match_field] not in index:
                    index[item[match_field]] = item

        out_parts: List[str] = []
        for _id in ids:
            if _id not in index:
                warnings_out.append(RenderWarning(
                    code="YAML_SELECT_ID_NOT_FOUND",
                    message=f"ID '{_id}' not found in '{from_key}.{list_path}', ignoring.",
                    context={"missing_id": _id, "from": from_key, "list_path": list_path},
                ))
                continue
            item = index[_id]
            val = item.get(take_field, "")
            if not isinstance(val, str):
                val = "" if val is None else str(val)
            out_parts.append(val)

        text = sep.join([p for p in out_parts if p.strip() != ""])

        # Optional second-pass rendering with registry refs (for {prev_summary} etc.)
        refs = select_def.get("render_with_refs", [])
        if refs:
            if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
                raise PromptConfigError("render_with_refs must be a list[str].")
            slot_vals: Dict[str, Any] = {}
            for r in refs:
                slot_vals[r] = self.resolver.resolve(r, runtime_vars, memory)
            # Only replace placeholders matching those keys; missing placeholders will raise,
            # so we render only those by providing a custom safe render:
            text = self._render_subset_placeholders(text, slot_vals)

        return text

    @staticmethod
    def _render_subset_placeholders(text: str, subset: Dict[str, Any]) -> str:
        """
        Render only placeholders that appear in 'subset'. Leave others untouched.
        This avoids forcing every {x} inside plan text to exist.
        """
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in subset:
                v = subset[key]
                return "" if v is None else str(v)
            return match.group(0)  # keep as-is

        return _PLACEHOLDER_PATTERN.sub(repl, text)

    def _pdf_pages_to_parts(
        self,
        select_def: Dict[str, Any],
        runtime_vars: Dict[str, Any],
        memory: Dict[str, Any],
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        from_key = select_def.get("from")
        pages_param = select_def.get("pages_param")
        output = select_def.get("output", {})

        if not isinstance(from_key, str):
            raise PromptConfigError("pdf_pages.select.from must be a registry key string.")
        if not isinstance(pages_param, str):
            raise PromptConfigError("pdf_pages.pages_param must be a string.")

        pdf_path = self.resolver.resolve(from_key, runtime_vars, memory)
        if not isinstance(pdf_path, Path):
            raise RenderError(f"Registry '{from_key}' did not resolve to a PDF Path.")

        pages_raw = params.get(pages_param)
        if pages_raw is None or pages_raw == "" or pages_raw == []:
            # No pages requested => no parts appended
            return []

        pages = self._parse_pages_1based(pages_raw)

        as_fmt = output.get("as", "openai_image_url_parts")
        if as_fmt != "openai_image_url_parts":
            raise PromptConfigError(f"Unsupported pdf output.as: {as_fmt}")

        dpi = int(output.get("dpi", 200))
        image_format = str(output.get("image_format", "png")).lower()
        media_type = "image/png" if image_format == "png" else "image/jpeg"

        renderer = PdfPageRenderer(self.resolver.cache)
        images_bytes = renderer.render_pages(pdf_path, pages_1based=pages, dpi=dpi, image_format=image_format)

        parts: List[Dict[str, Any]] = []
        for img in images_bytes:
            url = renderer.to_data_url(img, media_type=media_type)
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    @staticmethod
    def _parse_pages_1based(pages_raw: Union[int, str, List[int]]) -> List[int]:
        """
        Accept:
          - 3
          - [3,4,5]
          - "3-5"
        1-based pages.
        """
        if isinstance(pages_raw, int):
            if pages_raw < 1:
                raise RenderError("PDF page numbers must be 1-based positive integers.")
            return [pages_raw]

        if isinstance(pages_raw, list):
            if not all(isinstance(x, int) for x in pages_raw):
                raise RenderError("PDF pages list must be list[int].")
            pages = []
            for x in pages_raw:
                if x < 1:
                    raise RenderError("PDF page numbers must be 1-based positive integers.")
                pages.append(x)
            # keep order, allow duplicates? usually no; we can dedupe while preserving order
            seen = set()
            deduped = []
            for p in pages:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            return deduped

        if isinstance(pages_raw, str):
            s = pages_raw.strip()
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
            if not m:
                raise RenderError("PDF pages string must be of the form 'start-end', e.g. '3-5'.")
            a = int(m.group(1))
            b = int(m.group(2))
            if a < 1 or b < 1:
                raise RenderError("PDF page numbers must be 1-based positive integers.")
            if b < a:
                raise RenderError("PDF pages range must have end >= start.")
            return list(range(a, b + 1))

        raise RenderError(f"Unsupported pdf pages input type: {type(pages_raw)}")


# =========================
# PDF rendering (PyMuPDF)
# =========================

class PdfPageRenderer:
    """
    Render selected pages of a PDF into image bytes.
    Requires PyMuPDF (import fitz). Only invoked if pdf_pages selector used.
    """
    def __init__(self, cache: ResourceCache):
        self.cache = cache

    def _open_pdf(self, pdf_path: Path):
        key = str(pdf_path.resolve())
        if key in self.cache.pdf_docs:
            return self.cache.pdf_docs[key]
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RenderError(
                "PDF rendering requires PyMuPDF (pip install pymupdf). "
                "Import 'fitz' failed."
            ) from e
        doc = fitz.open(str(pdf_path))
        self.cache.pdf_docs[key] = doc
        return doc

    def render_pages(
        self,
        pdf_path: Path,
        pages_1based: List[int],
        dpi: int = 200,
        image_format: str = "png",
    ) -> List[bytes]:
        # Import fitz here as well because _open_pdf caches only Document;
        # Matrix is a module-level symbol in PyMuPDF and should not be accessed via private doc._fitz.
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RenderError(
                "PDF rendering requires PyMuPDF (pip install pymupdf). "
                "Import 'fitz' failed."
            ) from e

        doc = self._open_pdf(pdf_path)
        # PyMuPDF pages are 0-based
        page_count = doc.page_count
        out: List[bytes] = []

        # Convert dpi to zoom: 72 DPI is default
        zoom = dpi / 72.0

        fmt = image_format.lower()
        if fmt not in ("png", "jpg", "jpeg"):
            raise RenderError("image_format must be 'png' or 'jpg'/'jpeg'.")
        target = "png" if fmt == "png" else "jpeg"

        for p1 in pages_1based:
            p0 = p1 - 1
            if p0 < 0 or p0 >= page_count:
                raise RenderError(f"PDF page out of range: {p1} (pdf has {page_count} pages).")

            page = doc.load_page(p0)
            # Use public API (fitz.Matrix). 'doc._fitz' is not available on some PyMuPDF versions.
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # PyMuPDF has had minor API changes across versions.
            # Try the common signatures in order.
            try:
                img_bytes = pix.tobytes(target)
            except TypeError:
                try:
                    img_bytes = pix.tobytes(output=target)
                except TypeError:
                    # Fallback: some versions ignore output format; returns bytes in default format.
                    img_bytes = pix.tobytes()

            out.append(img_bytes)
        return out

    @staticmethod
    def to_data_url(img_bytes: bytes, media_type: str = "image/png") -> str:
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:{media_type};base64,{b64}"


# =========================
# PromptLoader (public API)
# =========================

class PromptLoader:
    def __init__(self, config: Dict[str, Any], config_path: Path):
        self.config_path = config_path
        self.config = config
        self.cache = ResourceCache()

        # Base dir is informational here; paths.base_dir already interpolated
        base_dir = config.get("paths", {}).get("base_dir")
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else config_path.parent

        self.resolver = RegistryResolver(config=self.config, base_dir=self.base_dir, cache=self.cache)
        self.selector = SelectorEngine(self.resolver)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PromptLoader":
        config_path = Path(path)
        raw = config_path.read_text(encoding="utf-8")
        cfg = yaml.safe_load(raw) if raw.strip() else {}
        if not isinstance(cfg, dict):
            raise PromptConfigError("Top-level YAML must be a mapping/dict.")

        # Interpolate paths + registry file paths
        cfg2 = interpolate_paths_and_registry(cfg)

        return cls(cfg2, config_path=config_path)

    def validate(self) -> List[PromptConfigError]:
        """
        Best-effort validation; returns list of issues (does not raise).
        """
        issues: List[PromptConfigError] = []

        try:
            # ensure templates exist
            templates = self.config.get("templates", {})
            if not isinstance(templates, dict):
                raise PromptConfigError("'templates' must be a dict.")
        except PromptConfigError as e:
            issues.append(e)
            return issues

        # Validate registry refs used in templates
        registry = self.config.get("registry", {})
        templates = self.config.get("templates", {})
        for tname, tdef in templates.items():
            if not isinstance(tdef, dict):
                issues.append(PromptConfigError(f"templates.{tname} must be a dict."))
                continue
            for mname, mdef in tdef.items():
                if not isinstance(mdef, dict):
                    issues.append(PromptConfigError(f"templates.{tname}.{mname} must be a dict."))
                    continue
                content = mdef.get("content", {})
                slots = mdef.get("slots", {})
                try:
                    self._validate_message(content, slots, registry, template=tname, message=mname)
                except PromptConfigError as e:
                    issues.append(e)
        return issues

    def _validate_message(
        self,
        content: Dict[str, Any],
        slots: Dict[str, Any],
        registry: Dict[str, Any],
        *,
        template: str,
        message: str,
    ) -> None:
        if not isinstance(content, dict):
            raise PromptConfigError(f"templates.{template}.{message}.content must be a dict.")
        kind = content.get("kind")
        if kind not in ("txt_template", "parts"):
            raise PromptConfigError(f"Unsupported content.kind '{kind}' in {template}.{message}.")

        # Validate refs in content
        if kind == "txt_template":
            ref = content.get("ref")
            if not isinstance(ref, str) or ref not in registry:
                raise PromptConfigError(f"{template}.{message}.content.ref '{ref}' not found in registry.")
        if kind == "parts":
            parts = content.get("parts", [])
            if not isinstance(parts, list) or not parts:
                raise PromptConfigError(f"{template}.{message}.content.parts must be a non-empty list.")
            # validate text part template refs
            for p in parts:
                if not isinstance(p, dict) or "type" not in p:
                    raise PromptConfigError(f"Invalid part in {template}.{message}.content.parts.")
                if p.get("type") == "text":
                    text_def = p.get("text", {})
                    if not isinstance(text_def, dict):
                        raise PromptConfigError(f"Text part missing 'text' object in {template}.{message}.")
                    if text_def.get("kind") != "txt_template":
                        raise PromptConfigError("Text part only supports txt_template.")
                    ref = text_def.get("ref")
                    if not isinstance(ref, str) or ref not in registry:
                        raise PromptConfigError(f"Text part ref '{ref}' not found in registry.")

        # Validate slot refs/select.from exist
        if not isinstance(slots, dict):
            raise PromptConfigError(f"templates.{template}.{message}.slots must be a dict.")
        for slot_name, slot_def in slots.items():
            if not isinstance(slot_def, dict):
                raise PromptConfigError(f"Slot '{slot_name}' in {template}.{message} must be a dict.")
            if "ref" in slot_def:
                ref = slot_def["ref"]
                if not isinstance(ref, str) or ref not in registry:
                    raise PromptConfigError(f"Slot '{slot_name}' ref '{ref}' not found in registry.")
            if "select" in slot_def:
                sel = slot_def["select"]
                if not isinstance(sel, dict):
                    raise PromptConfigError(f"Slot '{slot_name}'.select must be a dict.")
                if sel.get("kind") == "pdf_pages" and kind != "parts":
                    raise PromptConfigError("pdf_pages selector can only be used when content.kind == 'parts'.")
                # check sel.from
                sel_from = sel.get("from")
                if isinstance(sel_from, str) and sel_from not in registry:
                    raise PromptConfigError(f"Selector slot '{slot_name}' from '{sel_from}' not found in registry.")

    def render(
        self,
        *,
        template: str,
        message_names: Optional[List[str]] = None,
        runtime_vars: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> RenderResult:
        runtime_vars = runtime_vars or {}
        memory = memory or {}
        params = params or {}

        templates = self.config.get("templates", {})
        if template not in templates:
            raise RenderError(f"Template not found: {template}")

        tdef = templates[template]
        if not isinstance(tdef, dict):
            raise RenderError(f"Template '{template}' must be a dict.")

        if message_names is None:
            message_names = list(tdef.keys())

        warnings_out: List[RenderWarning] = []
        messages: List[OpenAIMessage] = []
        for mname in message_names:
            messages.append(
                self.render_message(
                    template=template,
                    message=mname,
                    runtime_vars=runtime_vars,
                    memory=memory,
                    params=params,
                    warnings_out=warnings_out,
                )
            )
        return RenderResult(messages=messages, warnings=warnings_out)

    def render_message(
        self,
        *,
        template: str,
        message: str,
        runtime_vars: Dict[str, Any],
        memory: Dict[str, Any],
        params: Dict[str, Any],
        warnings_out: Optional[List[RenderWarning]] = None,
    ) -> OpenAIMessage:
        warnings_out = warnings_out if warnings_out is not None else []

        templates = self.config.get("templates", {})
        tdef = templates.get(template)
        if not isinstance(tdef, dict) or message not in tdef:
            raise RenderError(f"Message '{message}' not found in template '{template}'.")

        mdef = tdef[message]
        if not isinstance(mdef, dict):
            raise RenderError(f"templates.{template}.{message} must be a dict.")

        role = mdef.get("role")
        content_def = mdef.get("content")
        slots_def = mdef.get("slots")

        if not isinstance(role, str):
            raise RenderError(f"templates.{template}.{message}.role must be a string.")
        if not isinstance(content_def, dict):
            raise RenderError(f"templates.{template}.{message}.content must be a dict.")
        if not isinstance(slots_def, dict):
            raise RenderError(f"templates.{template}.{message}.slots must be a dict.")

        kind = content_def.get("kind")
        allow_pdf_parts = kind == "parts"

        # Compute slots values + extra parts
        slot_values: Dict[str, Any] = {}
        extra_parts: List[Dict[str, Any]] = []

        for slot_name, slot_entry in slots_def.items():
            if not isinstance(slot_entry, dict):
                raise RenderError(f"Slot '{slot_name}' must be a dict.")
            if "value" in slot_entry:
                slot_values[slot_name] = slot_entry["value"]
            elif "ref" in slot_entry:
                ref_key = slot_entry["ref"]
                if not isinstance(ref_key, str):
                    raise RenderError(f"Slot '{slot_name}'.ref must be a string.")
                slot_values[slot_name] = self.resolver.resolve(ref_key, runtime_vars, memory)
            elif "select" in slot_entry:
                sel_def = slot_entry["select"]
                if not isinstance(sel_def, dict):
                    raise RenderError(f"Slot '{slot_name}'.select must be a dict.")
                text_val, parts = self.selector.eval_select(
                    sel_def,
                    runtime_vars=runtime_vars,
                    memory=memory,
                    params=params,
                    warnings_out=warnings_out,
                    allow_pdf_parts=allow_pdf_parts,
                )
                # The slot placeholder gets replaced with text_val (often "")
                slot_values[slot_name] = text_val
                # For pdf_pages, append parts
                if parts:
                    extra_parts.extend(parts)
            else:
                raise RenderError(
                    f"Slot '{slot_name}' must contain one of: value/ref/select."
                )

        # Render content
        if kind == "txt_template":
            ref = content_def.get("ref")
            if not isinstance(ref, str):
                raise RenderError("txt_template content.ref must be a string.")
            template_text = self.resolver.resolve_text_template(ref, runtime_vars, memory)
            rendered = render_text_template(template_text, slot_values)
            return {"role": role, "content": rendered}

        if kind == "parts":
            parts_def = content_def.get("parts")
            if not isinstance(parts_def, list) or not parts_def:
                raise RenderError("parts content.parts must be a non-empty list.")

            parts_out: List[Dict[str, Any]] = []
            for p in parts_def:
                if not isinstance(p, dict) or "type" not in p:
                    raise RenderError("Invalid part definition.")
                ptype = p["type"]
                if ptype == "text":
                    text_def = p.get("text", {})
                    if not isinstance(text_def, dict):
                        raise RenderError("Text part must include 'text' dict.")
                    if text_def.get("kind") != "txt_template":
                        raise RenderError("Text part only supports txt_template.")
                    ref = text_def.get("ref")
                    if not isinstance(ref, str):
                        raise RenderError("Text part ref must be a string.")
                    template_text = self.resolver.resolve_text_template(ref, runtime_vars, memory)
                    rendered_text = render_text_template(template_text, slot_values)
                    parts_out.append({"type": "text", "text": rendered_text})
                else:
                    raise RenderError(f"Unsupported part.type: {ptype}")

            # Append selector-generated parts (e.g. PDF pages)
            parts_out.extend(extra_parts)

            return {"role": role, "content": parts_out}

        raise RenderError(f"Unsupported content.kind: {kind}")


# =========================
# Minimal usage example
# =========================
if __name__ == "__main__":
    # Example:
    # loader = PromptLoader.from_yaml("config/70_Smile.yaml")
    # result = loader.render(
    #     template="generator",
    #     message_names=["system", "user"],
    #     runtime_vars={
    #         "turn_index": 3,
    #         "student_answer": "I brush my teeth.",
    #     },
    #     memory={"prev_summary": "Last time we learned family words."},
    #     params={
    #         "step_ids": ["greet", "cover"],
    #         "pdf_pages": "3-4",
    #     },
    # )
    # print(result.messages)
    # for w in result.warnings:
    #     print(w.code, w.message, w.context)
    pass
