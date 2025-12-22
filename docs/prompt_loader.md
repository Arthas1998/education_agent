# PromptLoader 使用说明

## 1. 你需要准备哪些东西

### 1.1 每门课一份 YAML 配置

* 例如：`./prompts/70_Smile.yaml`
* 这份 YAML 里会声明：模板文件路径、教案 YAML 路径、教材 PDF 路径、以及每条 message 的 slots 绑定方式。

### 1.2 模板文件（.txt）

* 例如： 
  * system 模板（纯文本）
  * user 模板（纯文本；可用于 text 或 parts 的 text part）

### 1.3 可选资源文件

* 教案 YAML：一份文件包含所有步骤（steps）
* 教材 PDF：一份文件包含整本教材

### 1.4 每次渲染时要提供的外部输入

渲染时由上层系统传入三类数据：

* `runtime_vars`：本轮即时输入（例如学生回答）
* `memory`：跨轮数据（例如上节课 summary）
* `params`：selector 的选择条件（例如 step id、pdf 页码）

---

## 2. 怎么写模板（.txt）

### 2.1 在模板里写占位符

模板中用 `{变量名}` 写占位符，例如：

```txt
[PLAN]
{plan}

[STUDENT]
{student_answer}

[MATERIAL]
{material_pdf}
```

### 2.2 模板变量应该写在哪里

* **模板里出现的每一个 `{变量名}`**，都必须在对应 message 的 `slots` 中提供同名键。
* 变量名完全由你决定；PromptLoader 只做“同名替换”。

---

## 3. 怎么写 YAML 配置

### 3.1 顶层结构

```yaml
course:
paths:
registry:
templates:
policies:
```

---

### 3.2 course

```yaml
course:
  id: "70_Smile"
```

---

### 3.3 paths（路径与插值）

```yaml
paths:
  base_dir: "./prompts/src"
  template_dir: "{paths.base_dir}/template"
  plan_dir: "{paths.base_dir}/lesson_plans"
  textbook_dir: "{paths.base_dir}/textbook"
```

规则：

* `{}` 插值只用于 YAML 的路径字符串。
* 只允许 `{paths.xxx}` / `{course.xxx}`。
* 不允许 `{base_dir}` 这种裸引用，必须写 `{paths.base_dir}`。

---

### 3.4 registry（注册模板、资源与外部变量）

registry 中注册所有渲染时可能用到的数据来源：

* 模板文件（text）
* 教案 YAML（yaml_object）
* 教材 PDF（pdf）
* 外部运行时变量（runtime）
* 外部记忆变量（memory）

示例：

```yaml
registry:
  system_template:
    type: text
    from: { kind: file, path: "{paths.template_dir}/gen_sys_.txt", encoding: "utf-8" }

  user_template:
    type: text
    from: { kind: file, path: "{paths.template_dir}/gen_user_.txt", encoding: "utf-8" }

  lesson_plan:
    type: yaml_object
    from: { kind: file, path: "{paths.plan_dir}/{course.id}.yaml", encoding: "utf-8" }

  textbook_pdf:
    type: pdf
    from: { kind: file, path: "{paths.textbook_dir}/{course.id}.pdf" }

  student_answer:
    type: text
    default: ""
    from: { kind: runtime, key: student_answer }

  prev_summary:
    type: text
    default: ""
    from: { kind: memory, key: prev_summary }
```

---

## 4. templates：定义要渲染的 message

每条 message 只有三个字段：

```yaml
role:
content:
slots:
```

### 4.1 system（纯文本）

```yaml
templates:
  generator:
    system:
      role: "system"
      content:
        kind: "txt_template"
        ref: "system_template"
      slots:
        course_id: { value: "70_Smile" }
```

* `content.kind: txt_template`：输出字符串
* `content.ref`：指向 registry 中的模板文件
* `slots`：为模板占位符提供值

### 4.2 user（多模态 parts）

```yaml
templates:
  generator:
    user:
      role: "user"
      content:
        kind: "parts"
        parts:
          - type: "text"
            text:
              kind: "txt_template"
              ref: "user_template"
      slots:
        turn_index: { ref: "turn_index" }
        student_answer: { ref: "student_answer" }
```

* `content.kind: parts`：输出 OpenAI parts（支持图片等多模态）
* `parts` 的第一个 part 通常是 text 模板

---

## 5. slots：把模板变量填上内容

在 message 的 `slots` 中，为模板占位符 `{变量名}` 提供值。

### 5.1 value（直接写死）

```yaml
some_var:
  value: "some text"
```

### 5.2 ref（从 registry 取值）

```yaml
student_answer:
  ref: "student_answer"
```

### 5.3 select（从大资源中按条件抽取）

#### 5.3.1 yaml_by_id_text：按 step id 抽取教案文本

```yaml
plan:
  select:
    kind: yaml_by_id_text
    from: lesson_plan
    list_path: "steps"
    match:
      field: "id"
      input_param: "step_ids"   # 外部 params.step_ids
    take:
      field: "plan_nl"
    join:
      sep: "

"
    render_with_refs: [prev_summary]
```

外部需要提供：

* `params.step_ids`：支持

  * `"greet"`
  * `["greet","review"]`

> 如果某个 id 找不到：会产生 warning 并忽略该 id。

---

## 6. selector：pdf_pages（把 PDF 页变成图片 parts）

要把教材 PDF 的某些页作为多模态输入：

1. user message 必须使用 `content.kind: parts`
2. 在模板中放一个占位符（例如 `{material_pdf}`）
3. 在 slots 中为该占位符配置 `pdf_pages`

```yaml
material_pdf:
  select:
    kind: pdf_pages
    from: textbook_pdf
    pages_param: "pdf_pages"       # 外部 params.pdf_pages
    output:
      as: openai_image_url_parts
      dpi: 200
      image_format: "png"
    placeholder_text: ""            # {material_pdf} 在文本中替换为空字符串
```

外部需要提供：

* `params.pdf_pages`（页码 1-based），支持：

  * `3`
  * `[3,4,5]`
  * `"3-5"`

---

## 7. 调用 PromptLoader 渲染

### 7.1 初始化

```python
from prompt_loader import PromptLoader

loader = PromptLoader.from_yaml("./prompts/70_Smile.yaml")
```

### 7.2 渲染单条 message

```python
msg = loader.render_message(
    template="generator",
    message="user",
    runtime_vars={"student_answer": "Hello"},
    memory={"prev_summary": "Last lesson summary"},
    params={"step_ids": "greet", "pdf_pages": 3}
)
```

### 7.3 渲染一组 messages

```python
result = loader.render(
    template="generator",
    message_names=["system", "user"],
    runtime_vars={
        "turn_index": 2,
        "student_answer": "I eat breakfast"
    },
    memory={
        "prev_summary": "We practiced family words"
    },
    params={
        "step_ids": ["greet", "cover"],
        "pdf_pages": "3-4"
    }
)

messages = result.messages
warnings = result.warnings
```

---

## 8. OpenAI message 输出格式

### 8.1 纯文本

```json
{
  "role": "system",
  "content": "..."
}
```

### 8.2 多模态 parts

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

---

## 9. warning 的处理

渲染结果会包含 warnings（例如：step id 未命中）。你可以：

* 打印到日志
* 上报监控
* 或在调试时直接输出

---

## 10. 常见检查清单

* 模板 `.txt` 中出现的 `{变量名}` 是否都在对应 message 的 `slots` 中提供？
* YAML 的路径插值是否只用了 `{paths.xxx}` / `{course.xxx}`？
* `pdf_pages` 是否只用于 `content.kind: parts` 的 message？
* `params.step_ids` / `params.pdf_pages` 是否按规范传入？
