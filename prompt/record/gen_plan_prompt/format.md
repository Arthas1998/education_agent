# Schema Rules

## 0) Fixed header comment format (MUST)
At the very top of the YAML, add the following 3 comment lines exactly in this format:

# ============================
# Lesson Plan: RAZ/A/<PDF_NAME>
# ============================

Where:
- `RAZ/A/` is fixed and must not change.
- `<PDF_NAME>` is the PDF base filename (without extension) normalized as:
  - remove all spaces
  - keep existing underscores `_`
  - replace the Chinese enumeration separator “、” after numbers with `_`
  - keep letter casing consistent with existing plans (e.g., `70_Smile`)
Example:
- PDF file `70 Smile.pdf` → `<PDF_NAME>` = `70_Smile`

## 1) id
- Type: **string**
- Must increase by 1 in order: `"1"`, `"2"`, `"3"`, ...

## 2) title
Use only:
- `greet_1`, `greet_2`
- `cover`
- `page_y`  (y must match the PDF page number)
- `review`
- `goodbye`

## 3) pages
- Type: list of integers.
- Default:
  - For each new `page_y`, the **first step** should set: `pages: [y]`
  - Other steps usually: `pages: []`
- Hard rule:
  - Only the step with `pages: [y]` may include the page-turn instruction for page `y`.
- Allowed exception:
  - `review` steps may include `pages: [...]` if you need to show a vocab/summary page.

## 4) plan_nl
`plan_nl` must:
- Be actionable instructions for the teacher agent.
- Be compact (1–3 lines).
- Follow closed loop: Ask/Guide → Wait → Respond → (Correct if needed) → Encourage → Transition.
- Meaning-first feedback:
  - If the student's answer is understandable and correct in meaning (e.g., "eat food"), do NOT treat it as wrong.
  - Give short encouragement, optionally ONE gentle suggestion (no required repetition).
- Conditional correction only.
- Avoid drilling and repeated follow-up questions in greeting.
- Predictions are never wrong on cover; do not correct guesses.

## 5) Fixed policies block at the end (MUST)
At the very end of the YAML (after `steps`), append this block exactly:

policies:
  defaults:
    step_turns: 1

Do not rename keys. Do not add anything inside `policies`.