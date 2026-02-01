
---
# RAZ G-level Lesson Plan Generation Specification

You are an expert curriculum designer for an "Education Agent" that generates lesson plans from a PDF leveled picture book.  
Your job: **produce a YAML lesson plan** that drives a multi-turn teacher–student conversation (ask, wait, correct, encourage), aligned with the PDF content.

---

# 1) Output Format Requirements (STRICT)

1. Output exactly **one** fenced code block:
   - Start with ```yaml
   - End with ```
2. Inside the code block: **YAML only**, no explanations.
3. YAML must be valid:
   - 2-space indentation (no tabs)
   - ASCII quotes `"`
   - LF newlines, no trailing spaces
4. YAML top-level must contain: `steps:`
5. Each step must contain **exactly**:
   - `id`
   - `title`
   - `pages`
   - `plan_nl`
6. `id` must be a quoted string: `"1"`, `"2"`, ...
7. `plan_nl` must use block scalar:
   - `plan_nl: |`
   - Content indented by **6 spaces**
8. `plan_nl` must be short (1–3 lines only).
9. Do NOT add extra top-level fields except the required `policies` block.

---

# 2) Fixed YAML Header Comment (MUST)

At the very top:

# ============================
# Lesson Plan: RAZ/G/<PDF_NAME>
# ============================

`<PDF_NAME>` = PDF filename without extension, normalized:
- The course ID and course name are connected by an underscore '_'
- the course name uses Upper Camel Case.
- Example: 02_TheFoodChain

---

# 3) Step Schema Rules

## Titles (only these allowed)
- `greet_1`
- `greet_2`
- `cover`
- `page_y` (y = page number)
- `review`
- `goodbye`

## Pages field
- List of integers.
- For each new page `y`, the first step must be `pages: [y]`
- Other steps usually `pages: []`
- Only the `pages: [y]` step may include the page-turn instruction.

---

# 4) Lesson Flow Rules

## A) Greeting Section (Fixed 4 steps, MUST)

### Step 1 — greet_1 (fixed sentence)
- Use this prompt at first:
  First, introduce yourself: "Hello, I am your English teacher today."
- Ask one greeting question.

### Step 2 — greet_1
- Brief response + short chat (NO topic chain).

### Step 3 — greet_2
- Respond + ask second greeting question.

### Step 4 — greet_2
- Brief chat only, then transition to cover.

Greeting must stay low-turn, no multi-follow-up.
Greeting questions must come from Greeting Questions Bank.
[CRITICAL] The greeting question **cannot** be the same for different lesson plans.
### Greeting Questions Bank:
- "How are you today?"
- "How was your day at school?"
- "What did you do after school today?"
- "What did you eat for breakfast?"
- "What did you eat for lunch?"
- "What is your favorite food?"
- "Do you like vegetables or fruit?"
- "What animal do you like best?"
- "Have you ever seen a frog or a fish?"
- "What animals can you find near your home?"
- "Do you have a pet? What is it?"
- "What do you usually do on weekends?"
- "What is your favorite game to play?"
- "What book did you read recently?"
- "What is your favorite subject at school?"
- "Do you like science? Why or why not?"
- "Have you ever been to a zoo?"
- "What is something fun you learned this week?"
- "Do you like being outside in nature?"
- "What makes you happy today?"

## B) Cover Section (Fixed 2 steps, MUST)
### Step 5 — cover (pages: [1])
Use this prompt at first:
- Clearly prompt student to turn to the cover page: "OK, let's begin the lesson and turn to the cover page."
Then:
- Ask ONE prediction question about the cover picture.

### Step 6 — cover (pages: [])
Use this prompt at first:
- Introduce the text:
Then followed by giving a fixed full-book overview sentence.
End with the fixed question:
- Then ask: "Let's start today’s lesson, OK?"
Cover predictions are never corrected.

## C) Main Reading Section (Page-by-page)
### Global Rule
- Every sentence on every main content page must be read aloud.
- Reading is the core; questions focus on text meaning.
- No using brackets such as ( ) or [ ].
- Each prompt that requires a question should be followed by a general prompt:
  - Wait for the student’s response and correct mistakes.
- [CRITICAL] Before moving to the next page, it must clearly prompt to turn the page by using sentences from Page Turning Bank. For example: Clearly guide the student to turn to page 5: "Now, let`s turn to page 5."
    #### Page Turning Prompt Bank:
    - Clearly guide the student to turn to page __: "Now, let`s turn to page __."
    - Clearly guide the student to turn to page __: "Turn to page __."
    The blank spaces should be filled with the correct page number.

Each page must include:
### 1. Reading Steps (Required)
There are two types of reading aloud: teacher-led reading and independent student reading. 
Each page only requires choosing one method for reading aloud.
The first few pages can be the teacher lead-reads and subsequent pages can be selected between the two method.
Reading steps must use prompts from the respective prompt banks below.
[CRITICAL] The teacher’s read-aloud content must be **completely** identical to the passage on the current page. Any reading of text from subsequent pages is **strictly prohibited**. You must repeatedly check that the read-aloud text is correct.
#### Prompt banks:
(a) Short page (1–2 sentences): Use 1 reading step.
- Teacher lead-reads prompt:
  - The teacher reads the text first: "Read aloud with me. __"   
  Tips: The blank space contains the text to be read aloud.
- Students read aloud independently prompt:
  - Ask the student to read the text: "Can you read this text?"     

(b) Long page (3–4+ sentences): Use 2 reading steps, first 2 sentences, then the rest.
- Teacher lead-reads prompt:
  - The teacher reads the text first: "Read aloud with me. __"
  - After reading the first two sentences, then the teacher reads the text: "Go on reading. __"   
  Tips: The blank space contains the text to be read aloud.
- Students read aloud independently prompt:
  - Ask the student to read the text: "Can you read first two sentences?"
  - After the student reads the first two sentences, then ask the student to read: "Now, can you read the remaining sentences?"

### 2. Text Question Step (Required)
Add 1 step asking ONE text-based question direction.
Do NOT write specific questions, only direction prompts:
- Ask a question about an important detail in the text.
- Ask a question about sequence (what happens first/next).
- Ask a question about cause and effect in the text.
- Ask a question to check the meaning of a key word/phrase from the text.
- Ask a question about what might happen next in the story.
- Ask a question about the actions in the text.
- Ask a question to focus on specific elements or actions mentioned in the page.

Short pages should NOT force main idea; use detail/vocab/picture instead.

### 3. Picture Question Step (Required)
Add 1 step asking ONE picture-based question direction.
Do NOT write specific questions, only direction prompts:
- Ask a question about the character's actions or their feeling.
- Ask a question about how two things in current and last pages are alike or different.
- Ask a question linking the text to the picture/diagram.
- Ask a question about the details of the picture.

### 4. Divergent Question Step (Optional, Limited)
Only once every ~3 pages or at natural breakpoints.
Use one short prompt:
- Ask a divergent question about __.
Attention: You should fill the blank with a topic that links between the story and the student’s life or experience.

## D) Review Section (After Last Page)
Add 2 step to review.
Ensure lesson plan variety and avoid repeating the same question multiple times.
### Review Step 1 — review
Ask ONE question, choose one of these prompts:
- Ask a question to review key vocabulary or concepts from the text.
- Ask a question about specific details from the story.
- Ask a question about the sentence structure or pattern used in the text.
- Ask a question about making a sentence using the same structure as in the text.
- Ask a question about a simple action in the text.
If there is a vocabulary list, sometime you can choose to jump to the page containing the vocabulary list and use the prompts:
  - Ask the student to read aloud these words from the vocabulary list.
  Read aloud the vocabulary words is just a optional choice as review step 1.

### Review Step 2 — review 
Ask ONE question, choose one of these prompts:
- Ask a question about the relationship between the character and another person/animal.
- Ask a question about what would happen if the story continued.
- Ask a question about what happens in the beginning of the story.
- Ask a question about the character felt during different actions
- Ask a question about the character's trying to achieve.
- Ask a question about the main idea of the whole text.

## E) Goodbye Section (Fixed 1 step)
Use this prompt at the end:
- The teacher gives a summary and ends the class: "Great job today! You smiled with every page! See you next time!"

---

# 5) Fixed Policies Block (MUST)

Append exactly at the end:

policies:
  defaults:
    step_turns: 1

Do not rename keys. Do not add anything inside `policies`.

---
# 7) Example
Here is an example of how a lesson plan was generated:
# ============================
# Lesson Plan: RAZ/F/01_GaggleHerdAndMurder
# ============================

steps:
  - id: "1"
    title: "greet_1"
    pages: []
    plan_nl: |
      First, introduce yourself: "Hello, I am your English teacher today."
      Then ask: "What makes you happy today?"

  - id: "2"
    title: "greet_1"
    pages: []
    plan_nl: |
      Briefly respond and chat with the student.
      Keep it short and encouraging.

  - id: "3"
    title: "greet_2"
    pages: []
    plan_nl: |
      Respond to the student.
      Then ask: "What animal do you like best?"

  - id: "4"
    title: "greet_2"
    pages: []
    plan_nl: |
      Briefly chat.

  - id: "5"
    title: "cover"
    pages: [1]
    plan_nl: |
      Clearly prompt student to turn to the cover page: "OK, let's begin the lesson and turn to the cover page."
      Ask ONE prediction question about the cover picture.

  - id: "6"
    title: "cover"
    pages: []
    plan_nl: |
      Introduce the text: "Today we are going to read a book called "Gaggle, Herd, and Murder." We will learn the special names for animal groups."
      Then ask: "Let's start today’s lesson, OK?"

  - id: "7"
    title: "page_3"
    pages: [3]
    plan_nl: |
      Clearly guide the student to turn to page 3: "Turn to page 3."
      The teacher reads the text first: "Read aloud with me. Animals are everywhere. Sometimes they gather in groups."

  - id: "8"
    title: "page_3"
    pages: []
    plan_nl: |
      After reading the first two sentences, then the teacher reads the text: "Go on reading. Animal groups have different names."
      Wait for the student’s response and correct mistakes.

  - id: "9"
    title: "page_3"
    pages: []
    plan_nl: |
      Ask a question about an important detail in the text.
      Wait for the student’s response and correct mistakes.

  - id: "10"
    title: "page_3"
    pages: []
    plan_nl: |
      Ask a question linking the text to the picture/diagram.
      Wait for the student’s response and correct mistakes.

  - id: "11"
    title: "page_4"
    pages: [4]
    plan_nl: |
      Clearly guide the student to turn to page 4: "Now, let`s turn to page 4."
      The teacher reads the text first: "Read aloud with me. A group of geese is a gaggle. A gaggle of geese stands together."

  - id: "12"
    title: "page_4"
    pages: []
    plan_nl: |
      Ask a question to focus on specific elements or actions mentioned in the page.
      Wait for the student’s response and correct mistakes.

  - id: "13"
    title: "page_4"
    pages: []
    plan_nl: |
      Ask a question about the details of the picture.
      Wait for the student’s response and correct mistakes.

  - id: "14"
    title: "page_5"
    pages: [5]
    plan_nl: |
      Clearly guide the student to turn to page 5: "Turn to page 5."
      The teacher reads the text first: "Read aloud with me. A group of bats is a colony. A colony of bats may leave its cave all at once."

  - id: "15"
    title: "page_5"
    pages: []
    plan_nl: |
      Ask a question about cause and effect in the text.
      Wait for the student’s response and correct mistakes.

  - id: "16"
    title: "page_5"
    pages: []
    plan_nl: |
      Ask a question about the details of the picture.
      Wait for the student’s response and correct mistakes.

  - id: "17"
    title: "page_6"
    pages: [6]
    plan_nl: |
      Clearly guide the student to turn to page 6: "Now, let`s turn to page 6."
      The teacher reads the text first: "Read aloud with me. A murder is a group of crows. This murder gathers at sunset."

  - id: "18"
    title: "page_6"
    pages: []
    plan_nl: |
      Ask a question to check the meaning of a key word/phrase from the text.
      Wait for the student’s response and correct mistakes.

  - id: "19"
    title: "page_6"
    pages: []
    plan_nl: |
      Ask a question linking the text to the picture/diagram.
      Wait for the student’s response and correct mistakes.

  - id: "20"
    title: "page_6"
    pages: []
    plan_nl: |
      Ask a divergent question about flock of birds.

  - id: "21"
    title: "page_7"
    pages: [7]
    plan_nl: |
      Clearly guide the student to turn to page 7: "Turn to page 7."
      Ask the student to read the text: "Can you read this text?"

  - id: "22"
    title: "page_7"
    pages: []
    plan_nl: |
      Ask a question about an important detail in the text.
      Wait for the student’s response and correct mistakes.

  - id: "23"
    title: "page_7"
    pages: []
    plan_nl: |
      Ask a question linking the text to the picture/diagram.
      Wait for the student’s response and correct mistakes.

  - id: "24"
    title: "page_8"
    pages: [8]
    plan_nl: |
      Clearly guide the student to turn to page 8: "Now, let`s turn to page 8."
      Ask the student to read the text: "Can you read this text?"

  - id: "25"
    title: "page_8"
    pages: []
    plan_nl: |
      Ask a question about cause and effect in the text.
      Wait for the student’s response and correct mistakes.

  - id: "26"
    title: "page_8"
    pages: []
    plan_nl: |
      Ask a question about how two things in current and last pages are alike or different.
      Wait for the student’s response and correct mistakes.

  - id: "27"
    title: "page_9"
    pages: [9]
    plan_nl: |
      Clearly guide the student to turn to page 9: "Turn to page 9."
      Ask the student to read the text: "Can you read this text?"

  - id: "28"
    title: "page_9"
    pages: []
    plan_nl: |
      Ask a question about an important detail in the text.
      Wait for the student’s response and correct mistakes.

  - id: "29"
    title: "page_9"
    pages: []
    plan_nl: |
      Ask a question about the details of the picture.
      Wait for the student’s response and correct mistakes.

  - id: "30"
    title: "page_9"
    pages: []
    plan_nl: |
      Ask a divergent question about visiting zoo.

  - id: "31"
    title: "page_10"
    pages: [10]
    plan_nl: |
      Clearly guide the student to turn to page 10: "Now, let`s turn to page 10."
      Ask the student to read the text: "Can you read this text?"

  - id: "32"
    title: "page_10"
    pages: []
    plan_nl: |
      Ask a question about cause and effect in the text.
      Wait for the student’s response and correct mistakes.

  - id: "33"
    title: "page_10"
    pages: []
    plan_nl: |
      Ask a question about the details of the picture.
      Wait for the student’s response and correct mistakes.

  - id: "34"
    title: "page_11"
    pages: [11]
    plan_nl: |
      Clearly guide the student to turn to page 11: "Turn to page 11."
      Ask the student to read the text: "Can you read first two sentences?"

  - id: "35"
    title: "page_11"
    pages: []
    plan_nl: |
      After the student reads the first two sentences, then ask the student to read: "Now, can you read the remaining sentences?"
      Wait for the student’s response and correct mistakes.

  - id: "36"
    title: "page_11"
    pages: []
    plan_nl: |
      Ask a question about an important detail in the text.
      Wait for the student’s response and correct mistakes.

  - id: "37"
    title: "page_11"
    pages: []
    plan_nl: |
      Ask a question linking the text to the picture/diagram.
      Wait for the student’s response and correct mistakes.

  - id: "38"
    title: "review"
    pages: [12]
    plan_nl: |
      Clearly guide the student to turn to page 12: "Turn to page 12."
      Ask the student to read aloud these words from the vocabulary list.

  - id: "39"
    title: "review"
    pages: [12]
    plan_nl: |
      Ask a question about the main idea of the whole text.
      Wait for the student’s response and correct mistakes.

  - id: "40"
    title: "goodbye"
    pages: []
    plan_nl: |
      The teacher gives a summary and ends the class: "Great job today! You smiled with every page! See you next time!"

policies:
  defaults:
    step_turns: 1

---

# Now Generate
Read the provided PDF content and page structure, then output the YAML lesson plan following ALL rules above **as a single ```yaml code block and nothing else**.
