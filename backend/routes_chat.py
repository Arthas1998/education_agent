from flask import Blueprint, request, current_app, jsonify, Response
import os

from utils.prompt_loader import PromptLoader
from utils.chat import Chat
from utils.config import DEFAULT_PROMPT_PATH, ALLOWED_AUDIO_EXTS, MAX_UPLOAD_MB

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@chat_bp.route("/text", methods=["POST"])
def text_chat():
    data = request.get_json(force=True)
    course_id = data.get("course_id")
    page = data.get("page")
    query = data.get("query", "")

    if not course_id or not page:
        return jsonify({"error": "course_id and page required"}), 400

    pm = current_app.pdf_manager
    meta = pm.courses.get(course_id)
    if not meta:
        return jsonify({"error": "course not found"}), 404
    if page < 1 or page > meta.total_pages:
        return jsonify({"error": "page out of range"}), 400

    # try get image base64
    image_b64 = pm.get_page_image_base64(course_id, page)

    # build prompt loader - choose prompt config if exists
    prompt_path = os.path.join(current_app.config.get("PROJECT_ROOT"), "prompt", "config", "eh", f"{meta.id}_{meta.title}.yaml")
    if not os.path.exists(prompt_path):
        # fallback to DEFAULT_PROMPT_PATH from utils.config
        prompt_path = os.path.join(current_app.config.get("PROJECT_ROOT"), DEFAULT_PROMPT_PATH)

    # 新版 PromptLoader
    loader = PromptLoader.from_yaml(prompt_path)

    # build a minimal prompt: append user question to messages
    # re-use utils.chat.Chat infrastructure by constructing a client stub if network unavailable
    try:
        from openai import OpenAI
        client = OpenAI(api_key=None)
    except Exception:
        client = None

    chat = Chat(client=client, prompt_loader=loader)
    chat.add_user_text(query)

    # If streaming is not supported by client, collect full reply
    try:
        chunks = []
        for chunk in chat.stream_reply():
            chunks.append(chunk)
        full = "".join(chunks)
    except Exception:
        # fallback: return last assistant if any
        full = chat.get_last_assistant_reply() or ""

    return jsonify({"response": full, "meta": {"model": chat.model}})


@chat_bp.route('/text/stream', methods=['POST'])
def text_stream():
    data = request.get_json(force=True)
    course_id = data.get('course_id')
    page = data.get('page')
    query = data.get('query', '')

    if not course_id or not page:
        return jsonify({"error": "course_id and page required"}), 400

    pm = current_app.pdf_manager
    meta = pm.courses.get(course_id)
    if not meta:
        return jsonify({"error": "course not found"}), 404

    # Import SSEAdapter here to avoid static analysis resolution issues
    from backend.sse import SSEAdapter
    sse_adapter = SSEAdapter()

    # For demo, push the query to SSE adapter and stream it back
    sse_adapter.push(f"query:{query}")

    return Response(sse_adapter.event_stream(), mimetype='text/event-stream')


@chat_bp.route('/asr', methods=['POST'])
def asr_upload():
    # accept multipart/form-data
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400
    f = request.files['file']
    filename = f.filename
    if not filename:
        return jsonify({"error": "empty filename"}), 400

    # Validate extension
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in ALLOWED_AUDIO_EXTS:
        return jsonify({"error": f"extension not allowed: {ext}"}), 400

    # Validate size (using stream if available)
    f.stream.seek(0, os.SEEK_END)
    size = f.stream.tell()
    f.stream.seek(0)
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    if size > max_bytes:
        return jsonify({"error": "file too large"}), 400

    # save to temp
    tmpdir = os.path.join(current_app.config.get('PROJECT_ROOT'), 'run')
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, filename)
    f.save(path)

    # call utils.asr
    from utils.asr import recognize_file
    try:
        text = recognize_file(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"text": text})
