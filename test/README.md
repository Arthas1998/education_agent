Test scripts for standalone testing of backend API endpoints.

Files:
- test_asr.py: Uploads `temp.wav` to POST /api/chat/asr and prints JSON result.
- test_courses.py: GET /api/courses and GET /api/course/<id>/page/1 to verify course listing and page retrieval.
- test_text_stream.py: Interactive tool that POSTs to /api/chat/text/stream and parses SSE-like streaming output.

Setup:
1. Ensure backend server is running (defaults to http://localhost:8000). To run the app locally:
   - python backend/app.py

2. Install dependencies (recommended inside a virtualenv):

```powershell
python -m pip install -r test/requirements.txt
```

Usage examples:

```powershell
# ASR test
python test/test_asr.py

# Courses test
python test/test_courses.py

# Interactive text stream (enter course id, page and queries)
python test/test_text_stream.py
```

Environment variables:
- TEST_BASE_URL: override base URL (default: http://localhost:8000)
- TEST_COURSE_ID: default course id used by test_text_stream
- TEST_PAGE: default page number used by test_text_stream

Notes:
- The ASR endpoint may depend on external credentials (DashScope). If ASR fails with 500, check environment/config.
- The scripts are lightweight and avoid extra dependencies; SSE parsing is implemented with plain `requests`.

