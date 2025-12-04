# test/test_text_stream.py
# Interactive test for POST /api/chat/text/stream using SSE-like response
import os
import sys
import time
import requests

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:19980")


def stream_text(course_id, page, query, timeout=30):
    url = f"{BASE_URL}/api/chat/text/stream"
    payload = {"course_id": course_id, "page": page, "query": query}
    headers = {"Content-Type": "application/json"}
    print(f"POST {url} payload={payload}")
    try:
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False

    print(f"HTTP {resp.status_code}, Content-Type: {resp.headers.get('Content-Type')}")
    if resp.status_code != 200:
        try:
            print(resp.text)
        except Exception:
            pass
        return False

    # Parse SSE-like stream: lines like 'data: <payload>\n\n' and comments starting with ':'
    try:
        buffer = ""
        start = time.time()
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                break
            line = raw.strip()
            if not line:
                # event boundary
                if buffer:
                    # print full event
                    print(f"EVENT: {buffer}")
                    buffer = ""
                continue
            # SSE comment
            if line.startswith(":"):
                # keep-alive comment
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                # print incrementally
                print(data)
            else:
                # unknown line
                print(line)
            # small timeout guard
            if time.time() - start > timeout:
                print("Timeout reached while streaming")
                break
    except requests.exceptions.ChunkedEncodingError:
        print("Stream ended")
    except Exception as e:
        print(f"Error reading stream: {e}")
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return True


def main():
    # Ask user for course id and page to use for testing
    print("Interactive text stream test. Press Ctrl+C to exit.")
    default_course = os.environ.get("TEST_COURSE_ID")
    default_page = int(os.environ.get("TEST_PAGE", "1"))

    if default_course:
        print(f"Using TEST_COURSE_ID from env: {default_course}")

    try:
        while True:
            course_id = input(f"course_id [{default_course or ''}]: ") or default_course
            if not course_id:
                print("course_id is required")
                continue
            page_in = input(f"page [{default_page}]: ") or str(default_page)
            try:
                page = int(page_in)
            except Exception:
                print("invalid page")
                continue
            query = input("query (single line): ")
            if not query:
                print("empty query, skipping")
                continue

            ok = stream_text(course_id, page, query)
            print("--- request finished ---\n")
    except KeyboardInterrupt:
        print('\nExiting')
        sys.exit(0)


if __name__ == '__main__':
    main()

