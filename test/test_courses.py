# test/test_courses.py
# Simple tests for listing courses and requesting a page image URL
import os
import sys
import json
import requests

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:19980")


def fail(msg, code=1):
    print("ERROR:", msg)
    sys.exit(code)


def main():
    url = f"{BASE_URL}/api/courses"
    print(f"GET {url}")
    resp = None
    try:
        resp = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        fail(f"Request failed: {e}", 2)

    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text)
        fail("Failed to list courses", 3)

    courses = []
    try:
        courses = resp.json()
    except Exception as e:
        fail(f"Failed to parse JSON: {e}", 4)

    print(json.dumps(courses, ensure_ascii=False, indent=2))
    if not isinstance(courses, list) or len(courses) == 0:
        print("No courses found, nothing more to test.")
        sys.exit(0)

    first = courses[0]
    cid = first.get("id")
    print(f"Testing course id: {cid}")
    page = 1
    page_url = f"{BASE_URL}/api/course/{cid}/page/{page}"
    print(f"GET {page_url}")
    resp2 = None
    try:
        resp2 = requests.get(page_url, timeout=10)
    except requests.exceptions.RequestException as e:
        fail(f"Request failed: {e}", 5)

    print(f"HTTP {resp2.status_code}")
    if resp2.status_code == 200:
        try:
            data = resp2.json()
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            print(resp2.text)
            fail("Failed to parse page response JSON", 6)
    elif resp2.status_code == 202:
        print("Page is being generated (202).")
        print(resp2.text)
    else:
        print(resp2.text)
        fail(f"Unexpected status code: {resp2.status_code}", 7)

    print("Courses test completed successfully.")
    sys.exit(0)


if __name__ == '__main__':
    main()
