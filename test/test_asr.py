# test/test_asr.py
# Upload test/temp.wav to POST /api/chat/asr and print result
import os
import sys
import json
import requests

BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:19980")


def main():
    here = os.path.dirname(__file__)
    wav_path = os.path.join(here, "temp.wav")
    if not os.path.exists(wav_path):
        print(f"ERROR: test audio not found: {wav_path}")
        sys.exit(2)

    url = f"{BASE_URL}/api/chat/asr"
    print(f"POST {url} with file: {wav_path}")

    try:
        with open(wav_path, "rb") as fh:
            files = {"file": (os.path.basename(wav_path), fh, "audio/wav")}
            resp = requests.post(url, files=files, timeout=60)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        sys.exit(3)

    print(f"HTTP {resp.status_code}")
    ct = resp.headers.get("Content-Type", "")
    if "application/json" in ct:
        try:
            data = resp.json()
        except Exception as e:
            print(f"Failed to decode JSON response: {e}")
            print(resp.text)
            sys.exit(4)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if resp.status_code == 200:
            print("ASR call succeeded")
            sys.exit(0)
        else:
            print("ASR call returned error")
            sys.exit(5)
    else:
        print("Non-JSON response:")
        print(resp.text)
        sys.exit(6)


if __name__ == "__main__":
    main()

