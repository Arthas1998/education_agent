import queue
from typing import Generator


class SSEAdapter:
    """Simple in-process SSE adapter using queue.Queue.

    Usage:
        adapter = SSEAdapter()
        stream = adapter.event_stream()  # generator
        adapter.push("hello")

    The generator yields strings suitable for Flask Response with mimetype 'text/event-stream'.
    """

    def __init__(self, maxsize: int = 1000):
        self.q = queue.Queue(maxsize=maxsize)

    def push(self, event: str):
        try:
            self.q.put(event, block=False)
        except Exception:
            # drop if full
            pass

    def event_stream(self, timeout: float = 0.1) -> Generator[str, None, None]:
        # keep yielding as long as server runs; this is a simple implementation
        while True:
            try:
                item = self.q.get(timeout=timeout)
                if item is None:
                    # sentinel for close
                    break
                # SSE format: data: <line>\n\n
                yield f"data: {item}\n\n"
            except Exception:
                # timeout -> keep connection alive with comment
                yield f": keep-alive\n\n"
