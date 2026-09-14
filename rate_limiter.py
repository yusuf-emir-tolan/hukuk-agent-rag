from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:


    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:

        now = time.time()
        with self._lock:
            dq = self._hits[key]

            while dq and dq[0] <= now - self.window_seconds:
                dq.popleft()

            if len(dq) >= self.max_requests:
                return False

            dq.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()