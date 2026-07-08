import time
from typing import Any, Optional


class MemoryCache:
    """
    Simple in-memory cache with TTL (Time To Live).

    Stores movie analysis results to avoid repeatedly
    fetching reviews and performing sentiment analysis.

    Example:
        cache.set("movie_603", result)

        result = cache.get("movie_603")
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.cache = {}

    # ---------------------------------------------------------
    # Save item
    # ---------------------------------------------------------

    def set(self, key: str, value: Any):

        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }

    # ---------------------------------------------------------
    # Retrieve item
    # ---------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:

        if key not in self.cache:
            return None

        item = self.cache[key]

        age = time.time() - item["timestamp"]

        if age > self.ttl:
            del self.cache[key]
            return None

        return item["value"]

    # ---------------------------------------------------------
    # Delete one item
    # ---------------------------------------------------------

    def delete(self, key: str):

        if key in self.cache:
            del self.cache[key]

    # ---------------------------------------------------------
    # Clear everything
    # ---------------------------------------------------------

    def clear(self):

        self.cache.clear()

    # ---------------------------------------------------------
    # Cleanup expired items
    # ---------------------------------------------------------

    def cleanup(self):

        now = time.time()

        expired = []

        for key, value in self.cache.items():

            if now - value["timestamp"] > self.ttl:
                expired.append(key)

        for key in expired:
            del self.cache[key]

    # ---------------------------------------------------------
    # Cache size
    # ---------------------------------------------------------

    def size(self):

        return len(self.cache)
