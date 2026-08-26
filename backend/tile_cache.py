"""
Server-side disk cache for WSI thumbnails and tiles.

Keyed by: slide_id / level / x / y  (or slide_id for thumbnails).
Stored as JPEG on disk with an LRU eviction policy based on access time.
"""

import hashlib
import io
import os
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock

from PIL import Image

DEFAULT_CACHE_DIR = Path("/tmp/hpc_tile_cache")
MAX_CACHE_SIZE_MB = 2000  # 2 GB default


class TileCache:
    def __init__(self, cache_dir: str | Path | None = None, max_size_mb: int = MAX_CACHE_SIZE_MB):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = Lock()
        self._lru: OrderedDict[str, int] = OrderedDict()  # key -> file size
        self._total_bytes = 0
        self._index_existing()

    def _index_existing(self):
        """Walk the cache dir once at startup to rebuild the LRU index."""
        entries = []
        for p in self.cache_dir.rglob("*.jpg"):
            try:
                stat = p.stat()
                entries.append((p.stem, int(stat.st_size), stat.st_atime))
            except OSError:
                continue
        entries.sort(key=lambda e: e[2])
        for name, size, _ in entries:
            self._lru[name] = size
            self._total_bytes += size

    @staticmethod
    def _make_key(slide_id: str, level: int | None = None,
                  x: int | None = None, y: int | None = None,
                  width: int | None = None, height: int | None = None,
                  kind: str = "tile") -> str:
        raw = f"{slide_id}|{kind}|{level}|{x}|{y}|{width}|{height}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path_for(self, key: str) -> Path:
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.jpg"

    def get(self, slide_id: str, **kwargs) -> Image.Image | None:
        key = self._make_key(slide_id, **kwargs)
        p = self._path_for(key)
        if not p.exists():
            return None
        try:
            os.utime(p)  # touch access time
            with self._lock:
                self._lru.move_to_end(key)
            return Image.open(p).convert("RGB")
        except Exception:
            return None

    def put(self, img: Image.Image, slide_id: str, quality: int = 85, **kwargs) -> None:
        key = self._make_key(slide_id, **kwargs)
        p = self._path_for(key)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        p.write_bytes(data)
        with self._lock:
            self._lru[key] = len(data)
            self._lru.move_to_end(key)
            self._total_bytes += len(data)
            self._evict()

    def _evict(self):
        while self._total_bytes > self.max_size_bytes and self._lru:
            oldest_key, oldest_size = self._lru.popitem(last=False)
            p = self._path_for(oldest_key)
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
            self._total_bytes -= oldest_size

    def clear(self):
        import shutil
        with self._lock:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._lru.clear()
            self._total_bytes = 0
