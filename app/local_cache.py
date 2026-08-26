"""
Client-side disk cache for images fetched from the tile server.

Sits between api_client.py and the network. Thumbnails and tiles are stored
as JPEG files keyed by a hash of the request parameters. On hit the network
round-trip is skipped entirely.
"""

import hashlib
import io
from pathlib import Path

from PIL import Image

DEFAULT_CACHE_DIR = Path.home() / ".hpc_chatbot_cache"
MAX_CACHE_MB = 1000  # 1 GB


class LocalImageCache:
    def __init__(self, cache_dir: str | Path | None = None, max_mb: int = MAX_CACHE_MB):
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_mb * 1024 * 1024

    @staticmethod
    def _key(*parts) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        sub = self.cache_dir / key[:2]
        sub.mkdir(exist_ok=True)
        return sub / f"{key}.jpg"

    def get_image(self, *key_parts) -> Image.Image | None:
        p = self._path(self._key(*key_parts))
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def get_bytes(self, *key_parts) -> bytes | None:
        p = self._path(self._key(*key_parts))
        if p.exists():
            try:
                return p.read_bytes()
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def put_image(self, img: Image.Image, *key_parts, quality: int = 85):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        self.put_bytes(buf.getvalue(), *key_parts)

    def put_bytes(self, data: bytes, *key_parts):
        p = self._path(self._key(*key_parts))
        p.write_bytes(data)

    def clear(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
