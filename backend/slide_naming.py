"""Shared helper for recovering a clean slide_id from an uploaded WSI path.

tile_server_v2.upload_slide() saves raw files as
"{slide_id}_{upload_uuid}_{original_filename}", so Path(...).stem alone
pulls the upload uuid and original filename in with it. tile_mask.py and
auto_tile_from_mask.py both need the same clean slide_id (it becomes the
mask filename prefix and the output folder name), so this lives in one
place instead of being reimplemented per script.
"""

import re
from pathlib import Path

_UUID4 = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

# App uploads: "{slide_id}_{upload_uuid}_{original_filename}"
_UPLOAD_UUID_RE = re.compile(rf"^(?P<slide_id>.+?)_{_UUID4}_")
# GDC/TCGA downloads: "{barcode}.{file_uuid}.svs"
_GDC_UUID_RE = re.compile(rf"^(?P<slide_id>[^.]+)\.{_UUID4}\.")


def slide_id_from_raw_path(raw_path) -> str:
    name = Path(raw_path).name
    for pattern in (_UPLOAD_UUID_RE, _GDC_UUID_RE):
        match = pattern.match(name)
        if match:
            return match.group("slide_id")
    return Path(raw_path).stem
