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


# GDC/TCGA downloads carry the file's own uuid between the barcode and the
# extension: "{barcode}.{file_uuid}.svs". wsi_registry has a file_uuid column
# for it, so recover it here — beside the pattern that already knows the shape
# — rather than re-deriving the same regex at the call site.
_GDC_FILE_UUID_RE = re.compile(rf"^[^.]+\.(?P<file_uuid>{_UUID4})\.")


def file_uuid_from_raw_path(raw_path):
    """The GDC file uuid in a downloaded slide's name, or None.

    None for anything not named that way — an app upload, a locally produced
    slide — because there is no uuid to report, not because one is missing.
    """
    match = _GDC_FILE_UUID_RE.match(Path(raw_path).name)
    return match.group("file_uuid") if match else None


# --- the tile_coordinates / tile_registry join key -------------------------
#
# slide_tile is "<slides>_<tiles>" upper-cased, and it is the primary key of
# both tile_coordinates and tile_registry. The two sides of the pipeline
# disagree about the tile name, which is why this is centralised:
#
#   auto_tile_from_mask.py  writes tiles as "24_10.jpeg"  (Stage 1 metadata CSV)
#   make_hpl_hdf5.py        writes tiles as "24_10"       (packaged .h5, and so
#                                                          the Stage 4 CSV too)
#   existing TCGA registry rows are        "..._18_15.JPEG"
#
# So a key built from Stage 4's CSV could never match a row registered from
# Stage 1's CSV, and neither could match the TCGA rows already loaded. Both
# sides go through here instead.
#
# This does NOT repair a tile name that is missing its extension. The fix for
# that belongs in make_hpl_hdf5.py, which is where the suffix was being dropped;
# quietly appending one here would let a .h5 packaged before that fix produce a
# key that matches, while its (slides, tiles) columns still disagree with Kai's
# reference CSV and so still fail the acceptance test. Use tiles_missing_suffix()
# to refuse such a file instead.
_TILE_SUFFIX = ".JPEG"

# Only the tile part is ever inspected for an extension. A slide name may itself
# contain dots — a real one is "BB232560 A3-1 - 2023-10-11 16.41.02" — so
# testing the concatenated key would read that timestamp's ".02" as a file
# extension.
_HAS_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,5}$")


def make_slide_tile(slide, tile) -> str:
    """The join key for one tile: "<slides>_<tiles>", upper-cased."""
    return f"{str(slide).strip().upper()}_{str(tile).strip().upper()}"


def make_slide_tile_series(slides, tiles):
    """make_slide_tile over two pandas Series, returning a Series.

    Kept beside the scalar version and pinned to it by test, so the vectorised
    path used for millions of rows cannot drift from the definition.
    """
    return (
        slides.astype(str).str.strip().str.upper()
        + "_"
        + tiles.astype(str).str.strip().str.upper()
    )


def _as_text(value) -> str:
    """Tile names arrive as str from a CSV and as bytes from HDF5. Decoding
    matters more than it looks: str(b"18_15.jpeg") is "b'18_15.jpeg'", whose
    last character is a quote, so an extension test against it says the suffix
    is missing on a file that has it."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace").strip()
    return str(value).strip()


def tiles_missing_suffix(tiles) -> bool:
    """True if these tile names have no file extension.

    The signature of a .h5 packaged before make_hpl_hdf5.py started storing
    "18_15.jpeg" rather than "18_15". Every consumer of such a file is wrong in
    the same way — the KB join matches nothing and --validate-against merges
    zero rows — so it is worth naming rather than working around.
    """
    sample = [t for t in (_as_text(v) for v in list(tiles)[:100]) if t]
    if not sample:
        return False
    return not any(_HAS_EXTENSION_RE.search(t) for t in sample)
