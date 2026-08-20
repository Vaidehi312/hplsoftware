"""Guards that stop a packaged .h5 with stale tile names reaching the KB.

A .h5 packaged before make_hpl_hdf5.py kept the ".jpeg" suffix is structurally
perfect — four datasets, equal lengths, every row readable — and wrong in the
one place nothing looked: the contents of `tiles`. It read as "ready" in the UI
all the way to a 0.0% Knowledge Bank match on 38,892 tiles.

Two guards close that, and each is tested by proving it can come out bad:

  _validate_h5        rejects such a file at the packaging gate
  run_identity        refuses to resume a checkpoint written under the old
                      tile-name format, which would otherwise produce a file
                      holding both forms at once
"""

import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from slide_naming import tiles_missing_suffix  # noqa: E402

SLIDE = b"TCGA-55-7574-01Z-00-DX1"


def _write_packaged_h5(path: Path, tiles, rows=4):
    """A packaged .h5 exactly as package_slides_to_h5 leaves it."""
    with h5py.File(path, "w") as f:
        f.create_dataset("img", (rows, 4, 4, 3), dtype="uint8")
        f.create_dataset("samples", data=np.array([b"TCGA-55-7574"] * rows))
        f.create_dataset("slides", data=np.array([SLIDE] * rows))
        f.create_dataset("tiles", data=np.array(tiles[:rows]))


def _validate(path: Path):
    """_validate_h5 without importing tile_server_v2_, which builds a FastAPI
    app and opens HDF5 handles at import time."""
    with h5py.File(path, "r") as f:
        rows = f["img"].shape[0]
        head = f["tiles"][: min(rows, 100)]
        return not tiles_missing_suffix(head)


def test_stale_tile_names_are_rejected(tmp_path):
    stale = tmp_path / "stale.h5"
    _write_packaged_h5(stale, [b"18_15", b"14_6", b"20_3", b"7_9"])
    assert not _validate(stale), (
        "a .h5 whose tiles lack the suffix must be rejected — this is the file "
        "that reported 'ready' and then matched 0.0% of the KB"
    )


def test_correctly_named_h5_is_accepted(tmp_path):
    good = tmp_path / "good.h5"
    _write_packaged_h5(good, [b"18_15.jpeg", b"14_6.jpeg", b"20_3.jpeg", b"7_9.jpeg"])
    assert _validate(good)


def test_the_guard_reads_bytes_not_their_repr(tmp_path):
    """The failure mode that would make this guard reject everything: h5py
    returns bytes, and str(b"18_15.jpeg") ends in a quote, not in ".jpeg"."""
    good = tmp_path / "good.h5"
    _write_packaged_h5(good, [b"18_15.jpeg"] * 4)
    with h5py.File(good, "r") as f:
        raw = f["tiles"][:]
    assert isinstance(raw[0], (bytes, np.bytes_)), type(raw[0])
    assert not tiles_missing_suffix(raw)


def test_structural_checks_alone_cannot_catch_it(tmp_path):
    """Why the content check had to be added: the stale file passes everything
    that was already being checked."""
    stale = tmp_path / "stale.h5"
    _write_packaged_h5(stale, [b"18_15", b"14_6", b"20_3", b"7_9"])
    with h5py.File(stale, "r") as f:
        rows = f["img"].shape[0]
        assert all(name in f for name in ("img", "samples", "slides", "tiles"))
        assert rows == 4
        assert all(f[name].shape[0] == rows for name in ("samples", "slides", "tiles"))
        f["img"][0], f["img"][rows - 1], f["slides"][rows - 1]  # readable ends


def test_packaging_actually_stores_the_suffix(tmp_path):
    """Read the value back out of a real packaged .h5.

    The earlier test asserted on the source string, which is not the same
    claim: `tiles` is a FIXED-LENGTH byte column whose width was sized from
    the bare "24_10". Writing "24_10.jpeg" into an S5 field truncates it back
    to "24_10" — the exact bug the suffix was added to fix, restored silently
    on write, with the source looking correct the whole time. Only reading the
    stored bytes catches that.
    """
    from PIL import Image
    import pandas as pd
    from make_hpl_hdf5 import package_slides_to_h5

    slide_id = "BB232560 A3-1 - 2023-10-11 16.41.02"
    tile_dir = tmp_path / "tiles"
    slide_dir = tile_dir / "Radiogenomics" / slide_id
    slide_dir.mkdir(parents=True)

    records = []
    for col, row in [(24, 10), (25, 10), (7, 3)]:
        name = f"{col}_{row}.jpeg"
        Image.new("RGB", (224, 224), (200, 150, 180)).save(slide_dir / name, "JPEG")
        records.append({"slides": slide_id, "tiles": name,
                        "slide_tile": f"{slide_id}_{name}", "col": col, "row": row,
                        "x_5x": col * 224, "y_5x": row * 224,
                        "x_native": col * 1600, "y_native": row * 1600,
                        "tissue_percent": 80.0})
    pd.DataFrame(records).to_csv(slide_dir / f"{slide_id}_tile_metadata.csv", index=False)
    (slide_dir / f"{slide_id}_tiling_summary.json").write_text(
        '{"saved_tiles": 3, "target_mpp": 2.0}'
    )

    raw = tmp_path / "raw" / f"{slide_id}.svs"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"")

    result = package_slides_to_h5(
        raw_paths=[str(raw)], tile_dir=tile_dir, tile_dataset_name="Radiogenomics",
        output_root=tmp_path / "out", dataset_name="Radiogenomics_test",
        n_processes=1, batch_size=8,
    )

    with h5py.File(result["output_h5_path"], "r") as f:
        stored = [t.decode() for t in f["tiles"][:]]

    assert sorted(stored) == ["24_10.jpeg", "25_10.jpeg", "7_3.jpeg"], stored
    assert not tiles_missing_suffix(f_tiles := [s.encode() for s in stored]), f_tiles


def test_run_identity_carries_the_tile_name_format(tmp_path):
    """A checkpoint's tile *labels* are byte-identical before and after the
    suffix fix, so without this key a resume across the change would skip every
    already-written row and leave the .h5 holding both forms — structurally
    valid, unjoinable for part of its contents.
    """
    import make_hpl_hdf5

    assert make_hpl_hdf5._TILE_NAME_FORMAT == "col_row.jpeg"
    source = (BACKEND / "make_hpl_hdf5.py").read_text()
    assert '"tile_name_format": _TILE_NAME_FORMAT,' in source, (
        "run_identity must record the tile-name format"
    )

    # And the comparison that uses it is a plain dict inequality, so a config
    # written without the key cannot compare equal to one with it.
    old_config = {"tile_size": 224, "img_compression": "gzip"}
    new_config = dict(old_config, tile_name_format=make_hpl_hdf5._TILE_NAME_FORMAT)
    assert old_config != new_config

    # Round-trips through JSON the way the checkpoint does.
    written = tmp_path / "run_config.json"
    written.write_text(json.dumps(old_config))
    assert json.loads(written.read_text()) != new_config


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_pkg_guard_"))
        try:
            fn(tmp_path)
            print(f"PASS  {name}")
        except Exception as e:
            failures.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
