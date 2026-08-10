"""Runs the real, patched real_encode_contrastive_from_checkpoint end to end.

Every other test around this change works on reproductions of the loop or on
the merge in isolation. This one imports the actual function out of the
HPL-LATTICeA subtree and executes it, with TensorFlow replaced by a stub whose
session.run returns a value derived from the pixels it was handed. That makes
the whole path observable — the prefetch thread, the slice writes, the row
offsets, the part filenames — and lets the sharded result be compared against
the unsharded one produced by the same code.

TensorFlow itself is not the thing under test and cannot be installed
alongside this repo's Python anyway; the encoder graph is frozen upstream code
that this change does not touch.

Skipped when the subtree or the encoder's own imports (matplotlib, sklearn)
are unavailable, since neither is a dependency of this repo.
"""

import shutil
import sys
import tempfile
import types
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HPL = Path(__file__).resolve().parents[2] / "HPL-LATTICeA"
ROWS, TILE = 500, 4
H_DIM, Z_DIM = 1536, 128


def _available() -> bool:
    if not (HPL / "models" / "evaluation" / "features.py").is_file():
        return False
    try:
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _install_tensorflow_stub():
    """A tensorflow whose session.run encodes which rows it was fed.

    outputs[i][r] = pixel_value(r) * (1..dim), so a row written to the wrong
    place is visible in the result rather than merely plausible.
    """
    tf = types.ModuleType("tensorflow")

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self, ops, feed_dict=None):
            if not isinstance(ops, list):
                return None
            batch = list(feed_dict.values())[0]
            keys = batch[:, 0, 0, 0].astype(np.float32)
            return [
                np.outer(keys, np.arange(1, op.shape[1] + 1)).astype(np.float32)
                for op in ops
            ]

    class _Saver:
        def restore(self, session, checkpoint):
            pass

    tf.Session = _Session
    tf.train = types.SimpleNamespace(Saver=lambda *a, **k: _Saver())
    tf.global_variables_initializer = lambda: None
    tf.placeholder = lambda **k: None
    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.contrib"] = tf


def _fake_model():
    tensor = lambda dim: types.SimpleNamespace(shape=(None, dim))  # noqa: E731
    return types.SimpleNamespace(
        model_name="BarlowTwins_3", z_dim=Z_DIM,
        h_rep_out=tensor(H_DIM), z_rep_out=tensor(Z_DIM), real_images_2="placeholder",
    )


def _fake_data():
    return types.SimpleNamespace(dataset="DS", patch_h=224, patch_w=224, n_channels=3)


def _write_input(path: Path):
    """Chunked and gzipped exactly as make_hpl_hdf5.py writes it, so the read
    path under test is the real one."""
    idx = (np.arange(ROWS) % 251 + 1).astype(np.uint8)
    images = np.broadcast_to(idx[:, None, None, None], (ROWS, TILE, TILE, 3)).copy()
    with h5py.File(path, "w") as f:
        f.create_dataset("img", data=images, chunks=(1, TILE, TILE, 3),
                         compression="gzip", compression_opts=6)
        f.create_dataset("samples", data=np.arange(ROWS))
        f.create_dataset("tiles", data=np.arange(ROWS))


def test_sharded_encoding_matches_unsharded(tmp_path):
    if not _available():
        return  # subtree or encoder deps absent; nothing to exercise

    _install_tensorflow_stub()
    sys.path.insert(0, str(HPL))
    from models.evaluation.features import (  # noqa: E402
        real_encode_contrastive_from_checkpoint as encode,
    )
    from merge_projection_shards import merge_projection_shards  # noqa: E402
    from submit_feature_extraction import shard_output_path, shard_ranges  # noqa: E402

    model, data = _fake_model(), _fake_data()
    src = tmp_path / "hdf5_DS_he_train.h5"
    _write_input(src)

    results = tmp_path / "results" / "BarlowTwins_3" / "DS" / "h224_w224_n3_zdim128"

    # One job over everything — the reference.
    encode(model=model, data=data, data_out_path=str(tmp_path), checkpoint="ckpt",
           real_hdf5=str(src), batches=128)
    reference = results / "hdf5_DS_he_train.h5"
    assert reference.is_file(), "unsharded run produced no output"

    # Four jobs over disjoint ranges of the same input.
    bounds = shard_ranges(ROWS, 4)
    for lo, hi in bounds:
        encode(model=model, data=data, data_out_path=str(tmp_path), checkpoint="ckpt",
               real_hdf5=str(src), batches=128, row_start=lo, row_stop=hi)

    # The encoder's part names must be the ones the submitter and merge expect.
    merge_dir = tmp_path / "merged"
    merge_dir.mkdir()
    final = merge_dir / "hdf5_DS_he_train.h5"
    for lo, hi in bounds:
        part = results / shard_output_path(final, lo, hi).name
        assert part.is_file(), f"no part for rows [{lo}, {hi}): looked for {part.name}"
        shutil.move(str(part), merge_dir / part.name)

    info = merge_projection_shards(final, expected_rows=ROWS)
    assert info["rows"] == ROWS and info["parts"] == 4

    with h5py.File(reference, "r") as ref, h5py.File(final, "r") as got:
        assert set(ref.keys()) == set(got.keys())
        for name in ref.keys():
            assert np.array_equal(ref[name][:], got[name][:]), f"{name} differs after sharding"

    # And the latents really are a function of the input row, not of position
    # within a shard — which is what a boundary-off-by-one would break.
    expected = (np.arange(ROWS) % 251 + 1).astype(np.float32) / np.float32(255.)
    with h5py.File(final, "r") as got:
        assert np.array_equal(got["img_h_latent"][:, 0], expected)


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_smoke_"))
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
