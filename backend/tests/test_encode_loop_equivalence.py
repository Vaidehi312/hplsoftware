"""The encoding loop in HPL-LATTICeA's features.py was rewritten for
throughput (see backend/patches/hpl-encode-io.patch). These tests exist to
prove the rewrite did not move a single embedding.

The risk being tested is not "is it faster" — it is that batching, slicing and
a prefetch thread are three separate chances to write row N's embedding into
row M. A misalignment here produces a file that passes every completeness
check the pipeline has (right shape, right dtype, no zero rows) and is silently
wrong: every downstream cluster assignment would be attached to the wrong tile.

Both loops are reproduced below rather than imported, because features.py
cannot be imported without TensorFlow. test_source_matches_these_loops guards
the copies against drift.
"""

import queue
import sys
import tempfile
import threading
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HPL_FEATURES = (
    Path(__file__).resolve().parents[2] / "HPL-LATTICeA" / "models" / "evaluation" / "features.py"
)

# Stand-ins for the encoder's two outputs. Deliberately a function of the
# tile's *global* index, so any row that ends up in the wrong place is visible
# rather than plausible.
H_DIM, Z_DIM = 1536, 128


def fake_encode(img_batch):
    """A 'session.run' whose output encodes which input rows it saw.

    The cast on the first line is not incidental — it is what makes this a
    fair comparison. model.real_images_2 is a float32 placeholder, so feeding
    it the old loop's float64 array casts to float32 before any arithmetic
    happens. Skipping that here would compare float64 maths against float32
    maths and report a difference the encoder never sees.
    """
    img_batch = np.asarray(img_batch, dtype=np.float32)
    keys = img_batch[:, 0, 0, 0].astype(np.float64)
    h = np.outer(keys, np.arange(1, H_DIM + 1)).astype(np.float32)
    z = np.outer(keys, -np.arange(1, Z_DIM + 1)).astype(np.float32)
    return [h, z]


def make_input(num_samples, tile=4):
    """Tiles whose every pixel is the tile's own index, so fake_encode can
    recover it. uint8 like the real packaged .h5, hence the modulo."""
    idx = np.arange(num_samples, dtype=np.uint8) % 251 + 1
    return np.broadcast_to(
        idx[:, None, None, None], (num_samples, tile, tile, 3)
    ).copy()


def old_loop(images, batches):
    """features.py as it was: read whole batch, write one row at a time."""
    num_samples = images.shape[0]
    h_storage = np.zeros((num_samples, H_DIM), np.float32)
    z_storage = np.zeros((num_samples, Z_DIM), np.float32)
    ind = 0
    while ind < num_samples:
        if (ind + batches) < num_samples:
            real_img_batch = images[ind: ind + batches, :, :, :] / 255.
        else:
            real_img_batch = images[ind:, :, :, :] / 255.
        outputs = fake_encode(real_img_batch)
        for i in range(batches):
            if ind == num_samples:
                break
            h_storage[ind] = outputs[0][i, :]
            z_storage[ind] = outputs[1][i, :]
            ind += 1
    return h_storage, z_storage, ind


def new_loop(images, batches):
    """features.py as it is now: prefetch thread, float32 read, slice writes."""
    num_samples = images.shape[0]
    h_storage = np.zeros((num_samples, H_DIM), np.float32)
    z_storage = np.zeros((num_samples, Z_DIM), np.float32)

    batch_queue = queue.Queue(maxsize=2)
    reader_error = []

    def _read_batches():
        try:
            for start in range(0, num_samples, batches):
                stop = min(start + batches, num_samples)
                batch = images[start:stop, :, :, :].astype(np.float32) / np.float32(255.)
                batch_queue.put((start, stop, batch))
        except BaseException as e:  # noqa: BLE001 - re-raised on the main thread
            reader_error.append(e)
        finally:
            batch_queue.put(None)

    reader = threading.Thread(target=_read_batches)
    reader.daemon = True
    reader.start()

    ind = 0
    while True:
        item = batch_queue.get()
        if item is None:
            break
        start, stop, real_img_batch = item
        outputs = fake_encode(real_img_batch)
        h_storage[start:stop] = outputs[0]
        z_storage[start:stop] = outputs[1]
        ind = stop

    reader.join()
    if reader_error:
        raise reader_error[0]
    return h_storage, z_storage, ind


# The shapes that matter: a ragged tail, an exact multiple, a batch larger than
# the dataset, and a single tile. The old loop's `range(batches)` with a break
# and the new loop's min() disagree on all of these if either is wrong.
SHAPES = [
    (1000, 64),    # ragged tail
    (1024, 64),    # exact multiple
    (1000, 1),     # degenerate batch
    (7, 64),       # batch larger than the dataset
    (64, 64),      # exactly one batch
    (65, 64),      # one full batch plus one row
    (999, 256),    # the batch size worth switching to
]


def test_rewrite_is_row_for_row_identical(tmp_path):
    del tmp_path
    for num_samples, batches in SHAPES:
        images = make_input(num_samples)
        h_old, z_old, ind_old = old_loop(images, batches)
        h_new, z_new, ind_new = new_loop(images, batches)

        assert ind_old == ind_new == num_samples, (num_samples, batches)
        # Bit-identical, not close: this is the same arithmetic on the same
        # inputs, so anything less would mean rows moved.
        assert np.array_equal(h_old, h_new), f"h differs at {num_samples}/{batches}"
        assert np.array_equal(z_old, z_new), f"z differs at {num_samples}/{batches}"


def test_every_row_holds_its_own_tile(tmp_path):
    """Equivalence to the old loop is only reassuring if the old loop was
    right. Check the mapping directly too."""
    del tmp_path
    num_samples, batches = 1000, 256
    images = make_input(num_samples)
    h, z, _ = new_loop(images, batches)

    expected_keys = images[:, 0, 0, 0].astype(np.float32) / np.float32(255.)
    assert np.array_equal(h[:, 0], expected_keys)
    assert np.array_equal(z[:, 0], -expected_keys)


def test_float32_read_matches_the_float64_one(tmp_path):
    """The read now casts before dividing instead of after. Exhaustive over
    every value a uint8 tile can hold, so this is a proof rather than a
    sample."""
    del tmp_path
    a = np.arange(256, dtype=np.uint8)
    assert np.array_equal(
        (a / 255.).astype(np.float32), a.astype(np.float32) / np.float32(255.)
    )


def test_slice_writes_match_row_writes_through_h5py(tmp_path):
    """numpy and h5py assignment could in principle differ on dtype coercion;
    the real code writes to an h5py dataset, so check that path too."""
    num_samples, batches = 500, 128
    images = make_input(num_samples)
    h_expected, z_expected, _ = old_loop(images, batches)

    path = tmp_path / "proj.h5"
    with h5py.File(path, "w") as f:
        h_storage = f.create_dataset("h", (num_samples, H_DIM), dtype=np.float32)
        z_storage = f.create_dataset("z", (num_samples, Z_DIM), dtype=np.float32)
        for start in range(0, num_samples, batches):
            stop = min(start + batches, num_samples)
            batch = images[start:stop].astype(np.float32) / np.float32(255.)
            outputs = fake_encode(batch)
            h_storage[start:stop] = outputs[0]
            z_storage[start:stop] = outputs[1]

    with h5py.File(path, "r") as f:
        assert np.array_equal(f["h"][:], h_expected)
        assert np.array_equal(f["z"][:], z_expected)


def test_reader_failure_is_not_mistaken_for_end_of_data(tmp_path):
    """The sentinel that ends the loop is also what a dying reader pushes. If
    the error were dropped, a read failure halfway through would produce a
    half-zero file that looks complete."""
    del tmp_path
    num_samples, batches = 1000, 64
    images = make_input(num_samples)
    batch_queue = queue.Queue(maxsize=2)
    reader_error = []

    def _read_batches():
        try:
            for start in range(0, num_samples, batches):
                if start >= 256:
                    raise OSError("simulated HDF5 read failure")
                stop = min(start + batches, num_samples)
                batch_queue.put((start, stop, images[start:stop]))
        except BaseException as e:  # noqa: BLE001
            reader_error.append(e)
        finally:
            batch_queue.put(None)

    reader = threading.Thread(target=_read_batches)
    reader.daemon = True
    reader.start()
    ind = 0
    while True:
        item = batch_queue.get()
        if item is None:
            break
        ind = item[1]
    reader.join()

    assert reader_error, "a failed read must not look like a finished one"
    assert ind < num_samples
    assert isinstance(reader_error[0], OSError)


def test_source_matches_these_loops(tmp_path):
    """Guards the reproduced loops above against drifting from the real file.

    Skipped rather than failed when the HPL clone is absent — this repo does
    not vendor it, and the test is about the patch, not the environment.
    """
    del tmp_path
    if not HPL_FEATURES.is_file():
        return  # HPL-LATTICeA not cloned alongside; nothing to compare against

    # Scoped to the one function the pipeline calls. features.py holds several
    # near-identical encoders (real_encode_from_checkpoint and friends) that
    # still carry the per-row writes; they are dead code for this pipeline and
    # patching them would widen the diff into paths nothing here exercises.
    whole = HPL_FEATURES.read_text()
    start = whole.index("def real_encode_contrastive_from_checkpoint")
    rest = whole[start:]
    end = rest.index("\ndef ", 1) if "\ndef " in rest[1:] else len(rest)
    src = rest[:end]

    # The rewrite's load-bearing lines.
    for fragment in (
        "batch_queue = queue.Queue(maxsize=2)",
        "h_storage[start:stop] = outputs[0]",
        "z_storage[start:stop] = outputs[1]",
        ".astype(np.float32)/np.float32(255.)",
        "if ind != num_samples:",
    ):
        assert fragment in src, f"features.py no longer contains: {fragment}"

    # The per-row writes this patch exists to remove.
    for fragment in (
        "h_storage[ind] = outputs[0][i, :]",
        "z_storage[ind] = outputs[1][i, :]",
        "storage[ind] = info_batch[i]",
    ):
        assert fragment not in src, f"per-row write is back in features.py: {fragment}"


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_encode_test_"))
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
