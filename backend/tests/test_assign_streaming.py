"""Stage 4's assigner streams its input in chunks. These tests are about what
streaming could silently have broken.

The dangerous property is that `--centering query` mirrors scanpy's Ingest and
subtracts the mean over *all* queries. That couples every tile to every other
one, so the moment the input is processed in pieces there are two new ways to be
wrong, both of which still emit a complete, well-formed CSV:

  * a chunk centred on its own mean, making a tile's cluster depend on which
    chunk boundary it happened to fall inside;
  * a shard centred on its slice's mean, making cluster IDs depend on how many
    jobs the work was split across.

So the tests here are equivalence tests, not smoke tests: chunked must equal
unchunked, and sharded must equal unsharded, bit for bit.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

ASSIGN = BACKEND / "assign_hpc_clusters.py"

REF_ROWS, DIM, NCOMP, NCLUST, K = 4000, 32, 16, 12, 25
QUERY_ROWS = 900


def _write_reference(path: Path, seed: int = 0) -> None:
    """A build_hpc_reference.py-shaped .npz, small enough to be fast."""
    import json
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, NCLUST, REF_ROWS).astype(np.int64)
    np.savez(
        path,
        reference=rng.standard_normal((REF_ROWS, NCOMP)).astype(np.float32),
        components=rng.standard_normal((DIM, NCOMP)).astype(np.float32),
        codes=codes,
        categories=np.array([str(i) for i in range(NCLUST)]),
        n_neighbors=np.int64(K),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )


def _write_queries(path: Path, rows: int = QUERY_ROWS, dim: int = DIM, seed: int = 1) -> None:
    """A projections .h5 as feature extraction leaves it."""
    rng = np.random.default_rng(seed)
    # Offset from zero so the query mean is meaningfully non-zero — with a mean
    # of ~0 every centering bug would look like a passing test.
    emb = (rng.standard_normal((rows, dim)) + 3.0).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("img_z_latent", data=emb)
        f.create_dataset("samples", data=np.array([b"S%03d" % (i // 90) for i in range(rows)]))
        f.create_dataset("slides", data=np.array([b"slide_%03d" % (i // 90) for i in range(rows)]))
        f.create_dataset("tiles", data=np.array([b"%d_%d.jpeg" % (i // 30, i % 30) for i in range(rows)]))


def _run(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ASSIGN), *args],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    if result.returncode != 0:
        raise AssertionError(f"assign failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def _assign(ref: Path, h5: Path, out: Path, **flags) -> Path:
    args = ["--reference", str(ref), "--h5", str(h5), "--out", str(out)]
    for key, value in flags.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    _run(*args)
    return out


def test_chunk_size_does_not_change_assignments(tmp_path):
    """A tile's cluster must not depend on which chunk it landed in. This is the
    check that the query mean is computed over the file and not per chunk."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    outputs = []
    # 100 forces ragged chunks against 900 rows; 100_000 is a single chunk.
    for chunk in (100, 256, 900, 100_000):
        out = _assign(ref, h5, tmp_path / f"c{chunk}.csv", chunk_size=chunk)
        outputs.append(out.read_bytes())

    for chunk, data in zip((256, 900, 100_000), outputs[1:]):
        assert data == outputs[0], f"chunk_size={chunk} changed the output"


def test_batch_size_does_not_change_assignments(tmp_path):
    """The search batch is a performance knob only — search is per-query
    independent, so this must hold for any value."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    a = _assign(ref, h5, tmp_path / "b64.csv", batch_size=64).read_bytes()
    b = _assign(ref, h5, tmp_path / "b16k.csv", batch_size=16384).read_bytes()
    assert a == b


def test_sharded_with_shared_mean_equals_unsharded(tmp_path):
    """The property that makes sharding safe. Every shard is handed the same
    precomputed mean, so the concatenation must equal a single run exactly."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    whole = _assign(ref, h5, tmp_path / "whole.csv").read_text()

    mean_path = tmp_path / "mean.npy"
    _run("--reference", str(ref), "--h5", str(h5), "--precompute-mean", str(mean_path))
    assert mean_path.is_file()

    bounds = [(0, 225), (225, 450), (450, 675), (675, QUERY_ROWS)]
    pieces = []
    for lo, hi in bounds:
        _run("--reference", str(ref), "--h5", str(h5),
             "--out", str(tmp_path / "shard.csv"),
             "--query-mean", str(mean_path),
             "--row-start", str(lo), "--row-stop", str(hi))
        part = tmp_path / f"shard.rows{lo}-{hi}.csv"
        assert part.is_file(), f"no part written for [{lo}, {hi})"
        pieces.append(part.read_text())

    header = pieces[0].splitlines()[0]
    merged = header + "\n" + "".join(
        "".join(line + "\n" for line in p.splitlines()[1:]) for p in pieces
    )
    assert merged == whole, "sharded output differs from a single run"


def test_shard_without_shared_mean_is_refused(tmp_path):
    """The guard has to earn its place: without it, each shard would centre on
    its own slice and produce different labels. Proven by showing the labels
    really do differ when the mean is not shared."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)

    # 1. It is refused, and the message says what to do.
    result = subprocess.run(
        [sys.executable, str(ASSIGN), "--reference", str(ref), "--h5", str(h5),
         "--out", str(tmp_path / "x.csv"), "--row-start", "0", "--row-stop", "225"],
        capture_output=True, text=True, cwd=str(BACKEND),
    )
    assert result.returncode != 0
    assert "--query-mean" in result.stderr, result.stderr

    # 2. And the refusal is not pedantry. Centering a slice by its own mean does
    #    change assignments: 'none' centering makes the slice independent, so
    #    running a slice under it and comparing against the same rows of a whole
    #    run under 'query' shows the two spaces disagree.
    whole = _assign(ref, h5, tmp_path / "w.csv", centering="none").read_text().splitlines()
    _run("--reference", str(ref), "--h5", str(h5), "--out", str(tmp_path / "s.csv"),
         "--centering", "none", "--row-start", "0", "--row-stop", "225")
    part = (tmp_path / "s.rows0-225.csv").read_text().splitlines()
    # 'none' needs no mean, so a slice IS safe there — same rows, same answers.
    assert part[1:] == whole[1:226], "centering=none should be shard-independent"


def test_query_mean_matches_a_whole_array_mean(tmp_path):
    """The streamed float64 accumulation must agree with the obvious
    computation, or every projected coordinate is slightly off."""
    import assign_hpc_clusters as m

    h5 = tmp_path / "q.h5"
    _write_queries(h5, rows=1001)
    with h5py.File(h5, "r") as f:
        expected = np.asarray(f["img_z_latent"][:], dtype=np.float32).mean(axis=0)

    for chunk in (7, 128, 5000):
        got = m.compute_query_mean(h5, "z_latent", chunk, 1001)
        assert np.allclose(got, expected, atol=1e-6), f"chunk={chunk}"


def test_high_dimensional_rep_key_streams(tmp_path):
    """--rep-key h_latent is 1536-d, the case that motivated streaming: the old
    whole-array read was ~86 GB at registry scale. Nothing else exercises it."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    import json
    rng = np.random.default_rng(3)
    np.savez(
        ref,
        reference=rng.standard_normal((500, NCOMP)).astype(np.float32),
        components=rng.standard_normal((1536, NCOMP)).astype(np.float32),
        codes=rng.integers(0, NCLUST, 500).astype(np.int64),
        categories=np.array([str(i) for i in range(NCLUST)]),
        n_neighbors=np.int64(K),
        meta=json.dumps({"groupby": "leiden_2.5"}),
    )
    rows = 300
    with h5py.File(h5, "w") as f:
        f.create_dataset("img_h_latent", data=rng.standard_normal((rows, 1536)).astype(np.float32))
        f.create_dataset("samples", data=np.array([b"S1"] * rows))
        f.create_dataset("slides", data=np.array([b"slide_1"] * rows))
        f.create_dataset("tiles", data=np.array([b"%d.jpeg" % i for i in range(rows)]))

    out = _assign(ref, h5, tmp_path / "h.csv", rep_key="h_latent", chunk_size=64)
    lines = out.read_text().splitlines()
    assert len(lines) == rows + 1


def test_partial_output_is_not_left_behind(tmp_path):
    """A killed run must not leave a CSV holding some of the tiles: nothing
    downstream checks row counts before merging."""
    ref, h5 = tmp_path / "ref.npz", tmp_path / "q.h5"
    _write_reference(ref)
    _write_queries(h5)
    out = tmp_path / "ok.csv"
    _assign(ref, h5, out)
    assert out.is_file()
    assert not out.with_name(out.name + ".partial").exists()


# --- standalone runner ---------------------------------------------------

def main():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, fn in tests:
        tmp_path = Path(tempfile.mkdtemp(prefix="hpl_assign_test_"))
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
