#!/usr/bin/env python3
"""Measure the read and write halves of HPL feature extraction on the actual
cluster filesystem, so a decision to split the dataset across N GPU jobs is
sized from real numbers rather than guessed.

Why this exists: the encoder is not obviously GPU-bound. The packaged .h5
stores one tile per gzip chunk (make_hpl_hdf5.py), so reading a batch is one
decompression per tile, on one thread, and on a laptop that caps out around
1.3k tiles/s — comparable to or slower than an H200 encodes them. Whether that
holds on cephfs with your data and your CPUs is exactly what this measures.

Needs only h5py and numpy, and no GPU, so it runs on a login node or inside
the container. It reads a sample of the file and writes only to a temporary
directory; the input is opened read-only and never modified.

Usage:
    python benchmark_encode_io.py --real-hdf5 /path/to/hdf5_X_he_train.h5

    # If you already know the GPU rate from a job log, get a projection:
    python benchmark_encode_io.py --real-hdf5 ... --gpu-rate 3000
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import h5py
import numpy as np

DEFAULT_BATCH_SIZES = (64, 128, 256, 512)
# Enough to be past cache warm-up without turning a benchmark into a job.
DEFAULT_SAMPLE_TILES = 4096


def _image_key(f: h5py.File) -> str:
    for key in f.keys():
        if "image" in key or "img" in key:
            return key
    raise KeyError(f"No image dataset in {f.filename} (keys: {list(f.keys())})")


def describe_input(path: Path) -> dict:
    """What the file's layout implies before any timing is done."""
    with h5py.File(path, "r") as f:
        key = _image_key(f)
        d = f[key]
        info = {
            "key": key,
            "tiles": int(d.shape[0]),
            "tile_shape": tuple(int(x) for x in d.shape[1:]),
            "dtype": str(d.dtype),
            "chunks": tuple(int(x) for x in d.chunks) if d.chunks else None,
            "compression": d.compression,
            "compression_opts": d.compression_opts,
            "other_datasets": [k for k in f.keys() if k != key],
        }
    info["file_size_gb"] = path.stat().st_size / 1e9
    raw = info["tiles"] * int(np.prod(info["tile_shape"]))
    info["compression_ratio"] = raw / path.stat().st_size if path.stat().st_size else 0.0
    return info


def time_reads(path: Path, key: str, batch: int, n_tiles: int) -> float:
    """Tiles/s for the read the encoder actually issues, cast included."""
    with h5py.File(path, "r") as f:
        d = f[key]
        n = min(n_tiles, d.shape[0])
        # One batch first so page cache / chunk cache warm-up is not counted.
        _ = d[0:min(batch, n)]
        t0 = time.time()
        read = 0
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            _ = d[start:stop].astype(np.float32) / np.float32(255.)
            read = stop
        return read / (time.time() - t0)


def time_reads_prefetched(path: Path, key: str, batch: int, n_tiles: int,
                          gpu_seconds_per_batch: float) -> tuple[float, float]:
    """Tiles/s with and without the prefetch thread, against a simulated
    forward pass. Shows the overlap the patch buys at a given GPU rate."""
    def consume(prefetch: bool) -> float:
        with h5py.File(path, "r") as f:
            d = f[key]
            n = min(n_tiles, d.shape[0])
            _ = d[0:min(batch, n)]
            starts = list(range(0, n, batch))
            t0 = time.time()
            if prefetch:
                q: queue.Queue = queue.Queue(maxsize=2)

                def reader():
                    for s in starts:
                        e = min(s + batch, n)
                        q.put(d[s:e].astype(np.float32) / np.float32(255.))
                    q.put(None)

                t = threading.Thread(target=reader)
                t.daemon = True
                t.start()
                while q.get() is not None:
                    time.sleep(gpu_seconds_per_batch)
                t.join()
            else:
                for s in starts:
                    e = min(s + batch, n)
                    _ = d[s:e].astype(np.float32) / np.float32(255.)
                    time.sleep(gpu_seconds_per_batch)
            return n / (time.time() - t0)

    return consume(False), consume(True)


def time_writes(tmp_dir: Path, n: int, batch: int, h_dim: int = 1536,
                z_dim: int = 128) -> tuple[float, float]:
    """Tiles/s for per-row versus sliced writes of the latents."""
    def run(sliced: bool) -> float:
        p = tmp_dir / ("sliced.h5" if sliced else "perrow.h5")
        if p.exists():
            p.unlink()
        with h5py.File(p, "w") as f:
            h = f.create_dataset("h", (n, h_dim), dtype=np.float32)
            z = f.create_dataset("z", (n, z_dim), dtype=np.float32)
            oh = np.zeros((batch, h_dim), np.float32)
            oz = np.zeros((batch, z_dim), np.float32)
            t0 = time.time()
            for start in range(0, n, batch):
                stop = min(start + batch, n)
                if sliced:
                    h[start:stop] = oh[:stop - start]
                    z[start:stop] = oz[:stop - start]
                else:
                    for i, ind in enumerate(range(start, stop)):
                        h[ind] = oh[i, :]
                        z[ind] = oz[i, :]
            dt = time.time() - t0
        p.unlink()
        return n / dt

    return run(False), run(True)


def _fmt_hours(tiles: int, rate: float) -> str:
    if rate <= 0:
        return "n/a"
    seconds = tiles / rate
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-hdf5", type=Path, required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    ap.add_argument("--sample-tiles", type=int, default=DEFAULT_SAMPLE_TILES,
                    help="Tiles to read per measurement. Larger is steadier and slower.")
    ap.add_argument("--gpu-rate", type=float, default=None,
                    help="Tiles/s the GPU encodes at, from a job log. Enables the "
                         "prefetch-overlap and split projections.")
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Where write benchmarks go. Default: system temp. Point this "
                         "at the same filesystem the results/ tree lives on for a "
                         "representative number.")
    args = ap.parse_args()

    if not args.real_hdf5.is_file():
        print(f"No such file: {args.real_hdf5}", file=sys.stderr)
        raise SystemExit(1)

    info = describe_input(args.real_hdf5)
    print("=== Input ===")
    print(f"  file            {args.real_hdf5}")
    print(f"  tiles           {info['tiles']:,}  {info['tile_shape']} {info['dtype']}")
    print(f"  on disk         {info['file_size_gb']:.1f} GB  ({info['compression_ratio']:.1f}x compressed)")
    level = "" if info["compression_opts"] is None else f" level {info['compression_opts']}"
    print(f"  chunks          {info['chunks']}  compression={info['compression']}{level}")
    print(f"  carried through {info['other_datasets']}")
    if info["chunks"] and info["chunks"][0] == 1:
        print("  note            one tile per chunk: a batch read is one decompression")
        print("                  per tile, single-threaded. This is the ceiling below.")
    print()

    print("=== Read + decode (per process) ===")
    read_rates = {}
    for b in args.batch_sizes:
        r = time_reads(args.real_hdf5, info["key"], b, args.sample_tiles)
        read_rates[b] = r
        print(f"  batch {b:5d}     {r:9,.0f} tiles/s     full dataset: {_fmt_hours(info['tiles'], r)}")
    best_batch = max(read_rates, key=read_rates.get)
    spread = max(read_rates.values()) / max(min(read_rates.values()), 1e-9)
    if spread < 1.15:
        print("  → flat across batch sizes: reads are decode-bound, not per-call-bound.")
        print("    Raising --batch-size alone will not speed this up.")
    print()

    tmp_dir = args.tmp_dir or Path(tempfile.mkdtemp(prefix="hpl_bench_"))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    created = args.tmp_dir is None
    try:
        print("=== Latent writes ===")
        per_row, sliced = time_writes(tmp_dir, min(info["tiles"], 50_000), best_batch)
        print(f"  per-row (old)   {per_row:9,.0f} tiles/s")
        print(f"  sliced  (new)   {sliced:9,.0f} tiles/s     {sliced / max(per_row, 1e-9):.1f}x")
        print()

        if args.gpu_rate:
            gpu_s_per_batch = best_batch / args.gpu_rate
            serial, prefetched = time_reads_prefetched(
                args.real_hdf5, info["key"], best_batch, args.sample_tiles, gpu_s_per_batch
            )
            print(f"=== Prefetch overlap (GPU assumed {args.gpu_rate:,.0f} tiles/s) ===")
            print(f"  serial          {serial:9,.0f} tiles/s     {_fmt_hours(info['tiles'], serial)}")
            print(f"  prefetched      {prefetched:9,.0f} tiles/s     {_fmt_hours(info['tiles'], prefetched)}")
            print(f"  speedup         {prefetched / max(serial, 1e-9):.2f}x")
            print()

            ceiling = min(read_rates[best_batch], args.gpu_rate)
            bound = "read/decode" if read_rates[best_batch] < args.gpu_rate else "GPU"
            print("=== Splitting across jobs ===")
            print(f"  bound by        {bound}")
            print(f"  per-job ceiling {ceiling:9,.0f} tiles/s")
            for n in (1, 2, 4, 8):
                print(f"  {n} job(s)       {_fmt_hours(info['tiles'], ceiling * n)}"
                      f"   ({info['tiles'] // n:,} tiles each)")
            print()
            print("  Separate processes are what parallelises decode — h5py serialises")
            print("  HDF5 calls on a global lock, so threads inside one job do not.")
        else:
            print("Pass --gpu-rate (tiles/s, from 'Processed N images' timings in a job")
            print("log) for prefetch-overlap and job-split projections.")
    finally:
        if created:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
