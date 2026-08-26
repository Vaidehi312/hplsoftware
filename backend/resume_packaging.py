#!/usr/bin/env python3
"""Resume an interrupted make_hpl_hdf5.py packaging run.

package_slides_to_h5() (see make_hpl_hdf5.py) checkpoints its own progress
as it goes — every tile that gets a durable outcome (written, or given up
on as corrupt) is logged to a sidecar file next to the output .h5. This
script exists purely so resuming after a crash/OOM/TIMEOUT/Ctrl-C doesn't
require retyping every original argument by hand: it reads back
<output_h5_path>.run_config.json (written by the interrupted attempt
itself) and replays that exact call. package_slides_to_h5 then detects its
own checkpoint and picks up only the tiles that don't already have a
durable outcome, instead of redecoding a multi-hour dataset from scratch.

If no checkpoint is found, either nothing was ever started for this output
path, or a prior attempt already finished successfully (its checkpoint is
deleted once it completes) — either way there's nothing to resume, and
this says so rather than silently starting a full fresh run under a
"resume" name.

Usage:
    # Locate by the .h5 path directly:
    python resume_packaging.py --output-h5-path /path/to/hdf5_TCGA_he_train.h5

    # Or by the same identifying arguments make_hpl_hdf5.py itself takes:
    python resume_packaging.py --output-root /path/to/model_input \\
        --dataset-name TCGA --marker he --split train --tile-size 224

    # Tuning knobs are safe to change from whatever the interrupted attempt
    # used — e.g. lower --processes if the original run OOM'd:
    python resume_packaging.py --output-h5-path ... --processes 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from make_hpl_hdf5 import _checkpoint_paths, hpl_h5_output_path, package_slides_to_h5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-h5-path", type=Path, default=None,
        help="The .h5 path an interrupted run was writing to. Either this, or "
             "--output-root together with --dataset-name (plus --marker/--split/"
             "--tile-size if they weren't left at their defaults), is required.",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--marker", type=str, default="he")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument(
        "--processes", type=int, default=None,
        help="Worker processes for this resume attempt — safe to differ from what the "
             "interrupted run used (e.g. lower this if it OOM'd). Defaults to "
             "min(cpu count, 8), same as make_hpl_hdf5.py.",
    )
    parser.add_argument("--threads-per-process", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.output_h5_path:
        output_h5_path = args.output_h5_path
    elif args.output_root and args.dataset_name:
        output_h5_path = hpl_h5_output_path(
            args.output_root, args.dataset_name, args.marker, args.split, args.tile_size
        )
    else:
        build_parser().error(
            "Provide --output-h5-path, or --output-root together with --dataset-name."
        )
        return

    ckpt = _checkpoint_paths(output_h5_path)
    if not ckpt["config"].is_file():
        print(
            f"No checkpoint found for {output_h5_path} — either nothing was ever started "
            "for this output, or a prior attempt already completed successfully (its "
            "checkpoint is cleaned up once it finishes). Nothing to resume.\n"
            "To start a fresh run instead, use make_hpl_hdf5.py directly."
        )
        sys.exit(1)

    try:
        call_args = json.loads(ckpt["config"].read_text())
    except Exception as e:
        print(f"Checkpoint config at {ckpt['config']} is unreadable ({e}) — can't resume safely.")
        sys.exit(1)

    print(
        f"Found a checkpoint for {output_h5_path} — resuming the original call's "
        f"{len(call_args['raw_paths'])} slide(s) targeted at "
        f"{call_args['dataset_name']}/{call_args['tile_dataset_name']}."
    )

    result = package_slides_to_h5(
        raw_paths=call_args["raw_paths"],
        slide_ids=call_args["slide_ids"],
        tile_dir=Path(call_args["tile_dir"]),
        tile_dataset_name=call_args["tile_dataset_name"],
        output_root=Path(call_args["output_root"]),
        dataset_name=call_args["dataset_name"],
        marker=call_args["marker"],
        split=call_args["split"],
        tile_size=call_args["tile_size"],
        n_processes=args.processes,
        threads_per_process=args.threads_per_process,
        batch_size=args.batch_size,
    )

    print(f"Resumed from checkpoint:  {result['resumed']}")
    print(f"Output:                   {result['output_h5_path']}")
    print(f"Total tiles packaged:     {result['total_tiles']}")
    print(f"Slides packaged:          {result['slides_packaged']}")
    if result["slides_missing_metadata"]:
        print(f"Slides with no tile metadata (skipped): {result['slides_missing_metadata']}")
    if result["skipped_tiles"]:
        print(
            f"Tiles skipped (missing/corrupt on disk): {len(result['skipped_tiles'])} "
            f"({result['skip_percentage']:.4f}%)"
        )


if __name__ == "__main__":
    main()
