#!/bin/bash
set -euo pipefail

# Usage:
#   bash submit_dataset_tiling.sh /path/to/dataset
#
# Optional:
#   bash submit_dataset_tiling.sh /path/to/dataset /path/to/masks /path/to/tiles

RAW_DIR="${1:-/hpc-home/home/users/vpandya/long-term-scratch/uploaded_wsi/raw}"
MASK_DIR="${2:-/hpc-home/home/users/vpandya/long-term-scratch/tissue_masks}"
TILE_DIR="${3:-/hpc-home/home/users/vpandya/long-term-scratch/processed_tiles}"

BACKEND_DIR="${BACKEND_DIR:-/hpc-home/home/users/vpandya/long-term-scratch/Work/backend}"
CONDA_ENV="${CONDA_ENV:-/hpc-home/home/users/vpandya/conda_envs/hpl_kb}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

SBATCH_FILE="${BACKEND_DIR}/mask_and_tile_array.sbatch"
MANIFEST_DIR="${BACKEND_DIR}/slurm_manifests"
LOG_DIR="${BACKEND_DIR}/slurm_logs"

mkdir -p "$MANIFEST_DIR" "$LOG_DIR" "$MASK_DIR" "$TILE_DIR"

if [[ ! -d "$RAW_DIR" ]]; then
    echo "Dataset directory does not exist: $RAW_DIR" >&2
    exit 1
fi

if [[ ! -f "$SBATCH_FILE" ]]; then
    echo "Slurm worker file does not exist: $SBATCH_FILE" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="${MANIFEST_DIR}/wsi_manifest_${TIMESTAMP}.txt"

# Recursive search allows datasets organised into cohort/patient subfolders.
python - "$RAW_DIR" "$MANIFEST" <<'PY'
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1]).expanduser().resolve()
manifest = Path(sys.argv[2]).expanduser().resolve()
extensions = {".svs", ".ndpi", ".mrxs", ".tif", ".tiff", ".scn"}

slides = sorted(
    (p.resolve() for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
    key=lambda p: str(p).lower(),
)

manifest.write_text("".join(f"{p}\n" for p in slides), encoding="utf-8")
print(len(slides))
PY

N_SLIDES=$(wc -l < "$MANIFEST" | tr -d ' ')

if [[ "$N_SLIDES" -eq 0 ]]; then
    echo "No supported WSI files found under: $RAW_DIR" >&2
    rm -f "$MANIFEST"
    exit 1
fi

ARRAY_MAX=$((N_SLIDES - 1))

echo "Dataset:          $RAW_DIR"
echo "Slides found:     $N_SLIDES"
echo "Manifest:         $MANIFEST"
echo "Mask directory:   $MASK_DIR"
echo "Tile directory:   $TILE_DIR"
echo "Concurrent tasks: $MAX_CONCURRENT"

# --export passes configuration to every array task.
sbatch \
    --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
    --chdir="$BACKEND_DIR" \
    --export="ALL,MANIFEST=$MANIFEST,BACKEND_DIR=$BACKEND_DIR,MASK_DIR=$MASK_DIR,TILE_DIR=$TILE_DIR,CONDA_ENV=$CONDA_ENV" \
    "$SBATCH_FILE"
