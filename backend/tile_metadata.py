#!/usr/bin/env python3
"""Reading and integrity-checking a slide's tile metadata CSV.

Shared by make_hpl_hdf5.py (deciding which slides have tiles worth packaging)
and find_missing_slides.py (deciding which slides need re-tiling) specifically
so the two can never disagree. They used to apply different rules to the same
file — find_missing_slides treated "the CSV exists" as done, while packaging
required it to actually parse — so a CSV left truncated by a killed tiling
task counted as complete for resume purposes (never re-tiled) *and* as absent
for packaging purposes (silently dropped from the .h5). Those slides
disappeared from the dataset with nothing reporting it.

Three outcomes are deliberately distinguished, because they need opposite
handling:

  OK      — parses, has rows, has the columns packaging indexes by.
  EMPTY   — the slide ran tiling and legitimately saved zero tiles (no tissue
            passed the threshold). NOT a failure and NOT re-tileable: running
            it again produces the same empty result, so treating this as
            "missing" would make every resume resubmit the same dead slides
            forever.
  CORRUPT — the file exists but can't be trusted (unparseable, or missing the
            columns packaging needs). This one DOES need re-tiling, and is
            exactly the case that used to be silently misfiled as done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Columns package_slides_to_h5 indexes tiles by — auto_tile_from_mask.py
# writes them (see the records dict there). A CSV missing these parses fine
# but blows up later with an AttributeError deep inside packaging, so it's
# checked here where it can still be reported as a per-slide problem.
REQUIRED_COLUMNS = ("col", "row")

MISSING = "missing"
EMPTY = "empty"
CORRUPT = "corrupt"
OK = "ok"


@dataclass
class TileMetadata:
    status: str
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    detail: str = ""

    @property
    def usable(self) -> bool:
        """Has real tile rows that packaging can index."""
        return self.status == OK

    @property
    def needs_retile(self) -> bool:
        """Whether re-running tiling for this slide could actually change the
        outcome. EMPTY is deliberately excluded — see the module docstring.
        """
        return self.status in (MISSING, CORRUPT)


def tile_metadata_path(tile_dir: Path, dataset_name: str, slide_id: str) -> Path:
    """Canonical location auto_tile_from_mask.py writes a slide's metadata to."""
    return Path(tile_dir) / dataset_name / slide_id / f"{slide_id}_tile_metadata.csv"


def read_tile_metadata(path: Path) -> TileMetadata:
    """Classify (and, when usable, return) one slide's tile metadata.

    Never raises for a bad file — a single unreadable CSV out of thousands of
    slides should be reported and skipped, not abort a multi-hour packaging
    run or a resume scan. Note the honest limit here: a CSV truncated exactly
    at a line boundary is indistinguishable from a short-but-complete one, so
    this catches corruption pandas can detect (unparseable rows, missing
    columns, null coordinates), not every possible truncation.
    """
    path = Path(path)
    if not path.is_file():
        return TileMetadata(MISSING, detail=f"no metadata CSV at {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # A slide that tiled but saved zero tiles still writes this file, and
        # pd.DataFrame([]).to_csv() has no columns to infer a header from, so
        # it comes out genuinely empty rather than header-only.
        return TileMetadata(EMPTY, detail="tiling ran but saved zero tiles")
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        return TileMetadata(CORRUPT, detail=f"unreadable metadata CSV: {e}")

    if df.empty:
        return TileMetadata(EMPTY, detail="metadata CSV has a header but no tile rows")

    absent = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if absent:
        return TileMetadata(
            CORRUPT, detail=f"metadata CSV missing required column(s): {absent}"
        )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        return TileMetadata(
            CORRUPT, detail="metadata CSV has null col/row values (likely truncated mid-write)"
        )

    return TileMetadata(OK, frame=df)
