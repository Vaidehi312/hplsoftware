# Patches to HPL-LATTICeA

`HPL-LATTICeA` is a clone of [K-Rakovic/HPL-LATTICeA](https://github.com/K-Rakovic/HPL-LATTICeA),
not part of this repository — it is untracked here and carries its own git
history. Changes made inside that clone are invisible to this repo, and a
fresh clone on a new machine silently loses them.

So each change we depend on lives here as a patch file as well as on a branch
in the clone. The patch is the copy that survives.

## hpl-encode-io.patch

Throughput of `real_encode_contrastive_from_checkpoint` in
`models/evaluation/features.py`: overlaps tile reads with the forward pass,
writes latents one slice per batch instead of one row per tile, and reads as
float32 rather than float64.

Changes no embedding — see `backend/tests/test_encode_loop_equivalence.py`,
and the reasoning in the commit message inside the patch.

Apply to a fresh clone:

```bash
cd $HPL_REPO_DIR
git apply --check ../backend/patches/hpl-encode-io.patch   # dry run
git apply         ../backend/patches/hpl-encode-io.patch
```

Or, to keep it as a commit with its message intact:

```bash
git am ../backend/patches/hpl-encode-io.patch
```

Verify afterwards with `python backend/tests/test_encode_loop_equivalence.py`,
whose last test reads the patched file and fails if the per-row writes came
back (e.g. after a `git pull` from upstream).

If upstream has moved and the patch no longer applies, `git apply -3` will
attempt a three-way merge; the branch `perf/encode-io` in the clone holds the
same change if you need to rebase it by hand.
