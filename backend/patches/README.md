# Patches to HPL-LATTICeA

`HPL-LATTICeA/` is a **git subtree** of
[K-Rakovic/HPL-LATTICeA](https://github.com/K-Rakovic/HPL-LATTICeA), imported
squashed from upstream `master` (`aec5145`). Its files are tracked in this
repository like any others — cloning this repo gets you the encoder, and our
changes to it are ordinary commits. There is no separate clone to keep in sync
and nothing to re-apply after checkout.

The patches here are therefore **not** how the change reaches your working
tree. They are kept for two things: sending the change upstream to Kai, and
re-applying it by hand if a `git subtree pull` ever clobbers it.

## Working with the subtree

```bash
# Pull upstream changes in (from the repo root, working tree clean):
git subtree pull --prefix=HPL-LATTICeA hpl master --squash

# The remote, if it is not configured on a fresh clone:
git remote add hpl https://github.com/K-Rakovic/HPL-LATTICeA.git
```

A `subtree pull` can revert our encoder change if upstream has touched the
same lines. `backend/tests/test_encode_loop_equivalence.py` fails loudly when
that happens — run it after any pull.

## hpl-encode-io.patch

Throughput of `real_encode_contrastive_from_checkpoint` in
`models/evaluation/features.py`: overlaps tile reads with the forward pass,
writes latents one slice per batch instead of one row per tile, and reads as
float32 rather than float64.

Changes no embedding — see `backend/tests/test_encode_loop_equivalence.py`,
and the reasoning in the commit message inside the patch.

**Already applied in the subtree.** Base: upstream `aec5145`.

To re-apply it after a `subtree pull` reverted it, from the repo root:

```bash
git apply --directory=HPL-LATTICeA --check backend/patches/hpl-encode-io.patch  # dry run
git apply --directory=HPL-LATTICeA       backend/patches/hpl-encode-io.patch
```

`--directory` is needed because the patch is written against the HPL repo
root, while the files now live one level down under `HPL-LATTICeA/`.

If upstream has moved and it no longer applies, `git apply -3` will attempt a
three-way merge.
