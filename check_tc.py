import h5py
import re

h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"
with h5py.File(h5_path, "r") as f:
    tiles = [t.decode("utf-8") for t in f["train_tiles"][:100]]

for t in tiles[:20]:
    print(t, " → ", bool(re.match(r'\d+[_-]\d+', t)))