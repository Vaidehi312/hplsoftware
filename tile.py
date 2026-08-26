
import h5py
import pandas as pd

h5_path = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"

with h5py.File(h5_path, "r") as f:
    h5_tiles = f["train_tiles"][:]  # n-length array of tile names

# Convert bytes → str
h5_tiles = [t.decode() if isinstance(t, bytes) else str(t) for t in h5_tiles]

df_h5 = pd.DataFrame({
    "tiles": h5_tiles,
    "h5_index": list(range(len(h5_tiles)))
})

df_h5.head()

df_coords = pd.read_csv("/Users/vaidehipandya/Desktop/Work/TCGA_55_7574_final_tile_coordinates.csv")

df_final = df_coords.merge(df_h5, on="tiles", how="left")
df_final.to_csv("tile_coords_with_h5_index.csv", index=False)

df_final.head()

df = _coords[tile_coords["slides"] == "TCGA-55-7574-01Z-00-DX1"]

print("Unique tile widths:", (df["x_native"].sort_values().diff().dropna().unique()))
print("Unique tile heights:", (df["y_native"].sort_values().diff().dropna().unique()))