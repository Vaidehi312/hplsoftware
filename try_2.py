import pandas as pd

TILE_SIZE = 224

origin_x = 3169   # bbox_x0
origin_y = 1568   # bbox_y0

df = pd.read_csv("tile_coordinates.csv")

df["true_x_px"] = origin_x + df["col"] * TILE_SIZE
df["true_y_px"] = origin_y + df["row"] * TILE_SIZE

df.to_csv("tile_coordinates_corrected.csv", index=False)

print("Saved corrected coordinates → tile_coordinates_corrected.csv")