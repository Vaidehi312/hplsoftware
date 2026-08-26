import pandas as pd
import h5py
import numpy as np
import cv2
import openslide
from PIL import Image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CSV_PATH = "tile_coordinates.csv"
H5_PATH = "/Users/vaidehipandya/Desktop/Work/subset_all_10000_images.h5"
WSI_PATH = "/Users/vaidehipandya/Desktop/Work/TCGA-55-7574-01Z-00-DX1.09639e6a-d85f-4d84-abbc-0f5a6d679683.svs"
SLIDE_ID = "TCGA-55-7574-01Z-00-DX1"

TILE_SIZE = 224
DISPLAY_LEVEL = 2   # level used for matching
# --------------------------------------------------


# --------------------------------------------------
# 1. LOAD CSV: FILTER THIS SLIDE ONLY
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)
df_slide = df[df["slide_id"] == SLIDE_ID].reset_index(drop=True)

print("Tiles belonging to slide:", len(df_slide))
print(df_slide.head())


# --------------------------------------------------
# 2. PICK ONE TILE FROM THIS SLIDE (FIRST ROW)
# --------------------------------------------------
chosen = df_slide.iloc[0]
tile_idx = int(chosen["tile_index"])
row = int(chosen["row"])
col = int(chosen["col"])

print("\nUsing tile:")
print(chosen)


# --------------------------------------------------
# 3. LOAD THIS TILE FROM H5
# --------------------------------------------------
with h5py.File(H5_PATH, "r") as f:
    tile = np.squeeze(f["train_img"][tile_idx]).astype(np.uint8)

Image.fromarray(tile).save("tile_for_matching.png")
print("\nTile saved as tile_for_matching.png")


# --------------------------------------------------
# 4. LOAD WSI AT DISPLAY LEVEL
# --------------------------------------------------
slide = openslide.OpenSlide(WSI_PATH)
dims = slide.level_dimensions[DISPLAY_LEVEL]
downsample = slide.level_downsamples[DISPLAY_LEVEL]

region = slide.read_region((0, 0), DISPLAY_LEVEL, dims).convert("RGB")
wsi = np.array(region)


# --------------------------------------------------
# 5. RESIZE TILE TO MATCH WSI LEVEL
# --------------------------------------------------
tile_resized = cv2.resize(tile, None, fx=1/downsample, fy=1/downsample,
                          interpolation=cv2.INTER_AREA)


# --------------------------------------------------
# 6. TEMPLATE MATCH
# --------------------------------------------------
result = cv2.matchTemplate(wsi, tile_resized, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

print("\nMatch score:", max_val)
print("Match location at level-2:", max_loc)

# convert back to level 0
true_x = int(max_loc[0] * downsample)
true_y = int(max_loc[1] * downsample)

print("Match location at level 0:", true_x, true_y)


# --------------------------------------------------
# 7. COMPUTE REAL ORIGIN OF THIS SLIDE
# --------------------------------------------------
origin_x = true_x - col * TILE_SIZE
origin_y = true_y - row * TILE_SIZE

print("\nRecovered origin:")
print("origin_x =", origin_x)
print("origin_y =", origin_y)


# --------------------------------------------------
# 8. ASSIGN TRUE COORDINATES FOR ALL TILES OF THIS SLIDE
# --------------------------------------------------
df_slide["true_x_px"] = origin_x + df_slide["col"] * TILE_SIZE
df_slide["true_y_px"] = origin_y + df_slide["row"] * TILE_SIZE


# --------------------------------------------------
# 9. SAVE CORRECTED COORDINATES
# --------------------------------------------------
df_slide.to_csv("tile_coordinates_corrected_for_slide.csv", index=False)
print("\nSaved tile_coordinates_corrected_for_slide.csv")

df = pd.read_csv("tile_coordinates.csv")
print(df[df["slide_id"] == "TCGA-55-7574-01Z-00-DX1"].head(20))