# from tifffile import TiffFile
# from PIL import Image
# import numpy as np
# from pathlib import Path

# slide_path = "/Users/vaidehipandya/Desktop/Work/TCGA-55-7574-01Z-00-DX1.09639e6a-d85f-4d84-abbc-0f5a6d679683.svs"
# out_dir = Path("svs_pages")
# out_dir.mkdir(exist_ok=True)

# MAX_PIXELS_TO_SAVE_FULL = 20_000_000  # ~20MP safe

# with TiffFile(slide_path) as tif:
#     print("Total pages:", len(tif.pages))

#     for i, page in enumerate(tif.pages):
#         shape = page.shape
#         pixels = shape[0] * shape[1]

#         print(f"Page {i}: {shape} pixels={pixels}")

#         try:
#             if pixels > MAX_PIXELS_TO_SAVE_FULL:
#                 # extract a thumbnail region instead of full image
#                 print("  -> too large, saving preview")
#                 arr = page.asarray(key=0)[::20, ::20]  # downsample preview
#                 img = Image.fromarray(arr.astype(np.uint8))
#                 img.save(out_dir / f"page_{i}_preview.png")
#             else:
#                 arr = page.asarray()
#                 img = Image.fromarray(arr.astype(np.uint8))
#                 img.save(out_dir / f"page_{i}.png")

#         except Exception as e:
#             print("  -> failed:", e)


import openslide
from pathlib import Path

SLIDE_PATH = "/Users/vaidehipandya/Desktop/Work/TCGA-55-7574-01Z-00-DX1.09639e6a-d85f-4d84-abbc-0f5a6d679683.svs"

out_dir = Path("svs_export")
out_dir.mkdir(exist_ok=True)

slide = openslide.OpenSlide(SLIDE_PATH)

print("Level count:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)
print("Associated images:", list(slide.associated_images.keys()))

# 1) Export each pyramid level as a single image (safe: these are already downsampled)
for lvl in range(slide.level_count):
    w, h = slide.level_dimensions[lvl]
    img = slide.read_region((0, 0), lvl, (w, h)).convert("RGB")
    img.save(out_dir / f"level_{lvl}_{w}x{h}.png")

# 2) Export label / macro etc if present
for name, img in slide.associated_images.items():
    img.convert("RGB").save(out_dir / f"associated_{name}.png")

print("Saved to:", out_dir.resolve())