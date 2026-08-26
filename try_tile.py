import openslide
import numpy as np
from PIL import Image
import cv2

WSI_PATH = "/Users/vaidehipandya/Desktop/Work/TCGA-55-7574-01Z-00-DX1.09639e6a-d85f-4d84-abbc-0f5a6d679683.svs"

def get_tissue_bbox(wsi_path, level=3, thresh=0.80):
    slide = openslide.OpenSlide(wsi_path)

    # read WSI at level 3
    dims = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, dims).convert("RGB")
    img = np.array(region)

    # convert to HSV and threshold saturation channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    _, mask = cv2.threshold(s, int(thresh * 255), 255, cv2.THRESH_BINARY)

    mask = mask.astype(np.uint8)

    # get bounding box
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    # convert back to level 0 coordinates
    downsample = slide.level_downsamples[level]

    bbox_x0 = int(x * downsample)
    bbox_y0 = int(y * downsample)

    print("Bounding box (level 0 coordinates):")
    print("bbox_x0:", bbox_x0)
    print("bbox_y0:", bbox_y0)

    return bbox_x0, bbox_y0

if __name__ == "__main__":
    get_tissue_bbox(WSI_PATH)