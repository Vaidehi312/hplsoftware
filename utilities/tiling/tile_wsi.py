import argparse
from aicsimageio import AICSImage
import json
import numpy as np
import os
import pickle
from PIL import Image
import sys

parser = argparse.ArgumentParser(description="Sort tiles into train/valid/test sets")

parser.add_argument('--config', type=str, default=None, help='Path to json file with arguments')
parser.add_argument('--input_path', type=str, default=None, help='Path to input file')
parser.add_argument('--tma_flag', type=bool, default=False, help='Whether to use the TMA tile extractor (written for 1mm cores)')
parser.add_argument('--override', action='store_true', default=False, help='Whether to overwrite files if they already exist')

args = parser.parse_args()
image = args.input_path
tma_flag = args.tma_flag
override = args.override

if args.config is not None:
    json_config = args.config
    with open(json_config, 'r') as file:
        args = json.load(file)

# # Tiling
# image = args['input_path']
output_path = args['output_path']
tile_size = args['tile_size']
pixel_size = args['pixel_size']
magnification = args['magnification']
background = args['background'] # percentage of background pixels tolerated in a tile; 100% means do not exclude any tiles
img_ext = args['img_ext']
sampleID = args['sampleID'] # number of characters your sample (patient) ID has
slideID = args['slideID'] # number of characters your slide ID has
# dataset = args['dataset'] # dataset name, e.g. TCGA-LUAD

#####################################################

def tile_extractor(image, tile_size, pixel_size, background):
    img = AICSImage(image, dask_tiles=True)
    print(f'Processing: {os.path.basename(image)}')
    lazy_t0 = img.get_image_dask_data("YXS")  
    img_array = lazy_t0.compute()
    img_x = img_array.shape[0]
    img_y = img_array.shape[1]
    num_channels = img_array.shape[2]
    mpp = img.physical_pixel_sizes[1]
    scale = pixel_size / mpp
    scaled_tile = int(np.round(tile_size * scale))
    tiles_range_x = int(np.floor(img_x / scaled_tile))
    tiles_range_y = int(np.floor(img_y / scaled_tile))
    slide_dict = dict()
    for i in range(0, tiles_range_x + 1):
        for j in range(0, tiles_range_y + 1):
            tile = img_array[j * scaled_tile:(j+1) * scaled_tile, i * scaled_tile:(i+1) * scaled_tile, :] 
            tile_key = f'{os.path.basename(image)}_{i}_{j}'
            # savetile_path = os.path.join(tiled_path, tile_filename)
            if tile.shape[0] == scaled_tile and tile.shape[1] == scaled_tile:
                jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
                gray = jpeg.convert('L')
                bw = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                bkg = np.average(bw)
                if bkg <= (background / 100):
                    slide_dict[tile_key] = np.uint8(jpeg)
                else:
                    continue
    return slide_dict

def tile_extractor_tma(image, tile_size, pixel_size):
    # tiled_path = os.path.join(output, os.path.basename(image).split(format)[0] + '_files', magnification)
    # if not os.path.exists(tiled_path):
        # os.makedirs(tiled_path)
    img = AICSImage(image, dask_tiles=True)
    print(f'Processing: {os.path.basename(image)}')
    lazy_t0 = img.get_image_dask_data("YXS")  
    img_array = lazy_t0.compute()
    img_x = img_array.shape[0]
    img_y = img_array.shape[1]
    mpp = img.physical_pixel_sizes[1]
    scale = pixel_size / mpp
    scaled_tile = int(np.round(tile_size * scale))
    # tiles_range_x = int(np.floor(img_x / scaled_tile))
    # tiles_range_y = int(np.floor(img_y / scaled_tile))
    core_centroid_x = img_x // 2
    core_centroid_y = img_y // 2
    upper_left = img_array[core_centroid_y-scaled_tile:core_centroid_y, core_centroid_x-scaled_tile:core_centroid_x]
    lower_left = img_array[core_centroid_y:core_centroid_y+scaled_tile, core_centroid_x-scaled_tile:core_centroid_x]
    upper_right = img_array[core_centroid_y-scaled_tile:core_centroid_y, core_centroid_x:core_centroid_x+scaled_tile]
    lower_right = img_array[core_centroid_y:core_centroid_y+scaled_tile, core_centroid_x:core_centroid_x+scaled_tile]
    tiles = [upper_left, lower_left, upper_right, lower_right]
    filenames = ['0_0', '0_1', '1_0', '1_1']
    slide_dict = dict()
    for i, tile in enumerate(tiles):
        if tile.shape[0] == scaled_tile and tile.shape[1] == scaled_tile:
            jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
            tile_filename = filenames[i]
            tile_key = f'{os.path.basename(image)}_{tile_filename}'
            slide_dict[tile_key] = np.uint8(jpeg)
        # savetile_path = os.path.join(tiled_path, tile_filename)
        # jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
        # jpeg.save(savetile_path, quality=90)
    return slide_dict
    
########################################################

# image = image
slide = os.path.basename(image).split(img_ext)[0]

if os.path.isfile(os.path.join(output_path, f'{slide}_tiles_{str(pixel_size).replace(".","p")}mpp_{magnification}x.pkl')):
    if override is False:
        sys.exit("File already exists")
    else:
        os.remove(os.path.join(output_path, f'{slide}_tiles_{str(pixel_size).replace(".","p")}mpp_{magnification}x.pkl'))

if tma_flag:
    slide_dict = tile_extractor_tma(image=image, tile_size=tile_size, pixel_size=pixel_size)
else:
    slide_dict = tile_extractor(image=image, tile_size=tile_size, pixel_size=pixel_size, background=background)

with open(os.path.join(output_path, f'{slide}_tiles_{str(pixel_size).replace(".","p")}mpp_{magnification}x.pkl'), 'wb') as file:
    pickle.dump(slide_dict, file)
