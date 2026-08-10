from aicsimageio import AICSImage
import h5py
import numpy as np
import os
from PIL import Image

def load_dask_image(image_path):
    wsi = AICSImage(image_path, dask_tiles=True)
    wsi_dask_array = wsi.get_image_dask_data("YXS")
    img_y = wsi_dask_array.shape[0]
    img_x = wsi_dask_array.shape[1]

    return img_y, img_x, wsi

def calculate_tile_bounds(wsi, img_y, img_x, pixel_size, tile_size):
    mpp = wsi.physical_pixel_sizes[1]
    scale = pixel_size / mpp
    scaled_tile = int(np.round(tile_size * scale))
    tiles_range_x = int(np.floor(img_x / scaled_tile))
    tiles_range_y = int(np.floor(img_y / scaled_tile))

    return scaled_tile, tiles_range_x, tiles_range_y

def extract_tiles_to_directory(img, x_range, y_range, tile_size, scaled_tile, output_path, th_background):
    for j in range(0, x_range + 1):
        for i in range(0, y_range + 1):
            # tile = img_array[j * scaled_tile:(j+1) * scaled_tile, i * scaled_tile:(i+1) * scaled_tile, :] 
            try:
                tile = img.get_image_dask_data("YXS", X=slice(j * scaled_tile, (j+1) * scaled_tile), Y=slice(i * scaled_tile, (i+1) * scaled_tile)).compute()
                # tile = img.get_image_dask_data("YXS", X=slice(i * scaled_tile, (i+1) * scaled_tile), Y=slice(j * scaled_tile, (j+1) * scaled_tile)).compute()
                tile_key = f'{j}_{i}.jpeg'
                # savetile_path = os.path.join(tiled_path, tile_filename)
                print(tile_key)
                if tile.shape[0] == scaled_tile and tile.shape[1] == scaled_tile:
                    jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
                    gray = jpeg.convert('L')
                    bw = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                    bkg = np.average(bw)
                    if bkg >= 1 - (th_background / 100):
                        savetile_path = os.path.join(output_path, tile_key)
                        print(savetile_path)
                        jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
                        jpeg.save(savetile_path, quality=90)
                    else:
                        continue
            except IndexError:
                continue

def extract_tiles(img, x_range, y_range, tile_size, scaled_tile, output_path, th_background):
    loadedImages = list()
    loadedTiles = list()
    
    for j in range(0, x_range + 1):
        for i in range(0, y_range + 1):
            try:
                tile = img.get_image_dask_data("YXS", X=slice(j * scaled_tile, (j+1) * scaled_tile), Y=slice(i * scaled_tile, (i+1) * scaled_tile)).compute()
                tile_key = f'{j}_{i}.jpeg'
                if tile.shape[0] == scaled_tile and tile.shape[1] == scaled_tile:
                    jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
                    gray = jpeg.convert('L')
                    bw = gray.point(lambda x: 0 if x < 220 else 1, 'F')
                    bkg = np.average(bw)
                    if bkg >= 1 - (th_background / 100):
                        savetile_path = os.path.join(output_path, tile_key)
                        print(savetile_path)
                        jpeg = Image.fromarray(tile, mode='RGB').resize((tile_size,tile_size))
                        loadedImages.append(jpeg)
                        loadedTiles.append(tile_key)
                    else:
                        continue
            except IndexError:
                continue
    
    return loadedImages, loadedTiles

def make_hdf5_from_tiles(all_images, all_image_paths, tile_size, pixel_size, slideID, sampleID, input_path, output_path, th_background):
    slide_ext = os.path.basename(input_path).split(".")[1]
    slide_name = os.path.basename(input_path).split(f".{slide_ext}")[0]
    sample_name = slide_name[:sampleID]
    num_images = len(all_images)

    loadedSlides = [slide_name]*num_images
    loadedSamples = [sample_name]*num_images

    output_file = f'hdf5_{slide_name}_mpp{str(pixel_size).replace(".", "p")}_background_{th_background}pct_he.h5'

    with h5py.File(os.path.join(output_path, output_file), 'w') as hdf5:
        dset1 = hdf5.create_dataset('img', (num_images, tile_size, tile_size, 3), dtype='uint8', maxshape=(None, tile_size, tile_size, 3))
        dset2 = hdf5.create_dataset('slides', (num_images, ), dtype = f'S{slideID}', maxshape=(None, ))
        dset3 = hdf5.create_dataset('tiles', (num_images, ), dtype = 'S12', maxshape=(None, ))
        dset4 = hdf5.create_dataset('samples', (num_images, ), dtype = f'S{sampleID}', maxshape=(None, ))

        loadedImages = []
        loadedTiles = []

        for i, img in enumerate(all_images):
            img = np.asarray(img, dtype=np.uint8)
            tile = all_image_paths[i]
            if img.shape[0] != tile_size or img.shape[1] != tile_size:
                print(f"Error with image {tile}: mismatch between image size ({img.shape[0]}, {img.shape[1]}) and specified size ({tile_size}, {tile_size})")
                break

            loadedImages.append(img)
            loadedTiles.append(tile)

        img_stack = np.stack(loadedImages, axis=0)
        tile_stack = np.stack(loadedTiles, axis=0).astype('S12')
        slide_stack = np.stack(loadedSlides, axis=0).astype(f'S{slideID}')
        sample_stack = np.stack(loadedSamples, axis=0).astype(f'S{sampleID}')

        dset1[:, ...] = img_stack
        dset2[:, ...] = slide_stack
        dset3[:, ...] = tile_stack
        dset4[:, ...] = sample_stack

def tile_wsi_to_hdf5(input_path, output_path, pixel_size, tile_size, sampleID, slideID, th_background):
    img_y, img_x, wsi = load_dask_image(input_path)
    scaled_tile, x_range, y_range = calculate_tile_bounds(wsi, img_y, img_x, pixel_size, tile_size)
    all_images, all_image_paths = extract_tiles(wsi, x_range, y_range, tile_size, scaled_tile, output_path, th_background)
    make_hdf5_from_tiles(all_images, all_image_paths, tile_size, pixel_size, slideID, sampleID, input_path, output_path, th_background)

