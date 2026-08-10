import pickle
import h5py
import numpy as np
import os

source_dir = '/path/to/your/pkl/files'  # Update this to your source directory
filename = 'hdf5_your_output_file_name.h5'

hdf5_file = os.path.join(source_dir, filename)
width = 224
height = 224

dir_files = list()

for pkl_file in os.scandir(source_dir):
    dir_files.append(pkl_file.path)

all_pkl = list()

for file in dir_files:    
    with open(file, 'rb') as inFile:
        pickle_tiles = pickle.load(inFile)
        all_pkl.append(pickle_tiles)

num_img = sum([len(pkl) for pkl in all_pkl])

with h5py.File(hdf5_file, 'w') as hdf5:
    images = list()
    samples = list()
    slides = list()
    tiles = list()

    dset1 = hdf5.create_dataset('img', (num_img, width, height, 3), dtype='uint8')
    dset2 = hdf5.create_dataset('samples', (num_img), dtype=f'S8')
    dset3 = hdf5.create_dataset('slides', (num_img), dtype=f'S30')
    dset4 = hdf5.create_dataset('tiles', (num_img), dtype=f'S8')

    for pkl in all_pkl:
        for key, img in pkl.items():
            sample = "_".join([key.split("_")[0], key.split("_")[1]])
            slide = key.split(".ndpi")[0]
            tile = "_".join([key.split("_")[-2], key.split("_")[-1]])

            images.append(img)
            samples.append(sample)
            slides.append(slide)
            tiles.append(tile)
        
    image_stack = np.stack(images, axis=0)
    sample_stack = np.stack(samples, axis=0).astype(f'S8')
    slide_stack = np.stack(slides, axis=0).astype(f'S30')
    tile_stack =  np.stack(tiles, axis=0).astype(f'S8')

    dset1[:, ...] = image_stack
    dset2[:, ...] = sample_stack
    dset3[:, ...] = slide_stack
    dset4[:, ...] = tile_stack
