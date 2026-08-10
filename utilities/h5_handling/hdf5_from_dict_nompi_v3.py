import argparse 
from collections import defaultdict
import gc
import h5py
from itertools import chain
import json
import numpy as np
import os
import pickle
import random
import sys

parser = argparse.ArgumentParser(description="Sort tiles into train/valid/test sets and package into hdf5")

parser.add_argument('--config', type=str, default=None, help='Path to json file with arguments')

args = parser.parse_args()
json_config = args.config

with open(json_config, 'r') as file:
    args = json.load(file)

input_path = args['input_path']
magnification = args['magnification']
sampleID = args['sampleID']
slideID = args['slideID']
perc_valid = args['perc_valid']
perc_test = args['perc_test']
balance = args['balance']
dataset = args['dataset']

output_path = args['output_path']
width = args['wSize']
height = args['wSize']

if len(args['subset']) < 1:
    subset = None
else:
    subset = args['subset'] # mode 2 only; can be 'train', 'valid', 'test', 'all' or 'combined' (combined=no sorting; all=process all three train/valid/test)

sort = args['sort'] # bool, whether to sort slides into train/valid/test; if false then subset must be combined

if args['maxIm'] < 1:
    max_images = None
else:
    max_images = args['maxIm']

img_ext = args['img_ext']
chunk_size = args['chunk_size'] # how many slides to load at a time; ignored if 0

##################################################################################################

def sanity_check_sets(sorted_dict):
    train_set = sorted_dict['train']
    valid_set = sorted_dict['valid']
    test_set = sorted_dict['test']
    intersec = train_set & valid_set & test_set
    if len(intersec) == 0:
        return True

##################################################################################################

if sort == 1:
    if perc_valid is not None:
        perc_train = 100 - perc_valid - perc_test
    else:
        perc_train = 100 - perc_test

    slides = [f.path for f in os.scandir(input_path) if f.name.endswith('.pkl')]

    random.shuffle(slides)  

    image_dict = defaultdict(set)

    for slide in slides:
        slide = os.path.basename(slide)
        sample = slide[:sampleID]
        image_dict[sample].add(slide)

    num_samples = len(image_dict)
    num_slides = sum(1 for x in chain.from_iterable([value for value in image_dict.values()]))

    proportions = [perc_train, perc_valid, perc_test]
    proportions = [perc / 100 for perc in proportions]

    target_samples = [int(num_samples * proportion) for proportion in proportions]
    target_slides = [int(num_slides * proportion) for proportion in proportions]

    if balance == 1: # balance by slides
        all_slides = [x for x in chain.from_iterable([values for values in image_dict.values()])]
        random.shuffle(all_slides)
        
        sorted_dict = defaultdict(set)
        cats = ['train', 'valid', 'test']

        for slide in all_slides:
            if len(sorted_dict['train']) < target_slides[0]:
                sorted_dict['train'].add(slide)
            elif len(sorted_dict['valid']) < target_slides[1]:
                sorted_dict['valid'].add(slide)
            elif len(sorted_dict['test']) < target_slides[2]:
                sorted_dict['test'].add(slide)
            else:
                len_train, len_valid, len_test = len(sorted_dict['train']), len(sorted_dict['valid']), len(sorted_dict['test'])
                least_slides = np.argmin([len_train, len_valid, len_test])
                sorted_dict[cats[least_slides]].add(slide)

    elif balance == 2: # balance by patients
        samples = [key for key in image_dict.keys()]
        random.shuffle(samples)

        sorted_dict = defaultdict(set)
        cats = ['train', 'valid', 'test']
        current_counts = [0, 0, 0]

        for sample in samples:
            if current_counts[0] < target_samples[0]:
                sorted_dict['train'].update(image_dict[sample])
                current_counts[0] += 1
            elif current_counts[1] < target_samples[1]:
                sorted_dict['valid'].update(image_dict[sample])
                current_counts[1] += 1
            elif current_counts[2] < target_samples[2]:
                sorted_dict['test'].update(image_dict[sample])
                current_counts[2] += 1
            else:
                # len_train, len_valid, len_test = len(sorted_dict['train']), len(sorted_dict['valid']), len(sorted_dict['test'])
                least_slides = np.argmin(current_counts)
                sorted_dict[cats[least_slides]].update(image_dict[sample])
        
        if sanity_check_sets(sorted_dict):
            print('All good')
        else:
            print("Error with set sorting")
            sys.exit(1)

    with open(os.path.join(output_path, f'{dataset}_train_set.txt'), 'w') as file:
        [file.write(x + '\n') for x in sorted_dict['train']]

    with open(os.path.join(output_path, f'{dataset}_valid_set.txt'), 'w') as file:
        [file.write(x + '\n') for x in sorted_dict['valid']]

    with open(os.path.join(output_path, f'{dataset}_test_set.txt'), 'w') as file:
        [file.write(x + '\n') for x in sorted_dict['test']]

    with open(os.path.join(output_path, f'{dataset}_sorted_tiles.pkl'), 'wb') as outFile:
        pickle.dump(sorted_dict, outFile)

    print('TRAIN: ', len(sorted_dict['train']))
    print('VALID: ', len(sorted_dict['valid']))
    print('TEST: ', len(sorted_dict['test']))

# print(sorted_dict['train'])

# print(slides[1])

##################################################################################################

if subset != 'combined':
    train_slides = [slide for slide in slides if os.path.basename(slide) in sorted_dict['train']]
    valid_slides = [slide for slide in slides if os.path.basename(slide) in sorted_dict['valid']]
    test_slides = [slide for slide in slides if os.path.basename(slide) in sorted_dict['test']]
else:
    train_slides = [f.path for f in os.scandir(input_path) if f.name.endswith('.pkl')]

print(len(train_slides))

if subset == 'all':
    subsets = ['train', 'valid', 'test']
    for i, slide_subset in enumerate([train_slides, valid_slides, test_slides]):
        subset_name = subsets[i]
        num_subset_slides = len(slide_subset)
        num_loops, remainder = divmod(num_subset_slides, chunk_size)
        print(f"Processing {subset_name} set")
        print(f"\t{num_subset_slides} slides in {num_loops + 1} loops")
        for j in range(num_loops + 1):
            if j == num_loops -1:
                slides = list()

                for slide_path in slide_subset[(j * chunk_size):]:
                    with open(slide_path, 'rb') as file:
                        slide = pickle.load(file)
                        slides.append(slide)
            else:
                slides = list()
                for slide_path in slide_subset[(j * chunk_size):(j + 1) * chunk_size]:
                    with open(slide_path, 'rb') as file:
                        slide = pickle.load(file)
                        slides.append(slide)

            all_tiles_keys = [x for x in chain.from_iterable([slide.keys() for slide in slides])]
            all_tiles_values = [x for x in chain.from_iterable([slide.values() for slide in slides])]

            del slides

            all_tiles_list = list()

            for k, v in zip(all_tiles_keys, all_tiles_values):
                all_tiles_list.append((k, v))

            del all_tiles_keys
            del all_tiles_values

            num_images = len(all_tiles_list)
            print(f"Loop {j}")
            print("\tTotal images: " + str(num_images))

            if max_images is not None:
                random.shuffle(all_tiles_list)
                all_tiles_list = all_tiles_list[0:max_images]
                print(f"Restricting to {max_images} images")
                print("Total images: " + str(num_images))

            output_file = f'hdf5_{dataset}_he_{subset_name}.h5'

            gc.collect()

            if not os.path.isfile(os.path.join(output_path, output_file)):
                with h5py.File(os.path.join(output_path, output_file), 'w') as hdf5:
                    dset1 = hdf5.create_dataset(subset_name + '_img', (num_images, width, height, 3), dtype='uint8', maxshape=(None, width, height, 3))
                    dset2 = hdf5.create_dataset(subset_name + '_slides', (num_images, ), dtype = f'S{slideID}', maxshape=(None, ))
                    dset3 = hdf5.create_dataset(subset_name + '_tiles', (num_images, ), dtype = 'S38', maxshape=(None, ))
                    dset4 = hdf5.create_dataset(subset_name + '_samples', (num_images, ), dtype = f'S{sampleID}', maxshape=(None, ))

                    loadedImages = []
                    loadedTiles = []
                    loadedSlides = []
                    loadedSamples = []

                    for i, img in enumerate(all_tiles_list):
                        image = img[1]
                        slide = img[0].split(img_ext)[0]
                        tile = img[0].split(img_ext)[1][1:] + '.jpeg'
                        print(slide, tile)

                        if image.shape[0] != width or image.shape[1] != height:
                            print(f"Error with image {img}: mismatch between image size ({image.shape[0]}, {image.shape[1]}) and specified size ({width}, {height})")
                            break

                        image = np.uint8(image)
                        sample = slide[:sampleID]
                        slide = slide[:slideID]

                        loadedImages.append(image)
                        loadedTiles.append(tile)
                        loadedSlides.append(slide)
                        loadedSamples.append(sample)

                    image_stack = np.stack(loadedImages, axis=0)
                    tile_stack =  np.stack(loadedTiles, axis=0).astype('S16')
                    slide_stack = np.stack(loadedSlides, axis=0).astype(f'S{slideID}')
                    sample_stack = np.stack(loadedSamples, axis=0).astype(f'S{sampleID}')

                    dset1[:, ...] = image_stack
                    dset2[:, ...] = slide_stack
                    dset3[:, ...] = tile_stack
                    dset4[:, ...] = sample_stack
            
            else:
                with h5py.File(os.path.join(output_path, output_file), 'a') as hdf5:
                    dset1 = hdf5[f'{subset_name}_img']
                    dset2 = hdf5[f'{subset_name}_slides']
                    dset3 = hdf5[f'{subset_name}_tiles']
                    dset4 = hdf5[f'{subset_name}_samples']

                    dset1.resize((dset1.shape[0] + num_images, width, height, 3))
                    for dset in [dset2, dset3, dset4]:
                        dset.resize((dset.shape[0] + num_images, ))

                    loadedImages = []
                    loadedTiles = []
                    loadedSlides = []
                    loadedSamples = []

                    for i, img in enumerate(all_tiles_list):
                        image = img[1]
                        slide = img[0].split(img_ext)[0]
                        tile = img[0].split(img_ext)[1][1:] + '.jpeg'
                        print(slide, tile[1:])

                        if image.shape[0] != width or image.shape[1] != height:
                            print(f"Error with image {img}: mismatch between image size ({image.shape[0]}, {image.shape[1]}) and specified size ({width}, {height})")
                            break

                        image = np.uint8(image)
                        sample = slide[:sampleID]
                        slide = slide[:slideID]

                        loadedImages.append(image)
                        loadedTiles.append(tile)
                        loadedSlides.append(slide)
                        loadedSamples.append(sample)

                    image_stack = np.stack(loadedImages, axis=0)
                    tile_stack =  np.stack(loadedTiles, axis=0).astype('S16')
                    slide_stack = np.stack(loadedSlides, axis=0).astype(f'S{slideID}')
                    sample_stack = np.stack(loadedSamples, axis=0).astype(f'S{sampleID}')

                    dset1[-num_images:, ...] = image_stack
                    dset2[-num_images:, ...] = slide_stack
                    dset3[-num_images:, ...] = tile_stack
                    dset4[-num_images:, ...] = sample_stack

elif subset == 'combined':
    subsets = ['combined']
    for i, slide_subset in enumerate([train_slides]):
        subset_name = subsets[i]
        num_subset_slides = len(slide_subset)
        num_loops, remainder = divmod(num_subset_slides, chunk_size)
        print(f"Processing {subset_name} set")
        print(f"\t{num_subset_slides} slides in {num_loops + 1} loops")
        for j in range(num_loops + 1):
            if j == num_loops - 1:
                slides = list()

                for slide_path in slide_subset[(j * chunk_size):]:
                    with open(slide_path, 'rb') as file:
                        slide = pickle.load(file)
                        slides.append(slide)
            else:
                slides = list()
                for slide_path in slide_subset[(j * chunk_size):(j + 1) * chunk_size]:
                    with open(slide_path, 'rb') as file:
                        slide = pickle.load(file)
                        slides.append(slide)

            all_tiles_keys = [x for x in chain.from_iterable([slide.keys() for slide in slides])]
            all_tiles_values = [x for x in chain.from_iterable([slide.values() for slide in slides])]

            del slides

            all_tiles_list = list()

            for k, v in zip(all_tiles_keys, all_tiles_values):
                all_tiles_list.append((k, v))

            del all_tiles_keys
            del all_tiles_values

            num_images = len(all_tiles_list)
            print(f"Loop {j}")
            print("\tTotal images: " + str(num_images))

            if max_images is not None:
                random.shuffle(all_tiles_list)
                all_tiles_list = all_tiles_list[0:max_images]
                print(f"Restricting to {max_images} images")
                print("Total images: " + str(num_images))

            output_file = f'hdf5_{dataset}_he_{subset_name}.h5'

            gc.collect()

            if not os.path.isfile(os.path.join(output_path, output_file)):
                with h5py.File(os.path.join(output_path, output_file), 'w') as hdf5:
                    dset1 = hdf5.create_dataset(subset_name + '_img', (num_images, width, height, 3), dtype='uint8', maxshape=(None, width, height, 3))
                    dset2 = hdf5.create_dataset(subset_name + '_slides', (num_images, ), dtype = f'S{slideID}', maxshape=(None, ))
                    dset3 = hdf5.create_dataset(subset_name + '_tiles', (num_images, ), dtype = 'S38', maxshape=(None, ))
                    dset4 = hdf5.create_dataset(subset_name + '_samples', (num_images, ), dtype = f'S{sampleID}', maxshape=(None, ))

                    loadedImages = []
                    loadedTiles = []
                    loadedSlides = []
                    loadedSamples = []

                    for i, img in enumerate(all_tiles_list):
                        image = img[1]
                        slide = img[0].split(img_ext)[0]
                        tile = img[0].split(img_ext)[1][1:] + '.jpeg'
                        # print(slide, tile)

                        if image.shape[0] != width or image.shape[1] != height:
                            print(f"Error with image {img}: mismatch between image size ({image.shape[0]}, {image.shape[1]}) and specified size ({width}, {height})")
                            break

                        image = np.uint8(image)
                        sample = slide[:sampleID]
                        slide = slide[:slideID]

                        loadedImages.append(image)
                        loadedTiles.append(tile)
                        loadedSlides.append(slide)
                        loadedSamples.append(sample)

                    image_stack = np.stack(loadedImages, axis=0)
                    tile_stack =  np.stack(loadedTiles, axis=0).astype('S16')
                    slide_stack = np.stack(loadedSlides, axis=0).astype(f'S{slideID}')
                    sample_stack = np.stack(loadedSamples, axis=0).astype(f'S{sampleID}')

                    dset1[:, ...] = image_stack
                    dset2[:, ...] = slide_stack
                    dset3[:, ...] = tile_stack
                    dset4[:, ...] = sample_stack
            
            else:
                with h5py.File(os.path.join(output_path, output_file), 'a') as hdf5:
                    dset1 = hdf5[f'{subset_name}_img']
                    dset2 = hdf5[f'{subset_name}_slides']
                    dset3 = hdf5[f'{subset_name}_tiles']
                    dset4 = hdf5[f'{subset_name}_samples']

                    dset1.resize((dset1.shape[0] + num_images, width, height, 3))
                    for dset in [dset2, dset3, dset4]:
                        dset.resize((dset.shape[0] + num_images, ))

                    loadedImages = []
                    loadedTiles = []
                    loadedSlides = []
                    loadedSamples = []

                    for i, img in enumerate(all_tiles_list):
                        image = img[1]
                        slide = img[0].split(img_ext)[0]
                        tile = img[0].split(img_ext)[1][1:] + '.jpeg'
                        # print(slide, tile[1:])

                        if image.shape[0] != width or image.shape[1] != height:
                            print(f"Error with image {img}: mismatch between image size ({image.shape[0]}, {image.shape[1]}) and specified size ({width}, {height})")
                            break

                        image = np.uint8(image)
                        sample = slide[:sampleID]
                        slide = slide[:slideID]

                        loadedImages.append(image)
                        loadedTiles.append(tile)
                        loadedSlides.append(slide)
                        loadedSamples.append(sample)

                    image_stack = np.stack(loadedImages, axis=0)
                    tile_stack =  np.stack(loadedTiles, axis=0).astype('S16')
                    slide_stack = np.stack(loadedSlides, axis=0).astype(f'S{slideID}')
                    sample_stack = np.stack(loadedSamples, axis=0).astype(f'S{sampleID}')

                    dset1[-num_images:, ...] = image_stack
                    dset2[-num_images:, ...] = slide_stack
                    dset3[-num_images:, ...] = tile_stack
                    dset4[-num_images:, ...] = sample_stack

