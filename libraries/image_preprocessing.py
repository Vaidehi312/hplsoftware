import h5py
import numpy as np

def create_hdf5_file(hdf5_file, all_tiles_list, num_images, width, height, slideID, sampleID, img_ext='.ndpi'):
    with h5py.File(hdf5_file, 'w') as hdf5:
        dset1 = hdf5.create_dataset('img', (num_images, width, height, 3), dtype='uint8', maxshape=(None, width, height, 3))
        dset2 = hdf5.create_dataset('slides', (num_images, ), dtype = f'S{slideID}', maxshape=(None, ))
        dset3 = hdf5.create_dataset('tiles', (num_images, ), dtype = 'S38', maxshape=(None, ))
        dset4 = hdf5.create_dataset('samples', (num_images, ), dtype = f'S{sampleID}', maxshape=(None, ))

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

def append_to_hdf5_file(hdf5_file, all_tiles_list, num_images, width, height, slideID, sampleID, img_ext='.ndpi'):
    with h5py.File(hdf5_file, 'a') as hdf5:
        dset1 = hdf5['img']
        dset2 = hdf5['slides']
        dset3 = hdf5['tiles']
        dset4 = hdf5['samples']

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