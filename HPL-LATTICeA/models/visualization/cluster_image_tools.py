import h5py
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import os

def img_dict_from_dataset(h5_path):
    data_dict = dict()
    print(f'Loading image dataset: {h5_path}')
    with h5py.File(h5_path, 'r') as hdf5_file:
        data_dict['images'] = {i: image for i, image in enumerate(hdf5_file['img'][:])}
        data_dict['samples'] = {i: sample for i, sample in enumerate(hdf5_file['samples'][:])}
        data_dict['slides'] = {i: slide for i, slide in enumerate(hdf5_file['slides'][:])}
        data_dict['tiles'] = {i: tile for i, tile in enumerate(hdf5_file['tiles'][:])}

    return data_dict

def get_tile_image(h5_path, idx):
    data_dict = dict()
    with h5py.File(h5_path, 'r') as hdf5_file:
        data_dict['images'] = hdf5_file['img'][idx]
        data_dict['samples'] = hdf5_file['samples'][idx]
        data_dict['slides'] = hdf5_file['slides'][idx]
        data_dict['tiles'] = hdf5_file['tiles'][idx]

        return data_dict

def cluster_set_images(frame, cluster_id, leiden, dataset_path=None, save_path=None, batches=1):
    if save_path is not None:
        save_path = os.path.join(save_path, f'images_{leiden.replace(".", "p")}')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
    all_indexes = list(frame[frame[leiden] == cluster_id].index)
    random.shuffle(all_indexes)
    combined_plot = sorted(all_indexes[:100 * batches])

    for batch in range(batches):
        images_cluster = list()
        meta_cluster = list()
        for index in combined_plot[batch * 100 : (batch + 1) * 100]:
            data_dict = get_tile_image(dataset_path, index)
            images_cluster.append(data_dict['images'])
            meta_cluster.append((index, data_dict['samples'], data_dict['slides'], data_dict['tiles']))

        sns.set_theme(style="white")
        fig = plt.figure(figsize=(30,6))
        fig.suptitle(f'Cluster {cluster_id}', fontsize=18, fontweight='bold')
        grid = ImageGrid(fig, 111, nrows_ncols=(5,20), axes_pad=0.1)

        for ax, im in zip(grid, images_cluster):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'cluster_{cluster_id}_{batch}.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)
            meta_frame = pd.DataFrame(meta_cluster, columns=['index', 'samples', 'slides', 'tiles'])
            meta_frame.to_csv(os.path.join(save_path, f'cluster_{cluster_id}_{batch}_meta.csv'))
            plt.close(fig)
            
        plt.show()

def read_data_file(h5_complete_path, meta_field, resolution):
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_field)
    main_cluster_path = os.path.join(main_cluster_path, 'adatas')
    adata_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + f'_leiden_{str(resolution).replace(".", "p")}__fold2.csv'

    data_file = pd.read_csv(os.path.join(main_cluster_path, adata_name))
    leiden_clusters = sorted(list(data_file[f'leiden_{resolution}'].unique()))

    return data_file, leiden_clusters, main_cluster_path
