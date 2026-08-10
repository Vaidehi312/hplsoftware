import argparse
import sys
sys.path.append('/nfs/home/users/krakovic/sharedscratch/HPL-LATTICeA/')

from models.visualization.cluster_image_tools import read_data_file, img_dict_from_dataset, cluster_set_images

parser = argparse.ArgumentParser(description='Retrieve cluster images')
parser.add_argument("--dataset_path", type=str, default='', help='Path to original hdf5 containing tile images')
parser.add_argument("--h5_complete_path", type=str, default='', help='Path to main hdf5')
parser.add_argument("--resolution", type=float, default=None, help='Leiden resolution clustering was done at')
parser.add_argument("--meta_field", type=str, default='', help='Name of the clustering run - for the results directory name only')
parser.add_argument("--num_batches", type=int, default='', help='Number of sets of 100 tiles to produce')

args = parser.parse_args()

dataset_path = args.dataset_path
h5_complete_path = args.h5_complete_path
resolution = args.resolution
meta_field = args.meta_field
batches = args.num_batches

data_file, leiden_clusters, main_cluster_path = read_data_file(h5_complete_path=h5_complete_path, meta_field=meta_field, resolution=resolution)
# data_dict = img_dict_from_dataset(h5_path=dataset_path)

for cluster in leiden_clusters:
    cluster_set_images(frame=data_file, cluster_id=cluster, leiden=f'leiden_{resolution}', dataset_path=dataset_path, save_path=main_cluster_path, batches=batches)
    # cluster_set_images(frame=data_file, data_dict=data_dict, cluster_id=cluster, leiden=f'leiden_{resolution}', save_path=main_cluster_path, batches=batches)