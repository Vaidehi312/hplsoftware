# Imports
import anndata as ad
import argparse
import copy
import csv
import gc
import json
import numpy as np
import os
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/mnt/cephfs/home/users/krakovic/sharedscratch/Histomorphological-Phenotype-Learning')

# Own libs.
# from models.clustering.leiden_representations import run_leiden
from models.evaluation.folds import load_existing_split
from models.clustering.data_processing import *

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def sanity_check_neighbors(adata, dim_columns, neighbors_key='nn_leiden', tabs='\t\t'):
    distances = adata.obsp['%s_distances' % neighbors_key]
    n_neighbors = adata.uns['nn_leiden']['params']['n_neighbors']

    # Get locations with issues.
    original_frame_locs = list()
    i = 0
    for row in distances.tolil().rows:
        if len(row) != n_neighbors - 1:
            original_frame_locs.append(i)
        i += 1

    # If there's no problematic instances, continue
    if len(original_frame_locs) == 0:
        return False, None

    print('%sFound %s problematic instances' % (tabs, len(original_frame_locs)))
    print('%sRe-running clustering.' % tabs)
    # Recover from adata
    frame_sub = pd.DataFrame(adata.X, columns=dim_columns)
    for column in adata.obs.columns:
        frame_sub[column] = adata.obs[column].values

    # Drop problematic instances
    frame_sub = frame_sub.drop(original_frame_locs)
    # return False, frame_sub
    return False

def assign_clusters(frame, dim_columns, rest_columns, groupby, adata, main_cluster_path, adata_name, include_connections=False, save_adata=False, tabs='\t\t'):
	# If frame excceds maximum size, divide into smaller chuncks to avoid memory overrun.
	max_size = 500000
	if frame.shape[0] > max_size:
		frame = frame.reset_index()
		num_chuncks = np.ceil(frame.shape[0]/max_size).astype(int)
		sub_frames = [frame.loc[i*max_size:(i+1)*max_size-1] for i in range(num_chuncks)]
	else:
		sub_frames = [frame]

	mapped_frames = list()
	add = 0
	for i, frame in enumerate(sub_frames):
		# Crete AnnData based on frame.
		print('%s%s File %s' % (tabs, adata_name, i))
		adata_test = anndata.AnnData(X=frame[dim_columns].to_numpy(), obs=frame[rest_columns].astype('category'))

		# Assign cluster based on nearest neighbors from reference frame assignations.
		print('%sNearest Neighbors on data' % tabs)
		sc.tl.ingest(adata_test, adata, obs=groupby, embedding_method='pca', neighbors_key='nn_leiden')

		# Looks and dumps surrounding tile leiden connections per tile.
		if include_connections:
			include_tile_connections(groupby, main_cluster_path, adata_name)

		# Keep H5ad file.
		if save_adata:
			adata_test.write(os.path.join(main_cluster_path, adata_name + '_%s.h5ad' % i), compression='gzip')

		mapped_frames.append(pd.DataFrame(adata_test.obs).copy(deep=True))
		del adata_test

	# Combine all frames again and save to CSV.
	frame = pd.concat(mapped_frames, axis=0)
	frame.to_csv(os.path.join(main_cluster_path, '%s.csv' % adata_name), index=False)

	print()

def run_clustering(frame, dim_columns, rest_columns, resolution, groupby, n_neighbors, main_cluster_path, adata_name, subsample=None, include_connections=False, save_adata=False, tabs='\t\t'):
    # Handling subsampling.
    subsample_orig = subsample
    if subsample is None or subsample > frame.shape[0]:
        subsample      = int(frame.shape[0])
        adata_name     = adata_name.replace('_subsample', '')

    print('%sSubsampling DataFrame to %s samples' % (tabs, subsample))
    frame_sub = frame.sample(n=subsample, random_state=1)

    problematic_flag = False

    while problematic_flag:
        problematic_flag = False
        print('%s%s File' % (tabs, adata_name))
        adata = anndata.AnnData(X=frame_sub[dim_columns].to_numpy(), obs=frame_sub[rest_columns].astype('category'))
        # Nearest Neighbors
        print('%sPCA' % tabs)
        sc.tl.pca(adata, svd_solver='arpack', n_comps=adata.X.shape[1] - 1)
        print('%sNearest Neighbors' % tabs)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.X.shape[1] - 1, method='rapids', metric='euclidean', key_added='nn_leiden')
        # Sanity check, some representation instances introduce a bug where the # NN is smaller than specified and breaks scanpy
        # problematic_flag, frame_sub = sanity_check_neighbors(adata, dim_columns, neighbors_key='nn_leiden', tabs=tabs)
        problematic_flag = sanity_check_neighbors(adata, dim_columns, neighbors_key='nn_leiden', tabs=tabs)

    print('%s%s File' % (tabs, adata_name))
    adata = anndata.AnnData(X=frame_sub[dim_columns].to_numpy(), obs=frame_sub[rest_columns].astype('category'))
    # Nearest Neighbors
    print('%sPCA' % tabs)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=adata.X.shape[1] - 1)
    print('%sNearest Neighbors' % tabs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=adata.X.shape[1] - 1, method='rapids', metric='euclidean', key_added='nn_leiden')

    # Leiden clustering.
    print('%sLeiden' % tabs, resolution)
    rsc.tl.leiden(adata, resolution=resolution, key_added=groupby, neighbors_key='nn_leiden')

    # Save to csv.
    adata_to_csv(adata, main_cluster_path, adata_name)

    # Looks and dumps surrounding tile leiden connections per tile.
    if include_connections: include_tile_connections(groupby, main_cluster_path, adata_name)

    # Keep H5ad file.
    if save_adata: adata.write(os.path.join(main_cluster_path, adata_name) + '.h5ad', compression='gzip')
    print()

    # adata = anndata.read_h5ad(os.path.join(main_cluster_path, adata_name) + '.h5ad')

    if subsample_orig is None or subsample_orig > frame.shape[0]:
        subsample = None

    return adata, subsample

def run_leiden(meta_field, matching_field, rep_key, h5_complete_path, h5_additional_path, folds_pickle, resolutions, n_neighbors=250, subsample=200000, include_connections=False, save_adata=False):
    # Get folds from existing split.
    folds = load_existing_split(folds_pickle)

    complete_frame, complete_dims, complete_rest = representations_to_frame(h5_complete_path, meta_field=meta_field, rep_key=rep_key)
    additional_frame, additional_dims, additional_rest = representations_to_frame(h5_additional_path, meta_field=meta_field, rep_key=rep_key)

    # Setup folder scheme
    main_cluster_path = h5_complete_path.split('hdf5_')[0]
    main_cluster_path = os.path.join(main_cluster_path, meta_field)
    main_cluster_path = os.path.join(main_cluster_path, 'adatas')

    if not os.path.isdir(main_cluster_path):
        os.makedirs(main_cluster_path)

    print()
    for resolution in resolutions:
        print('Leiden %s' % resolution)
        groupby = 'leiden_%s' % resolution
        for i, fold in enumerate(folds):
            print('\tFold', i)

            # Fold split.
            train_samples, valid_samples, test_samples = fold

            ### Train set.
            failed = False
            try:
                train_frame = complete_frame[complete_frame[matching_field].isin(train_samples)]
                if train_frame.shape[0] == 0:
                    print('No match between fold train samples [%s] and H5 file matching_field [%s]' % (train_samples[0], complete_frame[matching_field][0]))
                    exit()
                adata_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
                adata_train, subsample = run_clustering(train_frame, complete_dims, complete_rest, resolution, groupby, n_neighbors, main_cluster_path, '%s_subsample' % adata_name,
                                                        subsample=subsample, include_connections=include_connections, save_adata=True)

                if subsample is not None:
                    assign_clusters(train_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, adata_name, include_connections=include_connections, save_adata=save_adata)

            except Exception as ex:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                failed = True

                if hasattr(ex, 'message'):
                    print('\t\tException', ex.message)
                else:
                    print('\t\tException', ex)
            finally:
                gc.collect()

            # Do not even try if train failed.
            if failed:
                continue

            ### Validation set.
            try:
                if len(valid_samples) > 0:
                    valid_frame = complete_frame[complete_frame[matching_field].isin(valid_samples)]
                    assign_clusters(valid_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_valid' % adata_name, include_connections=include_connections,
                                    save_adata=save_adata)
            except Exception as ex:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                if hasattr(ex, 'message'):
                    print('\t\tException', ex.message)
                else:
                    print('\t\tException', ex)

            ### Test set.
            try:
                test_frame = complete_frame[complete_frame[matching_field].isin(test_samples)]
                assign_clusters(test_frame, complete_dims, complete_rest, groupby, adata_train, main_cluster_path, '%s_test' % adata_name, include_connections=include_connections,
                                save_adata=save_adata)
            except Exception as ex:
                print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                if hasattr(ex, 'message'):
                    print('\t\tException', ex.message)
                else:
                    print('\t\tException', ex)

            ### Additional set.
            if additional_frame is not None:
                try:
                    adata_name = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + '_%s__fold%s' % (groupby.replace('.', 'p'), i)
                    assign_clusters(additional_frame, additional_dims, additional_rest, groupby, adata_train, main_cluster_path, adata_name, include_connections=include_connections,
                                    save_adata=save_adata)
                except Exception as ex:
                    print('\t\tIssue running Leiden %s on fold %s Train Set' % (resolution, i))
                    if hasattr(ex, 'message'):
                        print('\t\tException', ex.message)
                    else:
                        print('\t\tException', ex)

            del adata_train
            gc.collect()

###########################################

parser = argparse.ArgumentParser(description="Cluster representations using Leiden algorithm with RAPIDS")

parser.add_argument('--config', type=str, default=None, help='Path to json file with arguments')

args = parser.parse_args()
json_config = args.config

with open(json_config, 'r') as file:
    args = json.load(file)

main_path           = '/mnt/cephfs/home/users/krakovic/sharedscratch/Histomorphological-Phenotype-Learning'

meta_field          = args['meta_field']
matching_field      = args['matching_field']
h5_complete_path    = args['h5_complete_path']

try:
    h5_additional_path  = args['h5_additional_path']
except KeyError:
    h5_additional_path = None

folds_pickle        = args['folds_pickle']
rep_key             = args['rep_key']
resolutions         = args['resolution']
n_neighbors         = args['n_neighbors']
subsample           = args['subsample']
include_connections = False

run_leiden(meta_field, matching_field, rep_key, h5_complete_path, h5_additional_path, folds_pickle, resolutions, n_neighbors=n_neighbors, subsample=subsample, include_connections=include_connections)