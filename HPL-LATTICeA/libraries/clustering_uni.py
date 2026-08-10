import anndata as ad
import h5py
import numpy as np
import os
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns

def subsample_h5ad(h5ad_path, num_subsample, subsample_method='samples'):
	adata = ad.read_h5ad(h5ad_path)
	if subsample_method == 'index':
		num_unique_groups = adata.obs.shape[0]
		all_groups = adata.obs.index
	else:
		num_unique_groups = adata.obs[subsample_method].nunique()
		all_groups = np.unique(adata.obs[subsample_method].values)

	if num_subsample > num_unique_groups:
		print(f'Size of adata[{subsample_method}] ({num_unique_groups}) is greater than subsample ({num_subsample})')
		print(f'Processing all {num_unique_groups} samples')
		return adata
	
	random_idx = np.random.choice(len(all_groups), size=num_subsample, replace=False)
	random_groups = all_groups[random_idx]
	if subsample_method == 'index':
		adata_subsample = adata[adata.obs.index.isin(random_groups)]
	else:
		adata_subsample = adata[adata.obs[subsample_method].isin(random_groups)]
	
	print(f'Subsampling adata to {num_subsample} {subsample_method}. Total size = {adata_subsample.shape[0]}')
	return adata_subsample

def run_clustering(adata, h5ad_path, resolution, n_neighbours, main_cluster_path, adata_name, subsample=False):
	print(f'Processing all {adata.shape[0]} samples for clustering at Leiden {resolution}')
	
	print(f'\t\tPCA')
	rsc.pp.pca(adata, svd_solver='auto', n_comps=adata.X.shape[1])
	
	print(f'\t\tNearest Neighbours (n = {n_neighbours})')
	rsc.pp.neighbors(adata, n_neighbors=n_neighbours, n_pcs=adata.X.shape[1], use_rep='X_pca', metric='euclidean', key_added='nn_leiden')

	print(f'\t\tClustering at Leiden {resolution}')
	rsc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution}', neighbors_key='nn_leiden')

	adata_df = adata.obs.copy(deep=True)
	adata_df.to_csv(os.path.join(main_cluster_path, adata_name), index=False)
		
	if subsample:
		adata.write_h5ad(os.path.join(main_cluster_path, adata_name.replace(".csv", "subsample.h5ad")))
		assign_leiden_clusters(adata_ref=adata, main_cluster_path=main_cluster_path, adata_name=adata_name, h5ad_path=h5ad_path, obs=f'leiden_{resolution}')

	else:
		adata.write_h5ad(os.path.join(main_cluster_path, adata_name.replace(".csv", ".h5ad")))

def run_leiden(h5ad_path, meta_field, n_neighbours=250, resolutions=[5.0], subsample=None, subsample_method=None):
	if subsample is not None:
		adata = subsample_h5ad(h5ad_path, subsample, subsample_method)
	else:
		adata = ad.read_h5ad(h5ad_path)
		
	main_cluster_path = h5ad_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_field)
	main_cluster_path = os.path.join(main_cluster_path, 'adatas')

	if not os.path.isdir(main_cluster_path):
		os.makedirs(main_cluster_path)

	for resolution in resolutions:
		adata_name = h5ad_path.split('/hdf5_')[1].split('.h5ad')[0] + f'_leiden_{str(resolution).replace(".", "p")}.csv'
		run_clustering(adata, h5ad_path, resolution, n_neighbours, main_cluster_path, adata_name, subsample=subsample)

def assign_leiden_clusters(adata_ref, main_cluster_path, adata_name, h5ad_path, obs, embedding_method='pca', neighbour_key='nn_leiden'):
	adata_test = ad.read_h5ad(h5ad_path)
	print(f'Assigning clusters to {adata_test.shape[0]} samples at {obs}')
	sc.tl.ingest(adata_test, adata_ref, obs=obs, embedding_method=embedding_method, neighbors_key=neighbour_key)
	obs_frame = adata_test.obs.copy(deep=True)
	obs_frame['clustering_set'] = obs_frame.index.apply(lambda x: 'train' if x in adata_ref.obs.index else 'test')
	obs_frame.to_csv(os.path.join(main_cluster_path, adata_name), index=False)


