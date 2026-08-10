import anndata as ad
import h5py
import numpy as np
import os
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns

def representations_to_frame(h5_path, rep_key='z_latent'):
	if h5_path is not None:
		print('Loading representations:', h5_path)
		with h5py.File(h5_path, 'r') as content:
			for key in content.keys():
				if rep_key in key:
					representations = content[key][:]
					dim_columns     = list(range(representations.shape[1]))
					frame = pd.DataFrame(representations, columns=dim_columns)
					break

			rest_columns = list()
			for key in content.keys():
				if 'latent' in key:
					continue
				key_ = key.replace('train_', '')
				key_ = key_.replace('valid_', '')
				key_ = key_.replace('test_',  '')
				if key_ not in frame.columns:
					frame[key_] = content[key][:].astype(str)
					rest_columns.append(key_)
	else:
		frame, dim_columns, rest_columns = None, None, None

	return frame, dim_columns, rest_columns

def run_clustering_all_samples(frame, dim_columns, rest_columns, resolution, n_neighbours, main_cluster_path, adata_name, save_h5ad=True):
	adata = ad.AnnData(X=frame[dim_columns].values, obs=frame[rest_columns])
	print(f'Processing all {adata.shape[0]} samples for clustering at Leiden {resolution}')
	
	print(f'\t\tPCA')
	rsc.pp.pca(adata, svd_solver='auto', n_comps=adata.X.shape[1])
	
	print(f'\t\tNearest Neighbours (n = {n_neighbours})')
	rsc.pp.neighbors(adata, n_neighbors=n_neighbours, n_pcs=adata.X.shape[1], use_rep='X_pca', metric='euclidean', key_added='nn_leiden')

	print(f'\t\tClustering at Leiden {resolution}')
	rsc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution}', neighbors_key='nn_leiden')

	adata_df = adata.obs.copy(deep=True)
	adata_df.to_csv(os.path.join(main_cluster_path, adata_name), index=False)
	if save_h5ad:
		adata.write_h5ad(os.path.join(main_cluster_path, adata_name.replace(".csv", ".h5ad")))

def run_leiden(h5_complete_path, h5_additional_path, meta_field, n_neighbours=250, resolutions=[5.0]):
	complete_frame, complete_dims, complete_rest = representations_to_frame(h5_complete_path)
	main_cluster_path = h5_complete_path.split('hdf5_')[0]
	main_cluster_path = os.path.join(main_cluster_path, meta_field)
	main_cluster_path = os.path.join(main_cluster_path, 'adatas')

	if not os.path.isdir(main_cluster_path):
		os.makedirs(main_cluster_path)

	for resolution in resolutions:
		adata_name = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + f'_leiden_{str(resolution).replace(".", "p")}.csv'
		run_clustering_all_samples(complete_frame, complete_dims, complete_rest, resolution, n_neighbours, main_cluster_path, adata_name)