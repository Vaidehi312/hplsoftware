import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns
import warnings

def subsample_cores(frame, subsample):
    unique_cores_arr = np.unique(frame['core'])
    np.random.seed(1)
    random_indices = np.random.choice(unique_cores_arr.size, subsample, replace=False)
    subsampled_cores = np.unique(unique_cores_arr[random_indices])
    train_frame = frame[frame['core'].isin(subsampled_cores)]
    return train_frame

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
	return True, frame_sub

def plot_leiden_clusters_umap(resolution, adata, csv_file, ax, subsample=None):
    csv_dir = os.path.dirname(csv_file)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

    sc.tl.umap(adata, min_dist=0.5, spread=1.0, n_components=2, neighbors_key='nn_leiden', method='rapids')
    sc.pl.umap(adata, color=f'leiden_{resolution}', show=False, ax=ax)
    plt.gcf()
    plt.tight_layout()
    groupby = str(resolution).replace('.', 'p')
    plt.savefig(fname=os.path.join(csv_dir, f'01_umap_leiden_{groupby}.png'))

def plot_leiden_clusters_boxplot(adata, resolution, markers, nrows, ncols, subsample, csv_file):
    csv_dir = os.path.dirname(csv_file)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*8,nrows*8), sharey=True)
    axs = axs.flatten()

    for i, leiden in enumerate(sorted(adata.obs[f'leiden_{resolution}'].unique())):
        subset = adata.X[adata.obs[f'leiden_{resolution}'] == leiden]
        sns.boxplot(data=subset, width=0.5, ax=axs[i])
        axs[i].set_title(f'{leiden}')
        axs[i].tick_params(axis='x', labelrotation=45)
        axs[i].set_xticks(ticks=axs[i].get_xticks(), labels=markers)
        axs[i].axhline(y=0)
    
    plt.tight_layout()
    plt.gcf()
    groupby = str(resolution).replace('.', 'p')
    
    if subsample == True:
        plt.savefig(fname=os.path.join(csv_dir, f'02_training_{subsample}_cores_{groupby}_boxplot.png'))
    else:
        plt.savefig(fname=os.path.join(csv_dir, f'03_test_all_cores_leiden_{groupby}_boxplot.png'))