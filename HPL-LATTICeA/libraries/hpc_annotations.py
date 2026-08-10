import pandas as pd
import sys
sys.path.append('/nfs/home/users/krakovic/sharedscratch/notebooks/latticea_he/libraries/')
from data_processing import load_topography
from supercluster_dictionary import *

def tma_hpc_database(database):
    bioclavis_frame = pd.read_csv('/mnt/cephfs/home/users/krakovic/sharedscratch/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/adatas/LATTICeA_BioclavisHE_4pc_5x_he_train_filtered_leiden_2p5__fold2.csv')
    grandslam_frame = pd.read_csv('/mnt/cephfs/home/users/krakovic/sharedscratch/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/adatas/LATTICeA_GrandSlam_4pc_5x_he_combined_filtered_leiden_2p5__fold2.csv')

    grandslam_frame = grandslam_frame.rename(columns={'combined_samples':'samples',
                                                      'combined_slides':'slides',
                                                      'combined_tiles':'tiles'})
    
    bioclavis_frame['core_ID'] = bioclavis_frame['slides'].apply(lambda x: x.split('_')[2])
    bioclavis_frame['TMA'] = bioclavis_frame['core_ID'].apply(lambda x: x.split('-')[0])
    grandslam_frame['core_ID'] = grandslam_frame['slides'].apply(lambda x: x.split('_')[2])
    grandslam_frame['TMA'] = grandslam_frame['core_ID'].apply(lambda x: x.split('-')[0])

    if database == 'bioclavis':
        return bioclavis_frame
    elif database == 'grandslam':
        return grandslam_frame
    else:
        raise KeyError(f'Database {database} not found')

def load_hpc_core_annotations(dataset_key, tma_numbers):
    clusters_frame = tma_hpc_database(database=dataset_key)
    clusters_frame = clusters_frame[clusters_frame['TMA'].isin(tma_numbers)]
    clusters_frame['core_ID'] = clusters_frame['slides'].apply(lambda x: x[9:])
    return clusters_frame

def return_pure_superclusters(clusters_frame):
    # To use the original morphology-based classification system
    pivot_clusters_frame = pd.get_dummies(clusters_frame[['samples', 'core_ID', 'leiden_2.5']], prefix='cluster', columns=['leiden_2.5'])
    pivot_clusters_frame = pivot_clusters_frame.groupby(by='core_ID').sum().reset_index()
    leiden_clusters, malignant_clusters, stromal_clusters = load_topography()

    # Calcualte max and sum of clusters
    majority_clusters_frame = pivot_clusters_frame.copy(deep=True)
    majority_clusters_frame = majority_clusters_frame[['core_ID']+[f'cluster_{col}' for col in malignant_clusters]]
    majority_clusters_frame['max'] = majority_clusters_frame[[f'cluster_{col}' for col in malignant_clusters]].max(numeric_only=True, axis=1)
    majority_clusters_frame['sum'] = majority_clusters_frame[[f'cluster_{col}' for col in malignant_clusters]].sum(numeric_only=True, axis=1)

    # Assign majority vote for individual HPCs
    cluster_pure_cores = majority_clusters_frame.loc[(majority_clusters_frame['max'] / majority_clusters_frame['sum']) > 0.5]
    cluster_pure_cores['cluster'] = cluster_pure_cores[[f'cluster_{col}' for col in malignant_clusters]].idxmax(numeric_only=True, axis=1)
    cluster_pure_cores['cluster'] = cluster_pure_cores['cluster'].apply(lambda x: 'HPC ' + str(x.split("_")[1]))

    # Count up supercluster occurances -- different HPCs may be in the same supercluster and these cores should be considered pure for the supercluster
    # majority_clusters_frame['fibroblast_sum'] = majority_clusters_frame.apply(lambda row: count_fibroblast(row), axis=1)
    # majority_clusters_frame['ecm_sum'] = majority_clusters_frame.apply(lambda row: count_ecm(row), axis=1)
    # majority_clusters_frame['lymphocyte_sum'] = majority_clusters_frame.apply(lambda row: count_lymphocyte(row), axis=1)
    # majority_clusters_frame['lepidic_sum'] = majority_clusters_frame.apply(lambda row: count_lepidic(row), axis=1)
    # majority_clusters_frame['lepidic-muc_sum'] = majority_clusters_frame.apply(lambda row: count_mucinous(row), axis=1)
    # majority_clusters_frame['acinar_sum'] = majority_clusters_frame.apply(lambda row: count_acinar(row), axis=1)
    # majority_clusters_frame['papillary_sum'] = majority_clusters_frame.apply(lambda row: count_papillary(row), axis=1)
    # majority_clusters_frame['cribriform_sum'] = majority_clusters_frame.apply(lambda row: count_cribriform(row), axis=1)
    # majority_clusters_frame['solid_sum'] = majority_clusters_frame.apply(lambda row: count_solid(row), axis=1)
    # majority_clusters_frame['sc_max'] = majority_clusters_frame[['fibroblast_sum', 'ecm_sum', 'lymphocyte_sum', 'lepidic_sum', 'lepidic-muc_sum', 'acinar_sum', 'papillary_sum', 'cribriform_sum', 'solid_sum']].max(numeric_only=True, axis=1)

    majority_clusters_frame['hot_cohesive'] = majority_clusters_frame.apply(lambda row: count_hot_cohesive(row), axis=1)
    majority_clusters_frame['hot_discohesive'] = majority_clusters_frame.apply(lambda row: count_hot_discohesive(row), axis=1)
    majority_clusters_frame['cold_cohesive'] = majority_clusters_frame.apply(lambda row: count_cold_cohesive(row), axis=1)
    majority_clusters_frame['cold_discohesive'] = majority_clusters_frame.apply(lambda row: count_cold_discohesive(row), axis=1)

    # Majority vote for superclusters
    majority_clusters_frame['sc_max'] = majority_clusters_frame[['hot_cohesive', 'hot_discohesive', 'cold_cohesive', 'cold_discohesive']].max(numeric_only=True, axis=1)

    supercluster_pure_cores = majority_clusters_frame.loc[(majority_clusters_frame['sc_max'] / majority_clusters_frame['sum']) > 0.5]
    supercluster_pure_cores['supercluster'] = supercluster_pure_cores[['hot_cohesive', 'hot_discohesive', 'cold_cohesive', 'cold_discohesive']].idxmax(numeric_only=True, axis=1)
    # supercluster_pure_cores['supercluster'] = supercluster_pure_cores[['fibroblast_sum', 'ecm_sum', 'lymphocyte_sum', 'lepidic_sum', 'lepidic-muc_sum', 'acinar_sum', 'papillary_sum', 'cribriform_sum', 'solid_sum']].idxmax(numeric_only=True, axis=1)
    supercluster_pure_cores['supercluster'] = supercluster_pure_cores['supercluster'].apply(lambda x: x.split("_")[0].capitalize())

    assert cluster_pure_cores[~cluster_pure_cores['core_ID'].isin(supercluster_pure_cores['core_ID'])].shape[0] == 0

    merged_frames = supercluster_pure_cores.merge(cluster_pure_cores[['core_ID', 'cluster']], on='core_ID', how='left')
    merged_frames = merged_frames[['core_ID', 'cluster', 'supercluster']]

    return merged_frames

def return_pure_clusters(clusters_frame):
    pivot_clusters_frame = pd.get_dummies(clusters_frame[['samples', 'core_ID', 'leiden_2.5']], prefix='cluster', columns=['leiden_2.5'])
    pivot_clusters_frame = pivot_clusters_frame.groupby(by='core_ID').sum().reset_index()
    leiden_clusters, malignant_clusters, stromal_clusters = load_topography()
    malignant_clusters = list(malignant_clusters)+[10,16]

    # Calcualte max and sum of clusters
    majority_clusters_frame = pivot_clusters_frame.copy(deep=True)
    majority_clusters_frame = majority_clusters_frame[['core_ID']+[f'cluster_{col}' for col in malignant_clusters]]
    majority_clusters_frame['max'] = majority_clusters_frame[[f'cluster_{col}' for col in malignant_clusters]].max(numeric_only=True, axis=1)
    majority_clusters_frame['sum'] = majority_clusters_frame[[f'cluster_{col}' for col in malignant_clusters]].sum(numeric_only=True, axis=1)

    # Assign majority vote for individual HPCs
    cluster_pure_cores = majority_clusters_frame.loc[(majority_clusters_frame['max'] / majority_clusters_frame['sum']) > 0.5]
    cluster_pure_cores['cluster'] = cluster_pure_cores[[f'cluster_{col}' for col in malignant_clusters]].idxmax(numeric_only=True, axis=1)
    cluster_pure_cores['cluster'] = cluster_pure_cores['cluster'].apply(lambda x: 'HPC ' + str(x.split("_")[1]))

    return cluster_pure_cores[['core_ID', 'cluster']]

