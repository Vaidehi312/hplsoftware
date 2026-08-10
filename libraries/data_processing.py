import numpy as np
import os
import pandas as pd
from skbio.stats.composition import clr, multiplicative_replacement
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import time

def generate_timestamp():
    current = time.gmtime()
    yr = str(current[0])
    mo = str(current[1])
    dy = str(current[2])
    hr = str(current[3])
    mt = str(current[4])
    sc = str(current[5])
    return (yr + mo + dy + '_' + hr + mt + sc)

def reformat_core_id(name):
    row_number = name.split('-')[0]
    col_number = name.split('-')[1]
    if len(row_number) < 2:
        new_row_number = '0' + str(row_number)
    else:
        new_row_number = str(row_number)
    return new_row_number + '-' + col_number

def reformat_tma_num(name):
    if len(str(name)) < 2:
        return ('0' + name)
    else:
        return name
    
def match_cores_to_samples(df, col_to_map, recipient_col):
    cores_cases = np.loadtxt('/mnt/cephfs/home/users/krakovic/sharedscratch/datasets/LATTICeA/latticea_he_bioclavis_cores', dtype=str, delimiter=None)
    cores_cases_df = pd.DataFrame(cores_cases, columns=['complete'])
    cores_cases_df['samples'] = cores_cases_df['complete'].apply(lambda x: x.split("_")[0] + "_" + x.split("_")[1])
    cores_cases_df['core'] = cores_cases_df['complete'].apply(lambda x: x.split("_")[2].split(".tif")[0])

    cores_cases_dict = cores_cases_df[['core', 'samples']].set_index('core').to_dict()
    df[recipient_col] = df[col_to_map].map(cores_cases_dict['samples'])
    df[recipient_col] = df[recipient_col].astype('category')
    return df

# def load_topography(return_remove=False):
#     cluster_topography = pd.read_csv('/mnt/cephfs/home/users/krakovic/sharedscratch/notebooks/latticea_he/base_dataset_work/cluster_topography_v1.csv')
#     tier1_malignant = cluster_topography[cluster_topography['tier_1'] == 'malignant']['cluster'].values
#     tier1_malignant_nonecrosis = tier1_malignant[np.where((tier1_malignant != 9) & (tier1_malignant != 24) & (tier1_malignant != 33) & (tier1_malignant != 35))]
#     tier_1_tier_2_values = cluster_topography.loc[cluster_topography['tier_1'] == 'malignant']['tier_2'].values
#     tier_1_tier_2_values = tier_1_tier_2_values[np.where(tier_1_tier_2_values != 'necrosis')]
#     tier1_reactive = cluster_topography[cluster_topography['tier_1'] == 'reactive']['cluster'].values
#     remove_clusters = cluster_topography[cluster_topography['tier_1'] != 'malignant']['cluster'].values

#     if return_remove:
#         return tier1_malignant_nonecrosis, remove_clusters
#     else:
#         return tier1_malignant_nonecrosis

def load_topography():
    cluster_topography = pd.read_csv('/nfs/home/users/krakovic/sharedscratch/notebooks/latticea_he/base/consensus_frame_JLQ_DD_KR.csv', index_col=0)

    leiden_clusters = np.unique(cluster_topography['cluster'].values)
    malignant_clusters = cluster_topography[cluster_topography['malignant'] == 1]['cluster'].values
    stromal_clusters = cluster_topography.loc[(cluster_topography['malignant'] == 0) & (cluster_topography['feature'].isin(['collagenosis', 'elastosis', 'Lymphocytic stroma', 'other']))]['cluster'].values

    return leiden_clusters, malignant_clusters, stromal_clusters

def load_hazard_ratios(which, survival_path=None):
    
    if which == 'malignant':
        survival_path = '/mnt/cephfs/sharedscratch/users/krakovic/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/rapids_2p5m_leiden_2.5_l1ratio_0p2_mintiles_100_os_event_ind_20241126_17335_new_tumour_HPL_vectors/'
    elif which == 'stroma':
        survival_path = '/mnt/cephfs/sharedscratch/users/krakovic/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/rapids_2p5m_leiden_2.5_l1ratio_0p2_mintiles_100_os_event_ind_20241126_174844_new_stroma_HPL_vectors/'
    elif which == 'all':
        survival_path = '/mnt/cephfs/sharedscratch/users/krakovic/Histomorphological-Phenotype-Learning/results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/rapids_2p5m_leiden_2.5_l1ratio_0p2_mintiles_100_os_event_ind_20241126_155025_all_abnormal_HPL_vectors/'
    else:
        raise ValueError('Model should be either malignant, stroma or all')

    average_HRs = pd.read_csv(os.path.join(survival_path, 'all_folds_summary.csv'))

    return average_HRs

def get_wsi_composition(frame, matching_field, fold, counts=False, transform=True, groupby='leiden_2.5', save=False):
    clusters_vector = frame.groupby(matching_field)[f'{groupby}'].apply(list).reset_index(name='hpc_vector')
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w{1,2}\b")
    vectorized_text = vectorizer.fit_transform(clusters_vector['hpc_vector'].astype(str))
    vector_array = vectorized_text.toarray()
    if counts == True:
        clusters_vector_df = pd.DataFrame(vector_array, columns=vectorizer.get_feature_names_out(), index=clusters_vector.index)
        clusters_vector_df = pd.merge(clusters_vector_df, clusters_vector, left_index=True, right_index=True)
    elif transform == True:
        normed_matrix = clr(multiplicative_replacement(normalize(vector_array, axis=1, norm='l1')))
        clusters_vector_df = pd.DataFrame(normed_matrix, columns=vectorizer.get_feature_names_out(), index=clusters_vector.index)
        clusters_vector_df = pd.merge(clusters_vector_df, clusters_vector, left_index=True, right_index=True)
    else:
        normed_matrix = normalize(vector_array, axis=1, norm='l1')
        clusters_vector_df = pd.DataFrame(normed_matrix, columns=vectorizer.get_feature_names_out(), index=clusters_vector.index)
        clusters_vector_df = pd.merge(clusters_vector_df, clusters_vector, left_index=True, right_index=True)
    if save == True:
        clusters_vector_df.to_csv('%s_fold_%s_annotations.csv' % (groupby.replace('.', 'p'), fold))
    return clusters_vector_df.iloc[:, :-1]

def read_csvs_forcefold(adatas_path, groupby, h5_complete_path, h5_additional_path, force_fold):
    if force_fold is not None:
        i = force_fold

    adata_name      = h5_complete_path.split('/hdf5_')[1].split('.h5')[0] + f'_{groupby.replace(".", "p")}__fold{i}'
    train_csv       = os.path.join(adatas_path, f'{adata_name}_train.csv')
    valid_csv       = os.path.join(adatas_path, f'{adata_name}_valid.csv')
    test_csv       = os.path.join(adatas_path, f'{adata_name}_test.csv')
    if not os.path.isfile(train_csv):
        train_csv = os.path.join(adatas_path, f'{adata_name}.csv')
    train_df        = pd.read_csv(train_csv)
    valid_df        = pd.read_csv(valid_csv)
    test_df         = pd.read_csv(test_csv)

    complete_df = pd.concat([train_df, valid_df, test_df])

    leiden_clusters = np.unique(train_df[groupby].values.astype(int))

    if h5_additional_path is not None:
        adata_name     = h5_additional_path.split('/hdf5_')[1].split('.h5')[0] + f'_{groupby.replace(".", "p")}__fold{i}'
        additional_csv = os.path.join(adatas_path, f'{adata_name}.csv')
        additional_df = pd.read_csv(additional_csv)
        return complete_df, additional_df, leiden_clusters

    return complete_df, leiden_clusters

def generate_frequency_vector(complete_df, matching_field, groupby, leiden_clusters, transform=True, meta_field=None, min_tiles=1, min_perc=0):
    if min_perc >= 1:
        raise ValueError('Minimum percentage should be in the range 0-1')
    
    lr_data = list()
    lr_label = list()
    lr_samples = list()

    for sample in pd.unique(complete_df[matching_field].unique()):
        samples_df = complete_df[complete_df[matching_field] == sample]
        samples_df = samples_df[samples_df[groupby].isin(leiden_clusters)]

        num_tiles = samples_df.shape[0]
        if num_tiles < min_tiles:
            print(f'Sample: {sample} - {num_tiles} tiles. Skipping')
            continue
        
        # samples_features = [0]*len(leiden_clusters)
        samples_features = dict()
        for clust_id in leiden_clusters:
            samples_features[clust_id] = 0

        clusters_slide, clusters_counts = np.unique(samples_df[groupby], return_counts=True)
        for clust_id, count in zip(clusters_slide, clusters_counts):
            # samples_features[int(clust_id)] = count

            if (count / num_tiles) > min_perc:
                samples_features[clust_id] = count
            else:
                samples_features[clust_id] = 0

        # samples_features = np.array(samples_features, dtype=np.float64)
        samples_features = np.fromiter(samples_features.values(), dtype=np.float64)
        samples_features = np.array(samples_features) / np.sum(samples_features)
        if transform:
            samples_features = multiplicative_replacement(np.reshape(samples_features, (1,-1)))
            samples_features = clr(np.reshape(samples_features, (1,-1)))

        lr_samples.append(sample)
        lr_data.append(samples_features)

        try:
            samples_label = samples_df[meta_field].values[0]
            lr_label.append(samples_label)
        except:
            continue
            
    sample_rep_df = pd.DataFrame(data=lr_data, columns=leiden_clusters)
    sample_rep_df[matching_field] = lr_samples

    if len(lr_label) > 0:
        sample_rep_df[meta_field] = lr_label
        lr_label = np.stack(lr_label)
        lr_data = np.stack(lr_data)
    
        return lr_data, lr_label, sample_rep_df

    else:
        return sample_rep_df