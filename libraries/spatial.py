import libpysal as ps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pointpats import PointPattern, as_window
from pointpats import PoissonPointProcess as csr
import pointpats.quadrat_statistics as qs
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import KDTree
from scipy.stats import spearmanr, entropy
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize


def create_buffer_df(df, x_col, y_col, buffer_distance):
    min_x, max_x = df[x_col].min(), df[x_col].max()
    min_y, max_y = df[y_col].min(), df[y_col].max()
    
    is_interior = (
        (df[x_col] > min_x + buffer_distance) & 
        (df[x_col] < max_x - buffer_distance) & 
        (df[y_col] > min_y + buffer_distance) & 
        (df[y_col] < max_y - buffer_distance)
    )
    
    return is_interior

def create_buffer(coords, buffer_distance):
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    is_interior = (
        (coords[:, 0] > min_x + buffer_distance) & 
        (coords[:, 0] < max_x - buffer_distance) & 
        (coords[:, 1] > min_y + buffer_distance) & 
        (coords[:, 1] < max_y - buffer_distance)
    )
    
    return is_interior

def create_tessellation(interior_coords, tile_size):
    min_x, min_y = interior_coords.min(axis=0)
    max_x, max_y = interior_coords.max(axis=0)

    n_tiles_x = int((max_x - min_x) // tile_size)
    n_tiles_y = int((max_y - min_y) // tile_size)

    tile_centroids = np.zeros((n_tiles_x * n_tiles_y, 2))

    for i in range(n_tiles_x):
        tile_x = min_x + (i + 0.5) * tile_size
        for j in range(n_tiles_y):
            tile_y = min_y + (j + 0.5) * tile_size
            tile_centroids[i * n_tiles_y + j] = [tile_x, tile_y]
    
    return tile_centroids

def plot_spatial_densities(cell_type1, cell_type2, density1, density2, mask1, mask2, ax):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    # ax.scatter(density1[mask1], density2[mask2])
    ax.scatter(density1, density2, alpha=0.5)
    ax.set_xlabel(cell_type1)
    ax.set_ylabel(cell_type2)

def spatial_correlation(coords, cell_types, cell_type1, cell_type2, radius, ax):
    distances = squareform(pdist(coords))
    
    mask1 = cell_types == cell_type1
    mask2 = cell_types == cell_type2
    
    density1 = np.sum((distances < radius) & mask1[np.newaxis, :], axis=1)
    density2 = np.sum((distances < radius) & mask2[np.newaxis, :], axis=1)
    
    corr, p_val = spearmanr(density1, density2)
    print(density1.shape)
    print(density2.shape)

    plot_spatial_densities(cell_type1, cell_type2, density1, density2, mask1, mask2, ax)

    return corr, p_val

def cell_spatial_correlation(core_df, x_col, y_col, phenotype_col, buffer_distance, radius):
    coordinates = core_df[[x_col, y_col]].values
    phenotypes = core_df[phenotype_col].values
    interior_mask = create_buffer(coordinates, buffer_distance)
    interior_coordinates = coordinates[interior_mask]
    interior_phenotypes = phenotypes[interior_mask]

    tree = KDTree(data=interior_coordinates)
    neighbours = tree.query_ball_point(interior_coordinates, r=radius, workers=12)

    for i, neighbours_list in enumerate(neighbours):
        neighbours_list = neighbours_list.remove(i) # remove self from the list of neighbours

    neighbours_phenotypes = [interior_phenotypes[neighbours[i]] for i in range(len(neighbours))]

    neighbours_summary_list = list()

    for i, neighbours in enumerate(neighbours_phenotypes):
        u, c = np.unique(neighbours, return_counts=True)
        o = interior_phenotypes[i]
        result_dict = {'phenotype':o}
        result_dict.update(dict(zip(u, c)))
        neighbours_summary_list.append(result_dict)

    neighbours_df = pd.DataFrame.from_dict(neighbours_summary_list, orient='columns')
    neighbours_df = neighbours_df.fillna(0)
    neighbours_df['CellX'] = interior_coordinates[:, 0]
    neighbours_df['CellY'] = interior_coordinates[:, 1]
    # neighbours_df = neighbours_df.drop(columns='Negative')
    # neighbours_df = neighbours_df[neighbours_df['phenotype'] != 'Negative']

    return neighbours_df

def count_nearest_neighbours(core_df, x_col, y_col, phenotype_col, tile_size, buffer_distance=50):
    coordinates = core_df[[x_col, y_col]].values
    phenotypes = core_df[phenotype_col].values

    interior_mask = create_buffer(coordinates, buffer_distance)
    interior_coordinates = coordinates[interior_mask]
    interior_phenotypes = phenotypes[interior_mask]

    tile_centroids = create_tessellation(interior_coordinates, tile_size)

    core_cell_counts = list()

    for centroid in tile_centroids:
        distances = cdist([centroid], core_df[[x_col, y_col]].values).flatten()
        cells_in_radius = core_df[distances <= (tile_size / 2)]
        counts = cells_in_radius[phenotype_col].value_counts().to_dict()
        tile_info = {'x': centroid[0], 'y':centroid[1], **counts}
        core_cell_counts.append(tile_info)
    
    df = pd.DataFrame(core_cell_counts).fillna(0)
    
    return df

def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(spearmanr(tmp[r], tmp[c])[1], 4)
    return pvalues

def calculate_cell_density(df, x_col, y_col, phenotypes_col, pixel_size=None): 
    # Calculate the reciprocal of the average distance between cells of the same phenotype -- surrogate for cell density
    # phenotypes is an array of the phenotypes, indexes match the cell x and y coordinates;  
    # QuPath coordinates are in pixels not microns
    
    points = np.column_stack((df[x_col].values, df[y_col].values))
    phenotypes = df[phenotypes_col].values
    unique_phenotypes = np.unique(phenotypes)
    densities = dict()

    for phenotype in unique_phenotypes:
        phenotype_mask = phenotypes == phenotype
        phenotype_cells = points[phenotype_mask]

        if len(phenotype_cells) > 1:
            tree = KDTree(phenotype_cells)
            distances, _ = tree.query(phenotype_cells, k=2)
            mean_distance = np.mean(distances[:, 1])
            densities[phenotype] = dict()
            densities[phenotype]['average'] = 1 / mean_distance if mean_distance > 0 else 0
            densities[phenotype]['distances'] = distances
            # densities[phenotype] = mean_distance * pixel_size
        else:
            densities[phenotype]['average'] = 0
            densities[phenotype]['distances'] = 0

    return densities

def nearest_neighbour(df, x_col, y_col, phenotypes_col, target_phenotype, k=1, pixel_size=None, workers=12, buffer_distance=50):
    # Enumerate cells of a particular phenotype colocalising with another
    # target_phenotype is the cell of interest

    points = np.column_stack((df[x_col].values, df[y_col].values))
    phenotypes = df[phenotypes_col].values

    interior_mask = create_buffer(points, buffer_distance)
    interior_points = points[interior_mask]
    interior_phenotypes = phenotypes[interior_mask]

    target_mask = interior_phenotypes == target_phenotype
    target_points = interior_points[target_mask]

    if len(target_points) == 0:
        return None
    
    non_target_points = interior_points[~target_mask]

    tree = KDTree(non_target_points)
    distances, indices = tree.query(target_points, k=k, workers=workers)
    interest_phenotypes = phenotypes[indices]

    if pixel_size is not None:
        distances  = distances * pixel_size

    result = np.zeros(len(target_points), dtype=[('x', float), ('y', float), 
                                                 ('nearest_phenotype', object), 
                                                 ('distance', float),
                                                 ('core', object)])
    
    result['x'] = target_points[:, 0]
    result['y'] = target_points[:, 1]
    result['nearest_phenotype'] = interest_phenotypes
    result['distance'] = distances.flatten()
    
    return result

def calculate_distance_matrix_two_phenotypes(df, x_col, y_col, phenotypes_col, phenotype1, phenotype2, buffer_distance=10, pixel_size=None):
    # Calculate all the distances between all pairs of cells of two phenotypes eg. CD8+ and SMA+
    
    points = np.column_stack((df[x_col].values, df[y_col].values))
    phenotypes = df[phenotypes_col].values
    interior_mask = create_buffer(points, buffer_distance=buffer_distance)

    interior_points = points[interior_mask]
    interior_phenotypes = phenotypes[interior_mask]

    X_a = interior_points[interior_phenotypes==phenotype1]
    X_b = interior_points[interior_phenotypes==phenotype2]

    distance_mat = cdist(X_a, X_b, metric='euclidean')
    distance_mat = np.unique(distance_mat)
    distance_mat = distance_mat[distance_mat != 0]

    return np.mean(distance_mat)

def compute_qs(df, x_col, y_col, phenotypes_col, buffer_distance=125):
    points = np.column_stack((df[x_col].values, df[y_col].values))
    phenotypes = df[phenotypes_col].values
    interior_mask = create_buffer(points, buffer_distance=buffer_distance)

    interior_points = points[interior_mask]
    interior_phenotypes = phenotypes[interior_mask]
    core_pp = PointPattern(interior_points)

    q_r = qs.QStatistic(core_pp,shape= "rectangle",nx = 10, ny = 10)
    return q_r

# def get_core_noise_prop(core_df, x_col, y_col, phenotype_col, phenotype, eps, min_samples):
#     points = core_df[core_df[phenotype_col] == phenotype]][[x_col, y_col]].values

#     clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
#     total_cells = clusters.shape[0]
#     noise_cells = clusters[clusters == -1].shape
#     prop_noise = noise_cells / total_cells

#     # Reassign labels -- merge all clusters (comparison is cohesive vs not)
#     clusters[clusters != -1] = 1

#     return clusters, prop_noise

# def find_neighbourhoods(df, x_col, y_col, phenotypes_col, radius):

#     points = np.column_stack((df[x_col].values, df[y_col].values))
#     phenotypes = df[phenotypes_col].values

#     interior_mask = create_buffer(coords=points, buffer_distance=75)
#     interior_coordinates = points[interior_mask]
#     interior_phenotypes = phenotypes[interior_mask]

#     tile_centroids = create_tessellation(interior_coords=interior_coordinates, tile_size=(radius*2))
    
#     tree = KDTree(points)
#     result = tree.query_ball_point(tile_centroids, r=radius)

#     neighbours_phenotypes = [interior_phenotypes[res] for res in result]

#     return result, tile_centroids, neighbours_phenotypes

def calculate_shannon_diversity(phenotypes): #1-D ndarray of phenotypes of the cells of interest
    unique_phenotypes, counts = np.unique(phenotypes, return_counts=True)
    proportions = counts / phenotypes.shape[0]
    shannon_div = -np.sum(proportions * np.log(proportions))
    return shannon_div

def calculate_simpson_diversity(phenotypes):
    unique_phenotypes, counts = np.unique(phenotypes, return_counts=True)
    proportions = counts / phenotypes.shape[0]
    simpson_div = 1 - np.sum(proportions ** 2)
    return simpson_div

def run_shannon_entropy(row):
    counts_cols = [col for col in neighbourhood_df.columns if col not in ['x', 'y', 'core', 'km_cluster', 'entropy']]
    counts_vals = row[counts_cols].values
    if np.sum(counts_vals) == 0:
        return 0
    else:
        # print(counts_vals.reshape(1, -1))
        norm_counts_vals = normalize(counts_vals.reshape(1, -1), norm='l1', axis=1)
        # print(norm_counts_vals.shape)
        return entropy(norm_counts_vals.ravel(), base=2)
    
def run_shannon_entropy_relative(row):
    counts_cols = [col for col in neighbourhood_df.columns if col not in ['x', 'y', 'core', 'km_cluster', 'entropy']]
    counts_vals = row[counts_cols].values
    if np.sum(counts_vals) == 0:
        return 0
    else:
        # print(counts_vals.reshape(1, -1))
        norm_counts_vals = normalize(counts_vals.reshape(1, -1), norm='l1', axis=1)
        # print(norm_counts_vals.shape)
        core_entropy = whole_core_entropy[row['core']]
        return entropy(norm_counts_vals.ravel(), core_entropy, base=2)
    
def nearest_neighbour_select_cells(df, x_col, y_col, phenotype_col, parent_phenotype, target_phenotype, upper_bounds=100, buffer_distance=50, k=1, workers=-1):
    interior_mask = create_buffer_df(df=df, x_col=x_col, y_col=y_col, buffer_distance=buffer_distance)
    interior_df = df.loc[interior_mask]

    parent_cells = interior_df[interior_df[phenotype_col] == parent_phenotype][[x_col, y_col]].values # distance to these cells
    target_cells = interior_df[interior_df[phenotype_col] == target_phenotype][[x_col, y_col]].values # by these cells

    parent_tree = KDTree(parent_cells)

    distances, indices = parent_tree.query(target_cells, distance_upper_bound=upper_bounds)

    results = pd.DataFrame({
        'CellX': target_cells[:, 0],
        'CellY': target_cells[:, 1],
        'distance': distances
    })

    return results