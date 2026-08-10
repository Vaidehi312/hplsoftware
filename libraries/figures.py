from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import os
import seaborn as sns
from skimage.transform import resize
from skimage.transform import rescale
from sklearn.preprocessing import normalize

def reconstruct_wsi(jpeg_path, tile_size):
    jpeg_tiles = []
    for tile in os.scandir(jpeg_path):
        jpeg_tiles.append(os.path.basename(tile))
    x_coord = []
    y_coord = []
    for t in jpeg_tiles:
        split_x = int((t.split('.jpeg')[0]).split('_')[0])
        x_coord.append(split_x)
        split_y = int((t.split('.jpeg')[0]).split('_')[1])
        y_coord.append(split_y)
    max_x = np.max(x_coord)
    max_y = np.max(y_coord)
    tile_size_x, tile_size_y = tile_size
    wsi = np.ones(((max_y + 1) * tile_size_y, (max_x + 1) * tile_size_x, 3), dtype=np.uint8) * 255
    for i in range(0, max_x + 1):
        for j in range(0, max_y + 1):
            tile_arr = np.asarray(Image.open((jpeg_path + '/' + f'{i}_{j}.jpeg')))
            if tile_arr.shape[0] == tile_size_x and tile_arr.shape[1] == tile_size_y:
                wsi[j * tile_arr.shape[0] : (j + 1) * tile_arr.shape[0], i * tile_arr.shape[1] : (i + 1) * tile_arr.shape[1], :] = tile_arr
            else:
                continue
    return wsi

def reconstruct_wsi_from_dataset(data_dicts, slide, slide_frame, tile_size, leiden, clusters):
    slide_rep = slide_frame[slide_frame['slides'] == slide]
    slide_clusters = slide_frame[slide_frame['slides'] == slide][f'leiden_{leiden}'].values.tolist()
    colors = sns.color_palette('tab20', len(clusters))
    tiles = []
    for i, row in slide_rep.iterrows():
        tiles.append((row['indexes'], row['original_set'], row['tiles']))
    x_coord = []
    y_coord = []
    for t in tiles:
        split_x = int((t[2].split('.jpeg')[0]).split('_')[0])
        x_coord.append(split_x)
        split_y = int((t[2].split('.jpeg')[0]).split('_')[1])
        y_coord.append(split_y)
    max_x = np.max(x_coord)
    max_y = np.max(y_coord)
    tile_size_x, tile_size_y = tile_size
    wsi = np.ones(((max_y + 1) * tile_size_y, (max_x + 1) * tile_size_x, 3), dtype=np.uint8) * 255
    wsi_c = np.ones(((max_y + 1) * tile_size_y, (max_x + 1) * tile_size_x, 3), dtype=np.uint8) * 255
    mask = np.ones((tile_size_x, tile_size_y))
    cluster_labs = []
    for index, original_set, tile in tiles:
        tile_arr = data_dicts[original_set][index]
        i, j = (tile.split('.jpeg')[0]).split('_')
        i = int(i)
        j = int(j)
        wsi[j * tile_arr.shape[0] : (j + 1) * tile_arr.shape[0], i * tile_arr.shape[1] : (i + 1) * tile_arr.shape[1], :] = tile_arr
        cluster = slide_rep[slide_rep['tiles'] == f'{i}_{j}.jpeg']
        color = colors[int(cluster[f'leiden_{leiden}'])]
        #cluster_lab_x = (j * tile_arr.shape[0] + (j + 1) * tile_arr.shape[0]) / 2 
        #cluster_lab_y = (i * tile_arr.shape[1] + (i + 1) * tile_arr.shape[1]) / 2 
        cluster_lab_x = (j + 1) * tile_arr.shape[0] - 112
        cluster_lab_y = (i) * tile_arr.shape[1] + 20
        cluster_lab_text = cluster[f'leiden_{leiden}'].values[0]
        cluster_labs.append([cluster_lab_x, cluster_lab_y, cluster_lab_text])
        wsi_c[j * tile_arr.shape[0] : (j + 1) * tile_arr.shape[0], i * tile_arr.shape[1] : (i + 1) * tile_arr.shape[1], :] = apply_mask(image=tile_arr.copy(), mask=mask, color=color, alpha=0.5)

    return wsi, wsi_c, slide_clusters, cluster_labs, colors

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def reconstruct_wsi_with_cluster_mask(jpeg_path, tile_size, frame, leiden=2.5):
    jpeg_tiles = []
    for tile in os.scandir(jpeg_path):
        jpeg_tiles.append(os.path.basename(tile))
    x_coord = []
    y_coord = []
    for t in jpeg_tiles:
        split_x = int((t.split('.jpeg')[0]).split('_')[0])
        x_coord.append(split_x)
        split_y = int((t.split('.jpeg')[0]).split('_')[1])
        y_coord.append(split_y)
    max_x = np.max(x_coord)
    max_y = np.max(y_coord)
    tile_size_x, tile_size_y = tile_size
    wsi_c = np.ones(((max_y + 1) * tile_size_y, (max_x + 1) * tile_size_x, 3), dtype=np.uint8) * 255
    mask = np.ones((tile_size_x, tile_size_y))
    for i in range(0, max_x + 1):
        for j in range(0, max_y + 1):
            tile_arr = np.asarray(Image.open((jpeg_path + '/' + f'{i}_{j}.jpeg')))
            if tile_arr.shape[0] == tile_size_x and tile_arr.shape[1] == tile_size_y:
                try:
                    cluster = frame[frame['tiles'] == f'{i}_{j}.jpeg']
                    color = colors[int(cluster[f'leiden_{leiden}'])]
                    wsi_c[j * tile_arr.shape[0] : (j + 1) * tile_arr.shape[0], i * tile_arr.shape[1] : (i + 1) * tile_arr.shape[1], :] = apply_mask(image=tile_arr.copy(), mask=mask, color=color, alpha=0.5)
                except KeyError:
                    color = [255,255,255]
                    wsi_c[j * tile_arr.shape[0] : (j + 1) * tile_arr.shape[0], i * tile_arr.shape[1] : (i + 1) * tile_arr.shape[1], :] = apply_mask(image=tile_arr.copy(), mask=mask, color=color, alpha=0.2)  
            else:
                continue
    return wsi_c

def hpc_annotations_legend(slide_df,  slide_clusters, colors, annotations_path, fontsize=30, markersize=15):
    if annotations_path is not None:
        annotations = pd.read_csv(annotations_path)
        annotations = annotations.rename(columns={'cluster':'HPC'})
        annotations = annotations.set_index('HPC')
        annotations.index = annotations.index.map(str)
    image_clusters, counts = np.unique(slide_clusters, return_counts=True)
    custom_lines = [Line2D([0], [0], color=colors[image_clusters[index]], lw=10) for index in np.argsort(-counts)]
    names_lines  = []
    for index in np.argsort(-counts):
        hpc = str(image_clusters[index])
        percentage = np.round(float(slide_df.loc[:, hpc]) * 100, 1)
        try:
            annotation = annotations.loc[hpc, 'phenotype']
            name = f'HPC {hpc} ({percentage}%): {annotation}'
            names_lines.append(name)
        except:
            name = f'HPC {hpc} ({percentage}%)'
            names_lines.append(name)
    return names_lines, custom_lines, image_clusters, counts

def plot_wsi_clusters(wsi, wsi_c, rescale_factor, slide, slide_clusters, wsi_composition, cluster_labs, colors, text_labs=True, legend=False, annotations_path=None, save_path=None, save=False):
    if wsi is None:
        wsi_d = rescale(wsi_c, rescale_factor)
        wsi_d = (wsi_d * 255).astype(np.uint8)
        fig = plt.figure(figsize=(wsi_d.shape[1] / 100 + 100, wsi_d.shape[0] / 100))
        ax  = fig.add_subplot(1, 1, 1)
        ax.set_title(slide, loc='center', fontdict={'size':30, 'weight':'bold'})
        ax.imshow(wsi_d/255.)
        ax.axis('off')
        if legend == False:
            if cluster_labs is not None:
                for x, y, l in cluster_labs:
                    ax.text(x=y * rescale_factor, y=x * rescale_factor, s=str(l), fontsize=22, fontweight='bold')
            if save:
                plt.gcf()
                filename = os.path.join(save_path, f'{slide}_clusters.png')
                plt.savefig(fname=filename, dpi='figure', format='png', bbox_inches='tight')
                plt.show()
        else:
            slide_df = wsi_composition[wsi_composition['slides'] == slide]
            names_lines, custom_lines, image_clusters, counts = hpc_annotations_legend(slide_df=slide_df, colors=colors, slide_clusters=slide_clusters, annotations_path=annotations_path, fontsize=30, markersize=15)
            legend = ax.legend(custom_lines, names_lines[:6], title='Histomorphological Phenotype Cluster (HPC)', frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.2),
                   prop={'weight':'bold', 'size':34}, title_fontproperties={'weight':'bold', 'size':40}, ncol=3)
            plt.subplots_adjust(left=0.6)
            plt.subplots_adjust(top=0.5)
            if cluster_labs is not None:
                for x, y, l in cluster_labs:
                    ax.text(x=y * rescale_factor, y=x * rescale_factor, s=str(l), fontsize=22, fontweight='bold')
            if save:
                plt.gcf()
                filename = os.path.join(save_path, f'{slide}_clusters.png')
                plt.savefig(fname=filename, dpi='figure', format='png', bbox_inches='tight')
            plt.show()
    if wsi is not None:
        wsi_d = rescale(wsi, rescale_factor)
        wsi_d = (wsi_d * 255).astype(np.uint8)
        fig = plt.figure(figsize=(wsi_d.shape[1] / 100 + 100, wsi_d.shape[0] / 100))
        ax  = fig.add_subplot(1, 1, 1)
        ax.set_title(slide, loc='center', fontdict={'size':30, 'weight':'bold'})
        ax.imshow(wsi_d/255.)
        ax.axis('off')
        if cluster_labs is not None:
            for x, y, l in cluster_labs:
                ax.text(x=y-112, y=x-112, s=str(l), fontsize=20, fontweight='bold')
        if save:
            plt.gcf()
            filename = os.path.join(save_path, f'{slide}_clusters.png')
            plt.savefig(fname=filename, dpi='figure', format='png', bbox_inches='tight')

def generate_wsi_overlay(data_dicts, slide, slide_frame,  clusters, rescale_factor, wsi_composition, legend, tile_size=(224,224), leiden=2.5, save_path=None, save=False, color=True):
    wsi, wsi_c, slide_clusters, cluster_labs, colors = reconstruct_wsi_from_dataset(data_dicts=data_dicts, slide=slide, slide_frame=slide_frame, tile_size=tile_size, leiden=leiden, clusters=clusters)
    if color:
        plot_wsi_clusters(wsi=None, wsi_c=wsi_c, rescale_factor=rescale_factor, slide=slide, slide_clusters=slide_clusters, wsi_composition=wsi_composition, colors=colors, cluster_labs=cluster_labs, legend=legend, save=save, save_path=save_path)
    else:
        plot_wsi_clusters(wsi=wsi, wsi_c=None, rescale_factor=rescale_factor, slide=slide, slide_clusters=slide_clusters, wsi_composition=wsi_composition, colors=colors, cluster_labs=cluster_labs, legend=legend, save=save, save_path=save_path)

