# Histomorphological Phenotype Learning - LATTICeA

---

**Abstract:**


---

## Citation
```

```

---

## Repository overview

This repository is a fork of the original Histomorphological Phenotype Learning (HPL) codebase, which can be found [here](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning) used to support the publication above. 
To build the environment and run the main code (feature extraction, mapping new data to existing clusters), please refer to the instructions in the original [readme](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/README.md). For clustering with rapids, and all analysis (survival, spatial biology etc.) please refer to [this file](./requirements_analysis.txt) for package version details.

---

## 1. WSI preprocessing
We preprocess whole slide images (WSIs) into small image tiles, of 224x224 pixels in diameter at a resolution of 1.8 microns per pixel (5x magnification) using [this script](./utilities/tiling/tile_wsi.py). A sample configuration file is provided [here](./utilities/tiling/configs/tile_wsi_config.json). This script will create a ```.pkl``` file for each WSI, containing the tile images and their metadata. 

They are then packaged into ```.h5``` files with [this script](./utilities/tiling/make_hdf5.py), with the following minimum dataset structure:  
- `img`: Tile images (as numpy arrays)  
- `tiles`: Tile coordinates/filenames  
- `slides`: Slide names  
- `samples`: Patient names   

Alternatively, you can use existing tiling code such as that used in the [DeepPath pipeline](https://github.com/ncoudray/DeepPATH) following the instructions in the [original readme](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/blob/master/README.md) or other code you are familiar with. The only requirement is that the resulting ```.h5``` file adheres to the above minimum structure. 

## 2. Workspace setup
The code relies on a specific directory structure and ```.h5``` file content to run the flow. The following sections detail the requirements for the workspace setup.

### Directory Structure
The code will make the following assumptions with respect to where the datasets, model training outputs (ie. model weights), and image representations are stored:  

- Datasets:  
    - Dataset folder follows the following structure:
        - datasets/**dataset_name**/**marker_name**/patches_h**tile_size**_w**tile_size** 
        - E.g.: `./datasets/LATTICeA_5x/he/patches_h224_w224`
- Data_model_output: \
    - Output folder for self-supervised trained models. 
    - Follows the following structure: \
        - data_model_output/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size** 
        - E.g.: `./data_model_output/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128` 
- Results: 
    - Output folder for self-supervised representations results. 
    - This folder will contain the representation and clustering data 
    - Follows the following structure: 
        - results/**model_name**/**dataset_name**/h**tile_size**_w**tile_size**_n3_zdim**latent_space_size** 
        - E.g.: `./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128` 

We use the following variable names to refer to the above directories: 
- **dataset_name**: `LATTICeA_5x` 
- **marker_name**: `he` 
- **tile_size**: `224` 

## 3. Feature extraction
This step extracts features from tiles using the self-supervised model. 

```
python ./run_representationspathology_projection.py 
--dataset LATTICeA_5x 
--checkpoint ./data_model_output/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128/checkpoints/BarlowTwins_3.ckt
--model BarlowTwins_3 
--real_hdf5 ./datasets/dataset_name/hdf5_dataset_name_he.h5
```

## 4. Background and artefact removal
We map an initial set of clusters to the self-supervised representations, which have been previously annotated with as to whether they contain background regions or artefacts such as areas of blurring, air bubbles or out-of-focus regions. 

The steps to do this are as follows: 
1. Download the cluster configuration  
2. Use [this notebook](./utilities/tile_cleaning/process_external_dataset_review_clusters.ipynb) to generate ```.pkl``` files containing the file indexes (in the original ```.h5``` file) that are to be removed. 
3. Remove these tiles from the ```.h5``` file: 

```
python3 ./utilities/tile_cleaning/remove_indexes_h5.py 
--pickle_file ./utilities/files/indexes_to_remove/your_dataset/complete.pkl 
--h5_file ./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128/hdf5_dataset_name_he.h5 
```

## 5. Set up directory with filtered representations
1. Create the directory ```./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered```
2. To this newly created directory, copy over the resulting ```.h5``` file produced by the previous step (found at ```./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128/hdf5_dataset_name_he_filtered.h5```)
3. Download the cluster configuration file from ...
4. Copy the configuration file to ```./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/adatas/```

## 6. Assigning clusters to the filtered representations
```
python ./run_representationsleiden_assignment.py 
--meta_field rapids_2p5m 
--resolution 2.5
--folds_pickle ./utilities/fold_creation/lattice_5x_folds.pkl 
--h5_complete_path ./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/hdf5_LATTICeA_5x_he_complete_filtered.h5 
--h5_additional_path ./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/hdf5_dataset_name_he_filtered.h5 
```
Note: You will see warnings for folds 0, 1, 3 and 4, which is expected. 

At this point, the result is a ```.csv``` file containing the cluster assignations for each tile, found at: \
```./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/adatas/dataset_name_he_filtered_leiden_2p5__fold2.csv```

From here, you can proceed to using these as a substrate for further analysis, such as survival analysis or integration with other data modalities. 

If you wish to visualise the tiles in each cluster from your data, you can use the following script: 

```
python ./utilities/visualizations/cluster_images.py
--dataset_path ./datasets/dataset_name/he/patches_h224_w224/hdf5_dataset_name_he_train.h5
--h5_complete_path ./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/hdf5_test_file_he_filtered.h5
--resolution 2.5
--meta_field rapids_2p5m
--num_batches 1 # this is the number of sets of 100 tiles to plot for each cluster
```

The image files will be found at `./results/BarlowTwins_3/LATTICeA_5x/h224_w224_n3_zdim128_filtered/rapids_2p5m/adatas/images_leiden_2p5` along with a `.csv` file detailing which tiles were used to create each graphic. 
