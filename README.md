# Signature Clustering

## Disclaimer
This program is in very early development and has only been tested on macOS.

## Initial Setup
It is highly recommend that you install all of the necessary dependencies in a conda virtual environment.
* Download Anaconda or Miniconda [here](https://www.anaconda.com/download/success)

Create a conda virtual environment with the necessary dependencies:
```
conda create -n signature_clustering python=3.11.7 pandas=2.1.0 matplotlib=3.7.1 scikit-learn=1.2.2 scipy=1.10.1 scikit-image=0.21.0 opencv=4.8.1 tqdm=4.65.0 hyperopt=0.2.7 rapidfuzz=3.6.2 hnswlib=0.8.0 -c conda-forge
```
Enter the conda virtual environment:
```
conda activate signature_clustering
```

### To run the clustering interface:
```
python path_to_repository/guided_clustering_interface.py
```
#### Screenshots of guided_clustering_interface.py
##### Discovery mode
![extra_files/discovery_mode_screenshot.png](https://raw.githubusercontent.com/benfertig/signature-clustering/refs/heads/main/extra_files/discovery_mode_screenshot.png?token=GHSAT0AAAAAADBYVPOWOW5BXWBHXKPSIOUEZ7TC4SA)
##### Completion mode
![extra_files/completion_mode_screenshot.png](https://raw.githubusercontent.com/benfertig/signature-clustering/refs/heads/main/extra_files/completion_mode_screenshot.png?token=GHSAT0AAAAAADBYVPOXVDDRUONR4KEWY6YMZ7TC7OQ)
##### Verification mode
![extra_files/verification_mode_screenshot.png](https://raw.githubusercontent.com/benfertig/signature-clustering/refs/heads/main/extra_files/verification_mode_screenshot.png?token=GHSAT0AAAAAADBYVPOXPRRNEGN3TFEYE6OMZ7TDAOA)
### To run the automated clustering script
```
python path_to_repository/signature_clustering.py
```
Parameters concerning the behavior of the clustering algorithm can be found in the configs (**default_config** and **test_configs**) near the top of **signature_clustering.py**.

Please note that the clustering algorithm will run for as many times as there are entries in **test_configs** (where each parameter dictionary corresponds to a single entry). If you only want the clustering to run once, then you should only have one dictionary in **test_configs**.

For each test_config, all set values will **override** the values in **default_config** (meaning that **default_config** is only there so that the user doesn't need to reenter values which are already the same for each of their configs in **test_configs**).

The main parameters you are going to need to edit are:
* **'SIGNATURES_DIR'**: The directory containing the signatures to be clustered
* **'CLUSTER_DIRECTORY_DEPTH'**: Set this to 0 if you want to cluster the entire dataset amongst itself. Otherwise, set this value to the directory depth you want the program to descend down from 'SIGNATURES_DIR' before treating each subdirectory at that depth as its own cluster 'pool' (which cannot be clustered with any of the images from any other pool).
You can also run the optimization portion of the automated clustering script on a directory of preclustered signatures, where each subdirectory corresponds to a cluster:
```
python path_to_repository/signature_clustering.py optimize /path/to/preclustered/dataset num_iterations
```
...where ```num_iterations``` is the number of rounds for the **hyperopt** optimization algorithm. You should set this to as high of a value as you have time for. The amount of time that the optimization algorithm will take will depend on:
* The speed of your processor
* The characteristics of your dataset
* The number of iterations your optimization algorithm runs for

The optimization algorithm saves its progress as it runs, meaning that, if the optimization gets interrupted, you can continue running it where it left off by executing the optimiztion command again.

When continuing an interrupted optimization, you should set the ```num_iterations``` argument in the ```optimize``` command to the number of iterations you want the optimization algorithm to run **from when you most recently ran the ```optimize``` command**, so you should subtract the number of iterations it already ran for from the number of total iterations you want.

If you don't know how many rounds you should run the optimization algorithm for, 4000 would probably be good enough in most cases.

Here is the top-ranked configuration from 1334 rounds of the optimization algorithm that ran on the "train" subdirectory from the [**Signature_Verification_Dataset**](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset):
```
{
    'name': 'Rank_1_Score_0.6909',
    'CLUSTER_SPLIT_PERCENTILE': 91,
    'DISTANCE_METRIC': 'correlation',
    'DISTANCE_THRESHOLD': 0.910171,
    'ENSEMBLE_METHODS': '['hierarchical', 'spectral']',
    'ENSEMBLE_WEIGHTS': '[0.36337456821539843, 0.6366254317846016]',
    'HIERARCHICAL_WEIGHT_RATIO': 0.363375,
    'HOG_WEIGHT': 0.859632,
    'HU_WEIGHT': 0.013746,
    'IMAGE_SIZE': '[320, 160]',
    'LBP_WEIGHT': 7.8551,
    'LINKAGE_METHOD': 'complete',
    'MERGE_METHOD': 'average',
    'MERGE_THRESHOLD': 0.677289,
    'MIN_CLUSTER_SIZE': 17,
    'NORMALIZE_FEATURES': True,
    'NORMALIZE_METHOD': 'standard',
    'SPECTRAL_AFFINITY': 'rbf',
    'SPECTRAL_N_CLUSTERS': 20,
    'USE_ADAPTIVE_THRESHOLD': False,
    'USE_ENHANCED_LBP': True,
    'USE_ENSEMBLE': True,
    'USE_GABOR': False,
    'USE_PCA_HOG': False,
    'USE_TWO_STAGE': True,
    'USE_ZERNIKE': False,
}
```
If you want to use this configuration (for both **signature_clustering.py** *and* **guided_clustering_interface.py**), then you should paste it as the *first* value of **test_configs** in **signature_clustering.py** (I currently do not have this configuration pasted in my version because the signature dataset that I am working with has different charcteristics, and so I trained the optimization algorithm on a preclustered subset of that dataset).

## Attribution
* ***Python* packages**
  * **pandas** ([BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/main/LICENSE))
  * **matplotlib** ([Matplotlib License](https://matplotlib.org/stable/users/project/license.html), BSD-compatible)
  * **scikit-learn** ([BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING))
  * **scipy** ([BSD 3-Clause License](https://github.com/scipy/scipy/blob/main/LICENSE.txt))
  * **scikit-image** ([BSD 3-Clause License](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt))
  * **opencv** ([Apache License 2.0](https://github.com/opencv/opencv/blob/master/LICENSE))
  * **tqdm** ([MIT/MPL-2.0 dual License](https://github.com/tqdm/tqdm/blob/master/LICENCE))
  * **hyperopt** ([License](https://github.com/hyperopt/hyperopt/blob/master/LICENSE.txt))
  * **rapidfuzz** ([MIT License](https://github.com/rapidfuzz/rapidfuzz/blob/main/LICENSE))
  * **hnswlib** ([Apache License 2.0](https://github.com/nmslib/hnswlib/blob/master/LICENSE))
* **Signature datasets**
  * **Signature_Verification_Dataset** ([CC0: Public Domain](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset))
* Values for **confusion_matrix** in the **_get_confusion_weights** method in **guided_clustering_interface.py** derived from research by the **Center of Excellence for Document Analysis and Recognition (CEDAR)** at **SUNY Buffalo** on handwriting character confusion patterns.
    * For more information: https://www.cedar.buffalo.edu/
