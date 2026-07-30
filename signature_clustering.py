# -*- coding: utf-8 -*-

"""
Advanced Signature Clustering Implementation (Hierarchical Version)

This script provides a comprehensive framework for clustering signature images, with flexible
directory handling, multiple feature extraction methods, and support for both testing and
production environments.

SETUP INSTRUCTIONS:
------------------------------------------------------------------------
conda create -n signature_clustering python=3.11.7 pandas=2.1.0 \
    matplotlib=3.7.1 scikit-learn=1.2.2 scipy=1.10.1 scikit-image=0.21.0 \
        opencv=4.8.1 tqdm=4.65.0 hyperopt=0.2.7 rapidfuzz=3.6.2 hnswlib=0.8.0 -c conda-forge
conda activate signature_clustering

You will need to edit the parameters in default_config to point to your directory of
signatures (SIGNATURES_DIR). Edit the other parameters to your liking as well, but keep in
mind that any of the parameters in default_config will be overridden by parameters set in
the individual configurations in test_configs (if they are provided).

Each configuration in test_config will run a separate instance of the clustering, so if you
only want to run the clustering on your images once, then only provide one configuration here.

The single provided configuration in test_configs represents a configuration
that was tuned after running the parameter optimization algorithm on the
"train" subdirectory of the Signature_Verification_Dataset from Kaggle
(https://www.kaggle.com/datasets/robinreni/signature-verification-dataset),
with _forg subdirectories excluded.

Then, to run the clustering simply run python signature_clustering.py.

SEMI-SUPERVISED MODE
------------------------------------------------------------------------

This script also includes a semi-supervised clustering implementation. This allows the program
to recluster a set of images that it has already clustered once, after a human has examined the
clusters and fine-tuned them.

The best part is that the human's revisions do not need to be perfect. They simply need to do
as many tweaks as they feel like in order to give the program a better understanding of what
constitutes clusters for their particular dataset.

The human should examine the program's output directory of clustered images and either:
    1. Remove images from clusters to which they clearly do not belong by placing the images
       in the next highest directory (i.e., the directory containing all of the cluster folders).
    2. Remove images from clusters to which they clearly do not belong AND THEN place those images
       into new cluster folders along with any other images which clearly belong in that cluster.

The human can do a mixture of 1 and 2 from above. Doing option 2 as much as possible obviously
helps the clustering algorithm better fine-tune its parameters for future rounds of clustering.

Do not worry about the name of the folder in which you place images. Leaving empty cluster folders
is fine too. As long as the images are in the same subdirectory, the program will know to interpret
them as a cluster.

Then, when you are ready to run the semi-supervised clustering algorithm, change these
parameters for your clustering configuration below in the following ways:
    1. Set SEMI_SUPERVISED_MODE to True.
    2. Set HUMAN_FEEDBACK_DIR to the directory of human-tweaked cluster folders.
    3. Set CONSTRAINT_WEIGHT to a value between 0 and 1 to configure how strongly you would like the
       program to factor in the human's modifications to the program's previous round of clustering.
            - A value of 1 means that the program will treat each cluster in HUMAN_FEEDBACK_DIR as
              pure, completely forbidding separation of images within that cluster.
            - A value of 1 also means that any images which the program had initially clustered
              together which the human then separated will never be clustered together again by the
              program.
            - In summary, only set CONSTRAINT_WEIGHT to 1 if you are completely confident that all
              of the human's changes to the program's clustering were strict improvements. Reduce
              this value proportional to your confidence level.

ADDITIONAL INSTRUCTIONS FOR PARAMETER OPTIMIZATION ALGORITHM
------------------------------------------------------------------------
If you are attempting to run the parameter optimization algorithm on a preclustered signature
dataset, enter the following command, replacing "/path/to/preclustered/dataset" with the
location of your preclustered dataset (where each subfolder represents its own cluster):

python signature_clustering.py optimize /path/to/preclustered/dataset num_iterations

...where num_iterations is the number of configurations on which to test the
optimization (200, 1000, etc.) (more is obvioulsy better, but will take longer).

When the optimization is finished, information about all of the tested
configurations will be saved in results/optimize/optimization_results_{YYMMDD}_{HHMMSS}.
The configurations will be ranked, best to worst.
"""

import os
import pickle
import time
import json
import shutil
import re
from copy import deepcopy
from datetime import datetime
from collections import Counter, defaultdict
import random
import hashlib
import gc
import sys
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform, euclidean
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, leaves_list
from skimage.feature import local_binary_pattern
from tqdm import tqdm

#=============================================================================
# CONFIGURATION SECTION
#=============================================================================

# Path and file settings
default_config = {

    #================================================================================
    # FILE HANDLING SETTINGS
    # (THESE ARE JUST DEFAULTS, THEY CAN BE OVERRIDEN BY VALUES SET IN TEST CONFIGS)
    #================================================================================

    'SIGNATURES_DIR': "/Users/benjamin/Documents/GitHub/signature-clustering/sources/images_by_state_county/Oklahoma/Craig",
    'CLUSTER_DIRECTORY_DEPTH': 0,            # How many levels to descend before clustering
                                             # Use 'max' to go to leaf directories

    'SEMI_SUPERVISED_MODE': False,  # Set to True for second pass after human feedback
    'HUMAN_FEEDBACK_DIR': "/Users/benjamin/Documents/GitHub/signature-clustering/results/unsupervised/clustering_results_20250304_235712/configuration_summaries/Rank_1_Score_0.6909",
    'CONSTRAINT_WEIGHT': 1.0,       # How strongly to enforce human constraints (min: 0, max: 1)

    'TESTING_ON_PRECLUSTERED_IMAGES': False,  # Set to False for unclustered data

    'COPY_IMAGES_TO_CLUSTERS': True,       # Whether to copy images to cluster directories
    'DATA_LIMIT': None,                    # Limit signatures for testing

    'VALID_FILE_ENDINGS': [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"],
    'FEEDBACK_FILE': "feedback_for_llm.txt",  # Compact file for LLM analysis
    'UNSUPERVISED_RESULTS_DIR': "results/unsupervised",  # Base directory for unsupervised clustering results
    'SEMI_SUPERVISED_RESULTS_DIR': "results/semi_supervised",      # Base directory for semi-supervised clustering results
    'OPTIMIZATION_RESULTS_DIR': "results/optimize",      # Base directory for parameter optimization results
    'METADATA_DIR': "results/metadata",                  # Base directory for metadata storage

    'METADATA_FILE': None,  # If None, will be auto-generated based on SIGNATURES_DIR

    'FEEDBACK_SEARCH_PATTERN': "clustered_signatures_iteration_*",  # Pattern to find feedback directories

    #================================================================================
    # IMAGE PREPROCESSING SETTINGS
    # (THESE ARE JUST DEFAULTS, THEY CAN BE OVERRIDEN BY VALUES SET IN TEST CONFIGS)
    #================================================================================

    # Enhanced Preprocessing Options
    'DYNAMIC_PREPROCESSING': True,          # Dynamically adjust preprocessing based on image properties
    'PREPROCESSING_METHOD': 'adaptive',     # 'otsu' or 'adaptive'
    'USE_CLAHE': True,                     # Apply contrast enhancement 
    'CLAHE_CLIP_LIMIT': 2.0,               # Clip limit for CLAHE (1.0-4.0)
    'ADAPTIVE_THRESHOLD_C': 2,             # Constant subtracted from mean for adaptive threshold
    'ADAPTIVE_BLOCK_SIZE': 11,             # Block size for adaptive threshold (must be odd)
    'MORPHOLOGICAL_OP': 'open',            # 'none', 'open', 'close', 'open_close', 'dilate'
    'MORPHOLOGICAL_KERNEL_SIZE': 2,        # Size of morphological kernel (must be odd)

    'IMAGE_SIZE': (224, 224),              # Size to resize signatures

    #================================================================================
    # FEATURE EXTRACTION SETTINGS
    # (THESE ARE JUST DEFAULTS, THEY CAN BE OVERRIDEN BY VALUES SET IN TEST CONFIGS)
    #================================================================================

    # Feature weights
    'HU_WEIGHT': 5.52,                  # Weight for Hu moments
    'LBP_WEIGHT': 5.52,                 # Weight for LBP features
    'HOG_WEIGHT': 0.505,                # Weight for HOG features
    'ZERNIKE_WEIGHT': 1.48,             # Weight for Zernike moments (0 to disable)
    'GABOR_WEIGHT': 0.0,                # Weight for Gabor filter features (0 to disable)
    'STROKE_FEATURE_WEIGHT': 0.0,       # Weight for stroke-based features (0 to disable)

    # Feature options
    'USE_ENHANCED_LBP': True,          # Use multi-scale LBP
    'USE_ZERNIKE': False,              # Use Zernike moments
    'USE_GABOR': False,                # Use Gabor filter features
    'USE_STROKE_FEATURES': False,      # Use stroke-based features
    'USE_PCA_HOG': False,              # Apply PCA to reduce HOG dimensions
    'PCA_HOG_COMPONENTS': 100,         # Number of PCA components for HOG

    # Normalization options
    'NORMALIZE_FEATURES': True,         # Apply feature normalization
    'NORMALIZE_METHOD': 'standard',     # 'standard', 'l1', 'l2', or 'robust'

    #================================================================================
    # CLUSTERING SETTINGS
    # (THESE ARE JUST DEFAULTS, THEY CAN BE OVERRIDEN BY VALUES SET IN TEST CONFIGS)
    #================================================================================

    # Distance and clustering settings
    'DISTANCE_METRIC': 'correlation',   # 'correlation', 'cosine', 'euclidean', 'cityblock'
    'DISTANCE_THRESHOLD': 0.436,        # Primary threshold for clustering
    'LINKAGE_METHOD': 'average',        # 'single', 'complete', 'average', 'ward'

    # Singleton handling (for datasets with many single-instance clusters)
    'SINGLETON_HANDLING': True,         # Enable special handling for singletons
    'SINGLETON_DETECTION_THRESHOLD': 0.6, # Threshold for detecting singletons
    'EXPECTED_SINGLETON_RATIO': 0.5,    # Expected proportion of singletons in dataset

    # Multi-stage clustering settings
    'USE_TWO_STAGE': True,              # Use two-stage clustering
    'MERGE_THRESHOLD': 0.636,           # Threshold for merging similar clusters
    'MERGE_METHOD': 'average',          # 'average', 'min', 'max', 'adaptive'
    'MIN_CLUSTER_SIZE': 4,              # Min size to consider for splitting
    'CLUSTER_SPLIT_PERCENTILE': 91,     # Percentile for internal cluster split threshold
    'USE_ADAPTIVE_THRESHOLD': False,    # Use adaptive thresholding based on data distribution
    'ADAPTIVE_PERCENTILE': 25,          # Percentile for adaptive threshold

    # Ensemble clustering settings
    'USE_ENSEMBLE': False,              # Use ensemble of multiple clustering methods
    'ENSEMBLE_METHODS': ['hierarchical', 'spectral'],  # Methods to include in ensemble
    'ENSEMBLE_WEIGHTS': [0.7, 0.3],     # Weights for ensemble methods
    'HIERARCHICAL_WEIGHT_RATIO': 0.7,   # Ratio for hierarchical vs. spectral (if USE_ENSEMBLE)

    # Spectral clustering settings (for ensemble or large cluster splitting)
    'SPECTRAL_N_CLUSTERS': 'auto',      # 'auto' or integer number of clusters
    'SPECTRAL_AFFINITY': 'rbf',         # 'rbf', 'nearest_neighbors', or 'precomputed'

    #================================================================================
    # SEMI-SUPERVISED CLUSTERING SETTINGS
    # (THESE ARE JUST DEFAULTS, THEY CAN BE OVERRIDEN BY VALUES SET IN TEST CONFIGS)
    #================================================================================

    # Semi-supervised refinement controls
    'SKIP_WEIGHT_LEARNING': False,      # Set to True to keep original weights without analysis
    'DISABLE_WEIGHT_LEARNING': False,   # More forceful - prevents any weight adjustments 
    'DISABLE_CLUSTER_REORDERING': False,# Set to True to skip cluster reordering
    'CONSERVATIVE_ADJUSTMENT': True,    # Use very conservative weight adjustments (30% max)
    'MAX_ADJUSTMENT_PERCENTAGE': 30,    # Maximum percentage to adjust weights by (0-100)

    #================================================================================
    # OUTPUT SETTINGS
    # (THESE ARE JUST DEFAULTS, THEY CAN BE OVERRIDEN BY VALUES SET IN TEST CONFIGS)
    #================================================================================

    # Text output settings
    'SAVE_RESULTS': False,              # Save results to disk
    'DETAILED_ANALYSIS': True,          # Run detailed cluster analysis
    'SAVE_COMPACT_FEEDBACK': True,      # Save compact feedback for LLM analysis
    'MIN_CLUSTER_SIZE_TO_PRINT': 1,     # Min cluster size to print to terminal

    # Visual output settings
    'MAX_CLUSTERS_TO_VISUALIZE': 10,    # Max number of clusters to visualize
    'MIN_CLUSTER_SIZE_TO_VIS': 2,       # Min cluster size to visualize
    'SAVE_VISUALIZATIONS': False,       # Save cluster visualizations
    'VISUALIZE_PREPROCESSING': False,    # Visualize preprocessed images
    'MAX_PREPROCESSED_SIGS_TO_VIS': 30  # Maximum number of preprocessed images to visualize

}

test_configs = [
    # Rank 1: Score=0.6904, Accuracy=85.98%, Purity=92.80%, Fragmentation=1.57, Singletons=190
    {
        'name': 'Rank_1_Score_0.6904',
        'ADAPTIVE_BLOCK_SIZE': 24,
        'ADAPTIVE_THRESHOLD_C': 10.1233,
        'DISTANCE_METRIC': 'cosine',
        'DISTANCE_THRESHOLD': 0.735693,
        'DYNAMIC_PREPROCESSING': False,
        'ENSEMBLE_METHODS': ['hierarchical', 'spectral'],
        'ENSEMBLE_WEIGHTS': [0.22001424584386844, 0.7799857541561316],
        'EXPECTED_SINGLETON_RATIO': 0.101623,
        'HIERARCHICAL_WEIGHT_RATIO': 0.220014,
        'HOG_WEIGHT': 0.295751,
        'HU_WEIGHT': 0.001623,
        'IMAGE_SIZE': [256, 96],
        'LBP_WEIGHT': 0.342197,
        'LINKAGE_METHOD': 'average',
        'MORPHOLOGICAL_KERNEL_SIZE': 2,
        'MORPHOLOGICAL_OP': 'none',
        'NORMALIZE_FEATURES': False,
        'NORMALIZE_METHOD': 'l2',
        'PCA_HOG_COMPONENTS': 328,
        'PREPROCESSING_METHOD': 'otsu',
        'SINGLETON_DETECTION_THRESHOLD': 0.701436,
        'SINGLETON_HANDLING': True,
        'SPECTRAL_AFFINITY': 'precomputed',
        'SPECTRAL_N_CLUSTERS': 20,
        'USE_ADAPTIVE_THRESHOLD': False,
        'USE_CLAHE': False,
        'USE_ENHANCED_LBP': False,
        'USE_ENSEMBLE': False,
        'USE_GABOR': False,
        'USE_PCA_HOG': True,
        'USE_STROKE_FEATURES': False,
        'USE_TWO_STAGE': False,
        'USE_ZERNIKE': False,
    }
]

#=============================================================================
# FEATURE EXTRACTION
#=============================================================================

class SignatureFeatureExtractor:
    """Feature extraction class for signature images with enhanced preprocessing."""

    def __init__(self, config):
        """Initialize the feature extractor with configuration parameters."""
        self.config = config

        # Core parameters from config
        self.size = config['IMAGE_SIZE']
        self.hu_weight = config['HU_WEIGHT']
        self.lbp_weight = config['LBP_WEIGHT']
        self.hog_weight = config['HOG_WEIGHT']
        self.zernike_weight = config['ZERNIKE_WEIGHT']
        self.gabor_weight = config.get('GABOR_WEIGHT', 0.0)
        self.stroke_weight = config.get('STROKE_FEATURE_WEIGHT', 0.0)

        # Feature extraction flags
        self.use_enhanced_lbp = config['USE_ENHANCED_LBP']
        self.use_zernike = config['USE_ZERNIKE']
        self.use_gabor = config.get('USE_GABOR', False)
        self.use_stroke_features = config.get('USE_STROKE_FEATURES', False)
        self.use_pca_hog = config.get('USE_PCA_HOG', False)
        self.pca_hog_components = config.get('PCA_HOG_COMPONENTS', 100)

        # Normalization parameters
        self.normalize_features = config.get('NORMALIZE_FEATURES', True)
        self.normalize_method = config.get('NORMALIZE_METHOD', 'standard')

        # Enhanced preprocessing parameters
        self.use_clahe = config.get('USE_CLAHE', False)
        self.clahe_clip_limit = config.get('CLAHE_CLIP_LIMIT', 2.0)
        self.dynamic_preprocessing = config.get('DYNAMIC_PREPROCESSING', False)
        self.preprocessing_method = config.get('PREPROCESSING_METHOD', 'adaptive')
        self.adaptive_threshold_c = config.get('ADAPTIVE_THRESHOLD_C', 2)
        self.adaptive_block_size = config.get('ADAPTIVE_BLOCK_SIZE', 11)
        self.morphological_op = config.get('MORPHOLOGICAL_OP', 'open')
        self.morphological_kernel_size = config.get('MORPHOLOGICAL_KERNEL_SIZE', 2)

        # Initialize PCA if needed
        self.pca_hog = None

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self):
        """Validate and adjust parameters to ensure compatibility."""
        # Ensure morphological kernel size is odd
        if self.morphological_kernel_size % 2 == 0:
            self.morphological_kernel_size += 1
            print(f"Adjusted morphological kernel size to {self.morphological_kernel_size} (must be odd)")

        # Ensure clahe parameters are valid
        if self.use_clahe and (self.clahe_clip_limit < 0.1 or self.clahe_clip_limit > 10.0):
            self.clahe_clip_limit = min(max(0.1, self.clahe_clip_limit), 10.0)
            print(f"Adjusted CLAHE clip limit to {self.clahe_clip_limit} (valid range: 0.1-10.0)")

        # Validate adaptive threshold parameters
        if self.adaptive_block_size % 2 == 0:
            self.adaptive_block_size += 1  # Must be odd
            print(f"Adjusted adaptive block size to {self.adaptive_block_size} (must be odd)")

        # Check image size is valid for HOG
        w, h = self.size
        cell_size = 8  # Standard HOG cell size
        if w % cell_size != 0 or h % cell_size != 0:
            print(f"Warning: Image size {self.size} not divisible by HOG cell size {cell_size}")
            print("This may cause issues with HOG feature extraction")

    def get_signature_bounds(self, img_path=None, img=None):
        """
        Extract only the bounding box of the signature.
        Can work with either an image path or already loaded image.
        
        Args:
            img_path: Path to the signature image (optional if img provided)
            img: Already loaded grayscale image (optional if img_path provided)
            
        Returns:
            Tuple of (x, y, width, height) or None if detection fails
        """
        try:
            # Handle input - either load from path or use provided image
            if img is None and img_path is not None:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Failed to load image: {img_path}")
                    return None
            elif img is None:
                print("Error: either img_path or img must be provided")
                return None

            # Create a copy to work with
            orig_img = img.copy()

            # Apply CLAHE if enabled
            if self.use_clahe:
                try:
                    clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
                    img = clahe.apply(img)
                except Exception as e:
                    print(f"CLAHE error: {e}, using original image")
                    img = orig_img.copy()

            # Analyze image characteristics to guide preprocessing
            if self.dynamic_preprocessing:
                # Calculate image statistics
                img_mean = np.mean(img)
                img_std = np.std(img)

                # Get thresholds from config
                faded_mean_threshold = self.config.get('FADED_MEAN_THRESHOLD', 200)
                faded_std_threshold = self.config.get('FADED_STD_THRESHOLD', 40)
                low_contrast_std_threshold = self.config.get('LOW_CONTRAST_STD_THRESHOLD', 30)

                # Detect if image is low contrast
                low_contrast = img_std < low_contrast_std_threshold

                # Detect if image is faded (high mean, low std)
                faded = img_mean > faded_mean_threshold and img_std < faded_std_threshold

                # Choose preprocessing approach based on image characteristics
                if faded:
                    preprocessing_method = 'adaptive'
                    adaptive_threshold_c = max(1, self.adaptive_threshold_c - 1)  # More aggressive
                elif low_contrast:
                    preprocessing_method = 'adaptive'
                    adaptive_threshold_c = self.adaptive_threshold_c
                else:
                    preprocessing_method = self.preprocessing_method
                    adaptive_threshold_c = self.adaptive_threshold_c
            else:
                preprocessing_method = self.preprocessing_method
                adaptive_threshold_c = self.adaptive_threshold_c

            # First threshold for finding bounds only
            if preprocessing_method == 'adaptive':
                binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, self.adaptive_block_size, adaptive_threshold_c)
            else:
                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find the bounding box of the signature
            coords = cv2.findNonZero(binary)
            if coords is None or len(coords) == 0:  # Empty or almost empty image
                return None

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(coords)

            return (x, y, w, h)
        except Exception as e:
            print(f"Error extracting signature bounds: {e}")
            return None

    def preprocess(self, img):
        """Enhanced preprocessing with adaptive techniques for historical signatures."""
        if img is None:
            raise ValueError("Failed to load image")

        # Create a copy to work with
        orig_img = img.copy()

        # Apply CLAHE if enabled (Contrast Limited Adaptive Histogram Equalization)
        if self.use_clahe:
            try:
                clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
                img = clahe.apply(img)
            except Exception as e:
                print(f"CLAHE error: {e}, using original image")
                img = orig_img.copy()

        # Use the get_signature_bounds method to detect signature bounds
        bounds = self.get_signature_bounds(img=img)

        if bounds is None:  # Empty or almost empty image
            # Return a blank image of the standard size
            return np.zeros(self.size[::-1], dtype=np.uint8)

        # Unpack bounds
        x, y, w, h = bounds

        # Extract the signature from the image
        cropped_img = img[y:y+h, x:x+w]

        # The rest of the preprocessing...
        # Determine which preprocessing method to use based on config
        if self.dynamic_preprocessing:
            # Calculate image statistics
            img_mean = np.mean(img)
            img_std = np.std(img)

            # Get thresholds from config
            faded_mean_threshold = self.config.get('FADED_MEAN_THRESHOLD', 200)
            faded_std_threshold = self.config.get('FADED_STD_THRESHOLD', 40)
            low_contrast_std_threshold = self.config.get('LOW_CONTRAST_STD_THRESHOLD', 30)

            # Detect if image is low contrast
            low_contrast = img_std < low_contrast_std_threshold

            # Detect if image is faded (high mean, low std)
            faded = img_mean > faded_mean_threshold and img_std < faded_std_threshold

            # Choose preprocessing approach based on image characteristics
            if faded:
                preprocessing_method = 'adaptive'
                adaptive_threshold_c = max(1, self.adaptive_threshold_c - 1)  # More aggressive
            elif low_contrast:
                preprocessing_method = 'adaptive'
                adaptive_threshold_c = self.adaptive_threshold_c
            else:
                preprocessing_method = self.preprocessing_method
                adaptive_threshold_c = self.adaptive_threshold_c
        else:
            preprocessing_method = self.preprocessing_method
            adaptive_threshold_c = self.adaptive_threshold_c

        # Apply final thresholding on the cropped image
        if preprocessing_method == 'adaptive':
            binary_result = cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, self.adaptive_block_size, adaptive_threshold_c)
        else:
            _, binary_result = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply morphological operations if specified
        if self.morphological_op != 'none':
            kernel = np.ones((self.morphological_kernel_size, self.morphological_kernel_size), np.uint8)

            if self.morphological_op == 'open':
                binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_OPEN, kernel)
            elif self.morphological_op == 'close':
                binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)
            elif self.morphological_op == 'open_close':
                binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_OPEN, kernel)
                binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_CLOSE, kernel)
            elif self.morphological_op == 'dilate':
                binary_result = cv2.dilate(binary_result, kernel, iterations=1)

        # Calculate target dimensions while preserving aspect ratio
        target_width, target_height = self.size

        # Calculate scaling factor
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize the image preserving aspect ratio
        try:
            resized = cv2.resize(binary_result, (new_w, new_h))
        except Exception as e:
            print(f"Resize error: {e}, using original binary")
            # Try a different approach if resize fails
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            resized = cv2.resize(binary, (new_w, new_h))

        # Create a blank canvas
        result = np.zeros((target_height, target_width), dtype=np.uint8)

        # Calculate position to center the signature
        x_offset = (target_width - new_w) // 2
        y_offset = (target_height - new_h) // 2

        # Place the signature on the canvas
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return result

    def visualize_preprocessing_comparison(self, image_paths, output_dir=None):
        """
        Enhanced visualization of preprocessing comparison for multiple signatures.
        Shows original, CLAHE-enhanced, and final preprocessed versions.
        
        Args:
            image_paths (list): List of paths to signature images
            output_dir (str): Directory to save comparisons
        """

        print(f"\nGenerating preprocessing visualizations for {len(image_paths)} images...")

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")

        # Process each image
        for i, img_path in enumerate(image_paths):
            print(f"Processing visualization {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")

            try:
                # Load image
                original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if original is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                # Apply contrast enhancement if enabled
                if self.use_clahe:
                    try:
                        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8))
                        enhanced = clahe.apply(original.copy())
                    except Exception as e:
                        print(f"CLAHE error: {e}, skipping enhancement")
                        enhanced = original.copy()
                else:
                    enhanced = original.copy()

                # Apply full preprocessing
                preprocessed = self.preprocess(original.copy())

                # Create figure with three panels
                plt.figure(figsize=(15, 5))

                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(original, cmap='gray')
                plt.title(f"Original")
                plt.axis('off')

                # Enhanced image (after CLAHE if used)
                plt.subplot(1, 3, 2)
                plt.imshow(enhanced, cmap='gray')
                plt.title("Contrast Enhanced" if self.use_clahe else "Original")
                plt.axis('off')

                # Final preprocessed image
                plt.subplot(1, 3, 3)
                plt.imshow(preprocessed, cmap='gray')
                method_name = self.preprocessing_method.capitalize()
                ops_name = self.morphological_op.replace('_', '+').capitalize() if self.morphological_op != 'none' else 'None'
                plt.title(f"Preprocessed\n({method_name}+{ops_name})")
                plt.axis('off')

                plt.tight_layout()
                plt.suptitle(f"Preprocessing: {os.path.basename(img_path)}", y=1.05)

                # Save or display
                if output_dir:
                    # Create a clean filename
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    # Remove any problematic characters
                    base_name = re.sub(r'[^\w\-\.]', '_', base_name)
                    # Limit length to avoid too long filenames
                    if len(base_name) > 50:
                        base_name = base_name[:50]
                    output_path = os.path.join(output_dir, f"{base_name}_comparison.png")

                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    print(f"Saved visualization to: {output_path}")
                else:
                    plt.show()

                plt.close()

            except Exception as e:
                print(f"Error visualizing {img_path}: {e}")
                continue

    def extract_stroke_features(self, img):
        """Extract features related to pen stroke characteristics."""
        try:
            if not self.use_stroke_features:
                return None

            # Find contours in the binary image
            try:
                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except ValueError:
                # OpenCV 3.x returns 3 values
                _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return np.zeros(10)  # Return zeros if no contours found

            # Calculate stroke features
            features = []

            # 1. Average stroke width using distance transform
            dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)
            if np.max(dist) > 0:
                mean_width = np.mean(dist[dist > 0]) * 2  # Diameter = 2 * radius
                max_width = np.max(dist) * 2
            else:
                mean_width = 0
                max_width = 0

            features.extend([mean_width, max_width])

            # 2. Calculate horizontal vs vertical stroke ratio
            horizontal_kernel = np.ones((1, 5), np.uint8)
            vertical_kernel = np.ones((5, 1), np.uint8)

            horizontal_strokes = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_strokes = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)

            h_pixels = np.sum(horizontal_strokes) / 255
            v_pixels = np.sum(vertical_strokes) / 255
            total_pixels = np.sum(img) / 255

            if total_pixels > 0:
                h_ratio = h_pixels / total_pixels
                v_ratio = v_pixels / total_pixels
            else:
                h_ratio = v_ratio = 0

            features.extend([h_ratio, v_ratio])

            # 3. Shape complexity using contour metrics
            perimeters = []
            areas = []
            solidity_values = []

            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)

                if area > 5:  # Ignore tiny contours
                    perimeters.append(perimeter)
                    areas.append(area)

                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)

                    if hull_area > 0:
                        solidity = area / hull_area
                        solidity_values.append(solidity)

            if areas:
                avg_perimeter = np.mean(perimeters)
                avg_area = np.mean(areas)
                complexity = np.sum(perimeters) / max(1, np.sum(areas))  # Perimeter to area ratio
            else:
                avg_perimeter = avg_area = complexity = 0

            avg_solidity = np.mean(solidity_values) if solidity_values else 0

            features.extend([complexity, avg_solidity])

            # 4. Signature slant estimation
            non_zero_pixels = np.column_stack(np.where(img > 0))
            if len(non_zero_pixels) > 10:
                # Fit a line to the signature points
                vx, vy, _, _ = cv2.fitLine(non_zero_pixels, cv2.DIST_L2, 0, 0.01, 0.01)
                slope = vy / (vx + 1e-10)  # Avoid division by zero
                angle = np.arctan(slope) * 180 / np.pi

                # Normalize to range -45 to 45
                normalized_angle = min(max(angle, -45), 45) / 45
            else:
                normalized_angle = 0

            features.append(normalized_angle)

            # Format the features as numpy array
            try:
                stroke_features = np.array([
                    float(x) if not isinstance(x, np.ndarray) else float(x.item())
                    for x in features
                ], dtype=np.float32)
            except ValueError:
                print(f"Warning: Features have inconsistent shapes: {features}")
                # Force consistent shape
                stroke_features = np.zeros(7, dtype=np.float32)  # Or whatever size is expected
                stroke_features[:min(len(features), 7)] = [float(f) for f in features[:7]]

            # Ensure consistent length for the feature vector (pad with zeros if needed)
            padded = np.zeros(10, dtype=np.float32)
            padded[:min(len(stroke_features), 10)] = stroke_features[:min(len(stroke_features), 10)]

            return padded

        except Exception as e:
            print(f"Error extracting stroke features: {e}")
            return np.zeros(10, dtype=np.float32)  # Return zeros on error

    def extract_hu_moments(self, img):
        """Extract Hu moments for shape features."""
        # Calculate moments
        moments = cv2.moments(img)

        # Extract Hu moments
        hu_moments = cv2.HuMoments(moments)

        # Apply log transform to reduce magnitude differences and improve scale invariance
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments.flatten()

    def extract_lbp(self, img):
        """Extract Local Binary Patterns for texture features."""
        if self.use_enhanced_lbp:
            # Multi-scale LBP for better feature extraction
            lbp1 = local_binary_pattern(img, 8, 1, method='uniform')
            lbp2 = local_binary_pattern(img, 16, 2, method='uniform')
            lbp3 = local_binary_pattern(img, 24, 3, method='uniform')

            # Calculate histograms at different scales
            hist1, _ = np.histogram(lbp1, density=True, bins=10, range=(0, 10))
            hist2, _ = np.histogram(lbp2, density=True, bins=18, range=(0, 18))
            hist3, _ = np.histogram(lbp3, density=True, bins=26, range=(0, 26))

            # Combine multi-scale histograms
            return np.concatenate([hist1, hist2, hist3])
        else:
            # Standard LBP
            lbp = local_binary_pattern(img, 8, 1, method='uniform')
            hist, _ = np.histogram(lbp, density=True, bins=10, range=(0, 10))
            return hist

    def extract_hog(self, img):
        """Extract Histogram of Oriented Gradients for gradient features."""
        # Configure HOG parameters
        win_size = self.size

        # Create HOG parameters that satisfy the constraint
        # Block size must be divisible by cell size
        cell_size = (8, 8)
        block_size = (16, 16)  # Must be a multiple of cell_size

        # Ensure block_stride divides evenly into (winSize - blockSize)
        # For reliability, make block_stride a divisor of (winSize - blockSize)
        width_diff = win_size[0] - block_size[0]
        height_diff = win_size[1] - block_size[1]

        # Find a divisor for width_diff and height_diff
        # Starting with 8 which is common and usually works well
        block_stride_width = 8
        while width_diff % block_stride_width != 0 and block_stride_width > 1:
            block_stride_width -= 1

        block_stride_height = 8
        while height_diff % block_stride_height != 0 and block_stride_height > 1:
            block_stride_height -= 1

        block_stride = (block_stride_width, block_stride_height)

        # Number of orientation bins
        n_bins = 9

        # Create HOG descriptor and compute features
        try:
            hog = cv2.HOGDescriptor(
                win_size, block_size, block_stride, cell_size, n_bins
            )
            hog_feats = hog.compute(img)
            return hog_feats.flatten()
        except cv2.error as e:
            # Handle HOG computation errors more gracefully
            print(f"Error computing HOG features: {e}")
            # Return a zero vector with a similar length to avoid disrupting the pipeline
            expected_length = 10000  # Approximate - adjust based on your typical HOG feature length
            return np.zeros(expected_length)

    def extract_zernike_moments(self, img, radius=21, degree=8):
        """Extract Zernike moments for rotation-invariant features."""
        if not self.use_zernike:
            return None

        # Find contours to focus on the signature region
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(degree * 2)

        # Create a mask from contours
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, 255, -1)

        # Calculate moments to find center of mass
        moments = cv2.moments(mask)
        if moments['m00'] == 0:  # Check to avoid division by zero
            return np.zeros(degree * 2)

        # Calculate center of mass
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        # Extract features based on center and radius
        features = []
        for i in range(degree):
            for _ in range(i+1):
                # Approximate Zernike moments using radial distance and angle
                dist = np.sqrt((np.arange(img.shape[0])[:, np.newaxis] - cy)**2 +
                            (np.arange(img.shape[1])[np.newaxis, :] - cx)**2)
                mask = dist <= radius
                if mask.any():
                    value = np.sum(img * mask) / np.sum(mask)
                    features.append(value)
                else:
                    features.append(0)

        return np.array(features)

    def extract_gabor_features(self, img, num_scales=4, num_orientations=8):
        """Extract Gabor filter features for multi-scale orientation analysis."""
        if not self.use_gabor:
            return None

        # Define Gabor filter parameters
        ksize = 31  # Filter size
        sigma = 4.0  # Standard deviation of Gaussian envelope
        gamma = 0.5  # Spatial aspect ratio
        psi = 0      # Phase offset

        # Scales and orientations
        scales = np.logspace(-1, 0.5, num_scales)
        orientations = np.linspace(0, np.pi, num_orientations)

        # Extract Gabor features
        gabor_features = []

        for scale in scales:
            lambd = 10.0 / scale  # Wavelength of sinusoidal factor
            for theta in orientations:
                # Apply Gabor filter
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)

                # Extract statistics from filtered image
                mean = np.mean(filtered)
                std = np.std(filtered)
                max_val = np.max(filtered)

                gabor_features.extend([mean, std, max_val])

        return np.array(gabor_features)

    def fit_pca_hog(self, hog_features_list):
        """Fit PCA model on HOG features from multiple images."""
        if self.use_pca_hog and len(hog_features_list) > 0:
            # Stack all HOG features
            all_hog_features = np.vstack(hog_features_list)

            # Fit PCA
            self.pca_hog = PCA(n_components=min(self.pca_hog_components, all_hog_features.shape[0], all_hog_features.shape[1]), random_state=42)
            self.pca_hog.fit(all_hog_features)

            print(f"PCA HOG: Reduced dimensions from {all_hog_features.shape[1]} to {self.pca_hog.n_components_}")
            print(f"PCA HOG: Explained variance ratio: {np.sum(self.pca_hog.explained_variance_ratio_):.4f}")

    def apply_pca_hog(self, hog_features):
        """Apply fitted PCA model to reduce HOG feature dimensions."""
        if self.use_pca_hog and self.pca_hog is not None:
            return self.pca_hog.transform(hog_features.reshape(1, -1)).flatten()
        return hog_features

    def normalize_feature_group(self, features, method='standard'):
        """Normalize a feature group using specified method."""
        if not self.normalize_features or features.shape[0] <= 1:
            return features

        if method == 'standard':
            # Standardize features (zero mean, unit variance)
            scaler = StandardScaler()
            return scaler.fit_transform(features)
        elif method == 'l1':
            # L1 normalization (sum of absolute values = 1)
            return normalize(features, norm='l1')
        elif method == 'l2':
            # L2 normalization (Euclidean norm = 1)
            return normalize(features, norm='l2')
        elif method == 'robust':
            # Robust scaling using median and interquartile range
            scaler = RobustScaler()
            return scaler.fit_transform(features)
        else:
            # Default to no normalization
            return features

    def extract_features(self, img_path):
        """Extract all feature types from a signature image."""
        try:
            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image: {img_path}")
                return None, None, None, None, None, None

            # Preprocess the image
            img = self.preprocess(img)

            # Extract different feature types
            hu_moments = self.extract_hu_moments(img)
            lbp_hist = self.extract_lbp(img)
            hog_feats = self.extract_hog(img)
            stroke_features = self.extract_stroke_features(img) if self.use_stroke_features else None

            # Conditionally extract Zernike moments
            zernike_moments = None
            if self.use_zernike:
                zernike_moments = self.extract_zernike_moments(img)

            # Conditionally extract Gabor features
            gabor_features = None
            if self.use_gabor:
                gabor_features = self.extract_gabor_features(img)

            return hu_moments, lbp_hist, hog_feats, zernike_moments, gabor_features, stroke_features

        except Exception as e:
            print(f"Error extracting features from {img_path}: {e}")
            return None, None, None, None, None, None


class SignatureDatabase:
    """Class for managing signature image collection with hierarchical directory support."""

    def __init__(self, root_dir, valid_file_endings, config):
        """Initialize with root directory, valid file types, and configuration."""
        self.root = root_dir
        self.valid_file_endings = valid_file_endings
        self.cluster_directory_depth = config['CLUSTER_DIRECTORY_DEPTH']
        self.testing_on_preclustered_images = config['TESTING_ON_PRECLUSTERED_IMAGES']

    def get_directory_depth(self, path):
        """Get the directory depth relative to the root directory."""
        rel_path = os.path.relpath(path, self.root)
        if rel_path == '.':  # If path is root
            return 0
        return len(rel_path.split(os.sep))

    def get_cluster_pools(self):
        """Get signature paths grouped by cluster candidate pools based on directory structure."""
        print(f"Organizing signatures into cluster pools using depth: {self.cluster_directory_depth}...")

        # Dictionary to hold signature paths by pool
        cluster_pools = defaultdict(list)

        # Get maximum directory depth if 'max' is specified
        max_depth = 0
        if self.cluster_directory_depth == 'max':
            for root, _, _ in os.walk(self.root):
                depth = self.get_directory_depth(root)
                max_depth = max(max_depth, depth)
            print(f"Determined maximum directory depth: {max_depth}")
            cluster_depth = max_depth
        else:
            cluster_depth = self.cluster_directory_depth

        # Walk through directory tree
        for root, _, files in os.walk(self.root):
            # Get depth of current directory
            current_depth = self.get_directory_depth(root)

            # Determine if this directory should be a cluster pool
            is_pool = False
            if self.cluster_directory_depth == 'max':
                # Only leaf directories are pools
                if not any(os.path.isdir(os.path.join(root, d)) for d in os.listdir(root) if not d.startswith('.')):
                    is_pool = True
            else:
                # Directories at the specified depth are pools
                if current_depth == cluster_depth:
                    is_pool = True
                # Directories below the specified depth inherit their parent's pool
                elif current_depth > cluster_depth:
                    is_pool = False
                # Directories above the specified depth aren't pools themselves
                else:
                    is_pool = False

            # If this is a pool, collect all valid signature files
            if is_pool:
                pool_id = root
                for file in files:
                    if any([file.endswith(ending) for ending in self.valid_file_endings]):
                        cluster_pools[pool_id].append(os.path.join(root, file))
            # If below cluster depth, add to parent pool
            elif current_depth > cluster_depth:
                # Find parent directory at cluster depth
                parts = os.path.relpath(root, self.root).split(os.sep)
                parent_path = os.path.join(self.root, *parts[:cluster_depth])

                for file in files:
                    if any([file.endswith(ending) for ending in self.valid_file_endings]):
                        cluster_pools[parent_path].append(os.path.join(root, file))

        # Flatten the pools if depth is 0 (all signatures in one pool)
        if cluster_depth == 0:
            all_signatures = []
            for signatures in cluster_pools.values():
                all_signatures.extend(signatures)
            cluster_pools = {self.root: all_signatures}

        print(f"Found {len(cluster_pools)} cluster pools with a total of {sum(len(sigs) for sigs in cluster_pools.values())} signatures")
        return cluster_pools

    def get_all_signatures(self, limit=None):
        """Get paths of all signature images with optional limit."""
        print(f"Loading signature paths from {self.root}...")
        signatures = []

        # Walk through directory tree
        for root, _, files in os.walk(self.root):
            for file in files:
                if any([file.endswith(ending) for ending in self.valid_file_endings]):
                    signatures.append(os.path.join(root, file))

        print(f"Found {len(signatures)} signatures.")

        # Optionally limit the number of signatures (useful for testing)
        if limit is not None and limit < len(signatures):
            print(f"Limiting to {limit} signatures for testing.")
            # Ensure we get a good mix by shuffling before limiting
            random.shuffle(signatures)
            signatures = signatures[:limit]

        return signatures

    def get_true_labels(self):
        """Get ground truth labels based on folder structure for testing."""
        if not self.testing_on_preclustered_images:
            print("Not using preclustered images, skipping true label extraction.")
            return {}

        print("Getting true labels from folder names...")
        true_labels = {}

        # Determine the appropriate depth level to extract labels from
        if self.cluster_directory_depth == 'max':
            # Use leaf directories
            for root, dirs, files in os.walk(self.root):
                if not dirs:  # This is a leaf directory
                    label = os.path.basename(root)
                    for file in files:
                        if any([file.endswith(ending) for ending in self.valid_file_endings]):
                            true_labels[os.path.join(root, file)] = label
        else:
            # Use directories at the specified depth
            for root, _, files in os.walk(self.root):
                # Get depth of current directory
                current_depth = self.get_directory_depth(root)

                if current_depth == self.cluster_directory_depth:
                    label = os.path.basename(root)
                    for file in files:
                        if any([file.endswith(ending) for ending in self.valid_file_endings]):
                            true_labels[os.path.join(root, file)] = label
                elif current_depth > self.cluster_directory_depth:
                    # Get the label from the parent directory at the specified depth
                    parts = os.path.relpath(root, self.root).split(os.sep)
                    if len(parts) > self.cluster_directory_depth:
                        label = parts[self.cluster_directory_depth - 1]
                        for file in files:
                            if any([file.endswith(ending) for ending in self.valid_file_endings]):
                                true_labels[os.path.join(root, file)] = label

        print(f"Extracted {len(true_labels)} true labels.")
        return true_labels


def comprehensive_parameter_optimization(config, perfect_clusters_dir,
                                        checkpoint_file=None,
                                        checkpoint_history_file=None,
                                        n_iterations=200,
                                        base_dir=None):
    """
    Comprehensive parameter optimization using hyperopt with conditional parameter support.
    
    Args:
        config: Base configuration dictionary
        perfect_clusters_dir: Directory containing perfectly clustered signatures
        checkpoint_file: File to store checkpoint data
        checkpoint_history_file: File to store detailed history
        n_iterations: Total number of iterations to run
    """

    from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
    from hyperopt.pyll import scope

    # If no results directory specified, create one
    if base_dir is None:
        # Use default directory
        base_dir = config.get('OPTIMIZATION_RESULTS_DIR', 'results/optimize')
        os.makedirs(base_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_dir, f"optimization_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Set default checkpoint files in results directory if not specified
    if checkpoint_file is None:
        checkpoint_file = os.path.join(base_dir, "optimization_checkpoint.json")

    if checkpoint_history_file is None:
        checkpoint_history_file = os.path.join(base_dir, "optimization_checkpoint_history.json")

    # Enhanced search space with preprocessing and feature extraction parameters
    space = {
        # Preprocessing parameters
        'PREPROCESSING_PARAMS': hp.choice('PREPROCESSING_PARAMS', [
            {
                'DYNAMIC_PREPROCESSING': hp.choice('DYNAMIC_PREPROCESSING', [
                    {
                        'value': False,
                        # No additional parameters needed
                    },
                    {
                        'value': True,
                        'FADED_MEAN_THRESHOLD': hp.uniform('FADED_MEAN_THRESHOLD', 150, 230),
                        'FADED_STD_THRESHOLD': hp.uniform('FADED_STD_THRESHOLD', 10, 60),
                        'LOW_CONTRAST_STD_THRESHOLD': hp.uniform('LOW_CONTRAST_STD_THRESHOLD', 10, 50)
                    }
                ]),
                'PREPROCESSING_METHOD': hp.choice('PREPROCESSING_METHOD', ['otsu', 'adaptive']),
                'USE_CLAHE': hp.choice('USE_CLAHE', [
                    {
                        'value': False,
                    },
                    {
                        'value': True,
                        'CLAHE_CLIP_LIMIT': hp.uniform('CLAHE_CLIP_LIMIT', 0.5, 12.0),
                    }
                ]),
                'ADAPTIVE_THRESHOLD_C': hp.uniform('ADAPTIVE_THRESHOLD_C', -5.0, 15.0),
                'ADAPTIVE_BLOCK_SIZE': scope.int(hp.quniform('ADAPTIVE_BLOCK_SIZE', 3, 51, 2)),
                'MORPHOLOGICAL_OP': hp.choice('MORPHOLOGICAL_OP', ['none', 'open', 'close', 'open_close', 'dilate']),
                'MORPHOLOGICAL_KERNEL_SIZE': scope.int(hp.quniform('MORPHOLOGICAL_KERNEL_SIZE', 1, 9, 2)),  # Must be odd
            }
        ]),

        # Image size parameters
        'IMAGE_SIZE': hp.choice('IMAGE_SIZE', [
            # Original options
            (100, 100), (150, 150), (200, 200), (224, 224), (256, 256),
            # Additional wide formats for signatures
            (160, 64), (192, 64), (224, 64), (256, 96), (320, 96),
            (320, 128), (384, 128), (400, 150), (448, 128), (480, 160),
            # Very wide options for long signatures
            (512, 128), (640, 160)
        ]),

        # Feature extraction flags - boolean choices
        'USE_ENHANCED_LBP': hp.choice('USE_ENHANCED_LBP', [False, True]),

        # Stroke features - conditional weight
        'USE_STROKE_FEATURES': hp.choice('USE_STROKE_FEATURES', [
            {
                'value': False,
                'STROKE_FEATURE_WEIGHT': 0.0  # When disabled, weight is always 0
            },
            {
                'value': True,
                'STROKE_FEATURE_WEIGHT': hp.uniform('STROKE_FEATURE_WEIGHT', 0.5, 10.0)  # Only explored when enabled
            }
        ]),

        # Zernike features - conditional weight
        'USE_ZERNIKE': hp.choice('USE_ZERNIKE', [
            {
                'value': False,
                'ZERNIKE_WEIGHT': 0.0  # When disabled, weight is always 0
            },
            {
                'value': True,
                'ZERNIKE_WEIGHT': hp.uniform('ZERNIKE_WEIGHT', 0.01, 10.0)  # Only explored when enabled
            }
        ]),

        # Gabor features - conditional weight
        'USE_GABOR': hp.choice('USE_GABOR', [
            {
                'value': False,
                'GABOR_WEIGHT': 0.0  # When disabled, weight is always 0
            },
            {
                'value': True,
                'GABOR_WEIGHT': hp.uniform('GABOR_WEIGHT', 0.01, 10.0)  # Only explored when enabled
            }
        ]),

        # PCA for HOG - conditional components
        'USE_PCA_HOG': hp.choice('USE_PCA_HOG', [
            {
                'value': False,
                # No components needed when disabled
            },
            {
                'value': True,
                'PCA_HOG_COMPONENTS': scope.int(hp.quniform('PCA_HOG_COMPONENTS', 10, 500, 1))
            }
        ]),

        # Feature weights
        'HU_WEIGHT': hp.loguniform('HU_WEIGHT', np.log(0.001), np.log(50.0)),
        'LBP_WEIGHT': hp.loguniform('LBP_WEIGHT', np.log(0.01), np.log(50.0)),
        'HOG_WEIGHT': hp.loguniform('HOG_WEIGHT', np.log(0.01), np.log(50.0)),

        # Clustering parameters
        'DISTANCE_METRIC': hp.choice('DISTANCE_METRIC', [
            {
                'name': 'correlation',
                'threshold_range': hp.uniform('correlation_threshold', 0.05, 1.2)
            },
            {
                'name': 'cosine', 
                'threshold_range': hp.uniform('cosine_threshold', 0.05, 1.2)
            },
            {
                'name': 'euclidean',
                'threshold_range': hp.loguniform('euclidean_threshold', np.log(1), np.log(500))
            },
            {
                'name': 'cityblock',
                'threshold_range': hp.loguniform('cityblock_threshold', np.log(1), np.log(500))
            }
        ]),
        'LINKAGE_METHOD': hp.choice('LINKAGE_METHOD', ['average', 'complete', 'ward']),

        # Singleton handling parameters
        'SINGLETON_HANDLING': hp.choice('SINGLETON_HANDLING', [
            {
                'value': False,
                # No distance threshold here anymore
            },
            {
                'value': True,
                'SINGLETON_DETECTION_THRESHOLD': hp.uniform('SINGLETON_DETECTION_THRESHOLD', 0.3, 0.9),
                'EXPECTED_SINGLETON_RATIO': hp.uniform('EXPECTED_SINGLETON_RATIO', 0.1, 0.9),
            }
        ]),

        # Two-stage parameters
        'USE_TWO_STAGE': hp.choice('USE_TWO_STAGE', [
            {
                'value': False,
                # No need for merge parameters when two-stage is disabled
            },
            {
                'value': True,
                'MERGE_THRESHOLD': hp.uniform('MERGE_THRESHOLD', 0.2, 1.1),
                'MERGE_METHOD': hp.choice('MERGE_METHOD', ['average', 'min', 'max', 'adaptive']),
                'CLUSTER_SPLIT_PERCENTILE': hp.choice('CLUSTER_SPLIT_PERCENTILE', 
                           [50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99]),
                'MIN_CLUSTER_SIZE': scope.int(hp.qloguniform('MIN_CLUSTER_SIZE', np.log(1), np.log(50), 1)),
            }
        ]),

        # Adaptive thresholding - conditional percentile
        'USE_ADAPTIVE_THRESHOLD': hp.choice('USE_ADAPTIVE_THRESHOLD', [
            {
                'value': False,
                # No percentile needed when disabled
            },
            {
                'value': True,
                'ADAPTIVE_PERCENTILE': scope.int(hp.quniform('ADAPTIVE_PERCENTILE', 5, 95, 1))
            }
        ]),

        # Ensemble methods - conditional weights and spectral parameters
        'USE_ENSEMBLE': hp.choice('USE_ENSEMBLE', [
            {
                'value': False,
                # No ensemble parameters needed when disabled
            },
            {
                'value': True,
                'HIERARCHICAL_WEIGHT_RATIO': hp.uniform('HIERARCHICAL_WEIGHT_RATIO', 0.01, 0.99),
                'SPECTRAL_N_CLUSTERS': hp.choice('SPECTRAL_N_CLUSTERS', ['auto', 3, 5, 8, 12, 20]),
                'SPECTRAL_AFFINITY': hp.choice('SPECTRAL_AFFINITY', ['rbf', 'nearest_neighbors', 'precomputed'])
            }
        ]),

        # Feature normalization
        'NORMALIZE_FEATURES': hp.choice('NORMALIZE_FEATURES', [False, True]),
        'NORMALIZE_METHOD': hp.choice('NORMALIZE_METHOD', ['standard', 'l1', 'l2', 'robust']),
    }

    def unpack_params(params):
        """
        Unpack nested hyperopt parameters into a flat dictionary for evaluation.
        """
        flat_params = {}

        try:
            # Process all parameters
            for key, value in params.items():
                if key == 'PREPROCESSING_PARAMS':
                    # Extract the dynamic preprocessing parameters
                    if 'DYNAMIC_PREPROCESSING' in value:
                        dp_value = value['DYNAMIC_PREPROCESSING']
                        flat_params['DYNAMIC_PREPROCESSING'] = dp_value['value']
                        if dp_value['value']:  # If dynamic preprocessing is enabled
                            flat_params['FADED_MEAN_THRESHOLD'] = dp_value.get('FADED_MEAN_THRESHOLD', 200)
                            flat_params['FADED_STD_THRESHOLD'] = dp_value.get('FADED_STD_THRESHOLD', 40)
                            flat_params['LOW_CONTRAST_STD_THRESHOLD'] = dp_value.get('LOW_CONTRAST_STD_THRESHOLD', 30)

                    # Copy all other preprocessing parameters
                    for preproc_key, preproc_value in value.items():
                        if preproc_key == 'DYNAMIC_PREPROCESSING':
                            continue  # Already handled
                        elif preproc_key == 'USE_CLAHE':
                            flat_params['USE_CLAHE'] = preproc_value['value']
                            if preproc_value['value']:
                                flat_params['CLAHE_CLIP_LIMIT'] = preproc_value['CLAHE_CLIP_LIMIT']
                        else:
                            flat_params[preproc_key] = preproc_value

                elif key == 'DISTANCE_METRIC':
                    # Extract metric name and threshold
                    flat_params['DISTANCE_METRIC'] = value['name']
                    flat_params['DISTANCE_THRESHOLD'] = value['threshold_range']

                elif key == 'SINGLETON_HANDLING':
                    flat_params['SINGLETON_HANDLING'] = value['value']
                    if value['value']:
                        flat_params['SINGLETON_DETECTION_THRESHOLD'] = value['SINGLETON_DETECTION_THRESHOLD']
                        flat_params['EXPECTED_SINGLETON_RATIO'] = value['EXPECTED_SINGLETON_RATIO']

                # Rest of your existing parameter handling
                elif key == 'USE_ZERNIKE':
                    flat_params['USE_ZERNIKE'] = value['value']
                    flat_params['ZERNIKE_WEIGHT'] = value['ZERNIKE_WEIGHT']
                elif key == 'USE_STROKE_FEATURES':
                    flat_params['USE_STROKE_FEATURES'] = value['value']
                    flat_params['STROKE_FEATURE_WEIGHT'] = value['STROKE_FEATURE_WEIGHT']
                elif key == 'USE_GABOR':
                    flat_params['USE_GABOR'] = value['value']
                    flat_params['GABOR_WEIGHT'] = value['GABOR_WEIGHT']
                elif key == 'USE_PCA_HOG':
                    flat_params['USE_PCA_HOG'] = value['value']
                    if value['value']:
                        flat_params['PCA_HOG_COMPONENTS'] = value['PCA_HOG_COMPONENTS']
                elif key == 'USE_TWO_STAGE':
                    flat_params['USE_TWO_STAGE'] = value['value']
                    if value['value']:
                        flat_params['MERGE_THRESHOLD'] = value['MERGE_THRESHOLD']
                        flat_params['MERGE_METHOD'] = value['MERGE_METHOD']
                        flat_params['CLUSTER_SPLIT_PERCENTILE'] = value['CLUSTER_SPLIT_PERCENTILE']
                        flat_params['MIN_CLUSTER_SIZE'] = value['MIN_CLUSTER_SIZE']
                elif key == 'USE_ADAPTIVE_THRESHOLD':
                    flat_params['USE_ADAPTIVE_THRESHOLD'] = value['value']
                    if value['value']:
                        flat_params['ADAPTIVE_PERCENTILE'] = value['ADAPTIVE_PERCENTILE']
                elif key == 'USE_ENSEMBLE':
                    flat_params['USE_ENSEMBLE'] = value['value']
                    if value['value']:
                        flat_params['HIERARCHICAL_WEIGHT_RATIO'] = value['HIERARCHICAL_WEIGHT_RATIO']
                        flat_params['SPECTRAL_N_CLUSTERS'] = value['SPECTRAL_N_CLUSTERS']
                        flat_params['SPECTRAL_AFFINITY'] = value['SPECTRAL_AFFINITY']
                        # Set ensemble methods and weights based on ratio
                        flat_params['ENSEMBLE_METHODS'] = ['hierarchical', 'spectral']
                        hier_weight = value['HIERARCHICAL_WEIGHT_RATIO']
                        spectral_weight = 1.0 - hier_weight
                        flat_params['ENSEMBLE_WEIGHTS'] = [hier_weight, spectral_weight]
                else:
                    # Direct copy for simple parameters
                    flat_params[key] = value

            return flat_params

        except Exception as e:
            print(f"ERROR in parameter unpacking: {e}")
            print(f"Problem parameter structure: {params}")
            # Return a basic configuration as fallback
            return {'DISTANCE_THRESHOLD': 0.5, 'DISTANCE_METRIC': 'correlation'}  # Basic defaults

    def save_checkpoint(all_trials, all_results, checkpoint_file, history_file):
        """
        Save optimization checkpoint data with improved error handling.
        
        Args:
            all_trials: Hyperopt Trials object
            all_results: List of all evaluations with details
            checkpoint_file: File to save checkpoint
            history_file: File to save detailed history
        """
        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file)) or '.', exist_ok=True)

            # Try to save trials using pickle
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(all_trials, f)
            except Exception as pickle_error:
                print(f"Warning: Could not pickle trials object: {pickle_error}")
                # Create a simplified version of trials that's more likely to pickle
                simplified_trials = deepcopy(all_trials)
                # Clear any problematic attributes
                if hasattr(simplified_trials, '_dynamic_trials'):
                    simplified_trials._dynamic_trials = []

                try:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(simplified_trials, f)
                    print("Saved simplified trials object")
                except Exception as e:
                    print(f"Error saving simplified trials: {e}")
                    # Last resort - save just the trial results as a list
                    simple_results = [{'tid': t['tid'], 'result': t['result']}
                                    for t in all_trials.trials if 'result' in t]
                    with open(f"{checkpoint_file}.results", 'wb') as f:
                        pickle.dump(simple_results, f)
                    print(f"Saved minimal trial results to {checkpoint_file}.results")

            # Save detailed history in JSON format for readability
            try:
                with open(history_file, 'w') as f:
                    # Convert complex objects to simple types
                    history_data = []
                    for result in all_results:
                        # Create a simplified copy with Python native types
                        simple_result = {}
                        for k, v in result.items():
                            if k == 'params':
                                # Handle parameters
                                simple_result[k] = {pk: (float(pv) if isinstance(pv, (np.floating, np.integer))
                                                else pv) for pk, pv in v.items()}
                            elif k == 'metrics':
                                # Handle metrics
                                if v:
                                    metrics_copy = {}
                                    for mk, mv in v.items():
                                        if mk == 'cluster_sizes' and isinstance(mv, dict):
                                            metrics_copy[mk] = {sk: float(sv) if isinstance(sv, (np.floating, np.integer))
                                                            else sv for sk, sv in mv.items()}
                                        else:
                                            metrics_copy[mk] = float(mv) if isinstance(mv, (np.floating, np.integer)) else mv
                                    simple_result[k] = metrics_copy
                                else:
                                    simple_result[k] = v
                            else:
                                # Handle other fields
                                simple_result[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v

                        history_data.append(simple_result)

                    # Use a try-except block for the JSON dump
                    try:
                        json.dump(history_data, f, indent=2)
                    except TypeError as json_error:
                        print(f"Warning: JSON serialization error: {json_error}")
                        # Try a more aggressive approach to make it JSON serializable
                        safe_history = []
                        for result in history_data:
                            safe_result = {}
                            for k, v in result.items():
                                try:
                                    # Test if this value is JSON serializable
                                    json.dumps({k: v})
                                    safe_result[k] = v
                                except TypeError:
                                    # If not serializable, convert to string
                                    safe_result[k] = str(v)
                            safe_history.append(safe_result)

                        json.dump(safe_history, f, indent=2)
                        print("Saved simplified JSON history")
            except Exception as history_error:
                print(f"Error saving history file: {history_error}")
                # Try to save in a simpler format
                try:
                    with open(f"{history_file}.simple", 'w') as f:
                        # Just save scores and basic metrics
                        simple_data = []
                        for r in all_results:
                            entry = {
                                'score': r.get('score', 0),
                                'time': r.get('timestamp', '')
                            }
                            if 'metrics' in r:
                                entry['accuracy'] = r['metrics'].get('accuracy', 0)
                                entry['fragmentation'] = r['metrics'].get('avg_fragmentation', 0)
                            simple_data.append(entry)
                        json.dump(simple_data, f)
                    print(f"Saved minimal history to {history_file}.simple")
                except Exception as e:
                    print(f"Failed to save even simplified history: {e}")

            print(f"Checkpoint saved with {len(all_trials.trials)} trials")
            return True

        except Exception as e:
            print(f"Error in checkpoint saving process: {e}")
            traceback.print_exc()

            # Try an emergency checkpoint - ensure it's in the same directory
            try:
                emergency_file = f"{checkpoint_file}.emergency"
                with open(emergency_file, 'w') as f:
                    f.write(f"Emergency checkpoint at {datetime.now()}\n")
                    f.write(f"Trials completed: {len(all_trials.trials)}\n")
                    f.write(f"Results collected: {len(all_results)}\n")

                    # Write top 5 results
                    f.write("\nTop results:\n")
                    sorted_results = sorted(all_results,
                                        key=lambda x: x.get('score', -float('inf')),
                                        reverse=True)
                    for i, r in enumerate(sorted_results[:5]):
                        f.write(f"{i+1}. Score: {r.get('score', 0)}\n")

                print(f"Emergency text checkpoint saved to {emergency_file}")
                return True
            except Exception as e2:
                print(f"Emergency save also failed: {e2}")
                return False

    def load_checkpoint(checkpoint_file):
        """
        Load existing checkpoint with enhanced error handling.
        
        Args:
            checkpoint_file: Path to checkpoint file
            
        Returns:
            Loaded Trials object or new Trials object if loading fails
        """
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    trials = pickle.load(f)
                print(f"Loaded checkpoint with {len(trials.trials)} trials")
                return trials, True
            except Exception as pickle_error:
                print(f"Error loading checkpoint with pickle: {pickle_error}")

                # Try alternative files
                results_file = f"{checkpoint_file}.results"
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'rb') as f:
                            trial_results = pickle.load(f)
                        print(f"Loaded {len(trial_results)} trial results from alternative file")

                        # Create a new Trials object and populate it
                        from hyperopt import Trials
                        trials = Trials()
                        # We'd need to reconstruct trials from the results
                        # This is a partial reconstruction - optimizer can continue but won't have full state

                        print("Created new Trials object with partial state from results")
                        return trials, True
                    except Exception as e:
                        print(f"Failed to load alternative results file: {e}")

        else:
            print("No checkpoint file found. Starting fresh.")

        # If we get here, all loading attempts failed
        from hyperopt import Trials
        return Trials(), False

    def load_history(history_file):
        """
        Load detailed evaluation history with enhanced error handling.
        
        Args:
            history_file: Path to history file
            
        Returns:
            List of evaluation results or empty list if loading fails
        """
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                print(f"Loaded history with {len(history)} evaluations")
                return history
            except json.JSONDecodeError as json_error:
                print(f"Error parsing JSON history file: {json_error}")

                # Try the simple backup file
                simple_file = f"{history_file}.simple"
                if os.path.exists(simple_file):
                    try:
                        with open(simple_file, 'r') as f:
                            simple_history = json.load(f)
                        print(f"Loaded simplified history with {len(simple_history)} entries")

                        # Convert simplified history to something usable
                        expanded_history = []
                        for entry in simple_history:
                            result = {
                                'score': entry.get('score', 0),
                                'timestamp': entry.get('time', ''),
                                'metrics': {
                                    'accuracy': entry.get('accuracy', 0),
                                    'avg_fragmentation': entry.get('fragmentation', 0)
                                }
                            }
                            expanded_history.append(result)

                        return expanded_history
                    except Exception as e:
                        print(f"Failed to load simplified history: {e}")
            except Exception as e:
                print(f"Error loading history: {e}")

        return []

    def objective(params):
        """
        Objective function for hyperopt to minimize.
        
        Args:
            params: Parameter configuration to evaluate
            
        Returns:
            Dictionary with loss (to minimize) and other metadata
        """
        # Start timing this evaluation
        eval_start_time = time.time()

        # Unpack nested hyperopt parameters
        flat_params = unpack_params(params)

        # Display configuration details
        print(f"\nEvaluating configuration {len(all_results) + 1}:")
        # Sort parameters alphabetically for consistent display
        for key in sorted(flat_params.keys()):
            print(f"  {key}: {flat_params[key]}")

        try:
            # Create a test configuration by updating base config with these parameters
            test_config = deepcopy(config)
            test_config.update(flat_params)

            # Create clusterer with this configuration
            clusterer = SignatureClustering(test_config)

            # Extract features
            feature_groups, valid_signatures = clusterer.extract_features_batch(all_signatures)

            # Compute distance matrix
            dist_matrix, feature_vectors = clusterer.compute_distances(
                feature_groups, test_config['DISTANCE_METRIC']
            )

            # Perform clustering based on configuration (SAME AS NORMAL MODE)
            if test_config.get('USE_ENSEMBLE', False):
                # Ensemble clustering approach
                clusters, ensemble_dist, linkage_matrix = clusterer.ensemble_clustering(
                    feature_vectors, dist_matrix, valid_signatures
                )
                # Use ensemble distance matrix for analysis
                dist_matrix = ensemble_dist
            elif test_config['USE_TWO_STAGE']:
                # Two-stage clustering
                clusters, linkage_matrix = clusterer.two_stage_clustering(
                    valid_signatures, dist_matrix, feature_vectors
                )
            else:
                # Single-stage clustering
                labels, linkage_matrix = clusterer.cluster_hierarchical(
                    dist_matrix.copy(),
                    test_config['DISTANCE_THRESHOLD'],
                    test_config['LINKAGE_METHOD']
                )
                clusters = clusterer.create_clusters_from_labels(labels, valid_signatures)

            # Evaluate clustering against true labels
            metrics = clusterer.evaluate_clustering(clusters, true_labels)

            # Enhanced scoring function that accounts for singleton-heavy datasets
            accuracy = metrics['accuracy']
            purity = metrics['avg_purity']
            ari = metrics.get('adjusted_rand_index', 0)
            nmi = metrics.get('normalized_mi', 0)
            fragmentation = metrics['avg_fragmentation']
            singleton_accuracy = metrics.get('singleton_accuracy', 0)  # May be 0 if not calculated

            # Get expected singleton ratio from configuration or use a default
            expected_singleton_ratio = test_config.get('EXPECTED_SINGLETON_RATIO', 0.5)

            # Calculate singleton ratio in the dataset
            if 'singleton_count' in metrics and metrics['total_signatures'] > 0:
                actual_singleton_ratio = metrics['singleton_count'] / metrics['total_signatures']
            else:
                actual_singleton_ratio = expected_singleton_ratio

            # Create a balanced score with appropriate weights
            # Adjust for many singletons by giving singleton accuracy more weight
            singleton_weight = min(0.7, actual_singleton_ratio)
            cluster_weight = 1.0 - singleton_weight

            # Calculate fragmentation penalty (don't penalize as much for singleton-heavy datasets)
            fragmentation_penalty = np.exp(-0.1 * max(0, fragmentation - 1.0))

            # Calculate weighted accuracy
            weighted_accuracy = (singleton_weight * singleton_accuracy +
                                cluster_weight * accuracy)

            # Final score combines weighted accuracy, purity, ARI, NMI with fragmentation penalty
            score = (0.4 * weighted_accuracy +
                    0.2 * purity +
                    0.2 * ari +
                    0.2 * nmi) * fragmentation_penalty

            # Record result (hyperopt minimizes, so negate score)
            loss = -score

            # Store detailed results for analysis
            result = {
                'params': flat_params,
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'eval_time': time.time() - eval_start_time,
                'status': STATUS_OK
            }

            # Add to our detailed history
            all_results.append(result)

            # Save checkpoint after each evaluation
            save_checkpoint(trials, all_results, checkpoint_file, checkpoint_history_file)

            gc.collect()

            # Return dict for hyperopt (must include 'loss' and 'status')
            return {
                'loss': loss,
                'status': STATUS_OK,
                'metrics': metrics,
                'eval_time': time.time() - eval_start_time,
                'score': score
            }

        except Exception as e:
            print(f"Error evaluating configuration: {e}")
            traceback.print_exc()

            # Record the failed evaluation with a very bad score
            result = {
                'params': flat_params,
                'score': -100.0,  # Very bad score for failed evaluations
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'fail'
            }

            all_results.append(result)

            # Save checkpoint after failure
            save_checkpoint(trials, all_results, checkpoint_file, checkpoint_history_file)

            gc.collect()

            # Return a very bad score for hyperopt
            return {
                'loss': 100.0,  # High loss (bad) for failed evaluations
                'status': STATUS_OK,  # Still mark as OK so hyperopt continues
                'error': str(e)
            }

    # Load checkpoint if exists
    trials, resumed = load_checkpoint(checkpoint_file)
    all_results = load_history(checkpoint_history_file)

    # Create database class for accessing signatures
    database = SignatureDatabase(perfect_clusters_dir, config['VALID_FILE_ENDINGS'], config)

    # Get all signatures in the perfect clusters directory
    all_signatures = database.get_all_signatures(limit=config.get('DATA_LIMIT'))
    if not all_signatures:
        print("No signatures found in the perfect clusters directory!")
        return None

    print(f"Found {len(all_signatures)} signatures in {perfect_clusters_dir}")

    # Extract perfect clustering ground truth from directory structure
    true_clusters = {}
    for root, _, files in os.walk(perfect_clusters_dir):
        # Skip the root directory
        if root == perfect_clusters_dir:
            continue

        # Get cluster name (directory name)
        cluster_id = os.path.basename(root)

        # Find signatures in this cluster
        cluster_signatures = []
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in config['VALID_FILE_ENDINGS']):
                sig_path = os.path.join(root, file)
                cluster_signatures.append(sig_path)

        # Store cluster if it has signatures
        if cluster_signatures:
            true_clusters[cluster_id] = cluster_signatures

    # Create true labels mapping for evaluation
    true_labels = {}
    for cluster_id, signatures in true_clusters.items():
        for sig in signatures:
            true_labels[sig] = cluster_id

    # Analyze dataset characteristics
    singleton_count = 0
    cluster_sizes = []
    for cluster_id, signatures in true_clusters.items():
        size = len(signatures)
        cluster_sizes.append(size)
        if size == 1:
            singleton_count += 1

    # Print dataset analysis
    if cluster_sizes:
        print(f"Dataset analysis:")
        print(f"  Total clusters: {len(true_clusters)}")
        print(f"  Singleton clusters: {singleton_count} ({singleton_count/len(true_clusters)*100:.1f}%)")
        print(f"  Cluster size: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.1f}")
        print(f"  Extracted {len(true_labels)} labeled signatures from {len(true_clusters)} clusters")
    else:
        print("Warning: No clusters found in the dataset!")

    # Determine max_evals based on resuming or not
    if resumed:
        # We're resuming, so add the remaining iterations
        max_evals = len(trials.trials) + n_iterations
        print(f"Resuming optimization with {len(trials.trials)} completed trials")
        print(f"Will run for {n_iterations} more iterations (total: {max_evals})")
    else:
        # Fresh start
        max_evals = n_iterations
        print(f"Starting fresh optimization for {max_evals} iterations")

    # Run the optimization with error handling
    try:
        class RandomStateWrapper:
            def __init__(self, seed=None):
                self.random_state = np.random.RandomState(seed)

            def __getattr__(self, name):
                # If integers is requested but not available, use randint instead
                if name == 'integers' and not hasattr(self.random_state, 'integers'):
                    return self.random_state.randint
                return getattr(self.random_state, name)

        best_result = None

        try:
            best_result = fmin(
                fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials,
                rstate=RandomStateWrapper(42),
                show_progressbar=False
            )
        except Exception as e:
            print(f"\nOptimization failed with error: {e}")
            print("Saving partial results...")

            # Try to save checkpoint before exiting
            if save_checkpoint(trials, all_results, checkpoint_file, checkpoint_history_file):
                print("Progress saved. You can resume later with the same checkpoint file.")

            # Return partial results
            return trials, None, results_dir

        # Only proceed with best parameter extraction if we got a result
        if best_result is not None:
            best_params = space_eval(space, best_result)
            flat_best = unpack_params(best_params)

            # Display optimization results
            display_optimization_results(trials, all_results, flat_best, config, results_dir)

            return trials, flat_best, results_dir
        else:
            print("Optimization did not complete successfully")
            return trials, None, results_dir

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

        # Save checkpoint before exiting
        if save_checkpoint(trials, all_results, checkpoint_file, checkpoint_history_file):
            print("Progress saved. You can resume later with the same checkpoint file.")

        return trials, None, results_dir

    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        traceback.print_exc()

        # Try to save checkpoint
        if all_results:
            if save_checkpoint(trials, all_results, checkpoint_file, checkpoint_history_file):
                print("Progress saved despite error.")

        return trials, None, results_dir


def display_optimization_results(trials, all_results, best_params, config, results_dir=None, n_top=10):
    """
    Display optimization results and save detailed reports.
    
    Args:
        trials: Hyperopt Trials object
        all_results: List of all evaluations with details
        best_params: Best parameters found
        config: Base configuration dictionary
        results_dir: Directory to save results (optional)
        n_top: Number of top configurations to display
    """
    # Create base optimization directory
    if results_dir is None:
        base_dir = config.get('OPTIMIZATION_RESULTS_DIR', 'results/optimize')
        os.makedirs(base_dir, exist_ok=True)

        # Create a timestamped subdirectory for this optimization run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(base_dir, f"optimization_results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" OPTIMIZATION RESULTS ".center(80, "="))
    print("=" * 80 + "\n")

    # Find best result in all_results
    best_result = None
    best_score = -float('inf')
    for result in all_results:
        if 'score' in result and result['score'] > best_score:
            best_score = result['score']
            best_result = result

    # Print best configuration
    print("BEST CONFIGURATION:")
    print("-" * 50)

    # Sort parameters for consistent display
    for key in sorted(best_params.keys()):
        print(f"  {key}: {best_params[key]}")

    print(f"\nBest score: {best_score:.4f}")

    # Print metrics of the best configuration
    if best_result and 'metrics' in best_result:
        metrics = best_result['metrics']
        print("\nPerformance Metrics:")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"  Average Purity: {metrics.get('avg_purity', 0):.2%}")
        print(f"  Fragmentation: {metrics.get('avg_fragmentation', 0):.2f} clusters per signer")
        print(f"  Adjusted Rand Index: {metrics.get('adjusted_rand_index', 0):.4f}")
        print(f"  Normalized Mutual Information: {metrics.get('normalized_mi', 0):.4f}")

        if 'singleton_count' in metrics:
            print(f"  Singleton Clusters: {metrics.get('singleton_count', 0)}")
            print(f"  Singleton Accuracy: {metrics.get('singleton_accuracy', 0):.2%}")

        if 'cluster_sizes' in metrics:
            sizes = metrics['cluster_sizes']
            print(f"\nCluster Size Distribution:")
            print(f"  Mean: {sizes.get('mean', 0):.1f}")
            print(f"  Min: {sizes.get('min', 0)}")
            print(f"  Max: {sizes.get('max', 0)}")

    # Prepare ranking data from all_results
    ranking_data = []
    for i, result in enumerate(all_results):
        if 'params' in result and 'score' in result:
            entry = {'idx': i, 'score': result['score']}

            # Add parameters
            for k, v in result['params'].items():
                entry[k] = v

            # Add metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                entry['accuracy'] = metrics.get('accuracy', 0)
                entry['avg_purity'] = metrics.get('avg_purity', 0)
                entry['avg_fragmentation'] = metrics.get('avg_fragmentation', 0)
                entry['adjusted_rand_index'] = metrics.get('adjusted_rand_index', 0)
                entry['normalized_mi'] = metrics.get('normalized_mi', 0)
                entry['num_clusters'] = metrics.get('num_clusters', 0)
                entry['singleton_count'] = metrics.get('singleton_count', 0)
                entry['singleton_accuracy'] = metrics.get('singleton_accuracy', 0)

                if 'cluster_sizes' in metrics:
                    sizes = metrics['cluster_sizes']
                    entry['cluster_mean_size'] = sizes.get('mean', 0)
                    entry['cluster_min_size'] = sizes.get('min', 0)
                    entry['cluster_max_size'] = sizes.get('max', 0)

            ranking_data.append(entry)

    # Sort all configurations by score (descending)
    ranking_data.sort(key=lambda x: x['score'], reverse=True)

    # Display top configurations in console
    print("\n" + "=" * 80)
    print(f" TOP {min(n_top, len(ranking_data))} CONFIGURATIONS ".center(80, "="))
    print("=" * 80)

    for rank, entry in enumerate(ranking_data[:n_top]):
        print(f"\n{rank+1}. Score: {entry['score']:.4f}")
        print("-" * 50)

        # Group parameters by category for better readability
        param_categories = {
            'Preprocessing': ['PREPROCESSING_METHOD', 'USE_CLAHE', 'CLAHE_CLIP_LIMIT', 
                           'DYNAMIC_PREPROCESSING', 'ADAPTIVE_THRESHOLD_C', 'ADAPTIVE_BLOCK_SIZE',
                           'MORPHOLOGICAL_OP', 'MORPHOLOGICAL_KERNEL_SIZE'],
            'Image': ['IMAGE_SIZE'],
            'Features': ['HU_WEIGHT', 'LBP_WEIGHT', 'HOG_WEIGHT', 'ZERNIKE_WEIGHT', 'GABOR_WEIGHT',
                       'STROKE_FEATURE_WEIGHT', 'USE_ENHANCED_LBP', 'USE_STROKE_FEATURES', 
                       'USE_ZERNIKE', 'USE_GABOR', 'USE_PCA_HOG', 'PCA_HOG_COMPONENTS'],
            'Normalization': ['NORMALIZE_FEATURES', 'NORMALIZE_METHOD'],
            'Distance': ['DISTANCE_METRIC', 'DISTANCE_THRESHOLD'],
            'Clustering': ['LINKAGE_METHOD', 'USE_TWO_STAGE', 'MERGE_THRESHOLD', 'MERGE_METHOD',
                         'CLUSTER_SPLIT_PERCENTILE', 'MIN_CLUSTER_SIZE', 'SINGLETON_HANDLING',
                         'SINGLETON_DETECTION_THRESHOLD', 'EXPECTED_SINGLETON_RATIO'],
            'Ensemble': ['USE_ENSEMBLE', 'HIERARCHICAL_WEIGHT_RATIO', 'SPECTRAL_N_CLUSTERS',
                        'SPECTRAL_AFFINITY'],
            'Adaptive': ['USE_ADAPTIVE_THRESHOLD', 'ADAPTIVE_PERCENTILE']
        }

        # Print parameters by category
        for category, param_keys in param_categories.items():
            category_params = [(key, entry[key]) for key in param_keys if key in entry]
            if category_params:
                print(f"  {category} Parameters:")
                for key, value in category_params:
                    print(f"    {key}: {value}")
                print()

        # Print metrics separately
        if 'accuracy' in entry:
            print(f"  Performance Metrics:")
            print(f"    Accuracy: {entry['accuracy']:.2%}")
            print(f"    Average Purity: {entry['avg_purity']:.2%}")
            print(f"    Fragmentation: {entry['avg_fragmentation']:.2f} clusters per signer")
            print(f"    Adjusted Rand Index: {entry['adjusted_rand_index']:.4f}")
            print(f"    Normalized MI: {entry['normalized_mi']:.4f}")
            print(f"    Num Clusters: {entry['num_clusters']}")

            if 'singleton_count' in entry:
                print(f"    Singleton Clusters: {entry['singleton_count']}")
                print(f"    Singleton Accuracy: {entry['singleton_accuracy']:.2%}")

            if 'cluster_mean_size' in entry:
                print(f"    Avg Cluster Size: {entry['cluster_mean_size']:.1f}")
                print(f"    Min/Max Size: {entry['cluster_min_size']}/{entry['cluster_max_size']}")

    # Save detailed CSV with ALL configurations
    csv_file = os.path.join(results_dir, f'all_configurations_ranking.csv')

    try:
        # Convert to DataFrame
        df = pd.DataFrame(ranking_data)

        # Rearrange columns for better readability
        # Metrics columns to place first
        first_cols = [
            'score', 'accuracy', 'avg_purity', 'avg_fragmentation',
            'adjusted_rand_index', 'normalized_mi', 'num_clusters',
            'singleton_count', 'singleton_accuracy',
            'cluster_mean_size', 'cluster_min_size', 'cluster_max_size'
        ]

        # Get all remaining columns except metadata
        other_cols = [col for col in df.columns if col not in first_cols + ['idx']]

        # Create a list of all columns in desired order
        all_cols = [col for col in first_cols if col in df.columns] + other_cols

        # Rearrange columns
        df = df[all_cols]

        # Format percentages for readability
        if 'accuracy' in df:
            df['accuracy'] = df['accuracy'].apply(lambda x: f"{x:.4f}")
        if 'avg_purity' in df:
            df['avg_purity'] = df['avg_purity'].apply(lambda x: f"{x:.4f}")
        if 'singleton_accuracy' in df:
            df['singleton_accuracy'] = df['singleton_accuracy'].apply(lambda x: f"{x:.4f}")

        # Add rank column
        df.insert(0, 'rank', range(1, len(df) + 1))

        # Save as CSV
        df.to_csv(csv_file, index=False)
        print(f"\nALL configurations saved to CSV: {csv_file}")
    except Exception as e:
        print(f"\nError saving CSV: {e}")
        # Fallback to simpler CSV format if pandas fails
        try:
            with open(csv_file, 'w') as f:
                # Write header
                header = ['rank', 'score', 'accuracy', 'avg_purity', 'avg_fragmentation']
                f.write(','.join(header) + '\n')

                # Write each config
                for i, entry in enumerate(ranking_data):
                    f.write(f"{i+1},{entry.get('score', '')},{entry.get('accuracy', '')},"
                           f"{entry.get('avg_purity', '')},{entry.get('avg_fragmentation', '')}\n")

            print(f"Simplified CSV saved to: {csv_file}")
        except:
            print("Failed to save CSV in any format.")

    # Save comprehensive text summary
    summary_file = os.path.join(results_dir, f'optimization_summary.txt')

    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" SIGNATURE CLUSTERING OPTIMIZATION SUMMARY ".center(80, "=") + "\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total configurations evaluated: {len(ranking_data)}\n")
        f.write(f"Optimization objective: Maximize clustering quality with balanced fragmentation\n\n")

        # Write overall statistics
        f.write("OPTIMIZATION STATISTICS\n")
        f.write("-" * 30 + "\n")
        scores = [entry['score'] for entry in ranking_data]
        f.write(f"Score range: {min(scores):.4f} to {max(scores):.4f}\n")
        f.write(f"Mean score: {np.mean(scores):.4f}\n")
        f.write(f"Median score: {np.median(scores):.4f}\n\n")

        # Write best configuration details
        f.write("BEST CONFIGURATION\n")
        f.write("-" * 30 + "\n")

        best_entry = ranking_data[0]

        # Group parameters by category for better readability
        param_categories = {
            'Preprocessing': ['PREPROCESSING_METHOD', 'USE_CLAHE', 'CLAHE_CLIP_LIMIT', 
                           'DYNAMIC_PREPROCESSING', 'ADAPTIVE_THRESHOLD_C', 'ADAPTIVE_BLOCK_SIZE',
                           'MORPHOLOGICAL_OP', 'MORPHOLOGICAL_KERNEL_SIZE'],
            'Image': ['IMAGE_SIZE'],
            'Features': ['HU_WEIGHT', 'LBP_WEIGHT', 'HOG_WEIGHT', 'ZERNIKE_WEIGHT', 'GABOR_WEIGHT',
                       'STROKE_FEATURE_WEIGHT', 'USE_ENHANCED_LBP', 'USE_STROKE_FEATURES', 
                       'USE_ZERNIKE', 'USE_GABOR', 'USE_PCA_HOG', 'PCA_HOG_COMPONENTS'],
            'Normalization': ['NORMALIZE_FEATURES', 'NORMALIZE_METHOD'],
            'Distance': ['DISTANCE_METRIC', 'DISTANCE_THRESHOLD'],
            'Clustering': ['LINKAGE_METHOD', 'USE_TWO_STAGE', 'MERGE_THRESHOLD', 'MERGE_METHOD',
                         'CLUSTER_SPLIT_PERCENTILE', 'MIN_CLUSTER_SIZE', 'SINGLETON_HANDLING',
                         'SINGLETON_DETECTION_THRESHOLD', 'EXPECTED_SINGLETON_RATIO'],
            'Ensemble': ['USE_ENSEMBLE', 'HIERARCHICAL_WEIGHT_RATIO', 'SPECTRAL_N_CLUSTERS',
                        'SPECTRAL_AFFINITY'],
            'Adaptive': ['USE_ADAPTIVE_THRESHOLD', 'ADAPTIVE_PERCENTILE']
        }

        # Write parameters by category
        for category, param_keys in param_categories.items():
            category_params = [(key, best_entry[key]) for key in param_keys if key in best_entry]
            if category_params:
                f.write(f"\n{category} Parameters:\n")
                for key, value in category_params:
                    f.write(f"  {key}: {value}\n")

        # Write performance metrics
        f.write("\nPerformance:\n")
        f.write(f"Score: {best_entry['score']:.4f}\n")

        if 'accuracy' in best_entry:
            f.write(f"Accuracy: {best_entry['accuracy']:.2%}\n")
            f.write(f"Average Purity: {best_entry.get('avg_purity', 0):.2%}\n")
            f.write(f"Fragmentation: {best_entry.get('avg_fragmentation', 0):.2f} clusters per signer\n")
            f.write(f"Adjusted Rand Index: {best_entry.get('adjusted_rand_index', 0):.4f}\n")
            f.write(f"Normalized Mutual Information: {best_entry.get('normalized_mi', 0):.4f}\n")
            f.write(f"Number of Clusters: {best_entry.get('num_clusters', 0)}\n")

            if 'singleton_count' in best_entry:
                f.write(f"Singleton Clusters: {best_entry.get('singleton_count', 0)}\n")
                f.write(f"Singleton Accuracy: {best_entry.get('singleton_accuracy', 0):.2%}\n")

            if 'cluster_mean_size' in best_entry:
                f.write(f"Average Cluster Size: {best_entry.get('cluster_mean_size', 0):.1f}\n")
                f.write(f"Min Cluster Size: {best_entry.get('cluster_min_size', 0)}\n")
                f.write(f"Max Cluster Size: {best_entry.get('cluster_max_size', 0)}\n")

        # Write top configurations in rank order
        f.write("\n\nALL CONFIGURATIONS RANKED\n")
        f.write("=" * 50 + "\n\n")

        for rank, entry in enumerate(ranking_data[:30]):  # Limit to top 30 for readability
            f.write(f"Rank {rank+1}: Score = {entry['score']:.4f}\n")
            f.write("-" * 50 + "\n")

            # Write key parameters by group
            for category, param_keys in param_categories.items():
                category_params = [(key, entry[key]) for key in param_keys if key in entry]
                if category_params:
                    f.write(f"\n{category} Parameters:\n")
                    for key, value in category_params:
                        f.write(f"  {key}: {value}\n")

            # Add metrics
            if 'accuracy' in entry:
                f.write(f"\nPerformance:\n")
                f.write(f"Accuracy: {entry['accuracy']:.2%}\n")
                f.write(f"Average Purity: {entry.get('avg_purity', 0):.2%}\n")
                f.write(f"Fragmentation: {entry.get('avg_fragmentation', 0):.2f} clusters per signer\n")
                f.write(f"Adjusted Rand Index: {entry.get('adjusted_rand_index', 0):.4f}\n")
                f.write(f"Normalized Mutual Information: {entry.get('normalized_mi', 0):.4f}\n")
                f.write(f"Number of Clusters: {entry.get('num_clusters', 0)}\n")

                if 'singleton_count' in entry:
                    f.write(f"Singleton Clusters: {entry.get('singleton_count', 0)}\n")
                    f.write(f"Singleton Accuracy: {entry.get('singleton_accuracy', 0):.2%}\n")

                if 'cluster_mean_size' in entry:
                    f.write(f"Average Cluster Size: {entry.get('cluster_mean_size', 0):.1f}\n")
                    f.write(f"Min Cluster Size: {entry.get('cluster_min_size', 0)}\n")
                    f.write(f"Max Cluster Size: {entry.get('cluster_max_size', 0)}\n")

            f.write("\n")

        # Add parameter descriptions for reference
        f.write("\nPARAMETER DESCRIPTIONS\n")
        f.write("-" * 30 + "\n")
        f.write("PREPROCESSING_METHOD: Method for binarization ('otsu' or 'adaptive')\n")
        f.write("USE_CLAHE: Apply Contrast Limited Adaptive Histogram Equalization\n")
        f.write("DYNAMIC_PREPROCESSING: Dynamically adjust preprocessing based on image properties\n")
        f.write("MORPHOLOGICAL_OP: Morphological operation type ('open', 'close', etc.)\n")
        f.write("HU_WEIGHT: Weight for shape features\n")
        f.write("LBP_WEIGHT: Weight for texture features\n")
        f.write("HOG_WEIGHT: Weight for gradient features\n")
        f.write("ZERNIKE_WEIGHT: Weight for rotation-invariant features\n")
        f.write("GABOR_WEIGHT: Weight for Gabor filter features\n")
        f.write("STROKE_FEATURE_WEIGHT: Weight for stroke analysis features\n")
        f.write("SINGLETON_HANDLING: Use special handling for singleton signatures\n")
        f.write("DISTANCE_THRESHOLD: Threshold for initial clustering\n")

    print(f"Detailed summary saved to: {summary_file}")

    # Create a comprehensive file with all configurations sorted by performance
    all_configs_file = os.path.join(results_dir, f'all_configurations_ranked.py')

    with open(all_configs_file, 'w') as f:
        f.write("# All configurations from optimization ranked by performance\n")
        f.write("# Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        f.write("# Format: Each dictionary is a complete configuration that can be used in test_configs\n")
        f.write("# Configurations are sorted from best to worst performance\n\n")
        f.write("all_configurations = [\n")

        # Parameters that affect clustering behavior (include these)
        clustering_params = set([
            'PREPROCESSING_METHOD', 'USE_CLAHE', 'CLAHE_CLIP_LIMIT', 'DYNAMIC_PREPROCESSING',
            'ADAPTIVE_THRESHOLD_C', 'ADAPTIVE_BLOCK_SIZE', 'MORPHOLOGICAL_OP', 'MORPHOLOGICAL_KERNEL_SIZE',
            'HU_WEIGHT', 'LBP_WEIGHT', 'HOG_WEIGHT', 'ZERNIKE_WEIGHT', 'GABOR_WEIGHT',
            'STROKE_FEATURE_WEIGHT', 'USE_ENHANCED_LBP', 'USE_STROKE_FEATURES', 
            'USE_ZERNIKE', 'USE_GABOR', 'USE_PCA_HOG', 'PCA_HOG_COMPONENTS',
            'DISTANCE_METRIC', 'DISTANCE_THRESHOLD', 'LINKAGE_METHOD',
            'USE_TWO_STAGE', 'MERGE_THRESHOLD', 'MERGE_METHOD', 'MIN_CLUSTER_SIZE',
            'CLUSTER_SPLIT_PERCENTILE', 'USE_ADAPTIVE_THRESHOLD', 'ADAPTIVE_PERCENTILE',
            'USE_ENSEMBLE', 'ENSEMBLE_METHODS', 'ENSEMBLE_WEIGHTS', 'SPECTRAL_N_CLUSTERS', 
            'SPECTRAL_AFFINITY', 'NORMALIZE_FEATURES', 'NORMALIZE_METHOD', 'IMAGE_SIZE',
            'HIERARCHICAL_WEIGHT_RATIO', 'SINGLETON_HANDLING', 'SINGLETON_DETECTION_THRESHOLD',
            'EXPECTED_SINGLETON_RATIO'
        ])

        # Parameters to exclude (non-clustering affecting)
        exclude_params = {
            'idx', 'score', 'accuracy', 'avg_purity', 'avg_fragmentation', 
            'adjusted_rand_index', 'normalized_mi', 'num_clusters', 'cluster_mean_size', 
            'cluster_min_size', 'cluster_max_size', 'UNSUPERVISED_RESULTS_DIR', 
            'SUPERVISED_RESULTS_DIR', 'OPTIMIZATION_RESULTS_DIR', 'METADATA_DIR',
            'SIGNATURES_DIR', 'FEEDBACK_FILE', 'COPY_IMAGES_TO_CLUSTERS', 
            'RESULTS_DIR', 'SAVE_RESULTS', 'SAVE_VISUALIZATIONS', 'DATA_LIMIT',
            'TESTING_ON_PRECLUSTERED_IMAGES', 'VALID_FILE_ENDINGS',
            'DETAILED_ANALYSIS', 'MAX_CLUSTERS_TO_VISUALIZE', 'MIN_CLUSTER_SIZE_TO_VIS',
            'singleton_count', 'singleton_accuracy'
        }

        # Write each configuration with its performance metrics as a comment
        for rank, entry in enumerate(ranking_data):
            # Add comment with rank and performance metrics
            f.write(f"    # Rank {rank+1}: Score={entry.get('score', 0):.4f}")
            if 'accuracy' in entry:
                f.write(f", Accuracy={entry.get('accuracy', 0):.2%}")
            if 'avg_purity' in entry:
                f.write(f", Purity={entry.get('avg_purity', 0):.2%}")
            if 'avg_fragmentation' in entry:
                f.write(f", Fragmentation={entry.get('avg_fragmentation', 0):.2f}")
            if 'singleton_count' in entry:
                f.write(f", Singletons={entry.get('singleton_count', 0)}")
            f.write("\n")

            # Start the configuration dictionary
            f.write("    {\n")

            # Add a name field with rank
            f.write(f"        'name': 'Rank_{rank+1}_Score_{entry.get('score', 0):.4f}',\n")

            # Determine which parameters to include based on feature flags

            # Create a set of parameters to skip based on configuration
            skip_params = set()

            # If Zernike moments are disabled, skip weight
            if 'USE_ZERNIKE' in entry and entry['USE_ZERNIKE'] is False:
                skip_params.add('ZERNIKE_WEIGHT')

            # If Gabor filters are disabled, skip weight
            if 'USE_GABOR' in entry and entry['USE_GABOR'] is False:
                skip_params.add('GABOR_WEIGHT')

            # If stroke features are disabled, skip weight
            if 'USE_STROKE_FEATURES' in entry and entry['USE_STROKE_FEATURES'] is False:
                skip_params.add('STROKE_FEATURE_WEIGHT')

            # If PCA for HOG is disabled, skip components
            if 'USE_PCA_HOG' in entry and entry['USE_PCA_HOG'] is False:
                skip_params.add('PCA_HOG_COMPONENTS')

            # If adaptive threshold is disabled, skip percentile
            if 'USE_ADAPTIVE_THRESHOLD' in entry and entry['USE_ADAPTIVE_THRESHOLD'] is False:
                skip_params.add('ADAPTIVE_PERCENTILE')

            # If singleton handling is disabled, skip related params
            if 'SINGLETON_HANDLING' in entry and entry['SINGLETON_HANDLING'] is False:
                skip_params.add('SINGLETON_DETECTION_THRESHOLD')
                skip_params.add('EXPECTED_SINGLETON_RATIO')

            # If two-stage clustering is disabled, skip related parameters
            if 'USE_TWO_STAGE' in entry and entry['USE_TWO_STAGE'] is False:
                skip_params.update(['MERGE_THRESHOLD', 'MERGE_METHOD', 'MIN_CLUSTER_SIZE', 'CLUSTER_SPLIT_PERCENTILE'])

            # If ensemble clustering is disabled, skip related parameters
            if 'USE_ENSEMBLE' in entry and entry['USE_ENSEMBLE'] is False:
                skip_params.update(['ENSEMBLE_METHODS', 'ENSEMBLE_WEIGHTS', 'SPECTRAL_N_CLUSTERS',
                                    'SPECTRAL_AFFINITY', 'HIERARCHICAL_WEIGHT_RATIO'])

            # If CLAHE is disabled, skip related parameters
            if 'USE_CLAHE' in entry and entry['USE_CLAHE'] is False:
                skip_params.add('CLAHE_CLIP_LIMIT')

            # Add all clustering-affecting parameters
            for key in sorted(entry.keys()):
                # Skip non-clustering parameters and metrics
                if key in exclude_params or key in skip_params:
                    continue

                # If we know it's a clustering parameter, include it
                # Also include unknown parameters not in exclusion list to be safe
                if key in clustering_params or key not in exclude_params:
                    # Format the value based on type
                    if isinstance(entry[key], bool):
                        value = str(entry[key])
                    elif isinstance(entry[key], (int, float)):
                        if isinstance(entry[key], float):
                            value = f"{entry[key]:.6f}".rstrip('0').rstrip('.') if '.' in f"{entry[key]:.6f}" else f"{entry[key]:.1f}"
                        else:
                            value = str(entry[key])
                    elif isinstance(entry[key], list) or isinstance(entry[key], tuple):
                        if isinstance(entry[key][0], (int, float)):
                            value = str(entry[key])
                        else:
                            value = f"{entry[key]}"  # Let Python format it properly
                    else:
                        value = f"'{entry[key]}'"

                    f.write(f"        '{key}': {value},\n")

            # End this configuration
            f.write("    },\n\n")

        # Close the list
        f.write("]\n")

    print(f"All configurations ranked by performance saved to: {all_configs_file}")

    # Create plots of relationships between parameters
    plots_dir = os.path.join(results_dir, 'analysis_plots')
    os.makedirs(plots_dir, exist_ok=True)

    try:
        # Create DataFrame for plotting
        df = pd.DataFrame([{k: v for k, v in r['params'].items() if k in clustering_params} for r in all_results])
        # Add scores
        df['score'] = [r.get('score', 0) for r in all_results]

        if len(all_results) > 0 and 'metrics' in all_results[0]:
            # Add performance metrics
            df['accuracy'] = [r.get('metrics', {}).get('accuracy', 0) for r in all_results]
            df['avg_purity'] = [r.get('metrics', {}).get('avg_purity', 0) for r in all_results]
            df['avg_fragmentation'] = [r.get('metrics', {}).get('avg_fragmentation', 0) for r in all_results]

            # Create plot of accuracy vs. fragmentation
            plt.figure(figsize=(10, 8))
            plt.scatter(df['avg_fragmentation'], df['accuracy'], c=df['score'], cmap='viridis', alpha=0.7)
            plt.colorbar(label='Score')
            plt.xlabel('Fragmentation (clusters per signer)')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Fragmentation')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'accuracy_vs_fragmentation.png'))
            plt.close()

            # Create plot of accuracy vs. purity
            plt.figure(figsize=(10, 8))
            plt.scatter(df['avg_purity'], df['accuracy'], c=df['score'], cmap='viridis', alpha=0.7)
            plt.colorbar(label='Score')
            plt.xlabel('Purity')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs. Purity')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'accuracy_vs_purity.png'))
            plt.close()

            # Plot feature weight impact
            if 'HU_WEIGHT' in df and 'LBP_WEIGHT' in df:
                plt.figure(figsize=(10, 8))
                plt.scatter(df['HU_WEIGHT'], df['LBP_WEIGHT'], c=df['score'], cmap='viridis', alpha=0.7)
                plt.colorbar(label='Score')
                plt.xlabel('HU Weight')
                plt.ylabel('LBP Weight')
                plt.title('Feature Weight Relationship')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, 'feature_weights.png'))
                plt.close()

            # Plot threshold impact
            if 'DISTANCE_THRESHOLD' in df:
                plt.figure(figsize=(10, 8))
                plt.scatter(df['DISTANCE_THRESHOLD'], df['avg_fragmentation'], c=df['score'], cmap='viridis', alpha=0.7)
                plt.colorbar(label='Score')
                plt.xlabel('Distance Threshold')
                plt.ylabel('Fragmentation')
                plt.title('Threshold Impact on Fragmentation')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, 'threshold_impact.png'))
                plt.close()

            print(f"Created analysis plots in {plots_dir}")

    except Exception as e:
        print(f"Error creating analysis plots: {e}")

    # Display parameter diversity statistics
    param_counts = {}
    for entry in ranking_data:
        for key in entry.keys():
            if key not in exclude_params and key in clustering_params:
                if key not in param_counts:
                    param_counts[key] = set()
                param_counts[key].add(str(entry.get(key)))

    print("\nParameter diversity (unique values tested):")
    for key, values in sorted(param_counts.items()):
        print(f"  {key}: {len(values)} values")

    print("\nFiles generated:")
    print(f"1. Complete CSV: {csv_file}")
    print(f"2. Detailed summary: {summary_file}")
    print(f"3. All configs template: {all_configs_file}")
    print(f"4. Analysis plots: {plots_dir}")

    return results_dir


#=============================================================================
# CLUSTERING ALGORITHMS
#=============================================================================

class SignatureClustering:
    """Main clustering class with multi-stage and ensemble approaches."""

    def __init__(self, config):
        """Initialize clustering with configuration parameters."""
        self.config = config
        self.extractor = SignatureFeatureExtractor(config)

    def extract_features_batch(self, signatures):
        """Extract features from all signatures with separate feature types."""
        print("\nExtracting features from all signatures. This may take a while...")
        start_time = time.time()

        # Initialize feature arrays
        all_hu_moments = []
        all_lbp_hist = []
        all_hog_feats = []
        all_zernike_moments = []
        all_gabor_features = []
        all_stroke_features = []
        valid_signatures = []
        failed_signatures = []

        # Process each signature image with progress bar
        for signature in tqdm(signatures, total=len(signatures)):
            try:
                hu_moments, lbp_hist, hog_feats, zernike_moments, gabor_features, stroke_features = \
                    self.extractor.extract_features(signature)

                # Only include successfully processed images
                if hu_moments is not None:
                    all_hu_moments.append(hu_moments)
                    all_lbp_hist.append(lbp_hist)
                    all_hog_feats.append(hog_feats)

                    if zernike_moments is not None and self.config['USE_ZERNIKE']:
                        all_zernike_moments.append(zernike_moments)

                    if gabor_features is not None and self.config.get('USE_GABOR', False):
                        all_gabor_features.append(gabor_features)

                    if stroke_features is not None and self.config.get('USE_STROKE_FEATURES', False):
                        # Verify consistent shape before adding
                        if isinstance(stroke_features, np.ndarray) and stroke_features.shape == (10,):
                            all_stroke_features.append(stroke_features)
                        else:
                            print(f"Warning: Skipping malformed stroke features with shape {getattr(stroke_features, 'shape', 'unknown')}")

                    valid_signatures.append(signature)
                else:
                    failed_signatures.append(signature)
            except Exception as e:
                print(f"Error processing {signature}: {str(e)}")
                failed_signatures.append(signature)

        end_time = time.time()
        print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
        print(f"Successfully extracted features from {len(valid_signatures)} signatures")

        if failed_signatures:
            print(f"Failed to extract features from {len(failed_signatures)} signatures")

        # If using PCA for HOG, fit the PCA model
        if self.config.get('USE_PCA_HOG', False) and len(all_hog_feats) > 0:
            self.extractor.fit_pca_hog(all_hog_feats)
            # Apply PCA to HOG features
            all_hog_feats = [self.extractor.apply_pca_hog(hog) for hog in all_hog_feats]

        # Convert to numpy arrays and group by feature type
        feature_groups = {
            'hu': np.array(all_hu_moments) if all_hu_moments else np.array([]),
            'lbp': np.array(all_lbp_hist) if all_lbp_hist else np.array([]),
            'hog': np.array(all_hog_feats) if all_hog_feats else np.array([])
        }

        if all_zernike_moments and self.config['USE_ZERNIKE']:
            feature_groups['zernike'] = np.array(all_zernike_moments)

        if all_gabor_features and self.config.get('USE_GABOR', False):
            feature_groups['gabor'] = np.array(all_gabor_features)

        if all_stroke_features and self.config.get('USE_STROKE_FEATURES', False):
            # Verify all features have the same shape
            if len(set(f.shape for f in all_stroke_features)) == 1:
                feature_groups['stroke'] = np.array(all_stroke_features)
            else:
                print("Warning: Inconsistent stroke feature shapes detected, skipping stroke features")

        return feature_groups, valid_signatures

    def compute_distances(self, feature_groups, metric='correlation'):
        """Compute pairwise distances with feature-block normalization and proper weighting."""
        print(f"Computing pairwise distances using {metric} metric with block normalization and weighting...")

        # Apply normalization to each feature group separately
        normalized_groups = {}
        for name, features in feature_groups.items():
            if features.size == 0:
                continue  # Skip empty feature groups

            normalized_groups[name] = self.extractor.normalize_feature_group(
                features, method=self.config.get('NORMALIZE_METHOD', 'standard')
            )

        # Apply weights to normalized features
        weighted_features = []

        # Add weighted Hu moments
        if 'hu' in normalized_groups:
            weighted_features.append(normalized_groups['hu'] * self.config['HU_WEIGHT'])
            print(f"Hu moments: {normalized_groups['hu'].shape[1]} "
                f"dimensions, weight={self.config['HU_WEIGHT']}")

        # Add weighted LBP features
        if 'lbp' in normalized_groups:
            weighted_features.append(normalized_groups['lbp'] * self.config['LBP_WEIGHT'])
            print(f"LBP features: {normalized_groups['lbp'].shape[1]} "
                f"dimensions, weight={self.config['LBP_WEIGHT']}")

        # Add weighted HOG features
        if 'hog' in normalized_groups:
            weighted_features.append(normalized_groups['hog'] * self.config['HOG_WEIGHT'])
            print(f"HOG features: {normalized_groups['hog'].shape[1]} "
                f"dimensions, weight={self.config['HOG_WEIGHT']}")

        # Add weighted Zernike moments if available
        if 'zernike' in normalized_groups:
            weighted_features.append(normalized_groups['zernike'] * self.config['ZERNIKE_WEIGHT'])
            print(f"Zernike moments: {normalized_groups['zernike'].shape[1]} "
                f"dimensions, weight={self.config['ZERNIKE_WEIGHT']}")

        # Add weighted Gabor features if available
        if 'gabor' in normalized_groups:
            weighted_features.append(normalized_groups['gabor'] * self.config.get('GABOR_WEIGHT', 0.0))
            print(f"Gabor features: {normalized_groups['gabor'].shape[1]} "
                f"dimensions, weight={self.config.get('GABOR_WEIGHT', 0.0)}")

        # Add weighted stroke features if available
        if 'stroke' in normalized_groups:
            weighted_features.append(normalized_groups['stroke'] * self.config.get('STROKE_FEATURE_WEIGHT', 0.0))
            print(f"Stroke features: {normalized_groups['stroke'].shape[1]} "
                f"dimensions, weight={self.config.get('STROKE_FEATURE_WEIGHT', 0.0)}")

        # Check if we have any features
        if not weighted_features:
            raise ValueError("No features available for distance computation. Check feature extraction and parameters.")

        # Concatenate all weighted features
        combined_features = np.hstack(weighted_features)

        # Handle DistanceMetric-Normalization compatibility
        if metric in ['correlation', 'cosine'] and self.config.get('NORMALIZE_METHOD', 'standard') not in ['standard', 'l2']:
            print(f"Warning: {metric} distance works best with 'standard' or 'l2' normalization.")
        elif metric == 'euclidean' and self.config.get('NORMALIZE_METHOD', 'standard') not in ['robust', 'standard']:
            print(f"Warning: {metric} distance works best with 'robust' or 'standard' normalization.")
        elif metric == 'cityblock' and self.config.get('NORMALIZE_METHOD', 'standard') != 'l1':
            print(f"Warning: {metric} distance works best with 'l1' normalization.")

        # Compute distances
        try:
            dist_matrix = squareform(pdist(combined_features, metric=metric))
        except Exception as e:
            print(f"Error computing distance matrix with {metric} metric: {e}")
            print("Falling back to euclidean distance...")
            dist_matrix = squareform(pdist(combined_features, metric='euclidean'))

        # Report distance matrix properties
        print(f"Distance matrix shape: {dist_matrix.shape}")

        # Handle the case of only one signature
        if dist_matrix.shape[0] <= 1:
            print("WARNING: Only one signature found. Cannot compute pairwise distances.")
            min_distance = 0.0
            max_distance = 0.0
        else:
            # Find non-zero distances (ignoring diagonal)
            nonzero_mask = dist_matrix > 0
            np.fill_diagonal(nonzero_mask, False)  # Exclude diagonal
            nonzero_distances = dist_matrix[nonzero_mask]

            if nonzero_distances.size > 0:
                min_distance = np.min(nonzero_distances)
                max_distance = np.max(dist_matrix)

                # Print distance statistics
                percentiles = [10, 25, 50, 75, 90]
                percentile_values = np.percentile(nonzero_distances, percentiles)

                print(f"Distance Statistics:")
                print(f"  Min Distance: {min_distance:.4f}")
                print(f"  Max Distance: {max_distance:.4f}")
                print(f"  Mean Distance: {np.mean(nonzero_distances):.4f}")
                for p, v in zip(percentiles, percentile_values):
                    print(f"  {p}th Percentile: {v:.4f}")
            else:
                min_distance = 0.0
                max_distance = 0.0

        print(f"Minimum distance: {min_distance}")
        print(f"Maximum distance: {max_distance}")

        # If using adaptive threshold, compute it now
        if self.config.get('USE_ADAPTIVE_THRESHOLD', False) and dist_matrix.shape[0] > 1:
            # Compute threshold based on distribution of distances
            flat_distances = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
            adaptive_percentile = self.config.get('ADAPTIVE_PERCENTILE', 25)
            adaptive_threshold = np.percentile(flat_distances, adaptive_percentile)
            print(f"Using adaptive threshold: {adaptive_threshold:.4f} "
                f"(percentile={adaptive_percentile}%)")
            self.config['DISTANCE_THRESHOLD'] = adaptive_threshold

        # If using singleton handling, compute singleton detection threshold
        if self.config.get('SINGLETON_HANDLING', False):
            flat_distances = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
            singleton_threshold = self.config.get('SINGLETON_DETECTION_THRESHOLD', 0.5)

            # Check if singleton threshold is reasonable
            if singleton_threshold < min_distance or singleton_threshold > max_distance:
                adjusted_threshold = np.percentile(flat_distances, 75)  # Use 75th percentile as fallback
                print(f"Warning: Singleton threshold {singleton_threshold:.4f} outside distance range, adjusted to {adjusted_threshold:.4f}")
                self.config['SINGLETON_DETECTION_THRESHOLD'] = adjusted_threshold

        return dist_matrix, combined_features

    def perform_spectral_clustering(self, features, dist_matrix, n_clusters=None):
        """
        Perform spectral clustering on signature features.
        
        Args:
            features: Feature vectors for signatures
            dist_matrix: Pre-computed distance matrix
            n_clusters: Number of clusters or None for auto-determination
            
        Returns:
            Array of cluster labels
        """
        try:
            print(f"Performing spectral clustering...")

            # Get affinity parameter from config
            affinity = self.config.get('SPECTRAL_AFFINITY', 'rbf')

            # Determine number of clusters
            if n_clusters is None:
                if self.config.get('SPECTRAL_N_CLUSTERS', 'auto') == 'auto':
                    # Estimate using silhouette score over a range
                    max_clusters = min(30, len(features) // 5)  # Practical upper limit
                    best_n_clusters = 2  # Default minimum
                    best_score = -1

                    # Try range of cluster counts
                    for n in range(2, min(10, max_clusters) + 1):
                        # Create affinity matrix if needed
                        if affinity == 'precomputed':
                            # Convert distance matrix to similarity matrix
                            similarity = 1 - dist_matrix / np.max(dist_matrix)
                            spectral = SpectralClustering(
                                n_clusters=n,
                                affinity='precomputed',
                                random_state=42
                            )
                            try:
                                labels = spectral.fit_predict(similarity)
                            except Exception as e:
                                print(f"Error in spectral clustering with {n} clusters: {e}")
                                continue
                        else:
                            spectral = SpectralClustering(
                                n_clusters=n,
                                affinity=affinity,
                                random_state=42
                            )
                            try:
                                labels = spectral.fit_predict(features)
                            except Exception as e:
                                print(f"Error in spectral clustering with {n} clusters: {e}")
                                continue

                        # Only calculate silhouette if we have multiple clusters
                        if len(np.unique(labels)) > 1:
                            score = silhouette_score(features, labels)
                            if score > best_score:
                                best_score = score
                                best_n_clusters = n

                    n_clusters = best_n_clusters
                    print(f"Auto-determined optimal clusters: {n_clusters}")
                else:
                    # Use configured value
                    n_clusters = int(self.config.get('SPECTRAL_N_CLUSTERS', 5))

            # Perform spectral clustering
            if affinity == 'precomputed':
                # Convert distance matrix to similarity matrix
                similarity = 1 - dist_matrix / np.max(dist_matrix)
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                )
                try:
                    labels = spectral.fit_predict(similarity)
                except Exception as e:
                    print(f"Error in spectral clustering: {e}")
                    print("Falling back to hierarchical clustering")
                    # Fall back to simple clustering based on thresholding
                    labels = np.zeros(len(features), dtype=int)
                    return labels
            else:
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity=affinity,
                    random_state=42
                )
                try:
                    labels = spectral.fit_predict(features)
                except Exception as e:
                    print(f"Error in spectral clustering: {e}")
                    print("Falling back to hierarchical clustering")
                    # Fall back to simple clustering based on thresholding
                    labels = np.zeros(len(features), dtype=int)
                    return labels

            print(f"Spectral clustering created {len(np.unique(labels))} clusters")
            return labels

        except (np.linalg.LinAlgError, MemoryError, ValueError) as e:
            print(f"Error in spectral clustering: {e}")
            print("Falling back to hierarchical clustering")
            return np.zeros(len(features), dtype=int)  # Return dummy labels
        except Exception as e:
            print(f"Unexpected error in spectral clustering: {e}")
            return np.zeros(len(features), dtype=int)

    def cluster_hierarchical(self, dist_matrix, threshold, linkage_method='average'):
        """Perform hierarchical agglomerative clustering."""
        print(f"\nPerforming hierarchical clustering with {linkage_method} linkage...")

        # Handle the case of only one signature
        if dist_matrix.shape[0] <= 1:
            print("WARNING: Only one signature found. Creating a single cluster.")
            return np.array([0]), None  # Return a single cluster label and no linkage matrix

        # Handle potential zeros in the matrix diagonal
        np.fill_diagonal(dist_matrix, 0)

        # Ensure distance matrix is symmetric
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        # Convert to condensed form (upper triangular)
        try:
            condensed_dist = squareform(dist_matrix)
        except ValueError as e:
            print(f"Error converting distance matrix: {e}")
            # Force symmetry more aggressively
            dist_matrix = np.maximum(dist_matrix, dist_matrix.T)
            np.fill_diagonal(dist_matrix, 0)

            # Try again
            try:
                condensed_dist = squareform(dist_matrix)
            except ValueError:
                # If still fails, use a different approach
                print("Using alternative approach to convert distance matrix")
                n = dist_matrix.shape[0]
                condensed_dist = np.zeros(n * (n - 1) // 2)
                k = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        condensed_dist[k] = dist_matrix[i, j]
                        k += 1

        # Verify the condensed distance array
        if np.isnan(condensed_dist).any():
            print("Warning: NaN values detected in distance matrix, replacing with max distance")
            max_dist = np.nanmax(condensed_dist)
            condensed_dist = np.nan_to_num(condensed_dist, nan=max_dist)

        if np.isinf(condensed_dist).any():
            print("Warning: Infinite values detected in distance matrix, replacing with max distance")
            max_finite = np.max(condensed_dist[~np.isinf(condensed_dist)])
            condensed_dist = np.where(np.isinf(condensed_dist), max_finite * 2, condensed_dist)

        # Perform hierarchical clustering with the specified linkage method
        try:
            z = linkage(condensed_dist, method=linkage_method)

            # Verify linkage matrix has no NaN or Inf values
            if np.isnan(z).any() or np.isinf(z).any():
                print("Warning: NaN or Inf values in linkage matrix, attempting to fix")
                z = np.nan_to_num(z, nan=0.0, posinf=np.max(z[~np.isinf(z)]) * 2)
        except Exception as e:
            print(f"Error during linkage calculation: {e}")
            print("Falling back to 'average' linkage method...")
            try:
                # Try with different linkage method
                z = linkage(condensed_dist, method='average')
            except Exception as e2:
                print(f"Linkage calculation failed again: {e2}")
                # Create a simple clustering based on thresholding the distance matrix
                print("Creating clusters directly from distance matrix...")
                clusters = []
                assigned = set()

                for i in range(dist_matrix.shape[0]):
                    if i in assigned:
                        continue

                    # Start a new cluster
                    cluster = [i]
                    assigned.add(i)

                    # Find all points within threshold
                    for j in range(dist_matrix.shape[0]):
                        if j not in assigned and dist_matrix[i, j] <= threshold:
                            cluster.append(j)
                            assigned.add(j)

                    clusters.append(cluster)

                # Convert to labels
                labels = np.zeros(dist_matrix.shape[0], dtype=int)
                for i, cluster in enumerate(clusters):
                    for j in cluster:
                        labels[j] = i

                return labels, None

        # Special handling for singleton clusters if enabled
        if self.config.get('SINGLETON_HANDLING', False):
            # Extract clusters using standard threshold
            labels_standard = fcluster(z, threshold, criterion='distance') - 1

            # Get singleton threshold from config
            singleton_threshold = self.config.get('SINGLETON_DETECTION_THRESHOLD', 0.5)

            # Identify potential singletons by checking if all distances are above threshold
            potential_singletons = []
            for i in range(dist_matrix.shape[0]):
                # Check if all distances (except to self) are above singleton threshold
                if np.all(dist_matrix[i, :i] >= singleton_threshold) and np.all(dist_matrix[i, i+1:] >= singleton_threshold):
                    potential_singletons.append(i)

            if potential_singletons:
                print(f"Identified {len(potential_singletons)} potential singleton signatures")

                # Create new labels with singletons
                labels = labels_standard.copy()
                next_label = np.max(labels) + 1

                # Assign new unique labels to singletons
                for idx in potential_singletons:
                    # Only break out if it's not already a singleton
                    if np.sum(labels == labels[idx]) > 1:
                        labels[idx] = next_label
                        next_label += 1
            else:
                labels = labels_standard
        else:
            # Standard clustering without singleton handling
            labels = fcluster(z, threshold, criterion='distance') - 1

        # Count clusters and return labels
        num_clusters = len(np.unique(labels))
        print(f"Found {num_clusters} clusters.")

        return labels, z

    def ensemble_clustering(self, features, dist_matrix, signatures):
        """
        Perform ensemble clustering using multiple methods and combine results.
        Currently supports hierarchical and spectral clustering.
        """
        try:
            print("\nPerforming ensemble clustering...")

            # Get methods and weights from config
            methods = self.config.get('ENSEMBLE_METHODS', ['hierarchical', 'spectral'])
            weights = self.config.get('ENSEMBLE_WEIGHTS', [0.7, 0.3])

            # Normalize weights
            weights = np.array(weights) / np.sum(weights)

            # Dictionary to store cluster probabilities
            cluster_probs = {}

            # Run each clustering method
            for method, weight in zip(methods, weights):
                if method == 'hierarchical':
                    # Hierarchical clustering
                    labels, _ = self.cluster_hierarchical(
                        dist_matrix.copy(),
                        self.config['DISTANCE_THRESHOLD'],
                        self.config['LINKAGE_METHOD']
                    )
                elif method == 'spectral':
                    # Spectral clustering
                    labels = self.perform_spectral_clustering(
                        features,
                        dist_matrix,
                        n_clusters=None  # Auto-determine
                    )
                else:
                    print(f"Unknown clustering method: {method}")
                    continue

                # Convert labels to co-occurrence matrix (probability of being in same cluster)
                n_samples = len(signatures)
                co_occurrence = np.zeros((n_samples, n_samples))

                for label in np.unique(labels):
                    # Get indices for this cluster
                    indices = np.where(labels == label)[0]
                    # Update co-occurrence matrix
                    for i in indices:
                        for j in indices:
                            co_occurrence[i, j] += weight

                # Add to overall probabilities
                cluster_probs[method] = co_occurrence

            # Combine probabilities
            combined_probs = np.zeros_like(dist_matrix)
            for method, weight in zip(methods, weights):
                combined_probs += cluster_probs[method] * weight

            # Convert probability matrix to distance matrix (invert)
            combined_dist = 1 - combined_probs
            np.fill_diagonal(combined_dist, 0)  # Zero out diagonal

            # Apply hierarchical clustering to combined distances
            final_labels, linkage_matrix = self.cluster_hierarchical(
                combined_dist,
                self.config['DISTANCE_THRESHOLD'],  # Use the configured threshold instead of fixed value
                self.config['LINKAGE_METHOD']
            )

            # Convert to clusters
            clusters = self.create_clusters_from_labels(final_labels, signatures)

            return clusters, combined_dist, linkage_matrix

        except Exception as e:

            print(f"ERROR in ensemble clustering: {e}")
            print("Falling back to standard hierarchical clustering")

            # Fallback to regular clustering
            labels, linkage_matrix = self.cluster_hierarchical(
                dist_matrix.copy(),
                self.config['DISTANCE_THRESHOLD'],
                self.config['LINKAGE_METHOD']
            )
            clusters = self.create_clusters_from_labels(labels, signatures)
            return clusters, dist_matrix, linkage_matrix


    def filter_empty_clusters(self, clusters):
        """Filter out any empty clusters."""
        original_count = len(clusters)
        filtered_clusters = {cluster_id: signatures for cluster_id, signatures in clusters.items() if signatures}
        removed_count = original_count - len(filtered_clusters)

        if removed_count > 0:
            print(f"Filtered out {removed_count} empty clusters")

        return filtered_clusters

    def two_stage_clustering(self, signatures, dist_matrix, feature_vectors=None):
        """
        Perform two-stage clustering:
        1. Initial clustering with conservative threshold
        2. Refine by splitting large clusters and merging similar ones
        """
        print("\nPerforming two-stage clustering...")

        # Stage 1: Initial clustering
        labels, linkage_matrix = self.cluster_hierarchical(
            dist_matrix.copy(),
            self.config['DISTANCE_THRESHOLD'],
            self.config['LINKAGE_METHOD']
        )

        # Convert labels to clusters
        clusters = self.create_clusters_from_labels(labels, signatures)
        print(f"Stage 1: Created {len(clusters)} initial clusters")

        # Stage 2a: Split oversized clusters
        split_clusters = self.split_oversized_clusters(
            clusters,
            dist_matrix,
            signatures,
            feature_vectors,
            self.config['CLUSTER_SPLIT_PERCENTILE']
        )
        print(f"Stage 2a: After splitting, have {len(split_clusters)} clusters")

        # Stage 2b: Merge similar clusters
        final_clusters = self.merge_similar_clusters(
            split_clusters,
            dist_matrix,
            signatures,
            self.config['MERGE_THRESHOLD']
        )
        print(f"Stage 2b: After merging, have {len(final_clusters)} clusters")

        return final_clusters, linkage_matrix

    def reorder_clusters_by_similarity(self, clusters, feature_groups, valid_signatures):
        """
        Reorder cluster IDs with strict adherence to configuration settings.
        """
        print("\nReordering clusters by similarity using exact configuration settings...")

        # Report active configuration settings being used
        print(f"Using distance metric: {self.config['DISTANCE_METRIC']}")
        print(f"Using linkage method: {self.config['LINKAGE_METHOD']}")
        print(f"Using feature weights: HU={self.config['HU_WEIGHT']:.2f}, " +
            f"LBP={self.config['LBP_WEIGHT']:.2f}, HOG={self.config['HOG_WEIGHT']:.2f}")

        # Skip if only one cluster or disabled in config
        if len(clusters) <= 1 or self.config.get('DISABLE_CLUSTER_REORDERING', False):
            return clusters

        # Create a mapping from paths to indices for quick lookup
        sig_to_index = {sig: i for i, sig in enumerate(valid_signatures)}

        # Compute cluster centroids for each feature group
        cluster_centroids = {}

        for cluster_id, signatures in clusters.items():
            # Get indices of signatures in this cluster
            indices = [sig_to_index[sig] for sig in signatures if sig in sig_to_index]

            if not indices:  # Skip empty clusters
                continue

            # Compute cluster centroid for each feature group
            cluster_feature_groups = {}
            for feature_name, feature_data in feature_groups.items():
                if feature_data.shape[0] > 0:  # Ensure we have features
                    cluster_feature_data = feature_data[indices]
                    # Apply the same normalization method as in the main algorithm
                    if self.config.get('NORMALIZE_FEATURES', True) and len(indices) > 1:
                        norm_method = self.config.get('NORMALIZE_METHOD', 'standard')
                        cluster_feature_data = self.extractor.normalize_feature_group(
                            cluster_feature_data,
                            method=norm_method
                        )
                    cluster_feature_groups[feature_name] = np.mean(cluster_feature_data, axis=0)

            cluster_centroids[cluster_id] = cluster_feature_groups

        # Generate weighted feature vectors for each cluster
        cluster_representatives = {}
        for cluster_id, centroid_groups in cluster_centroids.items():
            # When building weighted features, use config values directly:
            weighted_features = []
            if 'hu' in centroid_groups:
                weighted_features.append(centroid_groups['hu'].reshape(1, -1) * self.config['HU_WEIGHT'])
            if 'lbp' in centroid_groups:
                weighted_features.append(centroid_groups['lbp'].reshape(1, -1) * self.config['LBP_WEIGHT'])
            if 'hog' in centroid_groups:
                weighted_features.append(centroid_groups['hog'].reshape(1, -1) * self.config['HOG_WEIGHT'])
            if 'zernike' in centroid_groups and self.config['USE_ZERNIKE']:
                weighted_features.append(centroid_groups['zernike'].reshape(1, -1) * self.config['ZERNIKE_WEIGHT'])
            if 'gabor' in centroid_groups and self.config.get('USE_GABOR', False):
                weighted_features.append(centroid_groups['gabor'].reshape(1, -1) * self.config.get('GABOR_WEIGHT', 0.0))
            if 'stroke' in centroid_groups and self.config.get('USE_STROKE_FEATURES', False):
                weighted_features.append(centroid_groups['stroke'].reshape(1, -1) * self.config.get('STROKE_FEATURE_WEIGHT', 0.0))

            # Concatenate all weighted features
            if weighted_features:
                cluster_representatives[cluster_id] = np.hstack(weighted_features)

        # Build distance matrix between cluster representatives using configured distance metric
        cluster_ids = list(cluster_representatives.keys())
        distance_matrix = np.zeros((len(cluster_ids), len(cluster_ids)))

        for i, id1 in enumerate(cluster_ids):
            for j, id2 in enumerate(cluster_ids):
                if i == j:
                    continue

                # Use pdist with the configured distance metric
                combined = np.vstack([cluster_representatives[id1], cluster_representatives[id2]])
                try:
                    distance = pdist(combined, metric=self.config['DISTANCE_METRIC'])[0]
                    distance_matrix[i, j] = distance
                except Exception as e:
                    print(f"Error computing distance between clusters {id1} and {id2}: {e}")
                    # Use a default high distance in case of error
                    distance_matrix[i, j] = 1.0

        # Convert to condensed form
        try:
            condensed_dist = squareform(distance_matrix)
        except Exception as e:
            print(f"Error converting distance matrix to condensed form: {e}")
            # If ordering fails, return original clusters
            return clusters

        # Use the configured linkage method
        try:
            Z = linkage(condensed_dist, method=self.config['LINKAGE_METHOD'])
            # Get optimal leaf ordering
            ordered_indices = leaves_list(Z)
            ordered_cluster_ids = [cluster_ids[i] for i in ordered_indices]
        except Exception as e:
            print(f"Error in linkage or ordering: {e}")
            # If ordering fails, return original clusters
            return clusters

        # Create new cluster mapping with sequential IDs
        remapped_clusters = {}
        for new_id, old_id in enumerate(ordered_cluster_ids):
            remapped_clusters[new_id] = clusters[old_id]

        # Add any clusters that weren't included in the ordering
        next_id = len(remapped_clusters)
        for old_id, signatures in clusters.items():
            if old_id not in cluster_representatives:
                remapped_clusters[next_id] = signatures
                next_id += 1

        print(f"Reordered {len(clusters)} clusters using configuration {self.config.get('name', 'default')}")
        return remapped_clusters

    def cluster_signatures(self, signatures):
        """Main clustering method with single or two-stage approach."""

        print("\nClustering signatures with the following configuration:")
        print(f"- Distance metric: {self.config['DISTANCE_METRIC']}")
        print(f"- Linkage method: {self.config['LINKAGE_METHOD']}")
        print(f"- Feature weights: HU={self.config['HU_WEIGHT']}, LBP={self.config['LBP_WEIGHT']}, HOG={self.config['HOG_WEIGHT']}")
        print(f"- Distance threshold: {self.config['DISTANCE_THRESHOLD']}")
        print(f"- Merge threshold: {self.config['MERGE_THRESHOLD']}")
        print(f"- Using two-stage: {self.config['USE_TWO_STAGE']}")

        # Extract features
        feature_groups, valid_signatures = self.extract_features_batch(signatures)

        # Compute distance matrix
        dist_matrix, feature_vectors = self.compute_distances(feature_groups, self.config['DISTANCE_METRIC'])

        # Perform clustering based on configuration
        if self.config.get('USE_ENSEMBLE', False):
            # Ensemble clustering approach
            clusters, ensemble_dist, linkage_matrix = self.ensemble_clustering(
                feature_vectors, dist_matrix, valid_signatures
            )
            # Use ensemble distance matrix for analysis
            dist_matrix = ensemble_dist
        elif self.config['USE_TWO_STAGE']:
            # Two-stage clustering
            clusters, linkage_matrix = self.two_stage_clustering(valid_signatures, dist_matrix, feature_vectors)
        else:
            # Single-stage clustering
            labels, linkage_matrix = self.cluster_hierarchical(
                dist_matrix.copy(),
                self.config['DISTANCE_THRESHOLD'],
                self.config['LINKAGE_METHOD']
            )
            clusters = self.create_clusters_from_labels(labels, valid_signatures)

        # Filter out any empty clusters
        clusters = self.filter_empty_clusters(clusters)

        # Reorder clusters by similarity using already extracted features
        clusters = self.reorder_clusters_by_similarity(clusters, feature_groups, valid_signatures)

        return clusters, dist_matrix, feature_vectors, linkage_matrix, valid_signatures, feature_groups

    def copy_images_to_cluster_directories(self, clusters, source_dir, output_dir, cluster_depth):
        """Copy clustered images to output directories."""
        if not self.config['COPY_IMAGES_TO_CLUSTERS']:
            return

        print(f"\nCopying clustered images to {output_dir}...")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Calculate padding width based on total clusters
        num_clusters = len(clusters)
        padding_width = len(str(num_clusters - 1)) if num_clusters > 1 else 1

        # Determine how to organize the output
        if cluster_depth == 0 or cluster_depth == 'max':
            # Simple case: just organize by cluster
            for cluster_id, signatures in clusters.items():
                # Check if this is a numeric ID
                if isinstance(cluster_id, (int, float)) or (isinstance(cluster_id, str) and cluster_id.isdigit()):
                    # Format with padding
                    dir_name = f"cluster_{int(cluster_id):0{padding_width}d}"
                else:
                    # Non-numeric name - preserve it
                    dir_name = cluster_id

                # Create cluster directory
                cluster_dir = os.path.join(output_dir, dir_name)
                os.makedirs(cluster_dir, exist_ok=True)

                for sig_path in signatures:
                    # Generate a unique hash for the image
                    img_hash = hashlib.md5(os.path.abspath(sig_path).encode()).hexdigest()[:8]

                    # Create unique filename with hash
                    base_name = os.path.basename(sig_path)
                    name, ext = os.path.splitext(base_name)
                    unique_name = f"{name}_{img_hash}{ext}"

                    # Copy the file
                    shutil.copy2(sig_path, os.path.join(cluster_dir, unique_name))
        else:
            # Complex case: maintain directory structure up to cluster_depth
            for cluster_id, signatures in clusters.items():
                # Group signatures by their parent directories at cluster_depth
                parent_dirs = {}
                for sig_path in signatures:
                    rel_path = os.path.relpath(sig_path, source_dir)
                    parts = rel_path.split(os.sep)

                    if len(parts) <= cluster_depth:
                        # Signature is directly in a directory at cluster_depth
                        parent = os.path.dirname(rel_path)
                    else:
                        # Signature is in a subdirectory beyond cluster_depth
                        parent = os.path.join(*parts[:cluster_depth])

                    if parent not in parent_dirs:
                        parent_dirs[parent] = []
                    parent_dirs[parent].append(sig_path)

                # Create cluster directories for each parent
                for parent, sigs in parent_dirs.items():
                    # Create the parent directory structure
                    parent_output_dir = os.path.join(output_dir, parent)
                    # Create the cluster directory within the parent
                    cluster_dir = os.path.join(parent_output_dir, f"cluster_{cluster_id}")
                    os.makedirs(cluster_dir, exist_ok=True)

                    # Copy signatures to the cluster directory
                    for sig_path in sigs:
                        filename = os.path.basename(sig_path)
                        shutil.copy2(sig_path, os.path.join(cluster_dir, filename))

        print(f"Copied {sum(len(signatures) for signatures in clusters.values())} images to {len(clusters)} cluster directories.")

    def create_compact_feedback(self, metrics, config_name, feedback_file):
        """Create compact feedback file with essential information for LLM analysis."""
        if not self.config['SAVE_COMPACT_FEEDBACK']:
            return

        # Build the feedback content
        content = [
            f"CONFIGURATION: {config_name}",
            "=" * 50,
            "\nKEY PARAMETERS:",
            f"- HU_WEIGHT: {self.config['HU_WEIGHT']}",
            f"- LBP_WEIGHT: {self.config['LBP_WEIGHT']}",
            f"- HOG_WEIGHT: {self.config['HOG_WEIGHT']}",
            f"- ZERNIKE_WEIGHT: {self.config['ZERNIKE_WEIGHT']}",
            f"- GABOR_WEIGHT: {self.config.get('GABOR_WEIGHT', 0.0)}",
            f"- USE_ZERNIKE: {self.config['USE_ZERNIKE']}",
            f"- USE_GABOR: {self.config.get('USE_GABOR', False)}",
            f"- USE_ENHANCED_LBP: {self.config['USE_ENHANCED_LBP']}",
            f"- USE_PCA_HOG: {self.config.get('USE_PCA_HOG', False)}",
            f"- DISTANCE_THRESHOLD: {self.config['DISTANCE_THRESHOLD']}",
            f"- MERGE_THRESHOLD: {self.config['MERGE_THRESHOLD']}",
            f"- DISTANCE_METRIC: {self.config['DISTANCE_METRIC']}",
            f"- LINKAGE_METHOD: {self.config['LINKAGE_METHOD']}",
            f"- USE_TWO_STAGE: {self.config['USE_TWO_STAGE']}",
            f"- MERGE_METHOD: {self.config.get('MERGE_METHOD', 'average')}",
            f"- USE_ADAPTIVE_THRESHOLD: {self.config.get('USE_ADAPTIVE_THRESHOLD', False)}",
            f"- USE_ENSEMBLE: {self.config.get('USE_ENSEMBLE', False)}",
            f"- NORMALIZE_METHOD: {self.config.get('NORMALIZE_METHOD', 'standard')}",
        ]

        # Add metrics if testing on preclustered images
        if self.config['TESTING_ON_PRECLUSTERED_IMAGES'] and metrics:
            content.extend([
                "\nPERFORMANCE METRICS:",
                f"- Accuracy: {metrics['accuracy']:.4f}",
                f"- Average Purity: {metrics['avg_purity']:.4f}",
                f"- Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}",
                f"- Normalized Mutual Information: {metrics['normalized_mi']:.4f}",
                f"- Average Fragmentation: {metrics['avg_fragmentation']:.2f}",
                f"- Balanced Score: {metrics['balanced_score']:.4f}",
                f"- Number of Clusters: {metrics['num_clusters']}",
                f"- Total Signatures: {metrics['total_signatures']}",

                "\nCLUSTER SIZE DISTRIBUTION:",
                f"- Min: {metrics['cluster_sizes']['min']}",
                f"- Max: {metrics['cluster_sizes']['max']}",
                f"- Mean: {metrics['cluster_sizes']['mean']:.2f}",
                f"- Std Dev: {metrics['cluster_sizes']['std']:.2f}",
            ])
        elif not self.config['TESTING_ON_PRECLUSTERED_IMAGES']:
            content.extend([
                "\nUNCLUSTERED MODE:",
                "- Running on real data without ground truth",
                f"- CLUSTER_DIRECTORY_DEPTH: {self.config['CLUSTER_DIRECTORY_DEPTH']}",
                f"- COPY_IMAGES_TO_CLUSTERS: {self.config['COPY_IMAGES_TO_CLUSTERS']}"
            ])

        # Add the content to the feedback file
        with open(feedback_file, 'a') as f:
            f.write('\n'.join(content))
            f.write('\n\n' + '-' * 80 + '\n\n')

        print(f"Compact feedback appended to {feedback_file}")

    def process_cluster_pool(self, signatures, pool_id, output_dir=None, clustered_output_dir=None):
        """
        Process a single cluster pool.
        """
        print(f"\nProcessing cluster pool: {pool_id} with {len(signatures)} signatures")

        # Skip empty pools
        if not signatures:
            print("Empty pool, skipping.")
            return {}

        # Handle pools with only one signature
        if len(signatures) == 1:
            print("Pool with only one signature. Creating a single cluster.")
            return {0: signatures}  # Put the single signature in its own cluster

        # Perform clustering on this pool
        clusters, dist_matrix, feature_vectors, linkage_matrix, valid_signatures, feature_groups = self.cluster_signatures(signatures)

        # Create visualizations if output directory is provided
        if output_dir and self.config['SAVE_VISUALIZATIONS']:
            pool_name = os.path.basename(pool_id) if os.path.isdir(pool_id) else 'pool'
            vis_dir = os.path.join(output_dir, 'visualizations', pool_name)
            self.visualize_clusters(clusters, vis_dir)

            # Only visualize dendrogram if linkage_matrix is valid
            if linkage_matrix is not None and isinstance(linkage_matrix, np.ndarray) and linkage_matrix.size > 0:
                self.visualize_dendrogram(linkage_matrix, vis_dir)
            else:
                print("Skipping dendrogram visualization due to invalid linkage matrix")

        # Copy images to cluster directories if enabled
        if self.config['COPY_IMAGES_TO_CLUSTERS'] and clustered_output_dir:
            cluster_dir = clustered_output_dir

            # Determine the relative path within the source directory
            source_dir = self.config['SIGNATURES_DIR']
            rel_path = os.path.relpath(pool_id, source_dir)

            print(f"Writing clustered images to output directory.")
            print(f"Source pool: {pool_id}")
            print(f"Relative path: {rel_path}")

            if rel_path == '.':
                # Root directory case
                print(f"Writing to root output directory: {cluster_dir}")
                self.copy_images_to_cluster_directories(clusters, source_dir, cluster_dir, 0)
            else:
                # Subdirectory case
                pool_output_dir = os.path.join(cluster_dir, rel_path)
                print(f"Writing to subdirectory output: {pool_output_dir}")
                self.copy_images_to_cluster_directories(
                    clusters, pool_id, pool_output_dir, 0
                )
        else:
            if not self.config['COPY_IMAGES_TO_CLUSTERS']:
                print("Image copying disabled in configuration.")
            if not clustered_output_dir:
                print("No clustered output directory provided.")

        print(f"Completed processing pool: {pool_id} - Found {len(clusters)} clusters")
        return clusters

    def create_clusters_from_labels(self, labels, signatures):
        """Convert cluster labels to dictionary of signature lists."""
        clusters = {}
        for label, sig_path in zip(labels, signatures):
            # Skip noise points (DBSCAN can label some points as -1)
            if label >= 0:
                clusters.setdefault(label, []).append(sig_path)
        return clusters

    def split_oversized_clusters(self, clusters, dist_matrix, signatures, feature_vectors=None, percentile=90):
        """
        Split large clusters that might contain signatures from different people.
        Uses internal distance distribution and optional spectral clustering
        to identify potential splits.
        """
        print("\nSplitting oversized clusters...")

        min_size = self.config['MIN_CLUSTER_SIZE']
        new_clusters = {}
        next_cluster_id = max(clusters.keys()) + 1 if clusters else 0

        # Track clusters that were split
        split_count = 0

        # Process each cluster
        for cluster_id, cluster_signatures in clusters.items():
            # Skip small clusters
            if len(cluster_signatures) < min_size * 2:  # Only split significantly large clusters
                new_clusters[cluster_id] = cluster_signatures
                continue

            # Get indices of signatures in this cluster
            signature_indices = [signatures.index(sig) for sig in cluster_signatures]

            # Extract the distance submatrix for just this cluster
            cluster_dist = dist_matrix[np.ix_(signature_indices, signature_indices)]

            # Calculate internal distance statistics for this cluster
            internal_distances = cluster_dist[np.triu_indices(len(signature_indices), k=1)]

            # Skip clusters with no internal distances (should not happen)
            if len(internal_distances) == 0:
                new_clusters[cluster_id] = cluster_signatures
                continue

            # Get threshold based on percentile of internal distances
            split_threshold = np.percentile(internal_distances, percentile)

            # Skip if the cluster is already cohesive
            if split_threshold <= self.config['DISTANCE_THRESHOLD']:
                new_clusters[cluster_id] = cluster_signatures
                continue

            # Determine if we should use spectral clustering for very large clusters
            use_spectral = (len(cluster_signatures) >= min_size * 3 and
                           feature_vectors is not None and
                           self.config.get('USE_ENSEMBLE', False))

            if use_spectral:
                # Extract feature vectors for this cluster
                cluster_features = feature_vectors[signature_indices]

                # Determine number of clusters using silhouette score if 'auto'
                if self.config.get('SPECTRAL_N_CLUSTERS', 'auto') == 'auto':
                    max_n_clusters = min(len(cluster_signatures) // 2, 10)
                    best_score = -1
                    best_n_clusters = 2

                    # Try different numbers of clusters
                    for n_clusters in range(2, max_n_clusters + 1):
                        # Create affinity matrix based on config
                        spectral = SpectralClustering(
                            n_clusters=n_clusters,
                            affinity=self.config.get('SPECTRAL_AFFINITY', 'rbf'),
                            random_state=42
                        )

                        try:
                            cluster_labels = spectral.fit_predict(cluster_features)

                            # Only calculate silhouette if we have multiple clusters
                            if len(np.unique(cluster_labels)) > 1:
                                score = silhouette_score(cluster_features, cluster_labels)
                                if score > best_score:
                                    best_score = score
                                    best_n_clusters = n_clusters
                        except Exception as e:
                            print(f"Error in spectral clustering with {n_clusters} clusters: {e}")
                            continue

                    n_clusters = best_n_clusters
                else:
                    n_clusters = int(self.config.get('SPECTRAL_N_CLUSTERS', 2))

                # Apply spectral clustering
                try:
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=self.config.get('SPECTRAL_AFFINITY', 'rbf'),
                        random_state=42
                    )
                    sub_labels = spectral.fit_predict(cluster_features)

                    print(f"  Spectral clustering split cluster {cluster_id} ({len(cluster_signatures)} items) "
                          f"into {len(np.unique(sub_labels))} subclusters")
                except Exception as e:
                    print(f"Error in spectral clustering: {e}")
                    print("  Falling back to hierarchical clustering")
                    # Fall back to hierarchical clustering
                    try:
                        condensed_dist = squareform(cluster_dist)
                        z = linkage(condensed_dist, method=self.config['LINKAGE_METHOD'])
                        sub_labels = fcluster(z, split_threshold, criterion='distance') - 1
                    except Exception as e2:
                        print(f"Hierarchical clustering also failed: {e2}")
                        print("  Keeping cluster intact")
                        new_clusters[cluster_id] = cluster_signatures
                        continue
            else:
                # Perform hierarchical clustering on just this cluster
                try:
                    condensed_dist = squareform(cluster_dist)
                    z = linkage(condensed_dist, method=self.config['LINKAGE_METHOD'])
                    sub_labels = fcluster(z, split_threshold, criterion='distance') - 1
                except Exception as e:
                    print(f"Error in hierarchical clustering of cluster {cluster_id}: {e}")
                    print("  Keeping cluster intact")
                    new_clusters[cluster_id] = cluster_signatures
                    continue

            # Check if splitting actually occurred
            unique_labels = np.unique(sub_labels)
            if len(unique_labels) > 1:
                split_count += 1
                print(f"  Splitting cluster {cluster_id} ({len(cluster_signatures)} items) "
                      f"into {len(unique_labels)} subclusters")

                # Create new clusters from split
                for label in unique_labels:
                    sub_indices = [i for i, l in enumerate(sub_labels) if l == label]
                    sub_signatures = [cluster_signatures[i] for i in sub_indices]

                    if label == 0:
                        # Keep the first subcluster with the original ID
                        new_clusters[cluster_id] = sub_signatures
                    else:
                        # Assign new IDs to the other subclusters
                        new_clusters[next_cluster_id] = sub_signatures
                        next_cluster_id += 1
            else:
                # No split occurred, keep the cluster as is
                new_clusters[cluster_id] = cluster_signatures

        print(f"  Split {split_count} oversized clusters into subclusters")
        return new_clusters

    def merge_similar_clusters(self, clusters, dist_matrix, signatures, merge_threshold):
        """
        Merge similar small clusters that likely belong to the same person.
        Uses various distance methods between clusters to identify potential merges.
        """
        print("\nMerging similar clusters...")

        # Calculate cluster centroids and indices
        cluster_indices = {}
        for cluster_id, cluster_signatures in clusters.items():
            indices = [signatures.index(sig) for sig in cluster_signatures]
            cluster_indices[cluster_id] = indices

        # Create mapping from original cluster IDs to merged cluster IDs
        merged_mapping = {cluster_id: cluster_id for cluster_id in clusters.keys()}

        # Start with smaller clusters for merging
        cluster_sizes = {cid: len(sigs) for cid, sigs in clusters.items()}
        sorted_clusters = sorted(cluster_indices.keys(), key=lambda x: cluster_sizes[x])

        # Get merge method from config
        merge_method = self.config.get('MERGE_METHOD', 'average')

        # Count of merges performed
        merge_count = 0

        # Compare all pairs of clusters
        for i, cluster1 in enumerate(sorted_clusters):
            if len(cluster_indices[cluster1]) == 0:
                continue

            # Use the already merged ID for this cluster
            current_merged_id = merged_mapping[cluster1]

            for cluster2 in sorted_clusters[i+1:]:
                if len(cluster_indices[cluster2]) == 0:
                    continue

                # Skip if already merged into the same cluster
                if merged_mapping[cluster2] == current_merged_id:
                    continue

                # Calculate inter-cluster distances
                indices1 = cluster_indices[cluster1]
                indices2 = cluster_indices[cluster2]

                # Extract the distance submatrix between the two clusters
                cross_dist = dist_matrix[np.ix_(indices1, indices2)]

                # Calculate distance between clusters based on the chosen method
                if merge_method == 'min':
                    # Minimum distance between any two points
                    dist = np.min(cross_dist)
                elif merge_method == 'max':
                    # Maximum distance between any two points
                    dist = np.max(cross_dist)
                elif merge_method == 'adaptive':
                    # Adaptive method based on cluster sizes
                    # For larger clusters, we use a more conservative approach
                    size1, size2 = len(indices1), len(indices2)
                    if size1 > 5 or size2 > 5:
                        # For larger clusters, use a stricter criterion
                        percentile_val = 25  # Use 25th percentile (stricter than min)
                        dist = np.percentile(cross_dist.flatten(), percentile_val)
                    else:
                        # For smaller clusters, use average distance
                        dist = np.mean(cross_dist)
                else:  # 'average' is the default
                    # Average distance between all pairs of points
                    dist = np.mean(cross_dist)

                # Adjust threshold based on the merge method
                adjusted_threshold = merge_threshold
                if merge_method == 'min':
                    # Can use a lower threshold for minimum distance
                    adjusted_threshold = merge_threshold * 0.8
                elif merge_method == 'max':
                    # Need a higher threshold for maximum distance
                    adjusted_threshold = merge_threshold * 1.2
                elif merge_method == 'adaptive':
                    # Keep the standard threshold for adaptive approach
                    pass

                # If distance is below threshold, merge the clusters
                if dist < adjusted_threshold:
                    old_merged_id = merged_mapping[cluster2]
                    merge_count += 1

                    # Update mapping for all clusters that were mapped to the second cluster
                    for cid, merged_id in merged_mapping.items():
                        if merged_id == old_merged_id:
                            merged_mapping[cid] = current_merged_id

        # Apply the merging
        merged_clusters = {}
        for cluster_id, sigs in clusters.items():
            merged_id = merged_mapping[cluster_id]
            if merged_id not in merged_clusters:
                merged_clusters[merged_id] = []
            merged_clusters[merged_id].extend(sigs)

        # Renumber clusters sequentially
        new_clusters = {}
        for i, (_, sigs) in enumerate(merged_clusters.items()):
            new_clusters[i] = sigs

        print(f"  Merged {merge_count} cluster pairs, reducing from {len(clusters)} to {len(new_clusters)} clusters")

        return new_clusters

    def evaluate_clustering(self, clusters, true_labels):
        """
        Evaluate clustering results with multiple metrics including singleton handling.
        
        Args:
            clusters: Dictionary with cluster labels as keys and lists of signatures as values
            true_labels: Dictionary with signature paths as keys and true labels as values
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating clustering results...")

        # Prepare true labels and predicted labels
        all_signatures = list(true_labels.keys())
        true_label_list = [true_labels.get(sig, 'unknown') for sig in all_signatures]

        # Create predicted labels list
        predicted_labels = [-1] * len(all_signatures)
        for cluster_id, cluster_signatures in clusters.items():
            for sig in cluster_signatures:
                if sig in all_signatures:
                    predicted_labels[all_signatures.index(sig)] = cluster_id

        # Remove any signatures not in clusters
        valid_indices = [i for i, label in enumerate(predicted_labels) if label != -1]
        true_label_filtered = [true_label_list[i] for i in valid_indices]
        predicted_label_filtered = [predicted_labels[i] for i in valid_indices]

        # Calculate purity (accuracy) of each cluster
        correct = 0
        total = 0
        purities = []

        for cluster in clusters.values():
            if not cluster:  # Skip empty clusters
                continue

            # Count occurrences of each label in the cluster
            label_counts = {}
            for sig in cluster:
                if sig in true_labels:
                    label = true_labels[sig]
                    label_counts[label] = label_counts.get(label, 0) + 1

            # Get most common label
            if label_counts:
                most_common_label = max(label_counts, key=label_counts.get)
                most_common_count = label_counts[most_common_label]

                # Calculate cluster purity
                cluster_size = len(cluster)
                purity = most_common_count / cluster_size
                purities.append(purity)

                # Add to overall counts
                correct += most_common_count
                total += cluster_size

        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0
        avg_purity = np.mean(purities) if purities else 0

        # Calculate comprehensive clustering metrics
        if len(true_label_filtered) > 1:
            try:
                adjusted_rand = adjusted_rand_score(true_label_filtered, predicted_label_filtered)
                normalized_mi = normalized_mutual_info_score(true_label_filtered, predicted_label_filtered)
            except Exception as e:
                print(f"Error calculating clustering metrics: {e}")
                adjusted_rand = 0
                normalized_mi = 0
        else:
            adjusted_rand = 0
            normalized_mi = 0

        # Calculate distribution of cluster sizes
        sizes = [len(cluster) for cluster in clusters.values()]

        if sizes:
            size_mean = np.mean(sizes)
            size_std = np.std(sizes)
            size_min = np.min(sizes)
            size_max = np.max(sizes)
        else:
            size_mean = size_std = size_min = size_max = 0

        # Calculate fragmentation - how many clusters per signer
        signer_to_clusters = {}
        for label, signature_list in clusters.items():
            for sig in signature_list:
                if sig in true_labels:
                    signer = true_labels[sig]
                    if signer not in signer_to_clusters:
                        signer_to_clusters[signer] = set()
                    signer_to_clusters[signer].add(label)

        # Calculate average number of clusters per signer
        fragmentations = [len(clusters) for signer, clusters in signer_to_clusters.items()]
        avg_fragmentation = np.mean(fragmentations) if fragmentations else 0

        # Calculate singleton analysis for datasets with many single-instance clusters
        singleton_count = 0
        singleton_signers = set()
        singleton_correct = 0

        # Identify singleton clusters
        for cluster_id, signatures in clusters.items():
            if len(signatures) == 1:
                singleton_count += 1
                sig = signatures[0]

                if sig in true_labels:
                    signer = true_labels[sig]
                    singleton_signers.add(signer)

                    # Check if this singleton is correctly separated
                    # A singleton is correct if there are no other signatures from the same signer
                    other_sigs_from_signer = [s for s, l in true_labels.items() if l == signer and s != sig]

                    if not other_sigs_from_signer:
                        singleton_correct += 1

        # Calculate singleton accuracy
        singleton_accuracy = singleton_correct / singleton_count if singleton_count > 0 else 1.0

        # Calculate a balanced metric that combines accuracy and fragmentation
        # The goal is to maximize accuracy while minimizing fragmentation
        ideal_fragmentation = 1.0  # Ideally one cluster per signer
        fragmentation_penalty = np.exp(-0.1 * (avg_fragmentation - ideal_fragmentation))
        balanced_score = accuracy * fragmentation_penalty

        # Printing results
        print(f"Clustering Accuracy: {accuracy:.2%}")
        print(f"Average Cluster Purity: {avg_purity:.2%}")
        print(f"Adjusted Rand Index: {adjusted_rand:.4f}")
        print(f"Normalized Mutual Information: {normalized_mi:.4f}")
        print(f"Cluster Size Distribution: min={size_min}, max={size_max}, "
              f"mean={size_mean:.1f}, std={size_std:.1f}")
        print(f"Average Fragmentation (clusters per signer): {avg_fragmentation:.2f}")
        print(f"Singleton Analysis: {singleton_count} clusters with 1 signature")
        if singleton_count > 0:
            print(f"Singleton Accuracy: {singleton_accuracy:.2%}")
        print(f"Balanced Score: {balanced_score:.4f}")

        # Additional detailed analysis
        if self.config.get('DETAILED_ANALYSIS', False):
            # Find problematic signatures (frequently misclassified)
            misclassifications = {}
            for cluster_id, sigs in clusters.items():
                labels = [true_labels.get(sig) for sig in sigs if sig in true_labels]
                if not labels:
                    continue

                # Get most common label using Counter
                most_common = Counter(labels).most_common(1)[0][0]

                for sig in sigs:
                    if sig in true_labels and true_labels[sig] != most_common:
                        misclassifications[sig] = (true_labels[sig], most_common)

            if misclassifications:
                print(f"\nFound {len(misclassifications)} misclassified signatures")

            # Analyze distribution of signer fragmentation
            frag_dist = Counter(fragmentations)
            print("\nFragmentation distribution:")
            for frag, count in sorted(frag_dist.items()):
                print(f"  {frag} clusters: {count} signers")

        metrics = {
            'accuracy': accuracy,
            'avg_purity': avg_purity,
            'adjusted_rand_index': adjusted_rand,
            'normalized_mi': normalized_mi,
            'avg_fragmentation': avg_fragmentation,
            'balanced_score': balanced_score,
            'cluster_sizes': {
                'mean': size_mean,
                'std': size_std,
                'min': size_min,
                'max': size_max
            },
            'num_clusters': len(clusters),
            'total_signatures': total,
            'singleton_count': singleton_count,
            'singleton_accuracy': singleton_accuracy
        }

        return metrics

    def visualize_clusters(self, clusters, output_dir):
        """
        Create enhanced visualizations of clustering results with improved layout.
        
        Args:
            clusters: Dictionary with cluster labels as keys and lists of signatures as values
            output_dir: Directory to save visualizations
        """
        if not self.config['SAVE_VISUALIZATIONS']:
            return

        print(f"\nCreating cluster visualizations in '{output_dir}'...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        if not clusters:
            print("No clusters to visualize.")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'No clusters to visualize',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=14)
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, 'no_clusters.png'))
            plt.close()
            return

        # Visualize distribution of cluster sizes
        plt.figure(figsize=(10, 6))
        sizes = [len(sigs) for sigs in clusters.values()]
        bins = np.arange(min(sizes), max(sizes) + 2) - 0.5
        plt.hist(sizes, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Cluster Size')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cluster Sizes')
        plt.grid(True, alpha=0.3)

        # Add detailed stats to the plot
        if sizes:
            stats_text = (f"Total Clusters: {len(sizes)}\n"
                        f"Mean Size: {np.mean(sizes):.1f}\n"
                        f"Median Size: {np.median(sizes):.1f}\n"
                        f"Min: {min(sizes)}, Max: {max(sizes)}\n"
                        f"Singletons: {sizes.count(1)} ({sizes.count(1)/len(sizes)*100:.1f}%)")
            plt.figtext(0.75, 0.75, stats_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'))
        plt.close()

        # Get clusters ordered by size
        cluster_sizes = [(label, len(sigs)) for label, sigs in clusters.items()]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)

        # Limit number of visualized clusters
        max_clusters = min(self.config['MAX_CLUSTERS_TO_VISUALIZE'], len(clusters))
        min_size = self.config['MIN_CLUSTER_SIZE_TO_VIS']

        # Calculate total clusters to visualize
        clusters_to_vis = sum(1 for _, size in cluster_sizes
                            if size >= min_size)
        print(f"Visualizing {min(max_clusters, clusters_to_vis)} clusters out of {len(clusters)}...")

        # Visualize selected clusters
        for i, (label, size) in enumerate(cluster_sizes):
            if i >= max_clusters or size < min_size:
                break

            signatures = clusters[label]

            # Limit number of signatures to visualize if very large cluster
            if size > 30:
                print(f"Limiting visualization for cluster {label} ({size} signatures) to first 30")
                vis_signatures = signatures[:30]
            else:
                vis_signatures = signatures

            # Calculate grid size - ensure it's reasonable based on cluster size
            grid_cols = min(5, int(np.ceil(np.sqrt(len(vis_signatures)))))
            grid_rows = int(np.ceil(len(vis_signatures) / grid_cols))

            # Create figure
            plt.figure(figsize=(3 * grid_cols, 3 * grid_rows))
            plt.suptitle(f'Cluster {label} - Size: {size}', fontsize=16)

            # Create a subplot for each signature
            for j, sig_path in enumerate(vis_signatures):
                try:
                    # Load original image
                    img = cv2.imread(sig_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        plt.subplot(grid_rows, grid_cols, j+1)
                        plt.text(0.5, 0.5, f'Failed to load\n{os.path.basename(sig_path)}',
                                horizontalalignment='center', verticalalignment='center')
                        plt.axis('off')
                        continue

                    # Get preprocessed version
                    if hasattr(self, 'extractor'):
                        # If we have an extractor, use it to preprocess
                        preprocessed = self.extractor.preprocess(img.copy())

                        # Create two subplots - original and preprocessed
                        plt.subplot(grid_rows, grid_cols, j+1)

                        # Create a figure with two side-by-side images
                        plt.subplots_adjust(hspace=0.4)

                        # Use pcolormesh for better rendering of binary images
                        plt.imshow(img, cmap='gray')

                        # Add small preprocessed version in corner
                        ax_inset = plt.axes([0.65, 0.65, 0.3, 0.3], frameon=True)
                        ax_inset.imshow(preprocessed, cmap='gray')
                        ax_inset.set_xticks([])
                        ax_inset.set_yticks([])
                        ax_inset.set_title('Processed', fontsize=8)

                        plt.title(os.path.basename(sig_path), fontsize=8)
                        plt.axis('off')
                    else:
                        # Just show the original
                        plt.subplot(grid_rows, grid_cols, j+1)
                        plt.imshow(img, cmap='gray')
                        plt.title(os.path.basename(sig_path), fontsize=8)
                        plt.axis('off')
                except Exception as e:
                    print(f"Error visualizing {sig_path}: {e}")
                    plt.subplot(grid_rows, grid_cols, j+1)
                    plt.text(0.5, 0.5, f'Error: {str(e)[:20]}...',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=8)
                    plt.axis('off')
                    continue

            plt.tight_layout()
            plt.subplots_adjust(top=0.92)

            # Create a descriptive but safe filename
            filename = f'cluster_{label}_size_{size}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=150)
            plt.close()

        print(f"Created visualizations for {min(max_clusters, i+1)} clusters in {output_dir}")

    def visualize_dendrogram(self, linkage_matrix, output_dir):
        """Visualize the clustering dendrogram with enhancements for readability."""
        if not self.config['SAVE_VISUALIZATIONS']:
            return

        # Skip if no linkage matrix is provided or it's invalid
        if linkage_matrix is None:
            print("No linkage matrix available, skipping dendrogram visualization.")
            return

        # Ensure linkage matrix is valid
        try:
            # Check if linkage matrix is in proper format and has proper shape
            if not isinstance(linkage_matrix, np.ndarray):
                print("Invalid linkage matrix format, skipping dendrogram visualization.")
                return

            # Must have at least 2 rows to create a dendrogram
            if linkage_matrix.shape[0] < 1:
                print("Linkage matrix too small, skipping dendrogram visualization.")
                return

            # Check data type
            if not np.issubdtype(linkage_matrix.dtype, np.floating):
                print(f"Linkage matrix has invalid data type {linkage_matrix.dtype}, skipping dendrogram.")
                return
        except Exception as e:
            print(f"Error validating linkage matrix: {e}")
            return

        print("Generating dendrogram visualization...")

        # Create enhanced dendrogram visualizations
        try:
            # Create a full dendrogram for smaller datasets
            if linkage_matrix.shape[0] < 100:
                plt.figure(figsize=(12, 8))
                plt.title('Full Hierarchical Clustering Dendrogram', fontsize=14)
                plt.xlabel('Sample index or Cluster ID', fontsize=12)
                plt.ylabel('Distance', fontsize=12)

                # Create full dendrogram
                dendrogram(
                    linkage_matrix,
                    leaf_rotation=90.,
                    leaf_font_size=8.,
                )

                threshold = self.config['DISTANCE_THRESHOLD']
                plt.axhline(y=threshold, color='r', linestyle='--', \
                        label=f'Threshold: {threshold:.2f}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, 'dendrogram_full.png'))
                plt.close()

            # Create a truncated version for better visualization with larger datasets
            plt.figure(figsize=(12, 8))
            plt.title('Condensed Hierarchical Clustering Dendrogram', fontsize=14)
            plt.xlabel('Cluster size', fontsize=12)
            plt.ylabel('Distance', fontsize=12)

            # Create condensed dendrogram
            dendrogram(
                linkage_matrix,
                truncate_mode='lastp',  # Show only the last p merged clusters
                p=min(30, linkage_matrix.shape[0]),  # Show at most 30 merged clusters
                leaf_rotation=90.,
                leaf_font_size=10.,
                show_contracted=True,  # Show contracted nodes as one
                above_threshold_color='grey',
            )

            threshold = self.config['DISTANCE_THRESHOLD']
            merge_threshold = self.config.get('MERGE_THRESHOLD', threshold)

            # Add threshold lines with legend
            plt.axhline(y=threshold, color='r', linestyle='--',
                    label=f'Distance Threshold: {threshold:.3f}')

            if self.config.get('USE_TWO_STAGE', False):
                plt.axhline(y=merge_threshold, color='g', linestyle='-.',
                        label=f'Merge Threshold: {merge_threshold:.3f}')

            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(os.path.join(output_dir, 'dendrogram.png'))
            plt.close()

            print(f"Dendrogram saved to {output_dir}")

        except Exception as e:
            print(f"Error generating dendrogram: {e}")
            traceback.print_exc()


class SignatureMetadataManager:
    """Manages metadata for semi-supervised signature clustering."""

    def __init__(self, config):
        """Initialize the metadata manager with dataset-specific metadata."""
        self.config = config

        # Get source directory and create a consistent ID for this dataset
        source_dir = config.get('SIGNATURES_DIR', 'unknown_source')
        source_dir_norm = os.path.normpath(os.path.abspath(source_dir))
        dir_hash = hashlib.md5(source_dir_norm.encode()).hexdigest()[:8]
        source_name = os.path.basename(source_dir_norm)
        sanitized_name = re.sub(r'[^\w\-\.]', '_', source_name)

        # Create metadata directory using new parameter
        metadata_dir = config.get('METADATA_DIR', 'results/metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        # Dataset-specific metadata file
        self.metadata_file = os.path.join(metadata_dir, f"{sanitized_name}_{dir_hash}_metadata.json")

        print(f"Using metadata file: {self.metadata_file}")
        self.metadata = self._load_or_create_metadata()

    def _load_or_create_metadata(self):
        """Load existing metadata or create a new structure."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "images": {},
            "clusters": {},
            "constraints": {
                "must_link": [],
                "cannot_link": []
            }
        }

    def determine_current_iteration(self):
        max_iteration = 1
        for img_data in self.metadata["images"].values():
            for history in img_data.get("clustering_history", []):
                max_iteration = max(max_iteration, history.get("iteration", 1))
        return max_iteration + 1

    def save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved to {self.metadata_file}")

    def save_clustering_metadata(self, config_dir):
        """Save metadata to the correct location in config directory."""
        metadata_file = os.path.join(config_dir, "clustering_metadata.json")

        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"DEBUG: Saved metadata to {metadata_file}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to save metadata: {e}")
            return False

    def generate_unique_id(self, img_path):
        """Generate a consistent unique ID for an image based on path hash."""
        try:
            # Normalize path
            normalized_path = os.path.abspath(img_path)

            # Generate a simple hash from the path - MUST MATCH UNSUPERVISED IMPLEMENTATION
            path_hash = hashlib.md5(normalized_path.encode()).hexdigest()[:8]

            # Base name without extension
            base_name = os.path.basename(normalized_path)
            name_part = os.path.splitext(base_name)[0]

            # Return a simple ID that matches the unsupervised implementation
            return f"{name_part}_{path_hash}"
        except Exception as e:
            print(f"ERROR: Failed to generate ID for {img_path}: {e}")
            return f"fallback_{hashlib.md5(str(img_path).encode()).hexdigest()[:8]}"

    def register_image(self, img_path, original_cluster=None):
        """Register an image in the metadata system with robust path handling."""
        # Normalize path
        normalized_path = os.path.normpath(os.path.abspath(img_path))

        # Check for existing registration by path
        path_to_id = {}
        for img_id, img_data in self.metadata["images"].items():
            norm_orig = os.path.normpath(os.path.abspath(img_data["original_path"]))
            norm_curr = os.path.normpath(os.path.abspath(img_data["current_path"]))
            path_to_id[norm_orig] = img_id
            path_to_id[norm_curr] = img_id

        # If path already registered, return existing ID
        if normalized_path in path_to_id:
            return path_to_id[normalized_path]

        # Generate new ID
        img_id = self.generate_unique_id(normalized_path)

        # Check if this ID already exists (shouldn't happen, but handle it)
        if img_id in self.metadata["images"]:
            # This is a duplicate ID situation - add a distinguisher
            base_id = img_id
            for i in range(1, 100):
                img_id = f"{base_id}_{i}"
                if img_id not in self.metadata["images"]:
                    break

        # Register the image
        self.metadata["images"][img_id] = {
            "original_path": normalized_path,
            "original_cluster": original_cluster,
            "current_path": normalized_path,
            "clustering_history": [],
            "human_feedback": {
                "removed_from": [],
                "manually_assigned_to": None
            }
        }

        return img_id

    def register_initial_clustering(self, clusters, iteration=1):
        """Register the results of initial clustering."""
        print("Registering initial clustering results in metadata...")

        # Track all registered image IDs
        all_registered_ids = set()

        # Process each cluster
        for cluster_id, signatures in tqdm(clusters.items()):
            # Create cluster entry if it doesn't exist
            if cluster_id not in self.metadata["clusters"]:
                self.metadata["clusters"][cluster_id] = {
                    "is_pure": False,
                    "verified_by_human": False,
                    "members": []
                }

            cluster_members = []

            # Process each signature in this cluster
            for sig_path in signatures:
                img_id = self.register_image(sig_path)
                all_registered_ids.add(img_id)

                # Update image metadata
                self.metadata["images"][img_id]["clustering_history"].append({
                    "iteration": iteration,
                    "cluster_id": cluster_id,
                    "confidence": 1.0  # Default confidence
                })

                # Add to current cluster's members
                cluster_members.append(img_id)

            # Update cluster members
            self.metadata["clusters"][cluster_id]["members"] = cluster_members

        self.save_metadata()
        print(f"Registered {len(all_registered_ids)} images across {len(clusters)} clusters")

    def get_cluster_filename(self, img_id_or_path, cluster_dir):
        """Get a unique filename for an image in a cluster directory with robust handling."""
        # Determine if input is an ID or path and handle accordingly
        if "/" in img_id_or_path or "\\" in img_id_or_path:  # It's a path
            # Register it if needed and get ID
            img_id = self.register_image(img_id_or_path)
        else:  # It's an ID
            img_id = img_id_or_path
            # Register if not found
            if img_id not in self.metadata["images"]:
                print(f"Warning: Image ID {img_id} not found in metadata. Creating placeholder entry.")
                self.metadata["images"][img_id] = {
                    "original_path": f"unknown_{img_id}.png",
                    "original_cluster": None,
                    "current_path": f"unknown_{img_id}.png",
                    "clustering_history": [],
                    "human_feedback": {"removed_from": [], "manually_assigned_to": None}
                }

        # Generate filename
        original_path = self.metadata["images"][img_id]["original_path"]
        name, ext = os.path.splitext(os.path.basename(original_path))

        # Ensure extension exists
        if not ext:
            ext = ".png"  # Default extension

        # Create unique name
        unique_name = f"{name}_{img_id[-8:]}{ext}"
        return os.path.join(cluster_dir, unique_name)

    def extract_constraints_from_human_feedback(self):
        """Extract must-link and cannot-link constraints from human feedback."""
        print("Extracting constraints from human feedback...")

        must_link = []
        cannot_link = []

        # 1. Process verified pure clusters for must-link constraints
        pure_clusters = {
            cid: data for cid, data in self.metadata["clusters"].items()
            if data["is_pure"] and data["verified_by_human"]
        }

        print(f"DEBUG: Found {len(pure_clusters)} clusters marked as pure and verified by human")
        for cid, data in list(pure_clusters.items())[:3]:  # Print first 3 for debugging
            print(f"  Cluster {cid}: {len(data['members'])} members, is_pure={data.get('is_pure')}, verified={data.get('verified_by_human')}")
            print(f"  First few members: {data['members'][:3] if data['members'] else 'empty'}")

        # Generate must-link constraints within pure clusters
        print("DEBUG: Generating must-link constraints...")
        for cluster_id, cluster_data in pure_clusters.items():
            members = cluster_data["members"]
            print(f"  Cluster {cluster_id}: Processing {len(members)} members for must-link constraints")
            for i in range(len(members)):
                for j in range(i+1, len(members)):
                    must_link.append([members[i], members[j]])
                    if len(must_link) == 1:  # Print just the first one as an example
                        print(f"  Added must-link: {members[i]} <-> {members[j]}")

        # 2. Generate cannot-link constraints for images that were separated
        print("DEBUG: Generating cannot-link constraints...")

        # Count images with removal data
        images_with_removed_data = 0
        for img_id, img_data in self.metadata["images"].items():
            if img_data["human_feedback"]["removed_from"]:
                images_with_removed_data += 1
                if images_with_removed_data <= 5:  # Example for first 5
                    print(f"  Image {img_id} was removed from: {img_data['human_feedback']['removed_from']}")

        print(f"  Found {images_with_removed_data} images with 'removed_from' data")

        # Find images that were removed from clusters
        removed_images = {}
        for img_id, img_data in self.metadata["images"].items():
            removed_from = img_data["human_feedback"]["removed_from"]
            if removed_from:
                for cluster_id in removed_from:
                    if cluster_id not in removed_images:
                        removed_images[cluster_id] = []
                    removed_images[cluster_id].append(img_id)

        print(f"  DEBUG: Found removed images from {len(removed_images)} clusters: {list(removed_images.keys())}")

        # For each cluster with removed images, create cannot-link constraints
        for cluster_id, removed_img_ids in removed_images.items():
            print(f"  Processing cluster {cluster_id} with {len(removed_img_ids)} removed images")

            if cluster_id in self.metadata["clusters"]:
                # Get images that remained in the cluster
                remaining_img_ids = self.metadata["clusters"][cluster_id]["members"]
                print(f"    Cluster has {len(remaining_img_ids)} remaining members")

                # Create cannot-link constraints between removed and remaining images
                pairs_created = 0
                for removed_id in removed_img_ids:
                    for remaining_id in remaining_img_ids:
                        cannot_link.append([removed_id, remaining_id])
                        pairs_created += 1

                print(f"    Created {pairs_created} cannot-links between removed and remaining images")

                # Create cannot-link constraints between images assigned to different clusters
                different_cluster_pairs = 0
                for i, img1 in enumerate(removed_img_ids):
                    img1_assigned = self.metadata["images"][img1]["human_feedback"].get("manually_assigned_to")
                    for j in range(i+1, len(removed_img_ids)):
                        img2 = removed_img_ids[j]
                        img2_assigned = self.metadata["images"][img2]["human_feedback"].get("manually_assigned_to")

                        # If they were assigned to different non-null clusters, add a cannot-link
                        if img1_assigned and img2_assigned and img1_assigned != img2_assigned:
                            cannot_link.append([img1, img2])
                            different_cluster_pairs += 1

                print(f"    Created {different_cluster_pairs} cannot-links between images assigned to different clusters")
            else:
                print(f"    Warning: Cluster {cluster_id} not found in current metadata")

        print(f"DEBUG: must_link count before returning: {len(must_link)}")
        print(f"DEBUG: cannot_link count before returning: {len(cannot_link)}")

        # Update the constraints in metadata
        self.metadata["constraints"]["must_link"] = must_link
        self.metadata["constraints"]["cannot_link"] = cannot_link

        print(f"Extracted {len(must_link)} must-link and {len(cannot_link)} cannot-link constraints")
        self.save_metadata()

        return must_link, cannot_link

    def detect_human_feedback(self, feedback_dir):
        """
        Detect human feedback by comparing original clusters with modified clusters.
        """
        print(f"\nDETECTING HUMAN FEEDBACK from {feedback_dir}")

        # Step 1: Get original clustering information from metadata
        print("DEBUG: Extracting original clustering from metadata")
        orig_clusters = defaultdict(list)
        img_id_to_orig_info = {}  # Map image IDs to original information

        for img_id, img_data in self.metadata.get('images', {}).items():
            if 'clustering_history' in img_data and img_data['clustering_history']:
                # Get last cluster assignment
                last_cluster = img_data['clustering_history'][-1].get('cluster_id')
                if last_cluster is not None:
                    orig_clusters[str(last_cluster)].append(img_id)
                    img_id_to_orig_info[img_id] = {
                        'cluster_id': last_cluster,
                        'orig_path': img_data.get('original_path', ''),
                        'current_path': img_data.get('current_path', '')
                    }

        print(f"DEBUG: Found {len(orig_clusters)} original clusters in metadata")

        # Step 2: Get current clustering from directory structure
        print(f"DEBUG: Scanning {feedback_dir} for current clustering")
        current_clusters = {}
        unclustered_images = []

        # Create a mapping from filename to metadata ID
        filename_to_id = {}
        for img_id, info in img_id_to_orig_info.items():
            # Get just the base filename without path
            orig_filename = os.path.basename(info['orig_path'])
            current_filename = os.path.basename(info['current_path'])

            # Map both original and current filenames to this ID
            if orig_filename:
                filename_to_id[orig_filename] = img_id
            if current_filename and current_filename != orig_filename:
                filename_to_id[current_filename] = img_id

            # Also map the base name without hash to this ID
            base_parts = orig_filename.split('_')
            if len(base_parts) >= 2:
                base_name = '_'.join(base_parts[:-1])  # Remove the hash part
                if base_name not in filename_to_id:  # Don't overwrite if already exists
                    filename_to_id[base_name] = img_id

        print(f"DEBUG: Created mapping with {len(filename_to_id)} filenames mapped to metadata IDs")

        # Identify unclustered images
        unclustered_img_ids = []
        for file in os.listdir(feedback_dir):
            file_path = os.path.join(feedback_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext)
                                            for ext in self.config['VALID_FILE_ENDINGS']):
                unclustered_images.append(file_path)
                print(f"DEBUG: Found unclustered image: {file}")

                # Get ID for this unclustered image
                if file in filename_to_id:
                    unclustered_img_ids.append(filename_to_id[file])
                else:
                    # Try to extract the base name
                    base_parts = file.split('_')
                    if len(base_parts) >= 2:
                        base_name = '_'.join(base_parts[:-1])
                        if base_name in filename_to_id:
                            unclustered_img_ids.append(filename_to_id[base_name])

        # Check each cluster directory
        for item in os.listdir(feedback_dir):
            dir_path = os.path.join(feedback_dir, item)
            if os.path.isdir(dir_path):
                # Use the user-provided directory name as the cluster ID
                cluster_id = item  # This can be any name the user chooses

                cluster_images = []
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path) and any(file.lower().endswith(ext)
                                                    for ext in self.config['VALID_FILE_ENDINGS']):
                        # Try to find the metadata ID for this file
                        img_id = None

                        # First try the exact filename
                        if file in filename_to_id:
                            img_id = filename_to_id[file]
                        else:
                            # Try to extract the base name
                            base_parts = file.split('_')
                            if len(base_parts) >= 2:
                                base_name = '_'.join(base_parts[:-1])  # Remove the hash part
                                if base_name in filename_to_id:
                                    img_id = filename_to_id[base_name]

                        # If still not found, generate a new ID
                        if img_id is None:
                            img_id = self.generate_unique_id(file_path)
                            print(f"DEBUG: Generated new ID {img_id} for {file}")

                        cluster_images.append(img_id)

                if cluster_images:
                    current_clusters[cluster_id] = cluster_images
                    print(f"DEBUG: Cluster {cluster_id} has {len(cluster_images)} images")

        # Step 3: Determine which images were moved/regrouped
        must_link = []
        cannot_link = []

        # Generate must-link constraints for images in the same cluster
        for cluster_id, img_ids in current_clusters.items():
            print(f"DEBUG: Processing cluster {cluster_id} for must-link constraints")
            for i in range(len(img_ids)):
                for j in range(i+1, len(img_ids)):
                    must_link.append((img_ids[i], img_ids[j]))
                    if len(must_link) == 1 or len(must_link) % 100 == 0:
                        print(f"DEBUG: Added must-link: {img_ids[i]} <-> {img_ids[j]}")

        # NEW: Generate cannot-link constraints by comparing original clusters with current state
        print("DEBUG: Generating cannot-link constraints from separated images")

        # Create a mapping from image ID to current cluster
        current_locations = {}
        for cluster_id, img_ids in current_clusters.items():
            for img_id in img_ids:
                current_locations[img_id] = cluster_id

        # Mark unclustered images
        for img_id in unclustered_img_ids:
            current_locations[img_id] = "UNCLUSTERED"

        # For each original cluster, check if images have been separated
        for orig_cluster_id, orig_img_ids in orig_clusters.items():
            print(f"DEBUG: Checking original cluster {orig_cluster_id} for separations")

            # Skip clusters with only one image
            if len(orig_img_ids) <= 1:
                continue

            # Check each pair of images in this original cluster
            for i in range(len(orig_img_ids)):
                img1_id = orig_img_ids[i]

                # Skip if not found in current locations (shouldn't happen but just in case)
                if img1_id not in current_locations:
                    continue

                for j in range(i+1, len(orig_img_ids)):
                    img2_id = orig_img_ids[j]

                    # Skip if not found in current locations
                    if img2_id not in current_locations:
                        continue

                    # If they're in different locations now
                    if current_locations[img1_id] != current_locations[img2_id]:
                        # Only create cannot-link if they're not both unclustered
                        if not (current_locations[img1_id] == "UNCLUSTERED" and
                            current_locations[img2_id] == "UNCLUSTERED"):
                            cannot_link.append((img1_id, img2_id))
                            print(f"DEBUG: Added cannot-link between {img1_id} and {img2_id}")
                            print(f"       (Originally together in cluster {orig_cluster_id}, now in {current_locations[img1_id]} and {current_locations[img2_id]})")

        print(f"DEBUG: Generated {len(must_link)} must-link and {len(cannot_link)} cannot-link constraints")

        return current_clusters, unclustered_images, must_link, cannot_link

    def _update_metadata_from_feedback(self, pure_clusters, rejected_images):
        """Update metadata based on detected human feedback with robust ID tracking."""
        print("Updating metadata based on human feedback...")

        # Create a mapping of basenames to their original IDs and clusters
        basename_to_info = {}
        for img_id, img_data in self.metadata["images"].items():
            if "clustering_history" in img_data and img_data["clustering_history"]:
                # Get the last known cluster assignment
                last_assignment = img_data["clustering_history"][-1]
                cluster_id = last_assignment.get("cluster_id")

                # Get the basename from the current path
                basename = os.path.basename(img_data.get("current_path", ""))
                if basename:
                    # Store both ID and cluster
                    if basename not in basename_to_info:
                        basename_to_info[basename] = []
                    basename_to_info[basename].append({
                        "img_id": img_id,
                        "cluster_id": cluster_id
                    })

        print(f"Built mapping with {len(basename_to_info)} unique basenames")

        # Process each rejected image
        rejected_count = 0
        for img_path in rejected_images:
            basename = os.path.basename(img_path)

            # Try to find this basename in our mapping
            if basename in basename_to_info:
                # If there's only one match, use it directly
                if len(basename_to_info[basename]) == 1:
                    match_info = basename_to_info[basename][0]
                    img_id = match_info["img_id"]
                    cluster_id = match_info["cluster_id"]

                    # Update this image's metadata
                    if img_id in self.metadata["images"]:
                        if cluster_id and cluster_id not in self.metadata["images"][img_id]["human_feedback"]["removed_from"]:
                            self.metadata["images"][img_id]["human_feedback"]["removed_from"].append(cluster_id)
                            print(f"Recorded that {basename} (ID: {img_id}) was removed from cluster {cluster_id}")
                            rejected_count += 1

                            # Remove from cluster members if present
                            if cluster_id in self.metadata["clusters"]:
                                if img_id in self.metadata["clusters"][cluster_id]["members"]:
                                    self.metadata["clusters"][cluster_id]["members"].remove(img_id)
                else:
                    # Multiple matches - need to disambiguate
                    # For now, just use the first one and warn
                    print(f"Warning: Multiple matches for {basename}, using first match")
                    match_info = basename_to_info[basename][0]
                    img_id = match_info["img_id"]
                    cluster_id = match_info["cluster_id"]

                    # Update metadata as above
                    if img_id in self.metadata["images"]:
                        if cluster_id and cluster_id not in self.metadata["images"][img_id]["human_feedback"]["removed_from"]:
                            self.metadata["images"][img_id]["human_feedback"]["removed_from"].append(cluster_id)
                            print(f"Recorded that {basename} (ID: {img_id}) was removed from cluster {cluster_id}")
                            rejected_count += 1

                            # Remove from cluster members if present
                            if cluster_id in self.metadata["clusters"]:
                                if img_id in self.metadata["clusters"][cluster_id]["members"]:
                                    self.metadata["clusters"][cluster_id]["members"].remove(img_id)
            else:
                # No match found - this is a new file
                print(f"No previous assignment found for {basename}")
                # Register it without any previous cluster info
                self.register_image(img_path)

        print(f"Updated metadata for {rejected_count} rejected images")


class SemiSupervisedSignatureClustering(SignatureClustering):
    """Extends signature clustering with semi-supervised learning from human feedback."""

    def __init__(self, config):
        """Initialize the semi-supervised clustering with configuration."""
        super().__init__(config)
        self.metadata_manager = SignatureMetadataManager(config)

    def analyze_pure_clusters(self, feature_groups, pure_clusters, valid_signatures):
        """
        Analyze pure clusters to learn optimal feature weights while respecting configuration.
        """
        print("\nAnalyzing pure clusters to refine feature weights...")

        # Get original weights from configuration - this is the source of truth
        original_weights = {
            'hu': self.config['HU_WEIGHT'],  # Use config directly, not extractor property
            'lbp': self.config['LBP_WEIGHT'],
            'hog': self.config['HOG_WEIGHT'],
            'zernike': self.config['ZERNIKE_WEIGHT'],
            'gabor': self.config.get('GABOR_WEIGHT', 0.0)
        }

        # Print original weights
        print("Original weights from configuration:")
        for feature, weight in original_weights.items():
            if feature in feature_groups:
                print(f"  - {feature}: {weight:.4f}")

        if not pure_clusters or self.config.get('SKIP_WEIGHT_LEARNING', False):
            print("Using original configuration weights without modification.")
            return original_weights

        # Create a mapping from paths to indices for quick lookup
        path_to_index = {sig: i for i, sig in enumerate(valid_signatures)}

        # Get image indices by cluster
        cluster_indices = {}

        for cluster_id, img_paths in pure_clusters.items():
            # Map paths directly to indices in the valid_signatures list
            indices = []
            for img_path in img_paths:
                # Try to find this path in valid_signatures
                if img_path in path_to_index:
                    indices.append(path_to_index[img_path])

            if indices:
                cluster_indices[cluster_id] = indices
                print(f"Cluster {cluster_id}: Found {len(indices)} of {len(img_paths)} images in feature set")

        # Initialize containers for intra-cluster and inter-cluster distances
        intra_distances = {feature: [] for feature in feature_groups.keys()}
        inter_distances = {feature: [] for feature in feature_groups.keys()}

        # For each feature type, calculate distances
        for feature_name, feature_data in feature_groups.items():
            # Skip if no features of this type
            if feature_data.shape[0] == 0:
                continue

            # For each cluster, calculate internal distances
            for cluster_id, indices in cluster_indices.items():
                if len(indices) > 1:
                    # Extract features for this cluster
                    cluster_features = feature_data[indices]

                    # Calculate pairwise distances within cluster
                    cluster_dist = pdist(cluster_features, metric=self.config['DISTANCE_METRIC'])
                    intra_distances[feature_name].extend(cluster_dist)

            # Calculate inter-cluster distances (between different clusters)
            all_clusters = list(cluster_indices.keys())
            for i in range(len(all_clusters)):
                for j in range(i+1, len(all_clusters)):
                    cluster1 = cluster_indices[all_clusters[i]]
                    cluster2 = cluster_indices[all_clusters[j]]

                    # Sample pairs from different clusters to keep computation manageable
                    max_samples = 5  # Limit samples for efficiency
                    for idx1 in cluster1[:min(max_samples, len(cluster1))]:
                        for idx2 in cluster2[:min(max_samples, len(cluster2))]:
                            dist = euclidean(feature_data[idx1], feature_data[idx2])
                            inter_distances[feature_name].append(dist)

        # Calculate discrimination power of each feature
        feature_scores = {}
        for feature_name in feature_groups.keys():
            if not intra_distances[feature_name] or not inter_distances[feature_name]:
                # Skip features with no distance data
                feature_scores[feature_name] = 1.0
                continue

            # Calculate mean intra-cluster distance (lower is better)
            mean_intra = np.mean(intra_distances[feature_name]) if intra_distances[feature_name] else float('inf')

            # Calculate mean inter-cluster distance (higher is better)
            mean_inter = np.mean(inter_distances[feature_name]) if inter_distances[feature_name] else 0.0

            # Discrimination score: higher is better
            if mean_intra == 0 or mean_intra == float('inf'):
                # Perfect intra-cluster similarity or no data
                score = 1.0  # Neutral score
            else:
                # Ratio of inter to intra distances
                score = mean_inter / mean_intra

            feature_scores[feature_name] = score

        print(f"Feature discrimination scores: {', '.join([f'{k}={v:.2f}' for k, v in feature_scores.items()])}")

        # Apply a very conservative adjustment, preserving configuration weights
        adjusted_weights = {}

        # Maximum adjustment percentage (1.0 = 100%)
        adjustment_bound = 0.30  # Only allow 30% adjustment from original weights

        for feature in feature_scores:
            if feature not in original_weights:
                adjusted_weights[feature] = original_weights.get(feature, 1.0)
                continue

            # Start with original weight from configuration
            base_weight = original_weights[feature]

            # Apply a very conservative adjustment
            relative_score = feature_scores[feature] / max(feature_scores.values()) if feature_scores else 0.5
            adjustment_factor = (relative_score - 0.5) * adjustment_bound + 1.0

            # Ensure adjustment is within bounds
            adjustment_factor = max(1.0 - adjustment_bound, min(1.0 + adjustment_bound, adjustment_factor))

            # Apply conservative adjustment
            adjusted_weights[feature] = base_weight * adjustment_factor

        # Print the learned weights
        print("Refined feature weights (conservative adjustment from configuration):")
        for feature, weight in adjusted_weights.items():
            if feature in original_weights:
                pct_change = (weight / original_weights[feature] - 1.0) * 100
                print(f"  - {feature}: {weight:.4f} ({pct_change:+.1f}% adjustment)")
            else:
                print(f"  - {feature}: {weight:.4f}")

        # Add config option to disable weight learning entirely
        if self.config.get('DISABLE_WEIGHT_LEARNING', False):
            print("Weight learning disabled by configuration. Using original weights.")
            return original_weights

        return adjusted_weights

    def apply_constraints_to_distance_matrix(self, dist_matrix, must_link, cannot_link, valid_signatures):
        """
        Modify distance matrix based on constraints with detailed tracking.
        """
        print("\nDEBUG: Applying constraints to distance matrix")

        # Create a mapping from paths to original image IDs using metadata
        path_to_id = {}
        id_to_index = {}

        print("DEBUG: Building path-to-ID mapping from metadata")
        # First build mapping from paths to IDs using metadata
        for img_id, img_data in self.metadata_manager.metadata.get('images', {}).items():
            orig_path = img_data.get('original_path', '')
            curr_path = img_data.get('current_path', '')

            if orig_path:
                path_to_id[os.path.normpath(orig_path)] = img_id
            if curr_path and curr_path != orig_path:
                path_to_id[os.path.normpath(curr_path)] = img_id

            # Also map the basename (both with and without hash)
            if orig_path:
                basename = os.path.basename(orig_path)
                path_to_id[basename] = img_id

                # Try without hash if it has one
                parts = basename.split('_')
                if len(parts) >= 2:
                    base_name = '_'.join(parts[:-1])
                    path_to_id[base_name] = img_id

        print(f"DEBUG: Built path-to-ID mapping with {len(path_to_id)} entries")

        # Now map to indices in the feature matrix
        for i, path in enumerate(valid_signatures):
            norm_path = os.path.normpath(os.path.abspath(path))
            basename = os.path.basename(path)

            # Try several ways to find a match
            if norm_path in path_to_id:
                id_to_index[path_to_id[norm_path]] = i
            elif basename in path_to_id:
                id_to_index[path_to_id[basename]] = i
            else:
                # Use the original ID generation method as fallback
                img_id = self.metadata_manager.generate_unique_id(path)
                id_to_index[img_id] = i

        print(f"DEBUG: Mapped {len(id_to_index)} image IDs to feature indices")

        # NOW add the debug code AFTER id_to_index is populated
        print(f"DEBUG: First few constraints to apply:")
        for i, (id1, id2) in enumerate(must_link[:5]):
            print(f"  • {id1} <-> {id2}")
            print(f"    Found in ID mapping: {id1 in id_to_index}, {id2 in id_to_index}")

        # Get constraint weight from config
        constraint_weight = self.config.get('CONSTRAINT_WEIGHT', 0.7)
        print(f"DEBUG: Using constraint weight: {constraint_weight}")

        # Apply must-link constraints (reduce distances)
        must_link_applied = 0
        for img1_id, img2_id in must_link:
            if img1_id in id_to_index and img2_id in id_to_index:
                i, j = id_to_index[img1_id], id_to_index[img2_id]

                # Reduce distance by constraint weight
                original_dist = dist_matrix[i, j]
                dist_matrix[i, j] *= (1 - constraint_weight)
                dist_matrix[j, i] = dist_matrix[i, j]  # Ensure symmetry

                must_link_applied += 1
                if must_link_applied <= 5 or must_link_applied % 100 == 0:
                    print(f"DEBUG: Applied must-link between {img1_id} and {img2_id}")
                    print(f"       Distance changed from {original_dist:.4f} to {dist_matrix[i, j]:.4f}")
            else:
                if must_link_applied < 5:
                    missing = []
                    if img1_id not in id_to_index:
                        missing.append(img1_id)
                    if img2_id not in id_to_index:
                        missing.append(img2_id)
                    print(f"WARNING: Couldn't apply must-link constraint - IDs not found: {', '.join(missing)}")

        # Apply cannot-link constraints (increase distances)
        cannot_link_applied = 0
        for img1_id, img2_id in cannot_link:
            if img1_id in id_to_index and img2_id in id_to_index:
                i, j = id_to_index[img1_id], id_to_index[img2_id]

                # Increase distance by constraint weight
                original_dist = dist_matrix[i, j]
                dist_matrix[i, j] = min(1.0, dist_matrix[i, j] + (1 - dist_matrix[i, j]) * constraint_weight)
                dist_matrix[j, i] = dist_matrix[i, j]  # Ensure symmetry

                cannot_link_applied += 1
                if cannot_link_applied <= 5 or cannot_link_applied % 50 == 0:
                    print(f"DEBUG: Applied cannot-link between {img1_id} and {img2_id}")
                    print(f"       Distance changed from {original_dist:.4f} to {dist_matrix[i, j]:.4f}")

        total_constraints = len(must_link) + len(cannot_link)
        applied_constraints = must_link_applied + cannot_link_applied

        if total_constraints > 0:
            success_rate = (applied_constraints / total_constraints) * 100
            print(f"DEBUG: Applied {applied_constraints} out of {total_constraints} constraints ({success_rate:.1f}%)")

            if success_rate < 70:
                print("\nWARNING: Low constraint application rate. Common causes:")
                print("1. Images have been renamed or modified, changing their hash")
                print("2. Some images in the constraints aren't included in the current feature set")
                print("3. Path mapping issues between original and current directories")
        else:
            print("DEBUG: No constraints to apply")

        return dist_matrix

    def merge_with_pure_clusters(self, new_clusters, pure_cluster_paths, valid_signatures):
        """
        Merge newly created clusters with existing pure clusters.
        """
        print("\nMerging new clusters with existing pure clusters...")

        # Start with pure clusters
        merged_clusters = {}
        for cluster_id, paths in pure_cluster_paths.items():
            # Skip empty clusters
            if not paths:
                continue
            merged_clusters[cluster_id] = paths

        # Keep track of signatures that have been assigned
        assigned_signatures = set()
        for paths in pure_cluster_paths.values():
            assigned_signatures.update(paths)

        # Process each new cluster
        next_id = 0
        used_ids = set(merged_clusters.keys())

        for _, signatures in new_clusters.items():
            # Skip empty clusters
            if not signatures:
                continue

            # Filter out already assigned signatures
            unassigned = [sig for sig in signatures if sig not in assigned_signatures]

            if not unassigned:
                # Skip if all signatures are already assigned
                continue

            # Find a unique numeric ID for new clusters
            while str(next_id) in used_ids:
                next_id += 1

            # Use string IDs for new clusters, not integers
            new_id = str(next_id)
            merged_clusters[new_id] = unassigned
            used_ids.add(new_id)
            assigned_signatures.update(unassigned)
            next_id += 1

        # Print statistics
        print(f"Merged clusters: {len(merged_clusters)} non-empty clusters with {len(assigned_signatures)} signatures")
        print(f"Final count: {len(merged_clusters)} non-empty clusters with {len(assigned_signatures)} total signatures")

        return merged_clusters


def get_latest_output_directory(base_dir, completed_cluster_dir):
    """Find the most recent iteration directory based on naming pattern."""
    # Define the pattern for iteration directories
    pattern = re.compile(f"{re.escape(completed_cluster_dir)}_iteration_(\d+)$")

    # Find all matching directories and their iteration numbers
    matching_dirs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path):
                match = pattern.match(item)
                if match:
                    iteration = int(match.group(1))
                    matching_dirs.append((full_path, iteration))

    # Return the highest iteration directory, or the base directory if none found
    if matching_dirs:
        # Sort by iteration number (second element in tuple)
        matching_dirs.sort(key=lambda x: x[1])
        return matching_dirs[-1][0]  # Return path of highest iteration

    # If no iteration directories found, return the original directory path
    return os.path.join(base_dir, completed_cluster_dir)


def perform_semi_supervised_clustering(config, first_pass_results=None):
    """
    Run semi-supervised clustering using human feedback and configuration.
    """
    print("\n" + "="*50)
    print(" SEMI-SUPERVISED SIGNATURE CLUSTERING ".center(50, "="))
    print("="*50 + "\n")

    # Initialize the semi-supervised clusterer
    clusterer = SemiSupervisedSignatureClustering(config)

    # Force-update extractor weights from config for consistency
    clusterer.extractor.hu_weight = config['HU_WEIGHT']
    clusterer.extractor.lbp_weight = config['LBP_WEIGHT']
    clusterer.extractor.hog_weight = config['HOG_WEIGHT']
    clusterer.extractor.zernike_weight = config['ZERNIKE_WEIGHT']
    clusterer.extractor.gabor_weight = config.get('GABOR_WEIGHT', 0.0)
    clusterer.extractor.use_enhanced_lbp = config['USE_ENHANCED_LBP']
    clusterer.extractor.use_zernike = config['USE_ZERNIKE']
    clusterer.extractor.use_gabor = config.get('USE_GABOR', False)

    # Define the database for accessing signatures
    database = SignatureDatabase(config['SIGNATURES_DIR'], config['VALID_FILE_ENDINGS'], config)

    # Create output directories with proper structure
    config_dir, clustered_output_dir = create_semi_supervised_output_dir(config)

    # STEP 1: Determine feedback directory
    if 'HUMAN_FEEDBACK_DIR' in config and config['HUMAN_FEEDBACK_DIR']:
        feedback_base_dir = config['HUMAN_FEEDBACK_DIR']
        print(f"DEBUG: Using human-specified feedback directory: {feedback_base_dir}")
    else:
        print("ERROR: HUMAN_FEEDBACK_DIR must be specified for semi-supervised mode")
        return None, None, None, None, None

    # STEP 2: Locate metadata file and clustered_signatures directory
    metadata_file = os.path.join(feedback_base_dir, "clustering_metadata.json")
    clustered_dir = os.path.join(feedback_base_dir, "clustered_signatures")

    print(f"DEBUG: Looking for metadata file at: {metadata_file}")
    print(f"DEBUG: Looking for clustered signatures at: {clustered_dir}")

    if not os.path.exists(metadata_file):
        print(f"ERROR: Metadata file not found at {metadata_file}")
        return None, None, None, None, None

    if not os.path.exists(clustered_dir):
        print(f"ERROR: Clustered signatures directory not found at {clustered_dir}")
        return None, None, None, None, None

    print(f"SUCCESS: Found metadata file and clustered directory")

    # STEP 3: Load metadata from file (replace current metadata tracking)
    print(f"DEBUG: Loading metadata from {metadata_file}")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            clusterer.metadata_manager.metadata = metadata
            print(f"DEBUG: Successfully loaded metadata with {len(metadata.get('images', {}))} images")
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return None, None, None, None, None

    # NEW: Check if we should process by pools
    if config['CLUSTER_DIRECTORY_DEPTH'] != 0 and config['CLUSTER_DIRECTORY_DEPTH'] != 'max':
        print(f"\nDEBUG: Using pool-based processing with depth {config['CLUSTER_DIRECTORY_DEPTH']}")

        # Get all cluster pools
        cluster_pools = database.get_cluster_pools()
        print(f"DEBUG: Found {len(cluster_pools)} cluster pools to process separately")

        # Container for combined results
        all_clusters = {}
        total_constraints_applied = 0
        all_valid_signatures = []
        linkage_matrix = None
        largest_pool_size = 0

        # Process each pool separately
        for pool_idx, (pool_id, pool_signatures) in enumerate(cluster_pools.items()):
            if not pool_signatures:
                print(f"DEBUG: Skipping empty pool {pool_id}")
                continue

            print(f"\nDEBUG: Processing pool {pool_idx+1}/{len(cluster_pools)}: {pool_id} with {len(pool_signatures)} signatures")

            # Extract features for just this pool
            pool_feature_groups, pool_valid_signatures = clusterer.extract_features_batch(pool_signatures)
            all_valid_signatures.extend(pool_valid_signatures)

            # Look for human feedback specific to this pool
            pool_relative_path = os.path.relpath(pool_id, config['SIGNATURES_DIR'])
            pool_feedback_dir = os.path.join(clustered_dir, pool_relative_path) if pool_relative_path != '.' else clustered_dir

            print(f"DEBUG: Looking for pool-specific feedback in {pool_feedback_dir}")

            # Try to detect human feedback for this pool
            if os.path.exists(pool_feedback_dir):
                pool_clusters, pool_unclustered, pool_must_link, pool_cannot_link = \
                    clusterer.metadata_manager.detect_human_feedback(pool_feedback_dir)

                print(f"DEBUG: Found {len(pool_clusters)} clusters, {len(pool_must_link)} must-link and {len(pool_cannot_link)} cannot-link constraints for this pool")

                # Analyze features and learn weights (similar to non-pool version)
                if pool_clusters:
                    learned_weights = clusterer.analyze_pure_clusters(pool_feature_groups, pool_clusters, pool_valid_signatures)
                    if not config.get('DISABLE_WEIGHT_LEARNING', False):
                        print("DEBUG: Applying learned weights to feature extractor for this pool")
                        clusterer.extractor.hu_weight = learned_weights.get('hu', clusterer.extractor.hu_weight)
                        clusterer.extractor.lbp_weight = learned_weights.get('lbp', clusterer.extractor.lbp_weight)
                        clusterer.extractor.hog_weight = learned_weights.get('hog', clusterer.extractor.hog_weight)
                        clusterer.extractor.zernike_weight = learned_weights.get('zernike', clusterer.extractor.zernike_weight)
                        clusterer.extractor.gabor_weight = learned_weights.get('gabor', clusterer.extractor.gabor_weight)

                # Compute distance matrix for this pool
                pool_dist_matrix, pool_feature_vectors = clusterer.compute_distances(pool_feature_groups, config['DISTANCE_METRIC'])

                # Apply constraints for this pool
                if pool_must_link or pool_cannot_link:
                    modified_dist = clusterer.apply_constraints_to_distance_matrix(
                        pool_dist_matrix.copy(), pool_must_link, pool_cannot_link, pool_valid_signatures
                    )
                    total_constraints_applied += len(pool_must_link) + len(pool_cannot_link)
                else:
                    modified_dist = pool_dist_matrix

                # Cluster this pool with constraints
                if config.get('USE_ENSEMBLE', False):
                    pool_clusters, _, pool_linkage_matrix = clusterer.ensemble_clustering(
                        pool_feature_vectors, modified_dist, pool_valid_signatures
                    )
                    # Save linkage matrix from largest pool
                    if len(pool_valid_signatures) > largest_pool_size:
                        linkage_matrix = pool_linkage_matrix
                        largest_pool_size = len(pool_valid_signatures)
                elif config['USE_TWO_STAGE']:
                    pool_clusters, pool_linkage_matrix = clusterer.two_stage_clustering(
                        pool_valid_signatures, modified_dist, pool_feature_vectors
                    )
                    # Save linkage matrix from largest pool
                    if len(pool_valid_signatures) > largest_pool_size:
                        linkage_matrix = pool_linkage_matrix
                        largest_pool_size = len(pool_valid_signatures)
                else:
                    labels, pool_linkage_matrix = clusterer.cluster_hierarchical(
                        modified_dist.copy(),
                        config['DISTANCE_THRESHOLD'],
                        config['LINKAGE_METHOD']
                    )
                    # Save linkage matrix from largest pool
                    if len(pool_valid_signatures) > largest_pool_size:
                        linkage_matrix = pool_linkage_matrix
                        largest_pool_size = len(pool_valid_signatures)
                    pool_clusters = clusterer.create_clusters_from_labels(labels, pool_valid_signatures)

                # Filter and sort clusters
                pool_clusters = clusterer.filter_empty_clusters(pool_clusters)
                pool_clusters = clusterer.reorder_clusters_by_similarity(
                    pool_clusters, pool_feature_groups, pool_valid_signatures
                )

                # Add to overall results with unique IDs
                for cluster_id, signatures in pool_clusters.items():
                    # Create a unique ID that includes the pool
                    unique_id = f"{os.path.basename(pool_id)}_{cluster_id}"
                    all_clusters[unique_id] = signatures

                # Create output directory for this pool
                pool_output_dir = os.path.join(clustered_output_dir, pool_relative_path) if pool_relative_path != '.' else clustered_output_dir
                os.makedirs(pool_output_dir, exist_ok=True)

                # Calculate padding for this pool
                num_clusters = len(pool_clusters)
                padding_width = len(str(num_clusters - 1)) if num_clusters > 1 else 1

                # Copy images to cluster directories for this pool
                if config['COPY_IMAGES_TO_CLUSTERS']:
                    print(f"DEBUG: Copying clustered images with {padding_width}-digit padding for {num_clusters} clusters")
                    for i, (_, signatures) in enumerate(sorted(pool_clusters.items())):
                        if not signatures:
                            continue

                        # Create cluster directory with proper naming
                        cluster_name = f"cluster_{i:0{padding_width}d}"
                        cluster_dir = os.path.join(pool_output_dir, cluster_name)
                        os.makedirs(cluster_dir, exist_ok=True)

                        # Copy signatures (similar to non-pool version)
                        for sig_path in signatures:
                            # Generate unique ID and filename
                            img_id = clusterer.metadata_manager.generate_unique_id(sig_path)
                            base_name = os.path.basename(sig_path)
                            name, ext = os.path.splitext(base_name)
                            unique_name = f"{name}_{img_id[-8:]}{ext}"

                            # Copy file and update metadata
                            try:
                                shutil.copy2(sig_path, os.path.join(cluster_dir, unique_name))

                                # Update metadata
                                if img_id in clusterer.metadata_manager.metadata['images']:
                                    img_data = clusterer.metadata_manager.metadata['images'][img_id]
                                    if 'clustering_history' not in img_data:
                                        img_data['clustering_history'] = []
                                    img_data['clustering_history'].append({
                                        'iteration': len(img_data.get('clustering_history', [])) + 1,
                                        'cluster_id': cluster_name,
                                        'confidence': 1.0
                                    })
                            except Exception as e:
                                print(f"ERROR: Failed to copy {sig_path}: {e}")
            else:
                print(f"DEBUG: No feedback directory found for pool {pool_id}, skipping")

        # Save metadata with all pools processed
        print("DEBUG: Saving updated metadata")
        clusterer.metadata_manager.save_clustering_metadata(config_dir)

        # For testing mode, evaluate overall results
        if config['TESTING_ON_PRECLUSTERED_IMAGES']:
            print("DEBUG: Evaluating clustering results")
            true_labels = database.get_true_labels()
            metrics = clusterer.evaluate_clustering(all_clusters, true_labels)

            # Create feedback
            create_semi_supervised_feedback(
                metrics,
                config.get('name', 'Semi-Supervised Clustering'),
                clusterer.metadata_manager,
                config['FEEDBACK_FILE']
            )

        print(f"\nDEBUG: Pool-based semi-supervised clustering complete")
        print(f"DEBUG: Results saved to {config_dir}")
        print(f"DEBUG: Applied {total_constraints_applied} constraints across {len(cluster_pools)} pools")

        return all_clusters, None, None, linkage_matrix, all_valid_signatures

    else:
        # ORIGINAL CODE - Use global processing when not using pools

        # STEP 4: Detect human feedback from directory structure
        print(f"DEBUG: Detecting human feedback from {clustered_dir}")
        current_clusters, unclustered_images, must_link, cannot_link = clusterer.metadata_manager.detect_human_feedback(clustered_dir)

        print(f"DEBUG: Found {len(current_clusters)} clusters after human feedback")
        print(f"DEBUG: Found {len(unclustered_images)} unclustered images")
        print(f"DEBUG: Generated {len(must_link)} must-link and {len(cannot_link)} cannot-link constraints")

        # STEP 5: Get all signatures including unclustered ones
        signatures = database.get_all_signatures(limit=config.get('DATA_LIMIT'))
        print(f"DEBUG: Loaded {len(signatures)} signatures from source directory")

        # STEP 6: Extract features from all signatures
        print("DEBUG: Extracting features from all signatures")
        feature_groups, valid_signatures = clusterer.extract_features_batch(signatures)
        print(f"DEBUG: Successfully extracted features from {len(valid_signatures)} valid signatures")

        # STEP 7: Learn from human feedback if possible
        if current_clusters:
            # Learn optimal feature weights from human-created clusters
            learned_weights = clusterer.analyze_pure_clusters(feature_groups, current_clusters, valid_signatures)

            # Apply learned weights if weight learning is enabled
            if not config.get('DISABLE_WEIGHT_LEARNING', False):
                print("DEBUG: Applying learned weights to feature extractor")
                clusterer.extractor.hu_weight = learned_weights.get('hu', clusterer.extractor.hu_weight)
                clusterer.extractor.lbp_weight = learned_weights.get('lbp', clusterer.extractor.lbp_weight)
                clusterer.extractor.hog_weight = learned_weights.get('hog', clusterer.extractor.hog_weight)
                clusterer.extractor.zernike_weight = learned_weights.get('zernike', clusterer.extractor.zernike_weight)
                clusterer.extractor.gabor_weight = learned_weights.get('gabor', clusterer.extractor.gabor_weight)

        # STEP 8: Compute distance matrix with the current weights
        print("DEBUG: Computing distance matrix")
        dist_matrix, feature_vectors = clusterer.compute_distances(feature_groups, config['DISTANCE_METRIC'])

        # STEP 9: Apply constraints to modify distances
        if must_link or cannot_link:
            print("DEBUG: Applying constraints to distance matrix")
            modified_dist = clusterer.apply_constraints_to_distance_matrix(
                dist_matrix.copy(), must_link, cannot_link, valid_signatures
            )
        else:
            modified_dist = dist_matrix

        # STEP 10: Perform clustering using constraints
        print("\nDEBUG: Performing semi-supervised clustering")
        if config.get('USE_ENSEMBLE', False):
            print("DEBUG: Using ensemble clustering with constraints")
            clusters, ensemble_dist, linkage_matrix = clusterer.ensemble_clustering(
                feature_vectors, modified_dist, valid_signatures
            )
        elif config['USE_TWO_STAGE']:
            print("DEBUG: Using two-stage clustering with constraints")
            print(f"DEBUG: DISTANCE_THRESHOLD: {config['DISTANCE_THRESHOLD']}")
            print(f"DEBUG: MERGE_THRESHOLD: {config['MERGE_THRESHOLD']}")
            clusters, linkage_matrix = clusterer.two_stage_clustering(
                valid_signatures, modified_dist, feature_vectors
            )
        else:
            print("DEBUG: Using single-stage clustering with constraints")
            labels, linkage_matrix = clusterer.cluster_hierarchical(
                modified_dist.copy(),
                config['DISTANCE_THRESHOLD'],
                config['LINKAGE_METHOD']
            )
            clusters = clusterer.create_clusters_from_labels(labels, valid_signatures)

        # STEP 11: Filter out any empty clusters
        clusters = clusterer.filter_empty_clusters(clusters)
        print(f"DEBUG: Found {len(clusters)} non-empty clusters after filtering")

        # STEP 12: Sort clusters by similarity
        print("DEBUG: Sorting clusters by similarity")
        sorted_clusters = clusterer.reorder_clusters_by_similarity(clusters, feature_groups, valid_signatures)

        num_clusters = len(sorted_clusters)
        padding_width = len(str(num_clusters - 1)) if num_clusters > 1 else 1

        # STEP 13: Copy images to output directories with sequential naming
        if config['COPY_IMAGES_TO_CLUSTERS']:
            print(f"DEBUG: Copying clustered images with {padding_width}-digit padding for {num_clusters} clusters")
            for i, (cluster_id, signatures) in enumerate(sorted(sorted_clusters.items())):
                # Skip empty clusters
                if not signatures:
                    continue

                # Use a display name that preserves user naming if possible
                if str(cluster_id) in clusterer.metadata_manager.metadata["clusters"]:
                    # Get existing display name if it exists
                    display_name = clusterer.metadata_manager.metadata["clusters"][str(cluster_id)].get(
                        "display_name", f"cluster_{i:0{padding_width}d}")
                else:
                    # Create new display name with proper padding
                    display_name = f"cluster_{i:0{padding_width}d}"

                # Create cluster directory with proper naming
                cluster_dir = os.path.join(clustered_output_dir, display_name)
                os.makedirs(cluster_dir, exist_ok=True)

                # Copy each signature to the cluster directory
                for sig_path in signatures:
                    # Generate unique ID and filename
                    img_id = clusterer.metadata_manager.generate_unique_id(sig_path)

                    # Create a unique filename with the hash included
                    base_name = os.path.basename(sig_path)
                    name, ext = os.path.splitext(base_name)
                    unique_name = f"{name}_{img_id[-8:]}{ext}"

                    # Copy the file
                    try:
                        shutil.copy2(sig_path, os.path.join(cluster_dir, unique_name))

                        # Update metadata
                        if img_id in clusterer.metadata_manager.metadata['images']:
                            img_data = clusterer.metadata_manager.metadata['images'][img_id]
                            if 'clustering_history' not in img_data:
                                img_data['clustering_history'] = []
                            img_data['clustering_history'].append({
                                'iteration': len(img_data.get('clustering_history', [])) + 1,
                                'cluster_id': display_name,  # Fixed: previously was cluster_name
                                'confidence': 1.0
                            })
                    except Exception as e:
                        print(f"ERROR: Failed to copy {sig_path}: {e}")

        # STEP 14: Save updated metadata
        print("DEBUG: Saving updated metadata")
        clusterer.metadata_manager.save_clustering_metadata(config_dir)

        # STEP 15: Evaluate results if in testing mode
        if config['TESTING_ON_PRECLUSTERED_IMAGES']:
            print("DEBUG: Evaluating clustering results")
            true_labels = database.get_true_labels()
            metrics = clusterer.evaluate_clustering(sorted_clusters, true_labels)

            # Create feedback
            create_semi_supervised_feedback(
                metrics,
                config.get('name', 'Semi-Supervised Clustering'),
                clusterer.metadata_manager,
                config['FEEDBACK_FILE']
            )

        print(f"\nDEBUG: Semi-supervised clustering complete")
        print(f"DEBUG: Results saved to {config_dir}")

        return sorted_clusters, modified_dist, feature_vectors, linkage_matrix, valid_signatures


def create_semi_supervised_output_dir(config):
    """Create proper output directory structure for semi-supervised clustering."""
    base_dir = config.get('SEMI_SUPERVISED_RESULTS_DIR', 'results/semi_supervised')

    # Check if this is already a specific config directory
    if "configuration_summaries" in base_dir:
        # We're already at the configuration level, just add clustered_signatures
        config_dir = base_dir
    else:
        # Create the full path structure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(base_dir, f"semi_supervised_batch_{timestamp}")

        # Create configuration-specific directory
        config_name = config.get('name', 'default_config')
        config_dir = os.path.join(batch_dir, "configuration_summaries", config_name)

    # Make sure directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Create clustered_signatures directory
    clustered_dir = os.path.join(config_dir, "clustered_signatures")
    os.makedirs(clustered_dir, exist_ok=True)

    print(f"DEBUG: Created output directory structure at {config_dir}")

    return config_dir, clustered_dir


def create_semi_supervised_feedback(metrics, config_name, metadata_manager, feedback_file):
    """Create compact feedback file with essential information about semi-supervised clustering."""
    if not os.path.exists(os.path.dirname(feedback_file)):
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    # Build the feedback content
    content = [
        f"SEMI-SUPERVISED CONFIGURATION: {config_name}",
        "=" * 60,
        "\nKEY PARAMETERS:",
        f"- HU_WEIGHT: {metadata_manager.config['HU_WEIGHT']}",
        f"- LBP_WEIGHT: {metadata_manager.config['LBP_WEIGHT']}",
        f"- HOG_WEIGHT: {metadata_manager.config['HOG_WEIGHT']}",
        f"- ZERNIKE_WEIGHT: {metadata_manager.config['ZERNIKE_WEIGHT']}",
        f"- GABOR_WEIGHT: {metadata_manager.config.get('GABOR_WEIGHT', 0.0)}",
        f"- DISTANCE_THRESHOLD: {metadata_manager.config['DISTANCE_THRESHOLD']}",
        f"- MERGE_THRESHOLD: {metadata_manager.config['MERGE_THRESHOLD']}",
        f"- CONSTRAINT_WEIGHT: {metadata_manager.config.get('CONSTRAINT_WEIGHT', 0.7)}",

        "\nHUMAN FEEDBACK STATISTICS:",
        f"- Pure Clusters: {len([c for c, data in metadata_manager.metadata['clusters'].items() if data.get('is_pure', False)])}",
        f"- Total Rejected Images: {sum(len(data['human_feedback']['removed_from']) for data in metadata_manager.metadata['images'].values())}",
        f"- Must-Link Constraints: {len(metadata_manager.metadata['constraints'].get('must_link', []))}",
        f"- Cannot-Link Constraints: {len(metadata_manager.metadata['constraints'].get('cannot_link', []))}",
    ]

    # Add metrics if testing on preclustered images
    if metadata_manager.config['TESTING_ON_PRECLUSTERED_IMAGES'] and metrics:
        content.extend([
            "\nPERFORMANCE METRICS:",
            f"- Accuracy: {metrics['accuracy']:.4f}",
            f"- Average Purity: {metrics['avg_purity']:.4f}",
            f"- Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}" if 'adjusted_rand_index' in metrics else f"- Adjusted Rand Index: N/A",
            f"- Normalized Mutual Information: {metrics['normalized_mi']:.4f}" if 'normalized_mi' in metrics else f"- Normalized Mutual Information: N/A",
            f"- Average Fragmentation: {metrics['avg_fragmentation']:.2f}",
            f"- Balanced Score: {metrics['balanced_score']:.4f}",
            f"- Number of Clusters: {metrics['num_clusters']}",
            f"- Total Signatures: {metrics['total_signatures']}",

            "\nCLUSTER SIZE DISTRIBUTION:",
            f"- Min: {metrics['cluster_sizes']['min']}",
            f"- Max: {metrics['cluster_sizes']['max']}",
            f"- Mean: {metrics['cluster_sizes']['mean']:.2f}",
            f"- Std Dev: {metrics['cluster_sizes']['std']:.2f}",

            "\nIMPROVEMENT FROM UNSUPERVISED:",
            "- Note: Compare these metrics with unsupervised results in the primary feedback file"
        ])

    # Add the content to the feedback file
    with open(feedback_file, 'a') as f:
        f.write('\n'.join(content))
        f.write('\n\n' + '-' * 80 + '\n\n')

    print(f"Semi-supervised feedback appended to {feedback_file}")


def run_semi_supervised_batch_tests(default_config, test_configs):
    """Run multiple semi-supervised clustering tests with different configurations."""
    print(f"\n{'='*20} SEMI-SUPERVISED BATCH TESTING MODE {'='*20}")
    print(f"Running {len(test_configs)} semi-supervised test configurations\n")

    # Create root directory using the new parameter
    root_dir = default_config.get('SEMI_SUPERVISED_RESULTS_DIR', 'results/semi_supervised')
    os.makedirs(root_dir, exist_ok=True)

    # Create a new batch directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(root_dir, f"semi_supervised_results_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    # Create summary directories
    batch_summary_dir = os.path.join(batch_dir, "batch_summary")
    os.makedirs(batch_summary_dir, exist_ok=True)

    config_summaries_dir = os.path.join(batch_dir, "configuration_summaries")
    os.makedirs(config_summaries_dir, exist_ok=True)

    # Create feedback file
    feedback_file = os.path.join(batch_summary_dir, "semi_supervised_feedback.txt")
    with open(feedback_file, 'w', encoding="utf-8") as f:
        f.write(f"SEMI-SUPERVISED BATCH TESTING RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    results = []

    # Run each test configuration
    for i, test_config in enumerate(test_configs):
        # Create a copy of the default config and update with test-specific settings
        test_name = test_config['name']
        test_config_full = default_config.copy()
        test_config_full.update(test_config)

        # Verification step - print critical settings to confirm they're applied
        print(f"\nVerifying configuration for test {test_name}:")
        for key in ['HU_WEIGHT', 'LBP_WEIGHT', 'HOG_WEIGHT', 'DISTANCE_THRESHOLD', 'DISTANCE_METRIC']:
            if key in test_config:
                print(f"- {key}: {test_config[key]} (from test config)")
            else:
                print(f"- {key}: {test_config_full[key]} (from default config)")

        # Enable semi-supervised mode
        test_config_full['SEMI_SUPERVISED_MODE'] = True

        # Point to feedback file
        test_config_full['FEEDBACK_FILE'] = feedback_file

        # Create results directory for this config
        results_dir = os.path.join(config_summaries_dir, sanitize_directory_name(test_name))
        os.makedirs(results_dir, exist_ok=True)

        # Set both parameters to ensure compatibility
        test_config_full['UNSUPERVISED_RESULTS_DIR'] = results_dir
        test_config_full['SEMI_SUPERVISED_RESULTS_DIR'] = results_dir

        # Remove the old parameter if it exists
        if 'RESULTS_DIR' in test_config_full:
            del test_config_full['RESULTS_DIR']

        print(f"\nSemi-supervised test {i+1}/{len(test_configs)}: {test_name}")

        # Run the semi-supervised clustering with this config
        clustering_result = perform_semi_supervised_clustering(test_config_full)

        # Check if clustering was successful
        if clustering_result[0] is None:
            print(f"ERROR: Semi-supervised clustering failed for {test_name}")
            print("Skipping evaluation and continuing with next configuration")
            results.append({
                'name': test_name,
                'metrics': None,
                'dir': results_dir,
                'error': True
            })
            continue

        clusters, dist_matrix, feature_vectors, linkage_matrix, valid_signatures = clustering_result

        # Evaluate results if we're testing on preclustered images
        metrics = None
        if test_config_full['TESTING_ON_PRECLUSTERED_IMAGES']:
            database = SignatureDatabase(test_config_full['SIGNATURES_DIR'],
                                        test_config_full['VALID_FILE_ENDINGS'],
                                        test_config_full)
            true_labels = database.get_true_labels()

            # Create evaluator
            evaluator = SemiSupervisedSignatureClustering(test_config_full)

            # Evaluate results
            metrics = evaluator.evaluate_clustering(clusters, true_labels)

            # Save metrics
            metrics_file = os.path.join(results_dir, 'semi_supervised_metrics.json')
            with open(metrics_file, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                metrics_json = {k: v if not isinstance(v, (np.float32, np.float64, np.int32, np.int64))
                               else v.item() for k, v in metrics.items()}
                if 'cluster_sizes' in metrics_json:
                    metrics_json['cluster_sizes'] = \
                        {k: v.item() if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) \
                         else v for k, v in metrics_json['cluster_sizes'].items()}
                json.dump(metrics_json, f, indent=4)

            # Create compact feedback for LLM analysis
            create_semi_supervised_feedback(
                metrics,
                test_name,
                evaluator.metadata_manager,
                feedback_file
            )

        # Store results
        results.append({
            'name': test_name,
            'metrics': metrics,
            'dir': results_dir
        })

        # Create visualizations
        if test_config_full['SAVE_VISUALIZATIONS']:
            vis_dir = os.path.join(results_dir, 'visualizations')
            evaluator = SemiSupervisedSignatureClustering(test_config_full)
            evaluator.visualize_clusters(clusters, vis_dir)
            evaluator.visualize_dendrogram(linkage_matrix, vis_dir)

    # Create summary report if in testing mode
    if default_config['TESTING_ON_PRECLUSTERED_IMAGES'] and any(r['metrics'] for r in results):
        # Same summary code as in the original run_batch_tests function
        summary_file = os.path.join(batch_summary_dir, 'semi_supervised_summary.csv')
        summary = []
        for result in results:
            if result['metrics']:
                row = {
                    'Test Name': result['name'],
                    'Accuracy': result['metrics']['accuracy'],
                    'Purity': result['metrics']['avg_purity'],
                    'Fragmentation': result['metrics']['avg_fragmentation'],
                    'Balanced Score': result['metrics']['balanced_score'],
                    'Clusters': result['metrics']['num_clusters'],
                    'ARI': result['metrics'].get('adjusted_rand_index', 0),
                    'NMI': result['metrics'].get('normalized_mi', 0),
                    'Results Directory': result['dir']
                }
                summary.append(row)

        # Save summary as CSV
        if summary:
            pd.DataFrame(summary).to_csv(summary_file, index=False)

            # Create visual comparison
            plt.figure(figsize=(12, 8))
            summary_df = pd.DataFrame(summary).sort_values('Balanced Score', ascending=False)
            plt.scatter(summary_df['Fragmentation'], summary_df['Accuracy'],
                      s=100, alpha=0.7, c=summary_df['Balanced Score'], cmap='viridis')
            for i, row in summary_df.iterrows():
                plt.annotate(row['Test Name'], (row['Fragmentation'], row['Accuracy']),
                           xytext=(5, 5), textcoords='offset points')
            plt.colorbar(label='Balanced Score')
            plt.xlabel('Fragmentation (clusters per signer)')
            plt.ylabel('Accuracy')
            plt.title('Semi-Supervised Clustering Performance Comparison')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(batch_summary_dir, 'semi_supervised_performance_comparison.png'))

            # Print the best configurations
            print("\nTop 3 semi-supervised configurations by balanced score:")
            for i in range(min(3, len(summary_df))):
                row = summary_df.iloc[i]
                print(f"{i+1}. {row['Test Name']} - Accuracy: {row['Accuracy']:.2%}, "
                     f"Fragmentation: {row['Fragmentation']:.2f}, "
                     f"Balanced Score: {row['Balanced Score']:.4f}")

    return results, batch_dir


#=============================================================================
# MAIN EXECUTION
#=============================================================================

def sanitize_directory_name(name):
    """Sanitize a string to be used as a directory name."""
    # Replace any characters that might cause issues in directory names
    return re.sub(r'[^\w\-\.]', '_', name)

def save_config(config, test_name, results_dir):
    """Save configuration parameters to a file."""
    config_file = os.path.join(results_dir, 'configuration.json')

    # Add test name to config
    config_to_save = config.copy()
    config_to_save['test_name'] = test_name

    with open(config_file, 'w') as f:
        json.dump(config_to_save, f, indent=4)

    print(f"Configuration saved to {config_file}")

def run_single_test(config, test_name, config_dir):
    """Run a single clustering test with the given configuration."""
    # Create sanitized directory name from test_name
    safe_test_name = sanitize_directory_name(test_name)

    # Set the results directory to be inside the provided directory
    results_dir = os.path.join(config_dir, safe_test_name)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*20} Running test: {test_name} {'='*20}")
    print(f"Results will be saved to: {results_dir}")

    # Save configuration
    save_config(config, test_name, results_dir)

    # Initialize classes
    database = SignatureDatabase(config['SIGNATURES_DIR'], config['VALID_FILE_ENDINGS'], config)
    clusterer = SignatureClustering(config)

    if config.get('VISUALIZE_PREPROCESSING', False):
        print("\nGenerating preprocessing visualizations...")
        # Get a sample of signatures (limit by the config parameter)
        sample_signatures = database.get_all_signatures(limit=config.get('MAX_PREPROCESSED_SIGS_TO_VIS', 10))

        # Always store visualizations in the configuration's directory
        vis_dir = os.path.join(results_dir, 'preprocessing_visualizations')

        # Ensure directory exists
        os.makedirs(vis_dir, exist_ok=True)

        print(f"Storing preprocessing visualizations in: {vis_dir}")

        # Call visualization function
        clusterer.extractor.visualize_preprocessing_comparison(sample_signatures, vis_dir)

    # Define clustered output directory for this configuration
    clustered_output_dir = None
    if config['COPY_IMAGES_TO_CLUSTERS']:
        clustered_output_dir = os.path.join(results_dir, "clustered_signatures")
        os.makedirs(clustered_output_dir, exist_ok=True)

    # Process differently based on whether testing on preclustered images
    if config['TESTING_ON_PRECLUSTERED_IMAGES']:
        # Testing mode
        if config['CLUSTER_DIRECTORY_DEPTH'] == 0:
            # Process everything at once only if depth is 0
            signatures = database.get_all_signatures(limit=config.get('DATA_LIMIT'))
            true_labels = database.get_true_labels()

            # Perform clustering
            clusters, dist_matrix, feature_vectors, linkage_matrix, valid_signatures, feature_groups = clusterer.cluster_signatures(signatures)

            clusters = clusterer.reorder_clusters_by_similarity(clusters, feature_groups, valid_signatures)

            # Evaluate results
            metrics = clusterer.evaluate_clustering(clusters, true_labels)

            # Create visualizations
            vis_dir = os.path.join(results_dir, 'visualizations')
            clusterer.visualize_clusters(clusters, vis_dir)
            clusterer.visualize_dendrogram(linkage_matrix, vis_dir)

            # Copy images to cluster directories if enabled
            if config['COPY_IMAGES_TO_CLUSTERS'] and clustered_output_dir:
                clusterer.copy_images_to_cluster_directories(
                    clusters, config['SIGNATURES_DIR'], clustered_output_dir, config['CLUSTER_DIRECTORY_DEPTH']
                )

            # Create compact feedback for LLM analysis
            clusterer.create_compact_feedback(metrics, test_name, config['FEEDBACK_FILE'])

            # Save results
            if config['SAVE_RESULTS']:
                # Save clusters
                clusters_file = os.path.join(results_dir, 'clusters.pkl')
                with open(clusters_file, 'wb') as f:
                    pickle.dump(clusters, f)

                # Save metrics
                metrics_file = os.path.join(results_dir, 'metrics.json')
                with open(metrics_file, 'w') as f:
                    # Convert numpy values to Python native types for JSON serialization
                    metrics_json = {k: v if not isinstance(v, (np.float32, np.float64, np.int32, np.int64))
                                   else v.item() for k, v in metrics.items()}
                    if 'cluster_sizes' in metrics_json:
                        metrics_json['cluster_sizes'] = \
                            {k: v.item() if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) \
                             else v for k, v in metrics_json['cluster_sizes'].items()}
                    json.dump(metrics_json, f, indent=4)

        else:
            # Process each pool separately to save memory
            cluster_pools = database.get_cluster_pools()
            all_clusters = {}
            all_pool_results = []

            # Process each pool separately
            for pool_idx, (pool_id, signatures) in enumerate(cluster_pools.items()):
                print(f"\nProcessing pool {pool_idx+1}/{len(cluster_pools)}: {pool_id} with {len(signatures)} signatures")

                # Skip empty pools
                if not signatures:
                    print("Empty pool, skipping.")
                    continue

                # Extract true labels only for this pool
                pool_true_labels = {sig: label for sig, label in database.get_true_labels().items() if sig in signatures}

                # Process this pool
                pool_clusters, dist_matrix, feature_vectors, linkage_matrix, valid_signatures, feature_groups = clusterer.cluster_signatures(signatures)

                feature_groups, _ = clusterer.extract_features_batch(valid_signatures)
                pool_clusters = clusterer.reorder_clusters_by_similarity(pool_clusters, feature_groups, valid_signatures)

                # Evaluate results for this pool
                pool_metrics = clusterer.evaluate_clustering(pool_clusters, pool_true_labels)
                all_pool_results.append((pool_id, pool_metrics))

                # Create visualizations
                pool_name = os.path.basename(pool_id) if os.path.isdir(pool_id) else f'pool_{pool_idx}'
                vis_dir = os.path.join(results_dir, 'visualizations', pool_name)
                clusterer.visualize_clusters(pool_clusters, vis_dir)
                clusterer.visualize_dendrogram(linkage_matrix, vis_dir)

                # MODIFIED: Copy images to cluster directories immediately after processing this pool
                if config['COPY_IMAGES_TO_CLUSTERS'] and clustered_output_dir:
                    # Determine the relative path within the source directory
                    rel_path = os.path.relpath(pool_id, config['SIGNATURES_DIR'])
                    if rel_path == '.':
                        # Root directory case
                        output_dir = clustered_output_dir
                    else:
                        # Subdirectory case
                        output_dir = os.path.join(clustered_output_dir, rel_path)

                    # Create the output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    print(f"Writing clustered images for pool {pool_name} to {output_dir}")
                    clusterer.copy_images_to_cluster_directories(
                        pool_clusters, pool_id, output_dir, 0
                    )

                # Add to overall clusters collection with unique IDs
                for cluster_id, cluster_signatures in pool_clusters.items():
                    # Generate a unique ID that includes the pool
                    unique_id = f"{os.path.basename(pool_id)}_{cluster_id}"
                    all_clusters[unique_id] = cluster_signatures

                # Save this pool's results
                pool_clusters_file = os.path.join(results_dir, f'clusters_{pool_idx}_{pool_name}.pkl')
                with open(pool_clusters_file, 'wb') as f:
                    pickle.dump(pool_clusters, f)

                # Force garbage collection to free memory
                gc.collect()

                print(f"Completed processing pool {pool_idx+1}/{len(cluster_pools)}: {pool_id}")

            # Aggregate metrics across pools
            if all_pool_results:
                # Calculate overall metrics
                total_correct = sum(metrics['accuracy'] * metrics['total_signatures'] for _, metrics in all_pool_results)
                total_signatures = sum(metrics['total_signatures'] for _, metrics in all_pool_results)
                overall_accuracy = total_correct / total_signatures if total_signatures > 0 else 0

                avg_purity = np.mean([metrics['avg_purity'] for _, metrics in all_pool_results])
                avg_fragmentation = np.mean([metrics['avg_fragmentation'] for _, metrics in all_pool_results])

                # Create combined metrics
                metrics = {
                    'accuracy': overall_accuracy,
                    'avg_purity': avg_purity,
                    'avg_fragmentation': avg_fragmentation,
                    'balanced_score': overall_accuracy * np.exp(-0.1 * (avg_fragmentation - 1.0)),
                    'num_clusters': sum(metrics['num_clusters'] for _, metrics in all_pool_results),
                    'total_signatures': total_signatures,
                    'cluster_sizes': {
                        'mean': np.mean([metrics['cluster_sizes']['mean'] for _, metrics in all_pool_results]),
                        'std': np.mean([metrics['cluster_sizes']['std'] for _, metrics in all_pool_results]),
                        'min': min([metrics['cluster_sizes']['min'] for _, metrics in all_pool_results]),
                        'max': max([metrics['cluster_sizes']['max'] for _, metrics in all_pool_results])
                    }
                }

                # Create compact feedback for LLM analysis
                clusterer.create_compact_feedback(metrics, test_name, config['FEEDBACK_FILE'])

                # Save overall metrics
                metrics_file = os.path.join(results_dir, 'metrics.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=4)
            else:
                metrics = None
    else:
        # Production mode - process each pool separately
        cluster_pools = database.get_cluster_pools()
        all_clusters = {}
        pool_summaries = []

        # Process each pool separately
        for pool_idx, (pool_id, signatures) in enumerate(cluster_pools.items()):
            print(f"\nProcessing pool {pool_idx+1}/{len(cluster_pools)}: {pool_id} with {len(signatures)} signatures")

            # Skip empty pools
            if not signatures:
                print("Empty pool, skipping.")
                continue

            # MODIFIED: Pass the clustered_output_dir to the process_cluster_pool method
            # Process this pool and immediately write output
            pool_clusters = clusterer.process_cluster_pool(
                signatures, pool_id, results_dir, clustered_output_dir
            )

            # Collect pool statistics
            if pool_clusters:
                pool_summary = {
                    'pool_id': pool_id,
                    'num_signatures': len(signatures),
                    'num_clusters': len(pool_clusters),
                    'avg_cluster_size': np.mean([len(sigs) for sigs in pool_clusters.values()]),
                    'cluster_size_std': np.std([len(sigs) for sigs in pool_clusters.values()]),
                    'cluster_size_min': min([len(sigs) for sigs in pool_clusters.values()]),
                    'cluster_size_max': max([len(sigs) for sigs in pool_clusters.values()])
                }
                pool_summaries.append(pool_summary)

            # Add to overall clusters collection
            for cluster_id, cluster_signatures in pool_clusters.items():
                # Generate a unique ID that includes the pool
                unique_id = f"{os.path.basename(pool_id)}_{cluster_id}"
                all_clusters[unique_id] = cluster_signatures

            # Save this pool's results
            pool_name = os.path.basename(pool_id) if os.path.isdir(pool_id) else f'pool_{pool_idx}'
            pool_clusters_file = os.path.join(results_dir, f'clusters_{pool_idx}_{pool_name}.pkl')
            with open(pool_clusters_file, 'wb') as f:
                pickle.dump(pool_clusters, f)

            # Force garbage collection to free memory
            gc.collect()

            print(f"Completed processing pool {pool_idx+1}/{len(cluster_pools)}: {pool_id}")

        # Save summary information about the clustering
        if pool_summaries:
            summary = {
                'num_pools': len(cluster_pools),
                'num_clusters': sum(summary['num_clusters'] for summary in pool_summaries),
                'total_signatures': sum(summary['num_signatures'] for summary in pool_summaries),
                'avg_cluster_size': np.mean([summary['avg_cluster_size'] for summary in pool_summaries]),
                'cluster_size_std': np.mean([summary['cluster_size_std'] for summary in pool_summaries]),
                'cluster_size_min': min([summary['cluster_size_min'] for summary in pool_summaries]),
                'cluster_size_max': max([summary['cluster_size_max'] for summary in pool_summaries])
            }
        else:
            summary = {
                'num_pools': len(cluster_pools),
                'num_clusters': 0,
                'total_signatures': 0,
                'avg_cluster_size': 0,
                'cluster_size_std': 0,
                'cluster_size_min': 0,
                'cluster_size_max': 0
            }

        # Save summary to file
        summary_file = os.path.join(results_dir, 'clustering_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"\nClustering complete. Processed {summary['num_pools']} pools, "
              f"created {summary['num_clusters']} clusters with "
              f"{summary['total_signatures']} signatures.")

        # Use the summary as metrics for the feedback file
        metrics = {
            'cluster_sizes': {
                'mean': summary['avg_cluster_size'],
                'std': summary['cluster_size_std'],
                'min': summary['cluster_size_min'],
                'max': summary['cluster_size_max']
            },
            'num_clusters': summary['num_clusters'],
            'total_signatures': summary['total_signatures']
        }

        # Create compact feedback for LLM analysis
        clusterer.create_compact_feedback(metrics, test_name, config['FEEDBACK_FILE'])

        # Save overall clusters
        clusters_file = os.path.join(results_dir, 'all_clusters.pkl')
        with open(clusters_file, 'wb') as f:
            pickle.dump(all_clusters, f)

    # Generate and save metadata for future semi-supervised clustering
    if config['COPY_IMAGES_TO_CLUSTERS'] and clustered_output_dir:
        # Ensure clusters is defined for metadata generation
        if 'clusters' not in locals() and 'all_clusters' in locals():
            clusters = all_clusters
        elif 'clusters' not in locals():
            clusters = {}
        print("\nGenerating clustering metadata for future semi-supervised use...")
        metadata = {
            "images": {},
            "clusters": {},
            "constraints": {
                "must_link": [],
                "cannot_link": []
            }
        }

        # Register all images and their cluster assignments
        for cluster_id, signatures in clusters.items():
            # Check if this is a newly added cluster (numeric ID) or an existing named cluster
            if isinstance(cluster_id, (int, float)) or (isinstance(cluster_id, str) and cluster_id.isdigit()):
                # This is a numeric ID - add "cluster_" prefix and zero padding
                num_clusters = len(clusters)
                padding_width = len(str(num_clusters - 1)) if num_clusters > 1 else 1
                display_name = f"cluster_{int(cluster_id):0{padding_width}d}"
            else:
                # This is already a named cluster - preserve the name
                display_name = cluster_id

            # Initialize cluster entry with display name support
            metadata["clusters"][str(cluster_id)] = {
                "is_pure": False,
                "verified_by_human": False,
                "display_name": display_name,  # Add this field
                "members": []
            }

            # Process each signature in this cluster
            for sig_path in signatures:
                # Generate unique ID
                img_id = hashlib.md5(os.path.abspath(sig_path).encode()).hexdigest()[:12]

                # Calculate the destination path where the image was copied
                base_name = os.path.basename(sig_path)
                name, ext = os.path.splitext(base_name)
                unique_name = f"{name}_{img_id[:8]}{ext}"
                dest_path = os.path.join(clustered_output_dir, display_name, unique_name)

                # Add to cluster members
                metadata["clusters"][str(cluster_id)]["members"].append(img_id)

                # Register image
                metadata["images"][img_id] = {
                    "original_path": os.path.abspath(sig_path),
                    "current_path": os.path.abspath(dest_path),
                    "original_cluster": str(cluster_id),
                    "clustering_history": [
                        {
                            "iteration": 1,
                            "cluster_id": str(cluster_id),
                            "confidence": 1.0
                        }
                    ],
                    "human_feedback": {
                        "removed_from": [],
                        "manually_assigned_to": None
                    }
                }

        # Save metadata to the configuration directory
        metadata_file = os.path.join(results_dir, "clustering_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved clustering metadata to {metadata_file}")

    print(f"\nTest completed. Results saved to {results_dir}")
    return metrics, results_dir

def run_batch_tests(default_config, test_configs):
    """Run multiple clustering tests with different configurations."""
    print(f"\n{'='*20} BATCH TESTING MODE {'='*20}")
    print(f"Running {len(test_configs)} test configurations\n")

    # Create root directory using the new parameter
    root_dir = default_config.get('UNSUPERVISED_RESULTS_DIR', 'results/unsupervised')
    os.makedirs(root_dir, exist_ok=True)

    # Create a new batch directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(root_dir, f"clustering_results_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)

    # Create directories for batch summary and configuration results
    batch_summary_dir = os.path.join(batch_dir, "batch_summary")
    os.makedirs(batch_summary_dir, exist_ok=True)

    config_summaries_dir = os.path.join(batch_dir, "configuration_summaries")
    os.makedirs(config_summaries_dir, exist_ok=True)

    # Create the feedback file in the batch summary directory
    feedback_file = os.path.join(batch_summary_dir, "feedback_for_llm.txt")
    with open(feedback_file, 'w', encoding="utf-8") as f:
        f.write(f"BATCH TESTING RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    results = []

    # Run each test configuration
    for i, test_config in enumerate(test_configs):
        # Create a copy of the default config and update with test-specific settings
        test_name = test_config['name']
        test_config_full = default_config.copy()
        test_config_full.update(test_config)

        # Point the feedback file to the batch summary's feedback file
        test_config_full['FEEDBACK_FILE'] = feedback_file

        # Set the central preprocessing visualization directory
        test_config_full['TEST_INDEX'] = i + 1  # Add index for visualization filenames

        # Run the test, saving results to the config_summaries directory
        print(f"\nTest {i+1}/{len(test_configs)}: {test_name}")
        metrics, results_dir = run_single_test(test_config_full, test_name, config_summaries_dir)

        # Store results
        results.append({
            'name': test_name,
            'metrics': metrics,
            'dir': results_dir
        })

    # Create summary report if in testing mode
    if default_config['TESTING_ON_PRECLUSTERED_IMAGES']:
        summary_file = os.path.join(batch_summary_dir, 'batch_summary.csv')
        summary = []
        for result in results:
            if result['metrics']:
                row = {
                    'Test Name': result['name'],
                    'Accuracy': result['metrics']['accuracy'],
                    'Purity': result['metrics']['avg_purity'],
                    'Fragmentation': result['metrics']['avg_fragmentation'],
                    'Balanced Score': result['metrics']['balanced_score'],
                    'Clusters': result['metrics']['num_clusters'],
                    'ARI': result['metrics']['adjusted_rand_index'] if 'adjusted_rand_index' in result['metrics'] else 0,
                    'NMI': result['metrics']['normalized_mi'] if 'normalized_mi' in result['metrics'] else 0,
                    'Results Directory': result['dir']
                }
                summary.append(row)

        # Save summary as CSV
        if summary:
            pd.DataFrame(summary).to_csv(summary_file, index=False)

            # Create visual comparison
            plt.figure(figsize=(12, 8))

            # Sort by balanced score
            summary_df = pd.DataFrame(summary).sort_values('Balanced Score', ascending=False)

            # Plot accuracy vs fragmentation
            plt.scatter(summary_df['Fragmentation'], summary_df['Accuracy'],
                       s=100, alpha=0.7, c=summary_df['Balanced Score'], cmap='viridis')

            for i, row in summary_df.iterrows():
                plt.annotate(row['Test Name'],
                            (row['Fragmentation'], row['Accuracy']),
                            xytext=(5, 5), textcoords='offset points')

            plt.colorbar(label='Balanced Score')
            plt.xlabel('Fragmentation (clusters per signer)')
            plt.ylabel('Accuracy')
            plt.title('Clustering Performance Comparison')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(batch_summary_dir, 'performance_comparison.png'))

            # Save overall batch configuration
            batch_config_file = os.path.join(batch_summary_dir, 'batch_config.json')
            with open(batch_config_file, 'w', encoding="utf-8") as f:
                batch_info = {
                    'timestamp': timestamp,
                    'number_of_configurations': len(test_configs),
                    'config_names': [config['name'] for config in test_configs],
                    'testing_on_preclustered_images': default_config['TESTING_ON_PRECLUSTERED_IMAGES']
                }
                json.dump(batch_info, f, indent=4)

            print(f"\nBatch testing complete. Summary saved to {summary_file}")

            # Print the best configurations
            print(f"\nTop {min(3, len(summary_df))} configurations by balanced score:")
            for i in range(min(3, len(summary_df))):
                row = summary_df.iloc[i]
                print(f"{i+1}. {row['Test Name']} - Accuracy: {row['Accuracy']:.2%}, "
                      f"Fragmentation: {row['Fragmentation']:.2f}, "\
                        f"Balanced Score: {row['Balanced Score']:.4f}")
        else:
            print("\nBatch testing complete, but no metrics available to create summary.")
    else:
        # For production mode, still create a basic summary
        batch_info_file = os.path.join(batch_summary_dir, 'batch_info.json')
        with open(batch_info_file, 'w', encoding="utf-8") as f:
            batch_info = {
                'timestamp': timestamp,
                'number_of_configurations': len(test_configs),
                'config_names': [config['name'] for config in test_configs],
                'testing_on_preclustered_images': False
            }
            json.dump(batch_info, f, indent=4)

        print(f"\nBatch testing complete. Results saved to {batch_dir}")

    return results, batch_dir

def display_config_instructions():
    """Display instructions for configuring the script."""
    print("\nSIGNATURE CLUSTERING CONFIGURATION GUIDE")
    print("=======================================")
    print("\nTo configure the clustering algorithm, edit the "\
          "'default_config' dictionary at the top of the script.")
    print("The following parameters can be modified:")

    print("\nDirectory Structure Parameters:")
    print("  - TESTING_ON_PRECLUSTERED_IMAGES: Set to True for testing, False for production")
    print("  - CLUSTER_DIRECTORY_DEPTH: How many directory levels down to go before clustering (0, 1, 2, etc. or 'max')")
    print("  - COPY_IMAGES_TO_CLUSTERS: Whether to copy clustered images to output directories")

    print("\nFeature Extraction Settings:")
    print("  - HU_WEIGHT: Weight for shape features (default: 5.52)")
    print("  - LBP_WEIGHT: Weight for texture features (default: 5.52)")
    print("  - HOG_WEIGHT: Weight for gradient features (default: 0.505)")
    print("  - ZERNIKE_WEIGHT: Weight for rotation-invariant features (default: 1.48)")
    print("  - GABOR_WEIGHT: Weight for Gabor filter features (default: 0.0, disabled)")
    print("  - USE_ENHANCED_LBP: Use multi-scale LBP features (default: False)")
    print("  - USE_ZERNIKE: Include Zernike moments (default: True)")
    print("  - USE_GABOR: Include Gabor filter features (default: False)")
    print("  - USE_PCA_HOG: Apply PCA to reduce HOG dimensions (default: False)")

    print("\nNormalization Options:")
    print("  - NORMALIZE_METHOD: Feature normalization method - 'standard', 'l1', 'l2', 'robust'")

    print("\nClustering Settings:")
    print("  - DISTANCE_METRIC: Distance measure - 'correlation', "\
          "'cosine', 'euclidean', 'cityblock' (default: 'correlation')")
    print("  - DISTANCE_THRESHOLD: Threshold for initial clustering (default: 0.436)")
    print("  - LINKAGE_METHOD: Method for hierarchical clustering (default: 'average')")

    print("\nMulti-stage Settings:")
    print("  - USE_TWO_STAGE: Enable two-stage clustering (default: True)")
    print("  - MERGE_THRESHOLD: Threshold for merging similar clusters (default: 0.636)")
    print("  - MERGE_METHOD: Method for calculating inter-cluster distances - 'average', 'min', 'max', 'adaptive'")
    print("  - USE_ADAPTIVE_THRESHOLD: Use data-driven threshold (default: False)")

    print("\nEnsemble Settings:")
    print("  - USE_ENSEMBLE: Enable ensemble clustering (default: False)")
    print("  - ENSEMBLE_METHODS: List of methods to include in ensemble - 'hierarchical', 'spectral'")
    print("  - ENSEMBLE_WEIGHTS: Relative weights for ensemble methods")

    print("\nBatch Testing:")
    print("  - The script now runs in batch mode only, testing all configurations in test_configs")
    print("  - All results are saved in ./clustering_results/clustering_results_{date}_{time}/")
    print("  - Each configuration gets its own directory inside configuration_summaries/")
    print("  - Overall batch results are saved in batch_summary/")

    print("\nYou can also add new test configurations to the 'test_configs' list.")


def main():
    """Main execution function."""

    # Check for parameter optimization mode
    if len(sys.argv) > 1 and sys.argv[1] == "optimize":
        if len(sys.argv) < 3:
            print("Usage: python signature_clustering.py optimize <perfect_clusters_dir> [checkpoint_file]")
            print("  perfect_clusters_dir: Directory with perfectly clustered signatures")
            print("  checkpoint_file: Optional file to store checkpoint data (default: optimization_checkpoint.json in timestamped results directory)")
            return

        perfect_clusters_dir = sys.argv[2]

        # Get the base optimization directory from config
        base_dir = default_config.get('OPTIMIZATION_RESULTS_DIR', 'results/optimize')
        os.makedirs(base_dir, exist_ok=True)

        # Create a timestamped directory for this optimization run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(base_dir, f"optimization_results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # If user provided a number of iterations, use it; otherwise use the default of 200.
        if len(sys.argv) > 3:
            try:
                n_iterations = int(sys.argv[3])
            except ValueError:
                print("ERROR: The number of iterations must be a number. You "\
                      f"provided \"{sys.argv[3]}\". Leave blank for a default of 200.")
                return

        # If user provided a path, use it; otherwise use the timestamped directory
        if len(sys.argv) > 4:
            checkpoint_file = sys.argv[4]
            # If just a filename was provided (no path), put it in base_dir
            if not os.path.dirname(checkpoint_file):
                checkpoint_file = os.path.join(base_dir, checkpoint_file)
        else:
            checkpoint_file = os.path.join(base_dir, "optimization_checkpoint.json")

        # Set history file in the same directory
        checkpoint_history_file = os.path.join(os.path.dirname(checkpoint_file),
                                             "optimization_checkpoint_history.json")

        # Continue with optimization
        try:
            print("Starting comprehensive parameter optimization")
            print(f"Results will be saved to: {results_dir}")
            _, _, _ = comprehensive_parameter_optimization(
                default_config,
                perfect_clusters_dir,
                checkpoint_file=checkpoint_file,
                checkpoint_history_file=checkpoint_history_file,
                n_iterations=n_iterations,
                base_dir=base_dir
            )
        except Exception as e:
            print(f"Error during optimization: {e}")
            traceback.print_exc()

        return

    print("\n" + "="*80)
    print(" ADVANCED SIGNATURE CLUSTERING (HIERARCHICAL VERSION) ".center(80, "="))
    print("="*80 + "\n")

    # Display configuration instructions
    display_config_instructions()

    # Check if we're in semi-supervised mode based on the configuration
    if default_config.get('SEMI_SUPERVISED_MODE', False):
        print("\nRunning in SEMI-SUPERVISED BATCH mode with all configurations")
        run_semi_supervised_batch_tests(default_config, test_configs)
    else:
        # Run standard batch testing with all configurations
        print("\nRunning in UNSUPERVISED BATCH mode with all configurations")
        run_batch_tests(default_config, test_configs)


if __name__ == "__main__":
    main()
