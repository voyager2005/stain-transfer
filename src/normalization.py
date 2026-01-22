"""
The following script implements a stain transfer pipeline based on a
structure-preserving color normalization method proposed by: 
Abhishek Vahadane et al.,
"Structure-Preserving Color Normalization and Sparse Stain Separation
for Histological Images"

This code is intended as a faithful, from-scratch implementation, 
prioritizing mathematical clarity and reproducibility over algorithmic 
novelty.
"""

import numpy as np
import cv2
from sklearn.decomposition import NMF
from pathlib import Path
import os
from tqdm import tqdm


def rgb_to_od(im_rgb):
    """
    Convert an RGB image to optical density (OD) space.

    Parameters :
    im_rgb : np.ndarray
        Input RGB image of shape (H, W, 3) with values in [0, 255].

    Returns :
    im_od : np.ndarray
        Optical density image of shape (H, W, 3).
    """
    im_rgb = im_rgb.astype(np.float64)
    im_rgb[im_rgb == 0] = 1e-6
    return -np.log(im_rgb / 255.0)


def od_to_rgb(im_od):
    """
    Convert an optical density (OD) image back to RGB space.

    Parameters :
    im_od : np.ndarray
        Optical density image of shape (H, W, 3).

    Returns :
    im_rgb : np.ndarray
        Reconstructed RGB image of shape (H, W, 3) with dtype uint8.
    """
    im_od = np.maximum(im_od, 1e-6)
    im_rgb = 255 * np.exp(-im_od)
    return np.clip(im_rgb, 0, 255).astype(np.uint8)


def get_stain_matrix(im_od, n_stains=2, l1_ratio=0.1):
    """
    Estimate stain basis vectors using non-negative matrix factorization (NMF).

    The OD image is reshaped to a 2D matrix and factorized into stain 
    concentration and basis matrices. 

    Parameters: 
    im_od : np.ndarray
        Optical density image of shape (H, W, 3).
    n_stains : int, optional
        Number of stain components to estimate (default is 2).
    l1_ratio : float, optional
        L1 regularization ratio controlling sparsity (default is 0.1).

    Returns : 
    stain_matrix : np.ndarray
        Normalized stain basis matrix of shape (n_stains, 3).
    """
    od_flat = im_od.reshape((-1, 3))
    od_flat = od_flat[np.sum(od_flat, axis=1) > 0.15]

    model = NMF(
        n_components=n_stains,
        init='nndsvda',
        solver='mu',
        beta_loss='kullback-leibler',
        l1_ratio=l1_ratio,
        max_iter=500
    )

    W = model.fit_transform(od_flat)
    H = model.components_
    H /= np.linalg.norm(H, axis=1)[:, np.newaxis]
    return H


# Configuration (system specific config)

data_path = Path(r"C:\Projects\monuseg\Data\kmms_training")
input_images_path = data_path / "images_renamed"
output_path = data_path / "normalized_images_manual"
output_path.mkdir(parents=True, exist_ok=True)

TARGET_IMAGE_NAME = "image_1.tif"
target_image_path = input_images_path / TARGET_IMAGE_NAME

# Extracting Target Strain

target_stains = None
try:
    target_image_bgr = cv2.imread(str(target_image_path))
    if target_image_bgr is None:
        raise FileNotFoundError(
            f"File not found @: {target_image_path}"
        )

    target_image_rgb = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB)
    target_od = rgb_to_od(target_image_rgb)

    target_stains = get_stain_matrix(target_od)

except Exception as e:
    print(f"Error details: {e}")
    exit()


# Stain Transfer on Dataset

if target_stains is not None:
    image_files = [f for f in os.listdir(input_images_path) if f.endswith('.tif')]

    for image_name in tqdm(image_files, desc="Normalizing Images"):
        image_path = input_images_path / image_name

        source_image_bgr = cv2.imread(str(image_path))
        if source_image_bgr is None:
            continue

        source_image_rgb = cv2.cvtColor(source_image_bgr, cv2.COLOR_BGR2RGB)
        source_od = rgb_to_od(source_image_rgb)
        source_od_flat = source_od.reshape((-1, 3))

        source_stains_for_conc = get_stain_matrix(source_od)

        source_concentrations, _, _, _ = np.linalg.lstsq(
            source_stains_for_conc.T,
            source_od_flat.T,
            rcond=None
        )

        normalized_od_flat = np.dot(source_concentrations.T, target_stains)
        normalized_od = normalized_od_flat.reshape(source_image_rgb.shape)

        normalized_rgb = od_to_rgb(normalized_od)
        normalized_bgr = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2BGR)

        output_image_path = output_path / image_name
        cv2.imwrite(str(output_image_path), normalized_bgr)

    print(f"Normalized images saved @: {output_path}")
