import funcnodes as fn
from typing import Tuple
import numpy as np
from csbdeep.utils import normalize
from skimage.measure import regionprops, find_contours
import cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans


class SegmentModels(fn.DataEnum):
    model_1 = "2D_demo"
    model_2 = "2D_versatile_fluo"
    model_3 = "2D_paper_dsb2018"


def remove_background(img: np.ndarray) -> np.ndarray:
    """
    Removes the background from an image using K-Means clustering and morphological operations.

    Parameters:
    img (np.ndarray): The input grayscale image as a NumPy array.

    Returns:
    np.ndarray: A binary image where the particles are considered foreground.
    """
    # Flatten the image for clustering
    pixels = img.reshape(-1, 1)

    # Apply K-Means clustering (2 clusters: foreground and background)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(pixels)
    segmented = kmeans.labels_.reshape(img.shape)

    # Ensure the particles are the brighter region
    if np.mean(img[segmented == 0]) > np.mean(img[segmented == 1]):
        segmented = 1 - segmented

    # Convert to binary image
    binary_image = (segmented * 255).astype(np.uint8)

    # Apply morphological closing to remove small background regions inside particles
    kernel = np.ones((5, 5), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # Apply morphological opening to remove small noise in the background
    opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)
    # Apply median filtering to smooth the background
    smoothed_image = cv2.medianBlur(opened_image, 3)
    return smoothed_image


def process_contour(reg: regionprops, lbls: np.ndarray) -> list:
    """
    Processes a single region to extract contours.

    Parameters:
    reg (regionprops): A region object from skimage.measure.regionprops.
    lbls (np.ndarray): The segmentation labels as a NumPy array.

    Returns:
    list: A list of contour arrays.
    """
    contours = find_contours(lbls == reg.label, fully_connected="high")
    return [np.array(cnt[:, ::-1], dtype=np.int32)[:, None, :] for cnt in contours]


def process_contours_parallel(props: list, lbls: np.ndarray) -> list:
    """
    Uses multi-threading to compute contours in parallel.

    Parameters:
    props (list): A list of region objects from skimage.measure.regionprops.
    lbls (np.ndarray): The segmentation labels as a NumPy array.

    Returns:
    list: A list of contour arrays computed in parallel.
    """
    contours_ski = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda region: process_contour(region, lbls), props)
        for contour_list in results:
            contours_ski.extend(contour_list)
    return contours_ski


def filter_contours_by_background(foreground: np.ndarray, contours: list) -> list:
    """
    Filters contours based on background exclusion using the Otsu threshold.

    Parameters:
    foreground (ndarray): The binary image where foreground and background are defined.
    contours (list): A list of contour arrays to filter.

    Returns:
    list: A list of contour arrays that pass the background exclusion criteria.
    """
    filtered_contours = []
    for cont in contours:
        if isinstance(cont, np.ndarray):
            M = cv2.moments(cont)
            if M["m00"] != 0:
                Y = int(M["m10"] / M["m00"])
                X = int(M["m01"] / M["m00"])
                if foreground[X, Y]:
                    filtered_contours.append(cont)
    return filtered_contours


def compute_centroids(contours: list) -> list:
    """
    Computes the centroids of given contour arrays.

    Parameters:
    contours (list): A list of contour arrays to compute centroids for.

    Returns:
    list: A list of tuples representing the centroids of the contours.
    """
    centers = [
        (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if M["m00"] != 0
        else (0, 0)
        for c in contours
        for M in [cv2.moments(c)]
    ]
    return centers


def _contours(
    img: np.ndarray, segment_model, foreground: np.ndarray = None
) -> Tuple[list, list]:
    """
    Extracts contours from the image using a segmentation model.

    Parameters:
    img (np.ndarray): The input grayscale image as a NumPy array.
    segment_model: A pre-trained segmentation model.
    foreground (ndarray, optional): The binary foreground mask. If None, it will be computed automatically.

    Returns:
    tuple: Contour arrays and their corresponding centroids.
    """
    centers = []
    labels = segment_model.predict_instances(normalize(img))[0]
    props = regionprops(labels)

    # Compute contours in parallel
    contours_ski = process_contours_parallel(props, labels)

    # Convert to OpenCV format
    contours_cv = contours_ski.copy()

    # Exclude Small Contours (<5 points)
    contours_cv = [c for c in contours_cv if c.shape[0] >= 5]

    # Edge Exclusion (Vectorized)
    mask = np.array(
        [
            not (
                np.any(cont[:, 0, 0] >= img.shape[1] - 1)
                or np.any(cont[:, 0, 1] >= img.shape[0] - 1)
                or np.any(cont[:, 0, 0] == 0)
                or np.any(cont[:, 0, 1] == 0)
            )
            for cont in contours_cv
        ]
    )
    mask = mask.astype(bool)

    contours_cv = np.array(contours_cv, dtype=object)[mask].tolist()

    # Background Exclusion (Precompute Otsu Threshold)
    if foreground is not None:
        filtered_contours = filter_contours_by_background(foreground, contours_cv)
        contours_cv = filtered_contours
        centers = compute_centroids(contours_cv)

    return contours_cv, centers


def _contours_crop(
    img: np.ndarray, segment_model, tiling_factor: int, foreground: np.ndarray
) -> Tuple[list, list]:
    """
    Extracts contours from the image using a segmentation model in a tiled manner.

    Parameters:
    img (np.ndarray): The input grayscale image as a NumPy array.
    segment_model: A pre-trained segmentation model.
    tiling_factor (int): The number of tiles to divide the image into along width axis.
    foreground (ndarray, optional): The binary foreground mask. If None, it will be computed automatically.

    Returns:
    Tuple[list,list]: Contour arrays and their corresponding centroids.
    """
    tiling_factor = int(tiling_factor)

    H, W = img.shape[:2]
    h_step_factor = max(1, tiling_factor - 1)  # Prevent division by zero
    w_step_factor = max(1, tiling_factor)  # Ensure a valid integer
    h_step = int(H // h_step_factor)
    w_step = int(W // w_step_factor)

    crops = [
        (
            img[
                i * h_step : min((i + 1) * h_step, H),
                j * w_step : min((j + 1) * w_step, W),
            ],
            i * h_step,
            j * w_step,
        )
        for i in range(h_step_factor)
        for j in range(w_step_factor)
    ]

    all_contours, all_centers = [], []
    for cropped_img, y_offset, x_offset in crops:
        y_offset = int(y_offset)
        x_offset = int(x_offset)
        cnts, cents = _contours(cropped_img, segment_model, foreground=None)
        for cnt in cnts:
            if isinstance(cnt, np.ndarray):
                cnt[:, 0, 0] += x_offset
                cnt[:, 0, 1] += y_offset
        all_contours.extend(cnts)
        all_centers.extend([(c[0] + x_offset, c[1] + y_offset) for c in cents])

    all_contours = filter_contours_by_background(foreground, all_contours)
    all_centers = compute_centroids(all_contours)
    return all_contours, all_centers


def merge_unique_contours(
    contours_1: list, centers_1: list, contours_2: list, centers_2: list
) -> Tuple[list, list]:
    """
    Merges unique contours from two sets.

    Parameters:
    contours_1 (list): A list of contour arrays.
    centers_1 (list): A list of centroids corresponding to contours_1.
    contours_2 (list): Another list of contour arrays.
    centers_2 (list): Corresponding centroids for contours_2.

    Returns:
    tuple: Merged lists of contour arrays and their corresponding centroids.
    """
    for i, center in enumerate(centers_1):
        if not any(
            cv2.pointPolygonTest(
                np.array(cont, dtype=np.int32).reshape(-1, 1, 2), center, False
            )
            >= 0
            for cont in contours_2
        ):
            contours_2.append(contours_1[i])
            centers_2.append(center)
    return contours_2, centers_2
