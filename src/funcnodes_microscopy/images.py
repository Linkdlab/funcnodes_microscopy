import funcnodes as fn
from typing import Tuple
import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops, find_contours
import cv2
from skimage.filters import threshold_otsu
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans

import torch
from torch import Tensor
from super_image import PanModel


class SegmentModels(fn.DataEnum):
    model_1 = "2D_demo"
    model_2 = "2D_versatile_fluo"
    model_3 = "2D_paper_dsb2018"


def _process_image(image: np.ndarray) -> Tensor:
    # Check if the image is 1D or 3D
    if image.ndim == 3 and image.shape[2] == 3:  # RGB image
        lr = image.astype(np.float32).transpose([2, 0, 1]) / 255.0
    elif image.ndim == 2:  # Grayscale image (1 channel)
        # Convert the grayscale to RGB by repeating the single channel across all three channels
        lr = (
            np.stack([image] * 3, axis=-1).astype(np.float32).transpose([2, 0, 1])
            / 255.0
        )
    else:
        raise ValueError(
            "Input numpy array must have 2 or 3 dimensions (grayscale or RGB)"
        )

    return torch.as_tensor(np.array([lr]))


def _deprocess_image(pred: Tensor) -> np.ndarray:
    pred = pred.data.cpu().numpy()
    pred = pred[0].transpose((1, 2, 0)) * 255.0

    # Clamp values to the range [0, 255]
    pred = np.clip(pred, 0, 255)

    pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return pred


@fn.NodeDecorator(
    node_id="fn.microscopy.images.resolution",
    name="Increase Resolution",
    # outputs=[
    #     {"name": "out", "type": OpenCVImageFormat},
    # ],
    default_io_options={
        "resolution_factor": {"value_options": {"min": 2, "max": 4}},
        # "thresh": {"value_options": {"min": 0, "max": 1}},
        # "max_eccentricity": {"value_options": {"min": 0, "max": 1}},
    },
    # default_render_options={"data": {"src": "out"}},
)
def increase_resolution(image: np.ndarray, resolution_factor: int = 2) -> np.ndarray:
    """
    Increases the resolution of an input image using super-resolution.

    Args:
        image (np.ndarray): The input image represented as a NumPy array.
                           Must have 2 or 3 dimensions (grayscale or RGB).

        resolution_factor (int): The factor by which to increase the resolution.

    Returns:
        np.ndarray: The high-resolution image as a NumPy array.
    """

    # Load the pretrained Pan model
    res_model = PanModel.from_pretrained("eugenesiow/pan-bam", scale=resolution_factor)

    # Process the input image
    inputs = _process_image(image)

    # Generate the super-resolution image
    preds = res_model(inputs)

    # Convert the prediction to a higher resolution image
    high_res_image = _deprocess_image(preds)

    # Return the high-resolution image (grayscale channel if needed)
    return high_res_image[:, :, 0]  # Assuming you want the first channel for grayscale


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
    tiling_factor (int): The number of tiles to divide the image into along each axis.
    foreground (ndarray, optional): The binary foreground mask. If None, it will be computed automatically.

    Returns:
    Tuple[list,list]: Contour arrays and their corresponding centroids.
    """
    H, W = img.shape[:2]
    h_step_factor = int(tiling_factor - 1)
    w_step_factor = int(tiling_factor)
    h_step = H // h_step_factor
    w_step = W // w_step_factor

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


def _merge_unique_contours(
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


@fn.NodeDecorator(
    node_id="fn.microscopy.images.segment",
    name="Segment",
    outputs=[
        {"name": "contours"},
        {"name": "centers"},
    ],
    # default_io_options={
    # "resolution_factor": {"value_options": {"min": 2, "max": 4}},
    # "thresh": {"value_options": {"min": 0, "max": 1}},
    # "max_eccentricity": {"value_options": {"min": 0, "max": 1}},
    # },
    # default_render_options={"data": {"src": "out"}},
)
def segment(
    image: np.ndarray,
    model: SegmentModels = SegmentModels.model_1,
    exclude_background: bool = True,
    tiling: bool = False,
    tiling_factor: int = 4,
) -> Tuple[list, list]:
    """
    Detects particles in the input image using a super-resolution model and contour extraction.

    'https://arxiv.org/abs/1806.03535#:~:text=https%3A//doi.org/10.48550/arXiv.1806.03535'
    'https://arxiv.org/abs/2203.02284#:~:text=https%3A//doi.org/10.48550/arXiv.2203.02284'

    Parameters:
    image (np.ndarray): The input grayscale image as a NumPy array.
    model: A pre-trained segmentation model for particle detection.
    exclude_background (bool, optional): If True, excludes contours whose center lies in the background.
    Default is True.
    tiling (bool, optional): If True, uses tiling to improve performance on large images. Default is False.
    tiling_factor (int, optional): The number of tiles to divide the image into along each axis if tiling is used.
    Default is 4.

    Returns:
    tuple: Contour arrays and their corresponding centroids.
    """
    model = SegmentModels.v(model)
    pretrained_model = StarDist2D.from_pretrained(model)
    # pretrained_model.config.use_gpu = True
    threshold_global_otsu = threshold_otsu(image)
    original_foreground = image >= threshold_global_otsu

    if exclude_background:
        original_foreground = (
            remove_background(original_foreground) >= threshold_global_otsu
        )

    cnts_1, cents_1 = _contours(image, pretrained_model, original_foreground)
    all_conts, all_cents = cnts_1, cents_1

    if tiling:
        cnts_2, cents_2 = _contours_crop(
            image, pretrained_model, tiling_factor, original_foreground
        )
        all_conts, all_cents = _merge_unique_contours(cnts_1, cents_1, cnts_2, cents_2)

    print(f"{len(all_conts)} particles detected!")
    contours = all_conts
    centers = all_cents
    return contours, centers


IMAGE_NODE_SHELF = fn.Shelf(
    nodes=[increase_resolution, segment],
    subshelves=[],
    name="Image",
    description="Advanced Image Analysis",
)

# MICROSCOP_NODE_SHELF = fn.Shelf(
#     nodes=[],
#     subshelves=[IMAGE_NODE_SHELF],
#     name="Image",
#     description="Image advanced analysis",
# )
