import funcnodes as fn
from typing import Tuple
import numpy as np
from stardist.models import StarDist2D

import cv2
from skimage.filters import threshold_otsu
from sklearn.decomposition import PCA
from super_image import PanModel
from ._resolution import process_image, deprocess_image
from ._contours import (
    SegmentModels,
    remove_background,
    _contours,
    _contours_crop,
    merge_unique_contours,
)
from ._circles import (
    calculateMaxCircleDiameters,
    calculateMinCircleDiameters,
    calculateCircularityScores,
)


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
    inputs = process_image(image)

    # Generate the super-resolution image
    preds = res_model(inputs)

    # Convert the prediction to a higher resolution image
    high_res_image = deprocess_image(preds)

    return high_res_image


@fn.NodeDecorator(
    node_id="fn.microscopy.images.segment",
    name="Segment",
    outputs=[
        {"name": "contours"},
        {"name": "centers"},
    ],
)
def segment(
    image: np.ndarray,
    model: SegmentModels = SegmentModels.model_1,
    exclude_background: bool = True,
    tiling: bool = False,
    tiling_factor: int = 4,
) -> Tuple[list, list]:
    """
    Detects particles in the input image using a pretrained model and contour extraction.

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
    tiling_factor = int(tiling_factor)
    if image.ndim == 3:
        image = image[:, :, 0]
    model = SegmentModels.v(model)
    pretrained_model = StarDist2D.from_pretrained(model)
    # pretrained_model.config.use_gpu = True
    original_foreground = np.ones(image.shape, dtype=bool)
    if exclude_background:
        threshold_global_otsu = threshold_otsu(image)
        original_foreground = remove_background(image) >= threshold_global_otsu

    cnts_1, cents_1 = _contours(image, pretrained_model, original_foreground)
    all_conts, all_cents = cnts_1, cents_1

    if tiling:
        cnts_2, cents_2 = _contours_crop(
            image, pretrained_model, tiling_factor, original_foreground
        )
        all_conts, all_cents = merge_unique_contours(cnts_1, cents_1, cnts_2, cents_2)

    print(f"{len(all_conts)} particles detected!")
    contours = all_conts
    centers = all_cents
    return contours, centers


@fn.NodeDecorator(
    node_id="fn.microscopy.images.circles",
    name="Circles from Contours",
)
def calculate_circles(contours: list, centers: list, pixel_size: float = 1.0) -> dict:
    maxCircleX, maxCircleY, maxcircleR, maxCircleD = calculateMinCircleDiameters(
        contours, pixel_size
    )
    minCircleX, minCircleY, mincircleR, minCircleD = calculateMaxCircleDiameters(
        contours, centers, pixel_size
    )
    circularityScores = calculateCircularityScores(contours)
    circles = {
        "max_center_x": maxCircleX,
        "max_center_y": maxCircleY,
        "max_radius": maxcircleR,
        "max_diameter": maxCircleD,
        "min_center_x": minCircleX,
        "min_center_y": minCircleY,
        "min_radius": mincircleR,
        "min_diameter": minCircleD,
        "circularityScores": circularityScores,
    }
    return circles


@fn.NodeDecorator(
    node_id="fn.microscopy.images.ellipse",
    name="Ellipses from Contours",
)
def calculate_ellipses(contours: list) -> dict:
    """
    Calculates the ellipse parameters for each contour and aligns the ellipse orientation

    based on the principal Axes determined by PCA.
    """
    ellipseAngles = []
    ellipseAxes = []
    ellipseCenters = []
    eccentricities = []

    number = len(contours)

    for i in range(number):
        contour = contours[i]

        # Fit an initial ellipse to get preliminary parameters
        (center, (MA, ma), angle) = cv2.fitEllipse(contour)

        # Check if values are valid
        if np.isnan(MA) or np.isnan(ma) or np.isnan(center[0]) or np.isnan(center[1]):
            center, axes, angle = (None, None), None, None
        else:
            # Perform PCA on the contour points to find the principal Axes
            points = contour.reshape(-1, 2)
            pca = PCA(n_components=2)
            pca.fit(points)
            principal_direction = np.degrees(np.arctan2(*pca.components_[0]))

            # Adjust ellipse angle based on the PCA-derived direction
            angle_difference = (angle - principal_direction + 360) % 360
            if angle_difference > 90 and angle_difference < 270:
                angle = (angle + 180) % 360  # Flip ellipse direction
            center = tuple(map(int, center))
            axes = (int(MA / 2), int(ma / 2))
        # Calculate eccentricity only if both axes are valid
        if axes[0] > 0 and axes[1] > 0:
            ratio = ((axes[1]) ** 2) / ((2 * axes[0]) ** 2)
            eccentricity = np.sqrt(max(0, 1 - ratio))
        else:
            # Default eccentricity if ellipse fitting failed
            eccentricity = 0
        eccentricities.append(eccentricity)
        ellipseAngles.append(angle)
        ellipseAxes.append(axes)
        ellipseCenters.append(center)
    return {
        "center_x": [center[0] for center in ellipseCenters],
        "center_y": [center[1] for center in ellipseCenters],
        "axes_x": [axis[0] for axis in ellipseAxes],
        "axes_y": [axis[1] for axis in ellipseAxes],
        "angle": ellipseAngles,
        "eccentricities": eccentricities,
    }


IMAGE_NODE_SHELF = fn.Shelf(
    nodes=[increase_resolution, segment, calculate_circles, calculate_ellipses],
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
