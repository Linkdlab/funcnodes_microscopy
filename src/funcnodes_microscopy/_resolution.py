import numpy as np
import cv2
import torch
from torch import Tensor


def process_image(image: np.ndarray) -> Tensor:
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


def deprocess_image(pred: Tensor) -> np.ndarray:
    pred = pred.data.cpu().numpy()
    pred = pred[0].transpose((1, 2, 0)) * 255.0

    # Clamp values to the range [0, 255]
    pred = np.clip(pred, 0, 255)

    pred = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return pred
