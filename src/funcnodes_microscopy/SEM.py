from typing import Tuple, Dict, Any
from funcnodes import NodeDecorator, Shelf
import tifffile as tiff
import numpy as np
import io


def handle_metadata(sem_meta: Dict[str, Tuple[str, Any, str]]) -> Dict[str, Any]:
    """
    Converts the CZ_SEM dictionary into a more readable format.

    Parameters:
    ----------
    sem_meta : Dict[str, Tuple[str, Any, str]]
        The original CZ_SEM dictionary with metadata.

    Returns:
    -------
    Dict[str, Any]
        A dictionary with the format 'Description (Unit)': value.
    """
    readable_meta = {}
    for key, value in sem_meta.items():
        if isinstance(value, tuple) and len(value) == 3:
            description, val, unit = value
            readable_key = f"{description} ({unit})"
            readable_meta[readable_key] = val
        elif isinstance(value, tuple) and len(value) == 2:
            # Handle cases without units
            description, val = value
            readable_meta[description] = val
    return readable_meta


def _read_sem_image(data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads a TIFF (SEM) image file and returns the image data along with its metadata.

    Parameters:
    ----------
    file_path : str
        The path to the TIFF file.

    Returns:
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        - An ndarray (numpy array) representing the image data.
        - A dictionary with the metadata extracted from the TIFF file.

    Raises:
    ------
    ValueError:
        If the provided file is not in a .tif or .tiff format.
    """
    byteio = io.BytesIO(data)
    # Open the image using tifffile for metadata
    with tiff.TiffFile(byteio) as tif:
        meta_dict = tif.pages[0].tags  # Get the first page's tags (metadata)
        image = tif.asarray()  # Load the image data as a numpy array

    # Extract metadata into a dictionary
    metadata = {tag.name: tag.value for tag in meta_dict.values()}

    # Process the CZ_SEM metadata if it exists and add it to the main dictionary
    if "CZ_SEM" in metadata:
        readable_cz_sem = handle_metadata(metadata["CZ_SEM"])
        # Add the key-value pairs from CZ_SEM to the main metadata dictionary
        metadata.update(readable_cz_sem)
        # Remove the 'CZ_SEM' key to avoid redundancy
        del metadata["CZ_SEM"]
    return image, metadata


@NodeDecorator(
    id="microscopy.sem.upload",
    name="Process SEM Image",
    inputs=[
        {
            "name": "input",
        },
    ],
    outputs=[
        {
            "name": "image",
        },
        {
            "name": "metadata",
        },
    ],
)
def sem_image(input: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _read_sem_image(input)


SEM_NODE_SHELF = Shelf(
    nodes=[sem_image],
    subshelves=[],
    name="Scanning Electron Microscopy (SEM)",
    description="Handling of SEM images in tif (tiff) format",
)
