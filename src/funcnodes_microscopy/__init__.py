import funcnodes as fn
from .SEM import SEM_NODE_SHELF
from .images import IMAGE_NODE_SHELF

__version__ = "0.1.0"

NODE_SHELF = fn.Shelf(
    nodes=[],
    name="Microscopy",
    description="The nodes of Funcnodes Microscopy package",
    subshelves=[SEM_NODE_SHELF, IMAGE_NODE_SHELF],
)
