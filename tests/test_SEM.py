import funcnodes as fn
import unittest
import numpy as np
from funcnodes_microscopy.SEM import sem_image
import os


class TestSEM(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), "1908248.tif"), "rb") as f:
            self.tiffbytes = f.read()

    async def test_sem_image(self):
        load_sem: fn.Node = sem_image()
        load_sem.inputs["input"].value = self.tiffbytes
        self.assertIsInstance(load_sem, fn.Node)
        await load_sem
        image = load_sem.outputs["image"].value
        metadata = load_sem.outputs["metadata"].value
        self.assertIsInstance(image, np.ndarray)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata["Pixel Size (nm)"], 7.344)
