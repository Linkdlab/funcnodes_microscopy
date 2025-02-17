import funcnodes as fn
import unittest
from funcnodes_microscopy.SEM import upload_sem_image
from funcnodes_microscopy.images import (
    increase_resolution,
    segment,
    # ThresholdTypes,
    # RetrievalModes,
    # ContourApproximationModes,
)
import os

fn.config.IN_NODE_TEST = True


class TestSegmentation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with open(os.path.join(os.path.dirname(__file__), "1908248.tif"), "rb") as f:
            self.tiffbytes = f.read()

    async def test_classical_segmentation(self):
        load_sem: fn.Node = upload_sem_image()
        load_sem.inputs["input"].value = self.tiffbytes
        self.assertIsInstance(load_sem, fn.Node)
        res: fn.Node = increase_resolution()
        res.inputs["image"].connect(load_sem.outputs["image"])
        # res.inputs["resolution_factor"].value = 3
        seg: fn.Node = segment()
        seg.inputs["image"].connect(res.outputs["out"])

        # seg.inputs["iter"].value = 3
        # seg.inputs["pixel_size"].value = 7.344
        # seg.inputs["min_diameter"].value = 10

        # print()
        self.assertIsInstance(seg, fn.Node)
        await fn.run_until_complete(seg, res, load_sem)
        conts = seg.outputs["contours"].value
        # print(h_res.shape)
        self.assertIsInstance(conts, list)
