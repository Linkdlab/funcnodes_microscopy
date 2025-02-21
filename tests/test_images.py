import funcnodes as fn
import unittest
from funcnodes_core.testing import setup

from funcnodes_microscopy.SEM import sem_image
from funcnodes_microscopy.images import (
    increase_resolution,
    segment,
    calculate_circles,
    calculate_ellipses,
)
import os


class TestSegmentation(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        setup()
        with open(os.path.join(os.path.dirname(__file__), "1908248.tif"), "rb") as f:
            self.tiffbytes = f.read()

    async def test_images(self):
        load_sem: fn.Node = sem_image()
        load_sem.inputs["input"].value = self.tiffbytes
        self.assertIsInstance(load_sem, fn.Node)
        img_array = load_sem.outputs["image"]
        res: fn.Node = increase_resolution()
        res.inputs["image"].connect(img_array)
        seg: fn.Node = segment()
        seg.inputs["image"].connect(img_array)
        self.assertIsInstance(seg, fn.Node)
        circle: fn.Node = calculate_circles()
        circle.inputs["contours"].connect(seg.outputs["contours"])
        circle.inputs["centers"].connect(seg.outputs["centers"])
        self.assertIsInstance(circle, fn.Node)
        ellipse: fn.Node = calculate_ellipses()
        ellipse.inputs["contours"].connect(seg.outputs["contours"])
        self.assertIsInstance(ellipse, fn.Node)
        await fn.run_until_complete(ellipse, circle, seg, res, load_sem)
        self.assertEqual(img_array.value.shape, (768, 1024))
        conts = seg.outputs["contours"].value
        circle_dict = circle.outputs["out"].value
        ellipse_dict = ellipse.outputs["out"].value
        self.assertIsInstance(circle_dict, dict)
        self.assertIsInstance(ellipse_dict, dict)
        self.assertIsInstance(conts, list)
