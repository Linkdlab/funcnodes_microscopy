import sys
import os
import unittest
from tests import test_images, test_SEM  # noqa: E402

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)  # in case test folder is not in sys path
from all_nodes_test_base import TestAllNodesBase  # noqa: E402

# fn.config.IN_NODE_TEST = True


sub_test_classes = []

for mod in (test_SEM, test_images):
    for cls in mod.__dict__.values():
        if isinstance(cls, type) and issubclass(cls, unittest.IsolatedAsyncioTestCase):
            sub_test_classes.append(cls)


class TestAllNodes(TestAllNodesBase):
    # in this test class all nodes should be triggered at least once to mark them as testing

    # if you tests your nodes with in other test classes, add them here
    # this will automtically extend this test class with the tests in the other test classes
    # but this will also mean if you run all tests these tests might run multiple times
    # also the correspondinig setups and teardowns will not be called, so the tests should be
    # independently callable
    sub_test_classes = sub_test_classes

    # if you have specific nodes you dont want to test, add them here
    # But why would you do that, it will ruin the coverage?!
    # a specific use case would be ignore nodes that e.g. load a lot of data, but there we would recommend
    # to write tests with patches and not ignore them.
    ignore_nodes = []

    # async def test_first_node(self):
    #     node = fnmodule.FirstNode()
    #     node.inputs["x"].value = "foo"
    #     await node
    #     self.assertEqual(node.get_output("out").value, "bar")
