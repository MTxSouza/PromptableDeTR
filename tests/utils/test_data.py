"""
This module tests the data.py module from the utils package.
"""
from unittest import TestCase

import torch

from utils.data import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


# Classes.
class TestBox(TestCase):


    # Tests.
    def test_cxcy_to_xyxy_convertion_without_class(self):
        """
        Tests the box_cxcywh_to_xyxy function without the class column.
        """
        # Bounding boxes in (center_x, center_y, width, height) format.
        boxes = torch.tensor([
            [[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.125, 0.125]],
            [[0.75, 0.75, 0.375, 0.375], [0.125, 0.125, 0.5, 0.5]]
        ])

        # Image dimensions.
        width, height = 640, 480

        # Convert the bounding boxes.
        xyxy = box_cxcywh_to_xyxy(
            boxes=boxes, 
            width=width, 
            height=height
        )

        # Assertions.
        self.assertEqual(xyxy.size(), (2, 2, 4))
        self.assertEqual(xyxy[0, 0].long().tolist(), [240, 180, 400, 300])
        self.assertEqual(xyxy[0, 1].long().tolist(), [120, 90, 200, 150])
        self.assertEqual(xyxy[1, 0].long().tolist(), [360, 270, 600, 450])
        self.assertEqual(xyxy[1, 1].long().tolist(), [0, 0, 240, 180])
    

    def test_cxcy_to_xyxy_convertion_with_class(self):
        """
        Tests the box_cxcywh_to_xyxy function with the class column.
        """
        # Bounding boxes in (class, center_x, center_y, width, height) format.
        boxes = torch.tensor([
            [[0, 0.5, 0.5, 0.25, 0.25], [1, 0.25, 0.25, 0.125, 0.125]],
            [[0, 0.75, 0.75, 0.375, 0.375], [1, 0.125, 0.125, 0.5, 0.5]]
        ])

        # Image dimensions.
        width, height = 640, 480

        # Convert the bounding boxes.
        xyxy = box_cxcywh_to_xyxy(
            boxes=boxes, 
            width=width, 
            height=height
        )

        # Assertions.
        self.assertEqual(xyxy.size(), (2, 2, 5))
        self.assertEqual(xyxy[0, 0].long().tolist(), [0, 240, 180, 400, 300])
        self.assertEqual(xyxy[0, 1].long().tolist(), [1, 120, 90, 200, 150])
        self.assertEqual(xyxy[1, 0].long().tolist(), [0, 360, 270, 600, 450])
        self.assertEqual(xyxy[1, 1].long().tolist(), [1, 0, 0, 240, 180])


    def test_xyxy_to_cxcywh_convertion_without_class(self):
        """
        Tests the box_xyxy_to_cxcywh function without the class column.
        """
        # Bounding boxes in (x_min, y_min, x_max, y_max) format.
        boxes = torch.tensor([
            [[240, 180, 400, 300], [120, 90, 200, 150]],
            [[360, 270, 600, 450], [0, 0, 240, 180]]
        ])

        # Image dimensions.
        width, height = 640, 480

        # Convert the bounding boxes.
        cxcywh = box_xyxy_to_cxcywh(
            boxes=boxes, 
            width=width, 
            height=height
        )

        # Assertions.
        self.assertEqual(cxcywh.size(), (2, 2, 4))
        self.assertEqual(cxcywh[0, 0].tolist(), [0.5, 0.5, 0.25, 0.25])
        self.assertEqual(cxcywh[0, 1].tolist(), [0.25, 0.25, 0.125, 0.125])
        self.assertEqual(cxcywh[1, 0].tolist(), [0.75, 0.75, 0.375, 0.375])
        self.assertEqual(cxcywh[1, 1].tolist(), [0.1875, 0.1875, 0.375, 0.375])


    def test_xyxy_to_cxcywh_convertion_with_class(self):
        """
        Tests the box_xyxy_to_cxcywh function with the class column.
        """
        # Bounding boxes in (class, x_min, y_min, x_max, y_max) format.
        boxes = torch.tensor([
            [[0, 240, 180, 400, 300], [1, 120, 90, 200, 150]],
            [[0, 360, 270, 600, 450], [1, 0, 0, 240, 180]]
        ])

        # Image dimensions.
        width, height = 640, 480

        # Convert the bounding boxes.
        cxcywh = box_xyxy_to_cxcywh(
            boxes=boxes, 
            width=width, 
            height=height
        )

        # Assertions.
        self.assertEqual(cxcywh.size(), (2, 2, 5))
        self.assertEqual(cxcywh[0, 0].tolist(), [0, 0.5, 0.5, 0.25, 0.25])
        self.assertEqual(cxcywh[0, 1].tolist(), [1, 0.25, 0.25, 0.125, 0.125])
        self.assertEqual(cxcywh[1, 0].tolist(), [0, 0.75, 0.75, 0.375, 0.375])
        self.assertEqual(cxcywh[1, 1].tolist(), [1, 0.1875, 0.1875, 0.375, 0.375])
