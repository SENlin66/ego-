"""CPU-only checks for the local FoundationStereo compatibility helpers."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stereo"))

from Utils import freeze_model, get_resize_keep_aspect_ratio


class FakeModel:
    def __init__(self):
        self.evaluating = False
        self.trainable = True

    def eval(self):
        self.evaluating = True
        return self

    def requires_grad_(self, value):
        self.trainable = value
        return self


class UtilsTest(unittest.TestCase):
    def test_freeze_model(self):
        model = FakeModel()
        self.assertIs(freeze_model(model), model)
        self.assertTrue(model.evaluating)
        self.assertFalse(model.trainable)

    def test_resize_is_aligned_and_bounded(self):
        self.assertEqual(get_resize_keep_aspect_ratio(768, 1024), (768, 1024))
        self.assertEqual(get_resize_keep_aspect_ratio(1536, 2048), (928, 1232))
        with self.assertRaises(ValueError):
            get_resize_keep_aspect_ratio(768, 1024, divider=0)


if __name__ == "__main__":
    unittest.main()
