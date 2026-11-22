"""Depth diagnostics and loader contracts without downloading a model."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import unittest

import cv2
import numpy as np

from depth import DepthEstimator, DepthView, depth_summary, floor_panel, MODEL_REVISION


class DepthTests(unittest.TestCase):
    def test_pinned_loader_uses_local_cache_and_safe_weights(self):
        processor_type, model_type = MagicMock(), MagicMock()
        model = model_type.from_pretrained.return_value
        model.to.return_value = model
        model.eval.return_value = model
        transformers = SimpleNamespace(AutoImageProcessor=processor_type,
                                       AutoModelForDepthEstimation=model_type)
        with patch.dict("sys.modules", {"torch": MagicMock(), "transformers": transformers}):
            estimator = DepthEstimator(device="cpu", offline=True)
        options = model_type.from_pretrained.call_args.kwargs
        self.assertEqual(options["revision"], MODEL_REVISION)
        self.assertTrue(options["use_safetensors"])
        self.assertTrue(options["weights_only"])
        self.assertTrue(options["local_files_only"])
        self.assertFalse(estimator.metadata()["affects_path"])

    def test_display_range_stays_fixed_between_frames(self):
        values = np.tile(np.arange(120, dtype=np.float32), (100, 1))
        view = DepthView()
        first = view.render(values)
        limits = view.limits
        second = view.render(values + 25)
        self.assertEqual(view.limits, limits)
        self.assertFalse(np.array_equal(first[70, 50], second[70, 50]))
        low, high = limits
        index = int(np.clip((75 - low) / (high - low), 0, 1) * 255)
        expected = cv2.applyColorMap(np.array([[index]], np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        np.testing.assert_array_equal(second[70, 50], expected)

    def test_invalid_and_constant_depth_do_not_establish_display_scale(self):
        view = DepthView()
        values = np.full((80, 120), np.nan, np.float32)
        values[60, 50] = np.inf
        self.assertEqual(depth_summary(values), dict(valid_fraction=0., relative_min=None, relative_max=None))
        output = view.render(values)
        self.assertEqual(output.shape, (80, 120, 3))
        self.assertFalse(output[60:].any())
        view.render(np.ones((80, 120), np.float32))
        self.assertIsNone(view.limits)

    def test_unusable_frame_skips_model_and_returns_no_stale_depth(self):
        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator.synchronize = MagicMock()
        estimator.model = MagicMock()
        result = estimator.process(np.zeros((80, 120, 3), np.uint8))
        self.assertTrue(np.isnan(result["values"]).all())
        self.assertEqual(result["status"], "UNUSABLE FRAME")
        estimator.model.assert_not_called()

    def test_floor_panel_preserves_mask_below_title(self):
        floor = np.zeros((80, 120), bool)
        floor[40:, 50:70] = True
        panel = floor_panel(floor)
        np.testing.assert_array_equal(panel[40:, :, 0] != 0, floor[40:])
        self.assertEqual(panel.shape, (80, 120, 3))


if __name__ == "__main__":
    unittest.main()
