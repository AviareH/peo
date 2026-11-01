"""Offline label grouping tests; no model download needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from semantic import group_predictions, annotate_semantic, SemanticDetector, MODELS


class SemanticTests(unittest.TestCase):
    def test_pinned_checkpoint_formats_and_offline_loading(self):
        for name, safetensors in (("b0", True), ("b2", False)):
            with self.subTest(model=name):
                processor_type = MagicMock()
                model_type = MagicMock()
                model = model_type.from_pretrained.return_value
                model.to.return_value = model
                model.eval.return_value = model
                model.config.id2label = {0: "floor", 1: "wall"}
                fake_transformers = SimpleNamespace(SegformerImageProcessor=processor_type,
                                                     SegformerForSemanticSegmentation=model_type)
                with patch.dict("sys.modules", {"torch": MagicMock(), "transformers": fake_transformers}):
                    detector = SemanticDetector(device="cpu", offline=True, model_name=name)
                kwargs = model_type.from_pretrained.call_args.kwargs
                self.assertEqual(kwargs["revision"], MODELS[name][1])
                self.assertNotEqual(kwargs["revision"], "main")
                self.assertEqual(kwargs["use_safetensors"], safetensors)
                self.assertTrue(kwargs["weights_only"])
                self.assertTrue(kwargs["local_files_only"])
                self.assertEqual(detector.model_name, name)

    def test_label_names_scores_and_unknown(self):
        labels = np.array([[7, 4, 9, 3, 7, 9]])
        scores = np.array([[.9, .8, .7, .95, .3, np.nan]])
        masks = group_predictions(labels, scores,
                                  {"7": "floor", "4": "wall", "9": "chair", "3": "rug"}, .5)
        np.testing.assert_array_equal(masks["floor"], [[True, False, False, True, False, False]])
        np.testing.assert_array_equal(masks["wall"], [[False, True, False, False, False, False]])
        np.testing.assert_array_equal(masks["other"], [[False, False, True, False, False, False]])
        np.testing.assert_array_equal(masks["unknown"], [[False, False, False, False, True, True]])
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()), np.ones_like(labels))

    def test_incompatible_label_map_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "floor and wall"):
            group_predictions(np.zeros((2, 2)), np.ones((2, 2)), {0: "road"}, .5)

    def test_render_preserves_hole_and_image_dimensions(self):
        labels = np.zeros((100, 120), dtype=int)
        labels[60:80, 50:70] = 2
        masks = group_predictions(labels, np.ones_like(labels), {0: "floor", 1: "wall", 2: "chair"}, .5)
        result = dict(masks=masks, confidence=np.ones_like(labels), processing_ms=1,
                      status="SEMANTIC ESTIMATES")
        frame = np.full((100, 120, 3), 100, np.uint8)
        overlay, classes, confidence = annotate_semantic(frame, result)
        self.assertEqual(overlay.shape, frame.shape)
        self.assertEqual(confidence.shape, frame.shape)
        self.assertGreater(classes[90, 20, 1], classes[90, 20, 2])
        self.assertGreater(classes[70, 60, 2], classes[70, 60, 1])


if __name__ == "__main__":
    unittest.main()
