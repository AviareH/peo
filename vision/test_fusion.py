"""Synthetic fusion cases, including near floor, protrusions and missing depth."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from fusion import fuse_floor_depth
from playback import four_panel
from path_estimation import estimate_path, PATH_COLOUR
import run


class FusionTests(unittest.TestCase):
    def scene(self):
        floor = np.ones((120, 160), bool)
        depth = np.broadcast_to(.5 + 4 * np.arange(120, dtype=np.float32)[:, None] / 120,
                                floor.shape).copy()
        return floor, depth

    def test_near_floor_remains_valid_but_protrusion_is_rejected(self):
        floor, depth = self.scene()
        depth[30:70, 70:100] += 1.2
        floor_before, depth_before = floor.copy(), depth.copy()
        fused = fuse_floor_depth(floor, depth)
        self.assertEqual(fused["status"], "FUSED FLOOR + DEPTH")
        self.assertTrue(fused["mask"][100, 80])  # Near ground, not an obstacle.
        self.assertFalse(fused["mask"][50, 80])  # Object mislabelled as floor.
        self.assertTrue(fused["rejected"][50, 80])
        self.assertGreater(fused["cost"][50, 65], fused["cost"][50, 15])
        np.testing.assert_array_equal(floor, floor_before)
        np.testing.assert_array_equal(depth, depth_before)

    def test_relative_depth_scale_and_shift_do_not_change_decision(self):
        floor, depth = self.scene()
        depth[30:70, 70:100] += 1.2
        a = fuse_floor_depth(floor, depth)
        b = fuse_floor_depth(floor, depth * 7 + 25)
        np.testing.assert_array_equal(a["mask"], b["mask"])
        np.testing.assert_allclose(a["cost"], b["cost"], atol=1e-4)

    def test_no_floor_is_added_and_invalid_depth_cannot_be_used(self):
        floor, depth = self.scene()
        floor[:40] = False
        depth[75:85, 50:60] = np.nan
        fused = fuse_floor_depth(floor, depth)
        self.assertFalse(np.any(fused["mask"] & ~floor))
        self.assertFalse(fused["mask"][75:85, 50:60].any())
        self.assertTrue(np.isfinite(fused["cost"]).all())
        for invalid in (np.full_like(depth, np.nan), np.ones_like(depth)):
            self.assertFalse(fuse_floor_depth(floor, invalid)["mask"].any())

    def test_centre_gap_between_objects_is_reachable(self):
        floor, depth = self.scene()
        floor[:, :25] = False
        floor[:, 135:] = False
        floor[30:75, 30:60] = False
        floor[30:75, 100:130] = False
        depth[30:75, 30:60] += 1.2
        depth[30:75, 100:130] += 1.2
        fused = fuse_floor_depth(floor, depth)
        points = estimate_path(fused["mask"], cost_map=fused["cost"])
        self.assertTrue(points)
        self.assertLess(points[-1][1], 30)
        self.assertTrue(all(60 <= x < 100 for x, y in points if 30 <= y < 75))

    def test_cost_changes_route_without_leaving_floor(self):
        floor = np.ones((240, 320), bool)
        costs = np.zeros(floor.shape, np.float32)
        costs[30:190, :180] = 4
        points = estimate_path(floor, cost_map=costs)
        self.assertTrue(points)
        self.assertGreater(points[-1][0], 180)
        with self.assertRaises(ValueError):
            estimate_path(floor, cost_map=np.full(floor.shape, np.nan))

    def test_quad_layout_preserves_raw_and_individual_model_panels(self):
        raw = np.full((120, 160, 3), 100, np.uint8)
        segmentation = np.full_like(raw, (30, 60, 90))
        depth_panel = np.full_like(raw, (20, 40, 80))
        floor, depth = self.scene()
        fusion = fuse_floor_depth(floor, depth)
        output = four_panel(raw, segmentation, depth_panel, fusion, [(80, 90), (80, 50)], (1, 2, 3))
        self.assertEqual(output.shape, (240, 320, 3))
        np.testing.assert_array_equal(output[:120, 160:], segmentation)
        np.testing.assert_array_equal(output[120:, :160], depth_panel)
        np.testing.assert_array_equal(output[60, 20], raw[60, 20])
        np.testing.assert_array_equal(output[60, 80], PATH_COLOUR)
        self.assertTrue(np.all(raw == 100))

    def test_quad_video_and_compare_mode_with_independent_model_outputs(self):
        floor, depth = self.scene()
        depth[25:55, 70:100] += 1.2
        masks = {"floor": floor, "wall": np.zeros_like(floor),
                 "other": np.zeros_like(floor), "unknown": np.zeros_like(floor)}
        segmentation = dict(masks=masks, confidence=np.ones(floor.shape, np.float32),
                            boxes=[], status="TEST SEGMENTATION", processing_ms=1.)
        depth_result = dict(values=depth, status="RELATIVE DEPTH", processing_ms=2.)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "scene.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.full((120, 160, 3), 100, np.uint8))
            writer.release()
            for mode in ("fused", "compare"):
                semantic_model, depth_model = MagicMock(), MagicMock()
                semantic_model.device = "cpu"
                semantic_model.metadata.return_value = {"model": "test"}
                semantic_model.process.return_value = segmentation
                depth_model.metadata.return_value = {"model": "test", "affects_path": False}
                depth_model.process.return_value = depth_result
                command = ["run.py", str(video), "--method", "semantic", "--model", "b2",
                           "--depth", "--depth-mode", mode, "--output-dir", str(root / mode)]
                with patch("sys.argv", command), patch("semantic.SemanticDetector", return_value=semantic_model), \
                        patch("depth.DepthEstimator", return_value=depth_model):
                    run.run(run.parse_args())
                records = [json.loads(x) for x in (root / mode / "scene_detections.jsonl").read_text().splitlines()]
                self.assertEqual(records[0]["depth"]["affects_path"], mode == "fused")
                self.assertEqual(semantic_model.process.call_count, 3)
                self.assertEqual(depth_model.process.call_count, 3)
                self.assertTrue(records[1]["path"]["points"])
                if mode == "fused":
                    self.assertGreater(records[1]["fusion"]["rejected_floor_fraction"], 0)
                else:
                    self.assertEqual(records[1]["fusion"]["rejected_floor_fraction"], 0)
                capture = cv2.VideoCapture(str(root / mode / "scene_annotated.mp4"))
                for _ in range(3):
                    ok, output = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(output.shape, (240, 320, 3))
                self.assertFalse(capture.read()[0])
                capture.release()
        self.assertTrue(floor.all())


if __name__ == "__main__":
    unittest.main()
