"""Contracts at the perception/planning and replay-control boundaries."""

import unittest
from unittest.mock import patch

import numpy as np

from navigation import NavigationSettings
from pipeline import PathSettings, PerceptionMasks, plan_frame
from video_io import preview_stopped


class PipelineTests(unittest.TestCase):
    def test_compare_keeps_segmentation_route_and_independent_inputs(self):
        floor = np.ones((120, 160), bool)
        floor[30:60, 65:95] = False
        masks = PerceptionMasks(floor, ~floor, np.zeros_like(floor), True)
        # Nondefault footprint settings must survive comparison-mode wiring.
        settings = PathSettings(corridor_near=.15, corridor_far=.03)
        plain = plan_frame(masks, settings, NavigationSettings())
        depth = np.full(floor.shape, np.nan, np.float32)
        comparison = plan_frame(masks, settings, NavigationSettings(),
                                dict(values=depth), depth_mode="compare")
        self.assertTrue(plain.points)
        self.assertEqual(plain.points, comparison.points)
        self.assertEqual(plain.navigation, comparison.navigation)
        self.assertEqual(comparison.navigation["corridor_settings"]["corridor_near"], .15)
        self.assertFalse(comparison.fusion["rejected"].any())
        np.testing.assert_array_equal(comparison.floor_mask, floor)
        self.assertFalse(floor[30:60, 65:95].any())
        self.assertTrue(np.isnan(depth).all())

    def test_missing_fused_depth_does_not_reuse_previous_route(self):
        floor = np.ones((120, 160), bool)
        masks = PerceptionMasks(floor, ~floor, None, True)
        settings, navigation = PathSettings(), NavigationSettings()
        self.assertTrue(plan_frame(masks, settings, navigation).points)
        missing = dict(values=np.full(floor.shape, np.nan, np.float32))
        plan = plan_frame(masks, settings, navigation, missing)
        self.assertEqual(plan.points, [])
        self.assertEqual(plan.navigation["alternatives"], [])
        self.assertEqual(plan.navigation["decision"]["action"], "STOP")
        self.assertEqual(plan.fusion["status"], "DEPTH UNAVAILABLE")
        self.assertTrue(floor.all())

    def test_unusable_semantic_view_never_suggests_motion(self):
        floor = np.zeros((120, 160), bool)
        masks = PerceptionMasks.from_result(
            dict(masks=dict(floor=floor, wall=floor, other=floor, unknown=~floor),
                 status="UNUSABLE FRAME"), "semantic")
        plan = plan_frame(masks, PathSettings(), NavigationSettings())
        self.assertEqual(plan.points, [])
        self.assertEqual(plan.navigation["alternatives"], [])
        self.assertEqual(plan.navigation["decision"]["action"], "STOP")
        self.assertFalse(plan.navigation["decision"]["motion_authorized"])

    def test_preview_pause_quit_and_continue(self):
        image = np.zeros((80, 120, 3), np.uint8)
        for keys, stopped, delays in (([32, ord("q")], True, [100, 0]),
                                      ([27], True, [100]),
                                      ([32, 32], False, [100, 0]),
                                      ([-1], False, [100])):
            with self.subTest(keys=keys), patch("video_io.cv2.imshow"), \
                    patch("video_io.cv2.waitKey", side_effect=keys) as wait:
                self.assertEqual(preview_stopped(image, 10), stopped)
                self.assertEqual([call.args[0] for call in wait.call_args_list], delays)


if __name__ == "__main__":
    unittest.main()
