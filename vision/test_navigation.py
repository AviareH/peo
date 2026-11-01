"""Repositioning corridors and mock decisions never authorize actual motion."""
import unittest

import cv2
import numpy as np

from navigation import (alternative_corridors, choose_decision, plan_navigation,
                         draw_navigation, POTENTIAL_COLOUR)
from path_estimation import estimate_path


class NavigationTests(unittest.TestCase):
    def test_off_origin_floor_is_shown_without_connecting_through_obstacle(self):
        floor = np.zeros((240, 320), bool)
        floor[:, :110] = True
        self.assertEqual(estimate_path(floor), [])
        obstacle = np.zeros_like(floor)
        obstacle[160:, 145:175] = True
        navigation = plan_navigation(floor, [], obstacle)
        self.assertEqual(navigation["decision"]["action"], "REPOSITION_LEFT")
        self.assertFalse(navigation["decision"]["motion_authorized"])
        self.assertEqual(len(navigation["alternatives"]), 1)
        candidate = navigation["alternatives"][0]
        self.assertFalse(candidate["reachable_from_car"])
        self.assertLess(candidate["points"][0][0], 110)
        stroke = np.zeros(floor.shape, np.uint8)
        cv2.polylines(stroke, [np.asarray(candidate["points"], np.int32)], False, 255, 2)
        self.assertFalse(np.any((stroke != 0) & ~floor))
        self.assertFalse(stroke[:, 160].any())

    def test_both_sides_have_at_most_one_candidate_each(self):
        floor = np.ones((240, 320), bool)
        floor[:, 110:210] = False
        alternatives = alternative_corridors(floor)
        self.assertEqual({x["side"] for x in alternatives}, {"left", "right"})
        self.assertEqual(len(alternatives), 2)

    def test_forward_and_steering_follow_current_path(self):
        obstacle = np.zeros((240, 320), bool)
        for points, expected in (([(160, 220), (160, 180), (160, 140)], "FORWARD"),
                                 ([(160, 220), (100, 180), (80, 140)], "STEER_LEFT"),
                                 ([(160, 220), (220, 180), (240, 140)], "STEER_RIGHT")):
            decision = choose_decision(points, [], obstacle)
            self.assertEqual(decision["action"], expected)
            self.assertFalse(decision["motion_authorized"])

    def test_foreground_blockage_suggests_reversing_only_with_rear_check(self):
        obstacle = np.zeros((240, 320), bool)
        obstacle[100:, 60:260] = True
        decision = choose_decision([], [], obstacle)
        self.assertEqual(decision["action"], "BACK_UP")
        self.assertTrue(decision["rear_clearance_required"])
        self.assertFalse(decision["motion_authorized"])
        self.assertIn("CHECK REAR", decision["label"])

    def test_distant_background_and_unknown_do_not_trigger_backup(self):
        obstacle = np.zeros((240, 320), bool)
        obstacle[:100] = True
        self.assertEqual(choose_decision([], [], obstacle)["action"], "STOP")
        obstacle[:] = True
        unknown = np.ones_like(obstacle)
        self.assertEqual(choose_decision([], [], obstacle, unknown)["action"], "STOP")
        self.assertEqual(choose_decision([], [], obstacle, usable=False)["action"], "STOP")

    def test_no_alternative_for_narrow_pocket_and_no_stale_candidate(self):
        floor = np.zeros((240, 320), bool)
        floor[170:, :110] = True
        self.assertEqual(alternative_corridors(floor), [])
        floor[:, :110] = True
        self.assertTrue(alternative_corridors(floor))
        self.assertEqual(alternative_corridors(np.zeros_like(floor)), [])

    def test_cyan_drawing_is_distinct_and_does_not_change_floor_input(self):
        floor = np.zeros((240, 320), bool)
        floor[:, :110] = True
        navigation = plan_navigation(floor, [], np.zeros_like(floor))
        image = np.zeros((240, 320, 3), np.uint8)
        draw_navigation(image, navigation, 320)
        self.assertTrue(np.any(np.all(image == POTENTIAL_COLOUR, axis=2)))
        self.assertTrue(floor[:, :110].all())
        self.assertFalse(floor[:, 110:].any())


if __name__ == "__main__":
    unittest.main()
