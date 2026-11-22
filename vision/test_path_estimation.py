"""Deterministic route checks, independent of a camera or model download."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from path_estimation import (estimate_path, draw_path, PATH_COLOUR, DEFAULT_ENDPOINT_MARGIN,
                             corridor_mask, corridor_radius, corridor_envelope, draw_corridor)


class PathTests(unittest.TestCase):
    def assert_on_floor(self, floor, points):
        self.assertTrue(points)
        overlay = np.zeros((*floor.shape, 3), np.uint8)
        draw_path(overlay, points, floor.shape[1], floor_mask=floor)
        marked = np.any(overlay != 0, axis=2)
        self.assertFalse(np.any(marked & (floor == 0)))
        self.assertFalse(np.any((corridor_mask(floor.shape, points) != 0) & (floor == 0)))
        self.assertTrue(np.any(np.all(overlay == PATH_COLOUR, axis=2)))

    def test_open_floor_goes_straight(self):
        floor = np.full((240, 320), 255, np.uint8)
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertLess(max(x for x, _ in points) - min(x for x, _ in points), 4)
        self.assertLess(points[-1][1], 60)

    def test_default_path_extends_above_previous_search_limit(self):
        floor = np.full((240, 320), 255, np.uint8)
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertLess(points[-1][1], int(floor.shape[0] * .12))
        # The longer lookahead must retain endpoint clearance from the image edge.
        self.assertGreaterEqual(points[-1][1], floor.shape[1] * DEFAULT_ENDPOINT_MARGIN - 2)

    def test_path_search_has_no_top_crop(self):
        floor = np.ones((720, 160), np.uint8)
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertLess(points[-1][1], floor.shape[0] * .04)
        self.assertGreaterEqual(points[-1][1], floor.shape[1] * DEFAULT_ENDPOINT_MARGIN - 2)

    def test_retries_endpoints_after_full_resolution_rejection(self):
        floor = np.ones((120, 160), np.uint8)
        with patch("path_estimation._path_fits_floor", side_effect=[False, True]) as fits:
            points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertEqual(fits.call_count, 2)

    def test_obstacle_is_avoided_and_farther_floor_reached(self):
        floor = np.full((240, 320), 255, np.uint8)
        floor[60:120, 140:220] = 0
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertLess(points[-1][1], 60)
        self.assertTrue(any(x < 140 for x, y in points if 60 <= y <= 120))

    def test_slanted_floor_corridor_can_reach_farther_sideways(self):
        ys, xs = np.indices((240, 320))
        floor = (np.abs(xs - (160 + 1.3 * (ys - 201))) < 65).astype(np.uint8)
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertLess(points[-1][1], 90)
        self.assertLess(points[-1][0], 40)

    def test_lateral_steps_cannot_jump_a_thin_separating_wall(self):
        floor = np.ones((240, 320), np.uint8)
        floor[:, 100:102] = 0
        floor[:100, 102:] = 0
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertTrue(all(x > 102 and y > 100 for x, y in points))

    def test_barrier_and_disconnected_floor_are_not_crossed(self):
        floor = np.full((240, 320), 255, np.uint8)
        floor[110:125] = 0
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        self.assertGreater(min(y for _, y in points), 125)

    def test_blocked_start_or_short_reach_gives_no_path(self):
        floor = np.full((240, 320), 255, np.uint8)
        floor[180:, 140:180] = 0
        self.assertEqual(estimate_path(floor), [])
        floor[:] = 0
        floor[210:] = 255
        self.assertEqual(estimate_path(floor), [])
        self.assertEqual(estimate_path(np.zeros_like(floor)), [])

    def test_narrow_slit_is_rejected(self):
        floor = np.zeros((240, 320), np.uint8)
        floor[:, 159:162] = 255
        self.assertEqual(estimate_path(floor), [])

    def test_small_pocket_at_obstacle_base_is_not_a_forward_route(self):
        floor = np.zeros((240, 320), np.uint8)
        floor[175:] = 255
        floor[140:175, 145:175] = 255
        # This recreates the previous failure: a thin line can reach a pocket,
        # but there is insufficient space for a useful stand-off endpoint.
        legacy = estimate_path(floor, margin_fraction=.012, endpoint_margin_fraction=.012,
                               minimum_forward_fraction=.12, corridor_near=0,
                               corridor_far=0, corridor_buffer=0)
        self.assertTrue(legacy)
        self.assertEqual(estimate_path(floor), [])

    def test_endpoint_stays_back_from_obstacle(self):
        floor = np.full((240, 320), 255, np.uint8)
        floor[:110] = 0
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)
        distance = cv2.distanceTransform(np.pad((floor != 0).astype(np.uint8), 1),
                                         cv2.DIST_L2, 5)[1:-1, 1:-1]
        x, y = points[-1]
        self.assertGreaterEqual(distance[y, x], floor.shape[1] * DEFAULT_ENDPOINT_MARGIN - 2)

    def test_full_corridor_rejects_gap_that_centre_line_can_fit(self):
        floor = np.zeros((240, 320), np.uint8)
        floor[:, 145:175] = 255
        options = dict(margin_fraction=.005, endpoint_margin_fraction=.005)
        self.assertTrue(estimate_path(floor, corridor_near=0, corridor_far=0,
                                      corridor_buffer=0, **options))
        self.assertEqual(estimate_path(floor, **options), [])

    def test_safety_buffer_can_reject_an_otherwise_wide_corridor(self):
        floor = np.zeros((240, 320), np.uint8)
        floor[:, 120:200] = 255
        self.assertTrue(estimate_path(floor, corridor_buffer=0))
        self.assertEqual(estimate_path(floor, corridor_buffer=.08), [])

    def test_perspective_width_and_connecting_segments_are_swept(self):
        points = [(160, 200), (120, 80)]
        mask = corridor_mask((240, 320), points)
        self.assertGreater(np.count_nonzero(mask[200]), np.count_nonzero(mask[80]))
        self.assertTrue(mask[140, 140])
        self.assertGreater(corridor_radius(200, 240, 320), corridor_radius(80, 240, 320))

    def test_small_hole_inside_footprint_is_not_hidden_by_downsampling(self):
        floor = np.ones((240, 320), np.uint8)
        floor[100, 165] = 0
        points = estimate_path(floor)
        self.assert_on_floor(floor, points)

    def test_invalid_corridor_settings_are_rejected(self):
        floor = np.ones((240, 320), np.uint8)
        for options in (dict(corridor_far=.3), dict(corridor_buffer=-.01),
                        dict(corridor_near=float("nan"))):
            with self.assertRaises(ValueError):
                estimate_path(floor, **options)

    def test_display_envelope_flares_to_base_without_rounded_caps(self):
        points = [(160, 190), (160, 70)]
        mask, left, right = corridor_envelope((240, 320), points, extend_to_base=True)
        self.assertFalse(mask[:70].any())
        self.assertGreater(np.count_nonzero(mask[-1]), 3 * np.count_nonzero(mask[70]))
        self.assertEqual(left[-1][1], 239)
        self.assertEqual(right[-1][1], 239)
        hypothetical, _, _ = corridor_envelope((240, 320), points)
        self.assertFalse(hypothetical[191:].any())

    def test_display_envelope_cannot_cross_obstacles_or_disconnected_floor(self):
        floor = np.ones((240, 320), bool)
        floor[:, 200:220] = False
        floor[210:, 155:165] = False
        mask, _, right = corridor_envelope(floor.shape, [(160, 190), (160, 70)], floor, True)
        self.assertFalse(np.any((mask != 0) & ~floor))
        self.assertFalse(mask[:, 220:].any())
        self.assertFalse(mask[210:].any())
        self.assertLessEqual(max(x for x, y in right), 199)

    def test_dotted_sides_and_faint_infill(self):
        image = np.full((240, 320, 3), 100, np.uint8)
        points = [(160, 190), (160, 70)]
        draw_corridor(image, points, PATH_COLOUR, extend_to_base=True)
        self.assertLess(int(np.abs(image[150, 160].astype(int) - 100).max()), 15)
        _, left, _ = corridor_envelope((240, 320), points, extend_to_base=True)
        dots = [np.array_equal(image[y, x], PATH_COLOUR) for x, y in left]
        self.assertTrue(any(dots))
        self.assertFalse(all(dots))


if __name__ == "__main__":
    unittest.main()
