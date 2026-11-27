"""Deterministic two-boundary perception, measured geometry, and dry-run lifecycle."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from cli import parse_args
from tape import TapeSettings, detect_tape
from tape_geometry import Calibration, StepLimits, plan_between_edges, plan_step
from tape_proposal import MockProposalSink, Proposal, propose
from tape_run import run_tape
from tape_view import render_tape


def strip(*points):
    frame = np.zeros((120, 160, 3), np.uint8)
    if points:
        cv2.polylines(frame, [np.array(points, np.int32)], False, (255, 0, 0), 5)
    return frame


def lane(*centers):
    left = strip(*[(x-26, y) for x, y in centers])
    right = strip(*[(x+26, y) for x, y in centers])
    return cv2.bitwise_or(left, right)


def synthetic_calibration():
    # Synthetic replay only, 5 mm/pixel with the optical floor origin 5 cm forward.
    return dict(version=1, source_size=[160, 120], processed_size=[160, 120], rotate=0,
                valid_image_quad_px=[[0, 42], [159, 42], [159, 119], [0, 119]],
                image_to_ground_m=[[-.005, 0, .4], [0, -.005, .595], [0, 0, 1]],
                camera_offset_left_forward_m=[0, .05])


class DetectionTests(unittest.TestCase):
    def test_ordered_centres_on_straight_offset_diagonal_and_curved_strips(self):
        for xs in ((80, 80, 80), (61, 61, 61), (80, 67, 54), (85, 56, 80)):
            with self.subTest(xs=xs):
                ys = (119, 80, 46)
                frame = lane(*zip(xs, ys))
                cv2.circle(frame, (140, 108), 2, (255, 0, 0), -1)
                route = detect_tape(frame, TapeSettings())
                self.assertEqual(route.status, "tracked")
                for x, y in zip(xs, ys):
                    closest = min(route.points, key=lambda p: abs(p[1]-y))
                    self.assertAlmostEqual(closest[0], x, delta=4)
                    self.assertAlmostEqual(closest[1], y, delta=5)
                self.assertTrue(all(a[1] > b[1] for a, b in zip(route.points, route.points[1:])))
                self.assertIn(route.image_target, route.points)
                self.assertEqual(route.selected[108, 140], 0)
                self.assertGreater(route.evidence.quality, .7)
                self.assertEqual(route.points, detect_tape(frame, TapeSettings()).points)

    def test_missing_specks_far_only_and_roi(self):
        for frame in (lane(), lane((20, 70), (80, 80)), lane((40, 115), (40, 115))):
            self.assertFalse(detect_tape(frame, TapeSettings()).points)
        frame = lane((70, 119), (70, 10))
        self.assertEqual(int(detect_tape(frame, TapeSettings()).mask[:42].sum()), 0)
        red = frame[:, :, ::-1].copy()
        self.assertFalse(detect_tape(red, TapeSettings()).points)
        self.assertTrue(detect_tape(red, TapeSettings(hsv_low=(0, 80, 40),
                                                     hsv_high=(10, 255, 255))).points)

    def test_gap_bound(self):
        frame = lane((80, 119), (80, 44))
        frame[78:80] = 0
        route = detect_tape(frame, TapeSettings())
        self.assertLess(route.points[-1][1], 78)
        self.assertLess(route.evidence.continuity, 1)
        frame[69:86] = 0
        route = detect_tape(frame, TapeSettings())
        self.assertTrue(route.points)
        self.assertGreaterEqual(min(y for _, y in route.points), 86)
        self.assertGreaterEqual(route.image_target[1], 86)

    def test_fork_or_crossing_on_either_boundary(self):
        for a, b in [((54, 80), (76, 46)), ((35, 82), (73, 82)),
                     ((37, 110), (72, 54))]:
            frame = lane((80, 119), (80, 44))
            cv2.line(frame, a, b, (255, 0, 0), 5)
            route = detect_tape(frame, TapeSettings())
            self.assertIn("ambiguous", route.status)
            self.assertFalse(route.points)

    def test_outside_clutter_at_clipped_near_edge_leaves_the_straight_lane(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.line(frame, (0, 397), (232, 265), (255, 0, 0), 11)
        cv2.line(frame, (639, 397), (407, 265), (255, 0, 0), 11)
        cv2.line(frame, (54, 365), (0, 367), (255, 0, 0), 5)
        for image in (frame, cv2.flip(frame, 1)):
            route = detect_tape(image, TapeSettings())
            self.assertEqual((route.status, route.turn_direction), ("tracked", 0))
            self.assertTrue(all(abs(x - 320) < 5 for x, _ in route.points))
        cv2.line(frame, (54, 365), (132, 338), (255, 0, 0), 5)
        self.assertEqual(detect_tape(frame, TapeSettings()).status, "ambiguous_split_boundary")

    def test_single_strip_and_pair_on_same_side_stop(self):
        for frame in (strip((50, 119), (50, 42)), strip((110, 119), (110, 42)),
                      lane((38, 119), (38, 42))):
            route = detect_tape(frame, TapeSettings())
            self.assertEqual(route.status, "missing_boundary_pair")
            self.assertFalse(route.points)

    def test_single_outside_elbow_suggests_an_in_place_turn(self):
        frame = strip((159, 119), (119, 72), (0, 72))
        for direction, image in ((-1, frame), (1, cv2.flip(frame, 1))):
            with self.subTest(direction=direction):
                route = detect_tape(image, TapeSettings())
                self.assertEqual(route.status, "turn_only")
                self.assertEqual(route.turn_direction, direction)
                self.assertFalse(route.points)
                self.assertGreater(direction * (route.turn_arm[-1][0] - 80), 10)
        outward = strip((145, 119), (107, 72), (159, 72))
        self.assertEqual(detect_tape(outward, TapeSettings()).status, "missing_boundary_pair")

    def test_single_outside_elbow_can_enter_from_a_side_above_the_near_line(self):
        frame = strip((0, 92), (34, 69), (159, 66))
        for direction, image in ((1, frame), (-1, cv2.flip(frame, 1))):
            with self.subTest(direction=direction):
                route = detect_tape(image, TapeSettings())
                self.assertEqual((route.status, route.turn_direction), ("turn_only", direction))
                self.assertGreater(route.evidence.quality, .25)
                self.assertFalse(route.points)
        for image in (strip((16, 92), (45, 69), (159, 66)), strip((0, 92), (159, 54))):
            self.assertFalse(detect_tape(image, TapeSettings()).turn_arm)

    def test_short_side_entering_strip_turns_away_and_fork_does_not(self):
        frame = strip((0, 88), (41, 75))
        for direction, image in ((1, frame), (-1, cv2.flip(frame, 1))):
            route = detect_tape(image, TapeSettings())
            self.assertEqual((route.status, route.turn_direction), ("edge_turn", direction))
            self.assertTrue(route.turn_arm)
        cv2.line(frame, (14, 84), (50, 90), (255, 0, 0), 5)
        self.assertFalse(detect_tape(frame, TapeSettings()).turn_arm)

    def test_short_pair_can_start_where_one_boundary_exits_the_frame(self):
        left = strip((0, 119), (28, 61))
        clipped = strip((159, 86), (120, 69))
        route = detect_tape(left | clipped, TapeSettings())
        self.assertEqual(route.status, "tracked")
        self.assertTrue(route.short_pair)
        self.assertGreaterEqual(len(route.points), 3)
        self.assertLess(route.points[0][1], .80 * left.shape[0])
        floating = strip((150, 86), (120, 69))
        self.assertEqual(detect_tape(left | floating, TapeSettings()).status, "missing_boundary_pair")

    def test_clipped_pair_can_track_three_well_supported_crossings(self):
        frame = strip((0, 119), (55, 44)) | strip((159, 90), (117, 78))
        route = detect_tape(frame, TapeSettings())
        self.assertEqual(route.status, "tracked")
        self.assertTrue(route.short_pair)
        self.assertEqual(len(route.points), 3)
        self.assertEqual(route.evidence.continuity, 1.)

    def test_outer_lane_is_ignored_and_converging_edges_keep_the_midpoint(self):
        frame = strip((25, 119), (65, 46))
        frame |= strip((135, 119), (95, 46))
        frame |= strip((151, 119), (151, 46))
        route = detect_tape(frame, TapeSettings())
        self.assertEqual(route.status, "tracked")
        self.assertTrue(all(abs(x-80) < 2 for x, _ in route.points))
        self.assertEqual(route.selected[110, 151], 0)
        self.assertGreater(route.right_edge[0][0] - route.left_edge[0][0],
                           route.right_edge[-1][0] - route.left_edge[-1][0])

    def test_pale_blue_glare_connected_to_a_tape_tip_is_excluded(self):
        tape = lane((80, 119), (80, 58))
        frame = np.full_like(tape, 140)
        frame[43:60, 91:147] = (195, 178, 162)
        frame[tape[:, :, 0] != 0] = (255, 0, 0)
        route = detect_tape(frame, TapeSettings())
        self.assertEqual(route.status, "tracked")
        self.assertEqual(route.mask[50, 121], 0)
        self.assertTrue(all(tape[y, x, 0] for x, y in (*route.left_edge, *route.right_edge)))

    def test_clipped_slanted_corner_follows_the_observed_outside_stem(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.polylines(frame, [np.array([(0, 446), (120, 353), (0, 328)])], False, (255, 0, 0), 17)
        cv2.line(frame, (639, 362), (500, 244), (255, 0, 0), 15)
        cv2.line(frame, (502, 244), (120, 214), (184, 164, 153), 2)
        route = detect_tape(frame, TapeSettings())
        self.assertEqual((route.status, route.turn_direction), ("tracked", -1))
        self.assertTrue(route.short_pair)
        self.assertLess(route.points[-1][0], route.points[0][0] - 150)
        outside = np.array(route.right_edge)
        self.assertLess(np.linalg.norm(np.diff(outside, axis=0), axis=1).max(), 25)
        self.assertTrue(all(frame[y, x, 0] for x, y in route.right_edge))

    def test_both_sides_of_a_right_angle_corner(self):
        for sign in (1, -1):
            def x(v):
                return 80 + sign*(v-80)
            frame = strip((x(54), 119), (x(54), 49), (x(155), 49))
            frame |= strip((x(106), 119), (x(106), 86), (x(155), 86))
            route = detect_tape(frame, TapeSettings())
            self.assertEqual(route.status, "tracked")
            end_x, end_y = route.points[-1]
            self.assertGreater(sign*(end_x-80), 60)
            self.assertAlmostEqual(end_y, (49+86)/2, delta=2)
            self.assertTrue(all(frame[round(y), round(x), 0] == 0 for x, y in route.points))

    def test_slanted_corner_recovers_a_faint_outer_arm(self):
        frame = strip((0, 119), (44, 87), (0, 87)) | strip((159, 119), (112, 60))
        hsv = np.array([[[108, 43, 184]]], np.uint8)
        faint = tuple(map(int, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))
        cv2.line(frame, (112, 60), (27, 60), faint, 1)
        cv2.line(frame, (33, 48), (107, 48), faint, 1)  # Unconnected background feature.
        for turn, image in ((-1, frame), (1, cv2.flip(frame, 1))):
            with self.subTest(turn=turn):
                route = detect_tape(image, TapeSettings())
                self.assertEqual(route.status, "tracked")
                self.assertEqual(route.turn_direction, turn)
                self.assertGreater(route.mask[60, 80], 0)
                self.assertEqual(route.mask[48, 80], 0)
                self.assertIn(route.image_target, route.points[:route.approach_end+1])
                self.assertGreater(turn * (route.points[-1][0] - route.image_target[0]), 35)


class MetricTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "cal.json"
        self.path.write_text(json.dumps(synthetic_calibration()))
        self.cal = Calibration.load(self.path)
        self.limits = StepLimits()

    def test_target_distance_sign_and_offset(self):
        for x, left in [(68, .06), (92, -.06), (80, 0)]:
            route = detect_tape(lane((x, 119), (x, 44)), TapeSettings())
            step = plan_step(route.points, self.cal, self.limits)
            np.testing.assert_allclose(step.target_m, [left, .25], atol=1e-10)
            np.testing.assert_allclose(step.target_px, [x, 79], atol=1e-10)
            self.assertAlmostEqual(step.angle_rad, math.atan2(left, .25))
            self.assertAlmostEqual(step.distance_m, math.hypot(left, .25))

    def test_single_elbow_proposes_only_a_calibrated_bounded_turn(self):
        route = detect_tape(strip((159, 119), (119, 72), (0, 72)), TapeSettings())
        base = Proposal("test:1", 1, 1., 2., "valid")
        proposal = propose(base, route, self.cal, self.limits)
        self.assertEqual(proposal.status, "PROPOSED")
        self.assertEqual([action.kind for action in proposal.actions], ["TURN"])
        self.assertAlmostEqual(proposal.actions[0].angle_rad, self.limits.max_turn_rad)
        self.assertEqual(propose(replace(base, calibration="required"), route, None, self.limits).actions, ())
        outside = replace(route, turn_arm=((119, 40), (20, 40)), can_estimate_turn=False)
        self.assertEqual(propose(base, outside, self.cal, self.limits).reason, "corner_outside_calibrated_region")
        self.assertAlmostEqual(propose(base, replace(outside, can_estimate_turn=True), self.cal,
                                       self.limits).actions[0].angle_rad, math.radians(12))

    def test_sparse_right_turn_proposes_only_a_small_estimated_pivot(self):
        route = detect_tape(strip((0, 88), (41, 75)), TapeSettings())
        base = Proposal("test:1", 1, 1., 2., "valid")
        proposal = propose(base, route, self.cal, self.limits)
        self.assertEqual(proposal.reason, "estimated_edge_turn")
        self.assertEqual([action.kind for action in proposal.actions], ["TURN"])
        self.assertAlmostEqual(proposal.actions[0].angle_rad, -math.radians(12))
        tighter = replace(self.limits, max_turn_rad=math.radians(8))
        self.assertAlmostEqual(propose(base, route, self.cal, tighter).actions[0].angle_rad, -math.radians(8))
        self.assertFalse(propose(replace(base, calibration="invalid"), route, None, self.limits).actions)

    def test_clipped_right_turn_keeps_the_image_hint_if_metric_planning_stops(self):
        cfg = TapeSettings()
        frame = strip((0, 92), (34, 69), (159, 66))
        route = detect_tape(frame, cfg)
        base = Proposal("test:1", 1, 1., 2., "valid")
        proposal = propose(base, route, self.cal, self.limits)
        self.assertEqual([action.kind for action in proposal.actions], ["TURN"])
        self.assertAlmostEqual(proposal.actions[0].angle_rad, -self.limits.max_turn_rad)
        stopped = proposal.stop("insufficient_measured_turn_arm")
        with patch("tape_view.draw_text") as draw:
            render_tape(frame, route, stopped, self.cal, cfg, 0, 0, True)
        labels = [call.args[1] for call in draw.call_args_list]
        self.assertTrue(any("TURN RIGHT (image only)" in label for label in labels))
        self.assertIn("CV: STOP  insufficient_measured_turn_arm", labels)

    def test_non_affine_projection_and_ground_interpolation(self):
        data = synthetic_calibration()
        data["image_to_ground_m"][2] = [0, .005, .5]
        self.path.write_text(json.dumps(data))
        cal = Calibration.load(self.path)
        step = plan_step([(80, 119), (80, 50)], cal, self.limits)
        self.assertAlmostEqual(step.target_m[1], .25)
        self.assertAlmostEqual(step.target_px[1], (.595 - .2*.5) / (.005+.2*.005))

    def test_measured_floor_corners_fit_the_projection(self):
        data = synthetic_calibration()
        data["image_to_ground_m"][2] = [0, .005, .5]
        self.path.write_text(json.dumps(data))
        route = [(80, 119), (80, 50)]
        expected = plan_step(route, Calibration.load(self.path), self.limits)

        matrix = np.asarray(data.pop("image_to_ground_m"))
        corners = np.column_stack((data["valid_image_quad_px"], np.ones(4))) @ matrix.T
        data["ground_quad_left_forward_m"] = (corners[:, :2] / corners[:, 2, None]).tolist()
        self.path.write_text(json.dumps(data))
        measured = plan_step(route, Calibration.load(self.path), self.limits)
        np.testing.assert_allclose(measured.target_m, expected.target_m, atol=1e-7)
        np.testing.assert_allclose(measured.target_px, expected.target_px, atol=1e-5)

        for incomplete in ([[None, None]] * 4, [[0, 0]] * 4):
            with self.subTest(incomplete=incomplete), self.assertRaises(ValueError):
                self.path.write_text(json.dumps({**data, "ground_quad_left_forward_m": incomplete}))
                Calibration.load(self.path)

    def test_metric_midpoint_projects_each_edge_and_checks_both_are_measured(self):
        data = synthetic_calibration()
        data["image_to_ground_m"][2] = [.004, 0, 1]
        self.path.write_text(json.dumps(data))
        cal = Calibration.load(self.path)
        left = [(50, 119), (50, 60)]
        right = [(110, 119), (110, 60)]
        step = plan_between_edges(left, right, cal, self.limits)
        expected_left = (.15 / 1.2 - .15 / 1.44) / 2
        self.assertAlmostEqual(step.target_m[0], expected_left)
        with self.assertRaisesRegex(ValueError, "outside_calibrated_region"):
            plan_between_edges([(-1, 119), (-1, 60)], right, cal, self.limits)

    def test_bent_route_uses_arc_for_lookahead_and_chord_for_drive(self):
        # First leg is 5 cm left and 5 cm forward, second leg runs straight forward.
        step = plan_step([(80, 119), (70, 109), (70, 50)], self.cal, self.limits)
        forward = .10 + self.limits.lookahead_m - math.hypot(.05, .05)
        np.testing.assert_allclose(step.target_m, [.05, forward])
        self.assertAlmostEqual(step.distance_m, math.hypot(.05, forward))

    def test_long_drive_across_inside_corner_stops(self):
        limits = replace(self.limits, lookahead_m=.8, max_step_m=.8, max_turn_rad=math.radians(70))
        left = [(70, 119), (70, 65), (140, 65)]
        right = [(90, 119), (90, 85), (140, 85)]
        with self.assertRaisesRegex(ValueError, "drive_crosses_boundary"):
            plan_between_edges(left, right, self.cal, limits)
        self.assertGreater(plan_between_edges(left, right, self.cal, self.limits).distance_m, 0)

    def test_corner_step_uses_the_approach_until_the_turn_is_close(self):
        left = [(59, 119), (59, 100), (58, 92), (58, 80), (54, 80), (52, 80)]
        right = [(101, 119), (101, 100), (102, 92), (102, 55), (75, 55), (65, 55)]
        approach = plan_between_edges(left, right, self.cal, self.limits, approach_end=2)
        self.assertEqual(approach.angle_rad, 0)
        np.testing.assert_allclose(approach.target_px, (80, 92))

        # At a closer viewpoint the first visible pair is already at the elbow.
        moved = replace(self.cal, offset=np.array([0, -.11]))
        turn = plan_between_edges(left[2:], right[2:], moved, self.limits, approach_end=0)
        self.assertGreater(turn.angle_rad, self.limits.deadband_rad)
        self.assertLess(turn.angle_rad, self.limits.max_turn_rad)

    def test_distance_limit_intersects_route_and_deadband(self):
        limits = replace(self.limits, max_step_m=.17)
        step = plan_step([(79, 119), (79, 43)], self.cal, limits)
        self.assertAlmostEqual(step.distance_m, .17)
        self.assertAlmostEqual(step.target_m[0], .005)
        self.assertEqual(step.angle_rad, 0)
        for points, cfg, error in [
            ([(20, 119), (20, 43)], self.limits, "turn_exceeds_limit"),
            ([(79, 119), (79, 117)], replace(self.limits, min_step_m=.1), "step_too_short"),
            ([(80, 55), (80, 43)], replace(self.limits, max_step_m=.30), "beyond_step_limit"),
            ([(80, 15), (80, 10)], self.limits, "outside_calibrated_region"),
            ([(80, 119), (np.nan, 50)], self.limits, "non_finite")]:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                plan_step(points, self.cal, cfg)
        with self.assertRaisesRegex(ValueError, "behind_robot"):
            plan_step([(80, 119), (80, 80)], replace(self.cal, offset=np.array([0, -.2])), limits)

    def test_calibrated_run_ends_at_first_exit(self):
        points = [(80, 119), (80, 110), (80, 25), (80, 80)]
        step = plan_step(points, self.cal, self.limits)
        self.assertAlmostEqual(step.target_px[1], 110)

    def test_invalid_calibrations_and_geometry_binding(self):
        for field, value in [
            ("image_to_ground_m", [[0, 0, 0]]*3),
            ("image_to_ground_m", [[1, 0, 0], [0, 1, 0], [0, 1, -60]]),
            ("camera_offset_left_forward_m", [float("nan"), 0]),
            ("valid_image_quad_px", [[0, 0], [60, 60], [0, 60], [60, 0]]),
            ("valid_image_quad_px", [[0, 0], [160, 0], [159, 60], [0, 60]]),
            ("processed_size", [120]), ("rotate", 45)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.path.write_text(json.dumps({**synthetic_calibration(), field: value}))
                Calibration.load(self.path)
        self.assertTrue(self.cal.matches((160, 120), (160, 120), 0))
        for args in [((320, 240), (160, 120), 0), ((160, 120), (80, 60), 0),
                     ((160, 120), (160, 120), 180)]:
            self.assertFalse(self.cal.matches(*args))

    def test_loss_and_calibration_failures_replace_last_good(self):
        route = detect_tape(lane((70, 119), (70, 44)), TapeSettings())
        base = Proposal("run:1", 1, 10., 11., "valid")
        clock = MagicMock(return_value=10.1)
        sink = MockProposalSink(clock)
        good = propose(base, route, self.cal, self.limits)
        self.assertEqual([action.kind for action in sink.publish(good).actions], ["TURN", "DRIVE"])
        one_side = detect_tape(strip((54, 119), (54, 44)), TapeSettings())
        self.assertFalse(sink.publish(propose(base, one_side, self.cal, self.limits)).actions)
        self.assertEqual(sink.latest.reason, "missing_boundary_pair")
        missing = detect_tape(lane(), TapeSettings())
        self.assertFalse(sink.publish(propose(base, missing, self.cal, self.limits)).actions)
        self.assertEqual(sink.latest.reason, "no_reliable_tape")
        for state in ("required", "invalid", "incompatible"):
            stop = propose(replace(base, calibration=state), route, None, self.limits)
            self.assertFalse(stop.actions)
            self.assertEqual(stop.reason, "calibration_"+state)
        sink.publish(good)
        clock.return_value = 11.
        self.assertEqual(sink.latest.reason, "stale_frame")
        self.assertFalse(sink.publish(good).actions)


class RunnerTests(unittest.TestCase):
    def test_preview_arms_disarms_and_sends_only_stop_on_q(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = path / "cal.json"
            config.write_text(json.dumps(synthetic_calibration()))
            argv = ["run", "--method", "tape", "--stream", "http://camera/stream", "--preview",
                    "--actuate-encoders", "--calibration", str(config), "--output-dir", str(path)]
            with patch.object(sys, "argv", argv):
                args = parse_args()
            frame = lane((80, 119), (80, 43))
            capture = MagicMock(received_at=10., frame_id=1, error="", drive_mode="AUTONOMOUS")
            capture.read.return_value = True, frame
            capture.get.side_effect = lambda prop: 30. if prop == cv2.CAP_PROP_FPS else 0.
            sink = MagicMock()
            sink.publish.side_effect = lambda proposal: proposal
            with (patch("tape_run.LiveCapture", return_value=capture),
                  patch("esp32_drive.Esp32DriveSink", return_value=sink),
                  patch("tape_run.time.monotonic", return_value=10.01),
                  patch("tape_run.open_preview"), patch("tape_run.cv2.imshow"),
                  patch("tape_run.cv2.destroyAllWindows"),
                  patch("tape_run.preview_key", side_effect=[255, ord("a"), ord("s"), ord("a"), ord("q")])):
                sink.status = None
                run_tape(args, args.stream)
            commands = [call.args[0] for call in sink.publish.call_args_list]
            self.assertEqual([command.reason for command in commands],
                             ["actuation_disarmed", "dry_run", "actuation_disarmed", "dry_run", "user_exit"])
            self.assertEqual([bool(command.actions) for command in commands], [False, True, False, True, False])
            records = [json.loads(line) for line in next(path.glob("*.jsonl")).read_text().splitlines()]
            self.assertEqual([record["actuation_armed"] for record in records if record["type"] == "frame"],
                             [False, True, False, True])

    def test_default_tape_step_limit_depends_on_actuation(self):
        common = ["run.py", "--method", "tape", "--stream", "http://camera/stream",
                  "--calibration", "camera_calibration.json"]
        for flags, distance in (([], .65), (["--actuate-encoders"], .30),
                                (["--actuate", "1000", "2000"], .30),
                                (["--max-step-m", ".50"], .50)):
            with self.subTest(flags=flags), patch.object(sys, "argv", common + flags):
                self.assertAlmostEqual(parse_args().step_limits.max_step_m, distance)

    def test_replay_picker_opens_in_footage_and_hands_selection_to_tape_mode(self):
        from run import run

        footage = Path(__file__).resolve().parent / "footage"
        chosen = footage / "selected.mov"
        tk = MagicMock()
        tk.filedialog.askopenfilename.return_value = str(chosen)
        with (patch.dict(sys.modules, {"tkinter": tk}), patch("tape_run.run_tape") as runner,
              patch.object(sys, "argv", ["run.py", "--method", "tape", "--preview", "--debug"])):
            args = parse_args()
            run(args)
        self.assertEqual(tk.filedialog.askopenfilename.call_args.kwargs["initialdir"], str(footage))
        runner.assert_called_once_with(args, chosen)

    def test_recorded_cli_tracks_then_loses_tape_and_writes_video(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video = folder / "synthetic.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
            try:
                self.assertTrue(writer.isOpened())
                writer.write(lane((68, 119), (68, 44)))
                writer.write(lane())
            finally:
                writer.release()
            config = folder / "cal.json"
            config.write_text(json.dumps(synthetic_calibration()))
            command = [sys.executable, str(Path(__file__).with_name("run.py")), str(video),
                       "--method", "tape", "--calibration", str(config), "--debug", "--output-dir", str(folder)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            _, first, lost, end = map(json.loads, next(folder.glob("*.jsonl")).read_text().splitlines())
            self.assertGreater(first["proposal"]["actions"][0]["angle_rad"], 0)
            self.assertEqual(lost["proposal"]["actions"], [])
            self.assertEqual(end["proposal"]["reason"], "end_of_input")
            output = cv2.VideoCapture(str(next(folder.glob("*.mp4"))))
            try:
                self.assertEqual(output.get(cv2.CAP_PROP_FRAME_COUNT), 2)
                self.assertTrue(output.read()[0])
            finally:
                output.release()

    def run_live(self, mode):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        config = path / "cal.json"
        data = synthetic_calibration()
        if mode == "incompatible":
            data["rotate"] = 180
        config.write_text("broken" if mode == "invalid" else json.dumps(data))
        argv = ["run", "--method", "tape", "--stream", "http://camera/stream",
                "--output-dir", str(path), "--debug", "--max-frame-age-ms", "100"]
        if mode != "uncalibrated":
            argv.extend(["--calibration", str(config)])
        if mode not in ("disconnect", "processing_failure"):
            argv.extend(["--max-frames", "1"])
        with patch.object(sys, "argv", argv):
            args = parse_args()
        frame = lane((80, 119), (80, 43))
        source = MagicMock(received_at=10., frame_id=3, error="", drive_mode="AUTONOMOUS",
                           get=MagicMock(return_value=0.))
        def read():
            if source.read.call_count == 1 or mode == "processing_failure":
                return True, frame
            source.error = "simulated stream loss"
            return False, None
        source.read.side_effect = read
        clock = MagicMock(return_value=10.2 if mode == "stale_input" else 10.01)
        sink = MockProposalSink(clock)
        def render(*params):
            if mode == "stale_at_publish":
                clock.return_value = 10.2
            return render_tape(*params)
        def detect(*params):
            if mode == "processing_failure" and source.read.call_count > 1:
                raise ValueError("simulated processing failure")
            if mode == "stale_after_detection":
                clock.return_value = 10.2
            return detect_tape(*params)
        at_release = []
        source.release.side_effect = lambda: at_release.append(sink.latest)
        with (patch("tape_run.LiveCapture", return_value=source),
              patch("tape_run.time.monotonic", clock),
              patch("tape_run.detect_tape", side_effect=detect) as detector,
              patch("tape_run.render_tape", side_effect=render)):
            if mode in ("processing_failure", "disconnect"):
                with self.assertRaisesRegex(ValueError, "simulated"):
                    run_tape(args, args.stream, sink)
            else:
                run_tape(args, args.stream, sink)
        if mode == "stale_input":
            detector.assert_not_called()
        self.assertEqual(at_release, [sink.latest])
        self.assertEqual(sink.latest.status, "STOP")
        self.assertFalse(sink.latest.actions)
        records = [json.loads(s) for s in next(path.glob("*.jsonl")).read_text().splitlines()]
        self.assertEqual([r["type"] for r in records], ["run", "frame", "end"])
        self.assertEqual(records[1]["drive_mode"], "AUTONOMOUS")
        return records

    def test_terminal_stop_precedes_cleanup_after_loss_failure_and_shutdown(self):
        for mode, reason in [("disconnect", "source_failure"), ("processing_failure", "processing_failure"),
                             ("shutdown", "frame_limit")]:
            with self.subTest(mode=mode):
                _, frame, end = self.run_live(mode)
                self.assertEqual(frame["proposal"]["status"], "PROPOSED")
                self.assertEqual(end["proposal"]["reason"], reason)

    def test_rejection_stages(self):
        for mode, reason in [("stale_input", "stale_frame"), ("stale_after_detection", "stale_frame"),
                             ("stale_at_publish", "stale_frame"), ("invalid", "calibration_invalid"),
                             ("uncalibrated", "calibration_required"),
                             ("incompatible", "calibration_incompatible")]:
            with self.subTest(mode=mode):
                _, frame, _ = self.run_live(mode)
                self.assertEqual(frame["proposal"]["reason"], reason)
                self.assertFalse(frame["proposal"]["actions"])
                if mode != "stale_input":
                    self.assertTrue(frame["centreline_px"])
                    self.assertTrue(frame["image_target_px"])


if __name__ == "__main__":
    unittest.main()
