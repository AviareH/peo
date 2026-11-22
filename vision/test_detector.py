"""Hardware-free checks: segmentation, failure cases, and full video replay."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from detector import ObstacleDetector, annotate


class DetectorTests(unittest.TestCase):
    def floor(self):
        return np.full((360, 640, 3), 160, dtype=np.uint8)

    def test_clear_floor_and_objects(self):
        detector = ObstacleDetector()
        self.assertEqual(detector.process(self.floor())["boxes"], [])
        for colour in [(0, 0, 220), (40, 40, 40)]:
            frame = self.floor()
            cv2.rectangle(frame, (70, 170), (220, 280), colour, -1)
            result = detector.process(frame)
            self.assertEqual(len(result["boxes"]), 1)
            box = result["boxes"][0]
            self.assertLessEqual(abs(box["x"] - 70), 5)
            self.assertGreater(box["width"], 140)

    def test_top_region_and_tiny_noise_are_ignored(self):
        detector = ObstacleDetector()
        detector.process(self.floor())
        frame = self.floor()
        frame[0:100, 0:200] = 0
        frame[200:202, 80:82] = 0
        self.assertEqual(detector.process(frame)["boxes"], [])

    def test_unusable_frame_does_not_calibrate(self):
        detector = ObstacleDetector()
        for value in (0, 255):
            result = detector.process(np.full((360, 640, 3), value, dtype=np.uint8))
            self.assertEqual(result["status"], "UNUSABLE FRAME")
            self.assertEqual(result["boxes"], [])
            self.assertFalse(np.any(result["floor_mask"]))
            self.assertIsNone(detector.floor)

    def test_floor_follows_shape_preserves_holes_and_reaches_above_roi(self):
        frame = np.full((360, 640, 3), (40, 40, 100), dtype=np.uint8)
        polygon = np.array([[240, 90], [400, 90], [620, 359], [20, 359]])
        cv2.fillPoly(frame, [polygon], (160, 160, 160))
        # A floor-coloured island is not connected to the foreground floor.
        cv2.rectangle(frame, (20, 70), (80, 110), (160, 160, 160), -1)
        cv2.rectangle(frame, (290, 190), (350, 240), (0, 0, 220), -1)
        detector = ObstacleDetector()
        result = detector.process(frame)
        floor = result["floor_mask"]
        self.assertEqual(floor[110, 320], 255)  # Above obstacle ROI.
        self.assertEqual(floor[340, 80], 255)  # Well outside centre sample.
        self.assertEqual(floor[340, 560], 255)
        self.assertEqual(floor[215, 320], 0)  # Hole for the obstacle.
        self.assertEqual(floor[150, 50], 0)  # Outside trapezoid.
        self.assertEqual(floor[90, 50], 0)  # Disconnected island.
        overlay = annotate(frame, result, detector.settings)
        self.assertGreater(overlay[340, 80, 1], overlay[340, 80, 0])
        np.testing.assert_array_equal(overlay[215, 320], frame[215, 320])

    def test_edge_boundary_separates_floor_coloured_wall(self):
        frame = self.floor()
        cv2.line(frame, (0, 150), (639, 150), (20, 20, 20), 3)
        floor = ObstacleDetector().process(frame)["floor_mask"]
        self.assertEqual(floor[250, 320], 255)
        self.assertEqual(floor[100, 320], 0)

    def test_no_floor_when_foreground_reference_is_occluded(self):
        detector = ObstacleDetector()
        detector.process(self.floor())
        frame = np.full((360, 640, 3), (0, 0, 160), dtype=np.uint8)
        frame[80:140, 200:400] = 160
        self.assertFalse(np.any(detector.process(frame)["floor_mask"]))

    def test_red_hue_wrap_is_floor(self):
        hsv = np.full((360, 640, 3), (179, 200, 160), dtype=np.uint8)
        detector = ObstacleDetector()
        detector.process(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        hsv[:, :, 0] = 1
        self.assertEqual(detector.process(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))["boxes"], [])

    def test_video_cli_and_output_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "phone sample.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
            self.assertTrue(writer.isOpened())
            for index in range(6):
                frame = self.floor()
                if index:
                    cv2.rectangle(frame, (70, 170), (220, 280), (0, 0, 220), -1)
                writer.write(frame)
            writer.release()
            script = Path(__file__).with_name("run.py")
            command = [sys.executable, str(script), str(source), "--debug", "--output-dir", str(root / "out")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = [json.loads(line) for line in (root / "out/phone sample_detections.jsonl").read_text().splitlines()]
            frames = [r for r in records if r["type"] == "frame"]
            self.assertEqual(len(frames), 6)
            self.assertFalse(frames[0]["boxes"])
            self.assertTrue(all(r["boxes"] for r in frames[1:]))
            capture = cv2.VideoCapture(str(root / "out/phone sample_annotated.mp4"))
            count = 0
            while True:
                ok, output = capture.read()
                if not ok:
                    break
                self.assertEqual(output.shape[:2], (360, 1920))
                count += 1
            capture.release()
            self.assertEqual(count, 6)
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("already exist", retry.stderr)
            bad = subprocess.run([sys.executable, str(script), str(root / "missing.mov")],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("does not exist", bad.stderr)


if __name__ == "__main__":
    unittest.main()
