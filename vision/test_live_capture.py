"""Live capture must drop backlog and reject missing or stale frames."""

import queue
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from live_capture import LiveCapture, ModeMonitor


class FakeDecoder:
    def __init__(self):
        self.frames = queue.Queue()
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return self.frames.get(timeout=2)

    def release(self):
        self.released = True


class LiveCaptureTests(unittest.TestCase):
    def setUp(self):
        self.decoder = FakeDecoder()
        with (patch("live_capture.cv2.VideoCapture", return_value=self.decoder),
              patch("live_capture.ModeMonitor")):
            self.capture = LiveCapture("http://camera/stream", 50, 1000)

    def tearDown(self):
        self.decoder.frames.put((False, None))
        self.capture.release()
        self.assertTrue(self.decoder.released)

    def deliver(self, values):
        target = self.capture._sequence + len(values)
        for value in values:
            self.decoder.frames.put((True, np.full((8, 8, 3), value, np.uint8)))
        with self.capture._condition:
            self.assertTrue(self.capture._condition.wait_for(
                lambda: self.capture._sequence >= target, timeout=1))

    def test_slow_consumer_receives_newest_frame(self):
        self.deliver([1, 2, 3])
        ok, frame = self.capture.read()
        self.assertTrue(ok)
        self.assertTrue(np.all(frame == 3))
        self.assertGreater(self.capture.received_at, 0)

    def test_missing_next_frame_is_not_repeated(self):
        self.deliver([1])
        self.assertTrue(self.capture.read()[0])
        self.assertFalse(self.capture.read()[0])
        self.assertIn("Timed out", self.capture.error)

    def test_stale_frame_is_rejected(self):
        self.deliver([1])
        with self.capture._condition:
            self.capture._timestamp -= 2
        self.assertFalse(self.capture.read()[0])
        self.assertIn("stale", self.capture.error)

    def test_disconnect_discards_last_frame(self):
        self.deliver([1])
        self.decoder.frames.put((False, None))
        with self.capture._condition:
            self.assertTrue(self.capture._condition.wait_for(
                lambda: bool(self.capture.error), timeout=1))
        self.assertFalse(self.capture.read()[0])


class StreamPipelineTests(unittest.TestCase):
    def test_board_mode_changes_and_missing_status(self):
        class Status(BaseHTTPRequestHandler):
            value = b"MANUAL"

            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200 if self.path == "/mode" else 404)
                self.end_headers()
                self.wfile.write(self.value)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Status)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        monitor = ModeMonitor(f"http://127.0.0.1:{server.server_port-1}/stream")
        try:
            import time
            for value, expected in [(b"MANUAL", "MANUAL"), (b"AUTONOMOUS", "AUTONOMOUS"),
                                    (b"unavailable", "UNKNOWN")]:
                Status.value = value
                deadline = time.monotonic() + 2
                while monitor.mode != expected and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(monitor.mode, expected)
        finally:
            monitor.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_mjpeg_reaches_existing_pipeline_and_recording(self):
        self.exercise_http_pipeline("classical")

    def test_http_mjpeg_reaches_tape_detector_and_mock_sink(self):
        self.exercise_http_pipeline("tape")

    def exercise_http_pipeline(self, method):
        stopped = threading.Event()
        frame = np.full((120, 160, 3), 150, np.uint8)
        if method == "tape":
            for x in (50, 110):
                cv2.line(frame, (x, 119), (x-10, 44), (255, 0, 0), 6)
        ok, jpeg = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        payload = jpeg.tobytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not stopped.is_set():
                        header = (f"--frame\r\nContent-Type: image/jpeg\r\n"
                                  f"Content-Length: {len(payload)}\r\n\r\n").encode()
                        self.wfile.write(header + payload + b"\r\n")
                        self.wfile.flush()
                        stopped.wait(.03)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                command = [sys.executable, str(Path(__file__).with_name("run.py")),
                           "--stream", f"http://127.0.0.1:{server.server_port}/stream",
                           "--method", method, "--max-frames", "3", "--output-dir", directory]
                result = subprocess.run(command, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                log = next(Path(directory).glob("*_detections.jsonl"))
                records = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(records[0]["input_mode"], "live")
                frames = [record for record in records if record["type"] == "frame"]
                self.assertEqual(len(frames), 3)
                self.assertTrue(all(frame["drive_mode"] == "UNKNOWN" for frame in frames))
                self.assertTrue(all(0 <= frame["frame_age_ms"] < 2000 for frame in frames))
                if method == "tape":
                    self.assertTrue(all(frame["proposal"]["reason"] == "calibration_required"
                                        and frame["centreline_px"] for frame in frames))
                    self.assertEqual(records[-1]["proposal"]["reason"], "frame_limit")
                    self.assertFalse(records[-1]["proposal"]["actions"])
                else:
                    self.assertTrue(all(not frame["navigation"]["decision"]["motion_authorized"]
                                        for frame in frames))
                    self.assertTrue(records[-1]["frame_limit_reached"])
                recording = cv2.VideoCapture(str(next(Path(directory).glob("*.mp4"))))
                try:
                    self.assertTrue(recording.read()[0])
                finally:
                    recording.release()
        finally:
            stopped.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
