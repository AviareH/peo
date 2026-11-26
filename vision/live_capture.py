"""Read an HTTP camera stream continuously, keeping only the newest frame."""

import threading
import time
from http.client import HTTPConnection, HTTPSConnection, HTTPException
from urllib.parse import urlsplit

import cv2


class ModeMonitor:
    """Read the ESP32 status server independently of frame and control timing."""

    def __init__(self, stream_url):
        url = urlsplit(stream_url)
        connection = HTTPSConnection if url.scheme == "https" else HTTPConnection
        port = (url.port or connection.default_port) + 1
        self._connection = connection(url.hostname, port, timeout=.5)
        self._sample = ("UNKNOWN", 0.)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._closed.is_set():
            value = "UNKNOWN"
            try:
                self._connection.request("GET", "/mode")
                response = self._connection.getresponse()
                body = response.read(16)
                if response.status == 200 and body in (b"MANUAL", b"AUTONOMOUS"):
                    value = body.decode("ascii")
            except (HTTPException, OSError, ValueError, OverflowError):
                pass
            finally:
                self._connection.close()
            self._sample = (value, time.monotonic())
            self._closed.wait(.25)

    @property
    def mode(self):
        mode, sampled_at = self._sample
        return mode if time.monotonic() - sampled_at < 1.0 else "UNKNOWN"

    def close(self):
        self._closed.set()
        self._thread.join(timeout=1)


class LiveCapture:
    def __init__(self, url: str, timeout_ms: int, max_age_ms: int):
        self.timeout = timeout_ms / 1000
        self.max_age = max_age_ms / 1000
        self.received_at = 0.0
        self.started_at = time.monotonic()
        self.error = ""
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = 0
        self._consumed = 0
        self._closed = False
        self._capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
        ])
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError("Cannot open camera stream. Join PEO-Camera, close other stream viewers, "
                             "and check the URL.")
        self._mode_monitor = ModeMonitor(url)
        self._thread = threading.Thread(target=self._receive, daemon=True)
        self._thread.start()

    def _receive(self):
        try:
            while True:
                with self._condition:
                    if self._closed:
                        break
                ok, frame = self._capture.read()
                with self._condition:
                    if not ok:
                        self.error = "Camera disconnected or stopped delivering frames."
                        self._condition.notify_all()
                        break
                    self._frame = frame
                    self._timestamp = time.monotonic()
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self.error = f"Camera capture failed: {error}"
                self._condition.notify_all()
        finally:
            self._capture.release()

    def read(self):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._sequence > self._consumed or self.error or self._closed,
                timeout=self.timeout,
            )
            if not ready or self.error or self._closed:
                self.error = self.error or "Timed out waiting for a new camera frame."
                return False, None
            if time.monotonic() - self._timestamp > self.max_age:
                self.error = "Latest camera frame is stale."
                return False, None
            self._consumed = self._sequence
            self.received_at = self._timestamp
            return True, self._frame

    def isOpened(self):
        return not self._closed and not self.error

    @property
    def drive_mode(self):
        return self._mode_monitor.mode

    @property
    def frame_id(self):
        """Decoder sequence of the frame most recently handed to the consumer."""
        return self._consumed

    def set(self, property_id, value):
        # File orientation/seek settings do not apply to live capture.
        return False

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return max(0.0, self.received_at - self.started_at) * 1000
        # Stream FPS metadata is often unreliable. Use a nominal recording rate.
        return 0.0

    def release(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._mode_monitor.close()
        # The receiver owns the decoder and releases it after read returns.
        self._thread.join(timeout=self.timeout + 1)
