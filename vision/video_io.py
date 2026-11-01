"""Local video selection and frame preparation, independent of perception."""

import math
from pathlib import Path

import cv2
import numpy as np


def pick_video() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            name = filedialog.askopenfilename(title="Select robot test footage",
                                              filetypes=[("Video", "*.mov *.MOV *.mp4 *.MP4 *.avi"),
                                                         ("All files", "*")])
        finally:
            root.destroy()
        if not name:
            raise ValueError("No video selected.")
        return Path(name)
    except ImportError as exc:
        raise ValueError("File picker unavailable; pass a video path on the command line.") from exc
    except tk.TclError as exc:
        raise ValueError("Cannot open file picker; pass a video path on the command line.") from exc


def prepare(frame: np.ndarray, args) -> np.ndarray:
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if args.rotate:
        frame = cv2.rotate(frame, rotations[args.rotate])
    scale = min(1, args.max_width / frame.shape[1])
    # Even dimensions keep the MP4 encoder from silently trimming a row/column.
    width = max(2, int(frame.shape[1] * scale) // 2 * 2)
    height = max(2, int(frame.shape[0] * scale) // 2 * 2)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def start_capture(capture: cv2.VideoCapture, start_seconds: float) -> tuple[float, np.ndarray]:
    """Validate the decoder and read the first frame before loading models."""
    if not capture.isOpened():
        raise ValueError("Cannot open video. Try exporting an H.264 MP4 copy.")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
        print("Video FPS unavailable; using 30 FPS for output playback.")
    if start_seconds and not capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000):
        raise ValueError("Decoder cannot seek to --start-seconds.")
    ok, frame = capture.read()
    if not ok:
        raise ValueError("No decodable frames at start. Try H.264 MP4 or a smaller --start-seconds.")
    return fps, frame


def output_paths(source: Path, output_dir: Path) -> tuple[Path, Path]:
    """Keep filenames stable and reject existing results or input collisions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{source.stem}_annotated.mp4"
    log_path = output_dir / f"{source.stem}_detections.jsonl"
    if source in (video_path.resolve(), log_path.resolve()):
        raise ValueError("Output would overwrite input; choose another --output-dir.")
    # Preserve previous runs and never silently overwrite recordings/results.
    if video_path.exists() or log_path.exists():
        raise ValueError("Outputs already exist. Choose a new --output-dir.")
    return video_path, log_path


PREVIEW_TITLE = "Robot vision | Q quit | space pause"


def open_preview(width: int, height: int) -> None:
    cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_NORMAL)
    scale = min(1, 1280 / width, 800 / height)
    cv2.resizeWindow(PREVIEW_TITLE, round(width * scale), round(height * scale))


def preview_stopped(overlay: np.ndarray, fps: float) -> bool:
    """Show a frame; space waits for another key, Q/Esc exits replay."""
    cv2.imshow(PREVIEW_TITLE, overlay)
    key = cv2.waitKey(max(1, round(1000 / fps))) & 0xFF
    if key == ord(" "):
        key = cv2.waitKey(0) & 0xFF
    return key in (ord("q"), 27)
