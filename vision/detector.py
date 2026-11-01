"""Floor-colour segmentation with edge-supported obstacle candidates. No actuation."""

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from drawing import draw_text


@dataclass
class Settings:
    roi_top: float = 0.35
    floor_patch: tuple = (0.40, 0.80, 0.60, 0.95)
    hue_tolerance: int = 18
    saturation_tolerance: int = 65
    value_tolerance: int = 65
    canny_low: int = 50
    canny_high: int = 150
    min_area: float = 0.003  # Fraction of the region of interest.
    min_edge_support: float = 0.08


def patch_bounds(shape, patch):
    height, width = shape[:2]
    x1, y1, x2, y2 = patch
    return (int(x1 * width), int(y1 * height),
            max(int(x2 * width), int(x1 * width) + 1),
            max(int(y2 * height), int(y1 * height) + 1))


def segment_floor(different, edges, settings):
    """Keep foreground-connected floor across the frame; preserve holes and edge barriers."""
    candidate = (~different).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    barriers = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidate[barriers != 0] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    seeds = np.zeros_like(candidate, dtype=bool)
    # Side regions still seed the floor when an obstacle blocks the centre.
    seeds[int(candidate.shape[0] * .95):] = True
    x1, y1, x2, y2 = patch_bounds(candidate.shape, settings.floor_patch)
    seeds[y1:y2, x1:x2] = True
    seeded_labels = np.unique(labels[seeds & (candidate != 0)])
    keep = np.zeros(count, dtype=bool)
    keep[seeded_labels] = True
    keep[0] = False
    keep[stats[:, cv2.CC_STAT_AREA] < candidate.size * .001] = False
    return keep[labels].astype(np.uint8) * 255


class ObstacleDetector:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.floor = None

    def calibrate(self, frame):
        """Sample clear floor once; never automatically learn an obstacle as floor."""
        x1, y1, x2, y2 = patch_bounds(frame.shape, self.settings.floor_patch)
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)
        # Circular hue mean handles red samples spanning OpenCV's 0/179 boundary.
        angles = pixels[:, 0] * (2 * np.pi / 180)
        weights = pixels[:, 1] / 255
        hue = np.arctan2(np.sum(np.sin(angles) * weights),
                         np.sum(np.cos(angles) * weights)) % (2 * np.pi)
        self.floor = np.array([hue * 180 / (2 * np.pi),
                               np.median(pixels[:, 1]), np.median(pixels[:, 2])])

    def process(self, frame):
        started = perf_counter()
        s = self.settings
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        empty = np.zeros(gray.shape, np.uint8)
        if gray.mean() < 5 or gray.mean() > 250:
            return dict(boxes=[], mask=empty, edges=empty, floor_mask=empty, status="UNUSABLE FRAME",
                        processing_ms=(perf_counter() - started) * 1000)
        if self.floor is None:
            self.calibrate(frame)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_delta = np.abs(hsv[:, :, 0] - self.floor[0])
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        # Hue is unreliable for low-saturation surfaces; use S/V there.
        hue_diff = ((hue_delta > s.hue_tolerance) & (hsv[:, :, 1] > 40)
                    & (self.floor[1] > 40))
        different = (hue_diff
                     | (np.abs(hsv[:, :, 1] - self.floor[1]) > s.saturation_tolerance)
                     | (np.abs(hsv[:, :, 2] - self.floor[2]) > s.value_tolerance))
        top = int(frame.shape[0] * s.roi_top)
        mask = different.astype(np.uint8) * 255
        mask[:top] = 0
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), s.canny_low, s.canny_high)
        floor_mask = segment_floor(different, edges, s)
        edges[:top] = 0
        nearby_edges = cv2.dilate(edges, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum = s.min_area * (frame.shape[0] - top) * frame.shape[1]
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum:
                continue
            boundary = np.zeros_like(mask)
            cv2.drawContours(boundary, [contour], -1, 255, 2)
            support = np.count_nonzero(boundary & nearby_edges) / max(1, np.count_nonzero(boundary))
            if support < s.min_edge_support:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(dict(x=x, y=y, width=w, height=h,
                              area_px=float(area), edge_support=round(support, 3)))
        boxes.sort(key=lambda box: (box["x"], box["y"]))
        return dict(boxes=boxes, mask=mask, edges=edges, floor_mask=floor_mask,
                    status="CANDIDATES" if boxes else "NO CANDIDATES (not proof of clear path)",
                    processing_ms=(perf_counter() - started) * 1000)


def annotate(frame, result, settings, show_floor_sample=False):
    view = frame.copy()
    height, width = view.shape[:2]
    floor_mask = result["floor_mask"]
    selected = floor_mask != 0
    green = np.full_like(view, (60, 200, 60))
    tinted = cv2.addWeighted(view, .72, green, .28, 0)
    view[selected] = tinted[selected]
    # Draw actual region boundaries, including holes around obstacles. A convex
    # hull or bounding rectangle would incorrectly paint over obstacles/gaps.
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, contours, -1, (60, 230, 60), 2)
    cv2.line(view, (0, int(height * settings.roi_top)),
             (width, int(height * settings.roi_top)), (255, 200, 0), 2)
    if show_floor_sample:
        x1, y1, x2, y2 = patch_bounds(frame.shape, settings.floor_patch)
        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 255), 2)
        draw_text(view, "calibration sample", (x1, max(65, y1 - 6)), .45, (0, 220, 255))
    for box in result["boxes"]:
        x, y, w, h = (box[k] for k in ("x", "y", "width", "height"))
        cv2.rectangle(view, (x, y), (x + w, y + h), (0, 0, 255), 2)
        draw_text(view, "possible obstacle", (x, max(45, y - 6)), 0.5, (0, 0, 255))
    cv2.rectangle(view, (0, 0), (width, 54), (0, 0, 0), -1)
    draw_text(view, f"{result['status']} | {result['processing_ms']:.1f} ms", (8, 22))
    draw_text(view, "Green: estimated floor | Red: obstacle candidates", (8, 45), .45, (100, 230, 100))
    return view
