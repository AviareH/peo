"""Project prepared pixels inside a measured quadrilateral onto a level floor."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _array(data, key, shape):
    value = np.asarray(data[key])
    if value.shape != shape or value.dtype.kind not in "iuf":
        raise ValueError(f"Invalid numeric field {key}, expected shape {shape}")
    value = value.astype(float)
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite {key}")
    return value


def _image_size(data, key):
    size = _array(data, key, (2,))
    if (size < 2).any() or (size != np.floor(size)).any():
        raise ValueError(f"Invalid image size in {key}")
    return tuple(map(int, size))


def _project(matrix, points):
    vectors = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        projected = vectors[:, :2] / vectors[:, 2, None]
    if not np.isfinite(projected).all():
        raise ValueError("Non-finite homography projection")
    return projected


@dataclass(frozen=True)
class Calibration:
    source_size: tuple[int, int]
    processed_size: tuple[int, int]
    rotate: int
    matrix: np.ndarray
    inverse: np.ndarray
    region: np.ndarray
    offset: np.ndarray  # camera floor origin relative to robot turn centre, left/forward

    @classmethod
    def load(cls, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fields = {"version", "source_size", "processed_size", "rotate", "valid_image_quad_px",
                      "image_to_ground_m", "ground_quad_left_forward_m",
                      "camera_offset_left_forward_m", "description"}
            if not isinstance(data, dict) or set(data) - fields:
                raise ValueError("Unsupported calibration fields")
            if (type(data["version"]) is not int or data["version"] != 1 or
                    type(data["rotate"]) is not int or data["rotate"] not in (0, 90, 180, 270)):
                raise ValueError("Unsupported calibration version or rotation")
            source = _image_size(data, "source_size")
            size = _image_size(data, "processed_size")
            offset = _array(data, "camera_offset_left_forward_m", (2,))
            region = _array(data, "valid_image_quad_px", (4, 2))
            if ((region < 0).any() or (region > np.asarray(size)-1).any()
                    or not cv2.isContourConvex(region.astype(np.float32))):
                raise ValueError("Expected four ordered convex corners inside processed image")
            measured = "ground_quad_left_forward_m" in data
            if measured == ("image_to_ground_m" in data):
                raise ValueError("Provide exactly one of ground_quad_left_forward_m and image_to_ground_m")
            if measured:
                ground = _array(data, "ground_quad_left_forward_m", (4, 2))
                if not cv2.isContourConvex(ground.astype(np.float32)):
                    raise ValueError("Expected four convex floor corners in the same order as the pixels")
                matrix, _ = cv2.findHomography(region, ground, method=0)
                if matrix is None:
                    raise ValueError("Could not fit a projection to the four floor corners")
            else:
                matrix = _array(data, "image_to_ground_m", (3, 3))
            if not np.linalg.cond(matrix) <= 1e12:
                raise ValueError("Singular or ill-conditioned image_to_ground_m")
            matrix = matrix / np.abs(matrix).max()
            denominator = region @ matrix[2, :2] + matrix[2, 2]
            if (denominator.min() * denominator.max() <= 0 or
                    np.abs(denominator).min() < 1e-6 * np.abs(denominator).max()):
                raise ValueError("Calibrated region touches the projection horizon")
            _project(matrix, region)
            return cls(source, size, data["rotate"], matrix, np.linalg.inv(matrix),
                       region.astype(np.float32), offset)
        except (OSError, ValueError, TypeError, OverflowError, KeyError, np.linalg.LinAlgError, cv2.error) as exc:
            raise ValueError(f"Invalid calibration: {exc}") from exc

    def matches(self, source_size, processed_size, rotate):
        return (source_size, processed_size, rotate) == (self.source_size, self.processed_size, self.rotate)

    def contains(self, pixel):
        return bool(np.isfinite(pixel).all() and
                    cv2.pointPolygonTest(self.region, tuple(map(float, pixel)), False) >= 0)

    def robot_to_image(self, point):
        pixel = _project(self.inverse, [np.asarray(point) - self.offset])[0]
        return tuple(map(float, pixel)) if self.contains(pixel) else None


@dataclass(frozen=True)
class StepLimits:
    lookahead_m: float = .20
    max_step_m: float = .65
    min_step_m: float = .03
    max_turn_rad: float = math.radians(40)
    deadband_rad: float = math.radians(2)

    def __post_init__(self):
        if not (all(math.isfinite(x) for x in vars(self).values()) and self.lookahead_m > 0
                and 0 < self.min_step_m <= self.max_step_m
                and 0 <= self.deadband_rad < self.max_turn_rad <= math.pi / 2):
            raise ValueError("Invalid tape lookahead, step limits or steering angles")


@dataclass(frozen=True)
class GroundStep:
    target_m: tuple[float, float]
    target_px: tuple[float, float]
    angle_rad: float
    distance_m: float


def plan_boundary_turn(arm, direction, cal: Calibration, limits: StepLimits) -> float:
    """Face along the measured outgoing tape with a bounded turn in place."""
    pixels = [point for point in arm if cal.contains(point)]
    if len(pixels) < 2:
        raise ValueError("corner_outside_calibrated_region")
    ground = _project(cal.matrix, (pixels[0], pixels[-1])) + cal.offset
    delta = ground[1] - ground[0]
    if np.linalg.norm(delta) < limits.min_step_m or ground[:, 1].min() <= 0:
        raise ValueError("insufficient_measured_turn_arm")
    angle = math.atan2(*delta)
    if -direction * angle <= limits.deadband_rad:
        raise ValueError("no_measured_turn_in_detected_direction")
    return math.copysign(min(abs(angle), limits.max_turn_rad), angle)


def plan_step(points, cal: Calibration, limits: StepLimits) -> GroundStep:
    """Walk the first continuously calibrated run, stopping at arc/step limits."""
    supported = []
    for pixel in points:
        if not np.isfinite(pixel).all():
            raise ValueError("non_finite_route")
        if cal.contains(pixel):
            supported.append(pixel)
        elif supported:
            break
    if len(supported) < 2:
        raise ValueError("outside_calibrated_region")
    return _plan_ground(_project(cal.matrix, supported) + cal.offset, cal, limits)


def plan_between_edges(left, right, cal: Calibration, limits: StepLimits, *,
                       approach_end: int | None = None) -> GroundStep:
    """Only use spans with two measured endpoints. Average in the ground plane."""
    supported = []
    approach_count = 0
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if not np.isfinite((a, b)).all():
            raise ValueError("non_finite_route")
        if cal.contains(a) and cal.contains(b):
            supported.append((a, b))
            if approach_end is not None and i <= approach_end:
                approach_count += 1
        elif supported:
            break
    if len(supported) < 2:
        raise ValueError("boundaries_outside_calibrated_region")
    pixels = np.asarray(supported).reshape(-1, 2)
    edges = _project(cal.matrix, pixels).reshape(-1, 2, 2) + cal.offset
    centreline = edges.mean(axis=1)
    if 1 < approach_count < len(centreline):
        end = centreline[approach_count-1]
        inbound = end - centreline[0]
        length = np.linalg.norm(inbound)
        if length >= limits.min_step_m and end @ inbound >= limits.min_step_m * length:
            # Take one step along the remaining approach. A fresh frame can
            # choose the next target once the bend is close.
            centreline = centreline[:approach_count]
    step = _plan_ground(centreline, cal, limits)
    # A chord to a midpoint past a tight corner can cut across the inside tape.
    # Test the actual post-deadband drive against both measured edge polylines.
    direction = np.array([math.sin(step.angle_rad), math.cos(step.angle_rad)])
    along = edges @ direction
    side = edges @ np.array([direction[1], -direction[0]])
    start, end = side[:-1], side[1:]
    crosses = (np.minimum(start, end) <= 0) & (np.maximum(start, end) >= 0)
    fraction = np.divide(start, start-end, out=np.zeros_like(start), where=start != end)
    hit = along[:-1] + fraction * np.diff(along, axis=0)
    collinear = (start == 0) & (end == 0)
    overlap = (np.maximum(along[:-1], along[1:]) >= 0) & (
        np.minimum(along[:-1], along[1:]) <= step.distance_m)
    if ((crosses & (hit >= 0) & (hit <= step.distance_m)) | (collinear & overlap)).any():
        raise ValueError("drive_crosses_boundary")
    return step


def _plan_ground(ground, cal, limits):
    if not np.isfinite(ground).all():
        raise ValueError("non_finite_route")
    target = ground[0]
    if target[1] <= 0:
        raise ValueError("target_behind_robot")
    if np.linalg.norm(target) > limits.max_step_m:
        raise ValueError("nearest_tape_beyond_step_limit")
    remaining = limits.lookahead_m
    for end in ground[1:]:
        delta = end - target
        length = math.hypot(*delta)
        if length == 0:
            continue
        direction = delta / length
        advance = min(length, remaining)
        if np.linalg.norm(target + advance * direction) > limits.max_step_m:
            # Forward intersection of this segment with the robot-centred step circle.
            along = float(target @ direction)
            discriminant = along**2 + limits.max_step_m**2 - float(target @ target)
            advance = min(advance, max(0., math.sqrt(max(0., discriminant)) - along))
        target = target + advance * direction
        if target[1] <= 0:
            raise ValueError("target_behind_robot")
        remaining -= advance
        if advance < length or remaining <= 0:
            break
    pixel = cal.robot_to_image(target)
    if pixel is None:
        raise ValueError("target_outside_calibrated_region")
    distance = math.hypot(*target)
    angle = math.atan2(*target)  # atan2(left, forward), positive left
    if distance < limits.min_step_m:
        raise ValueError("step_too_short")
    if abs(angle) > limits.max_turn_rad:
        raise ValueError("turn_exceeds_limit")
    return GroundStep(tuple(map(float, target)), pixel,
                      0. if abs(angle) <= limits.deadband_rad else angle, distance)
