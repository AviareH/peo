"""Experimental floor/depth corridor scoring, without metric calibration."""

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class FusionSettings:
    depth_tolerance: float = .10  # Fraction of the current depth distribution.
    side_weight: float = 3.0
    side_margin_fraction: float = .045
    minimum_fit_fraction: float = .55


def fit_ground(floor, inverse_depth, tolerance):
    """Fit an affine floor-depth heuristic with RANSAC; this is not calibrated 3D."""
    height, width = floor.shape
    gw = min(width, 160)
    gh = max(2, round(height * gw / width))
    mask = cv2.resize(floor.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) >= .999
    # Resizing NaNs can contaminate neighbours; exclude those samples explicitly.
    depth = cv2.resize(inverse_depth, (gw, gh), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(mask & np.isfinite(depth))
    if len(xs) < 60 or np.ptp(ys) / gh < .12:
        return None, 0.
    rng = np.random.default_rng(0)
    if len(xs) > 2500:
        chosen = rng.choice(len(xs), 2500, replace=False)
        ys, xs = ys[chosen], xs[chosen]
    design = np.column_stack((xs / gw, ys / gh, np.ones(len(xs))))
    target = depth[ys, xs]
    best, best_count, best_error = None, 0, np.inf
    for _ in range(80):
        chosen = rng.choice(len(xs), 3, replace=False)
        matrix = design[chosen]
        if abs(np.linalg.det(matrix)) < .005:
            continue
        coefficients = np.linalg.solve(matrix, target[chosen])
        errors = np.abs(design @ coefficients - target)
        inliers = errors <= tolerance
        count = int(inliers.sum())
        error = float(np.minimum(errors, tolerance).mean())
        if count > best_count or (count == best_count and error < best_error):
            best, best_count, best_error = inliers, count, error
    if best is None:
        return None, 0.
    coefficients, _, _, _ = np.linalg.lstsq(design[best], target[best], rcond=None)
    fraction = float((np.abs(design @ coefficients - target) <= tolerance).mean())
    expected = (coefficients[0] * np.arange(width)[None, :] / width
                + coefficients[1] * np.arange(height)[:, None] / height + coefficients[2])
    return expected.astype(np.float32), fraction


def fuse_floor_depth(floor_mask, depth_values, confidence=None, settings=None):
    settings = settings or FusionSettings()
    if floor_mask.shape != depth_values.shape:
        raise ValueError("Floor and depth outputs must have matching image coordinates.")
    floor = floor_mask != 0
    valid = np.isfinite(depth_values)
    empty = np.zeros(floor.shape, bool)

    def unavailable(status, fit_fraction=0.):
        return dict(mask=empty.copy(), cost=np.zeros(floor.shape, np.float32),
                    rejected=floor.copy(), status=status, fit_fraction=fit_fraction, settings=asdict(settings))

    if not valid.any():
        return unavailable("DEPTH UNAVAILABLE")
    low, high = np.percentile(depth_values[valid], [2, 98])
    if high - low < 1e-6:
        return unavailable("DEPTH HAS NO CONTRAST")
    normalized = np.full(floor.shape, np.nan, np.float32)
    normalized[valid] = (depth_values[valid] - low) / (high - low)
    expected, fraction = fit_ground(floor & valid, normalized, settings.depth_tolerance)
    if expected is None or fraction < settings.minimum_fit_fraction:
        return unavailable("GROUND FIT UNCERTAIN", fraction)

    residual = normalized - expected
    # Nearby floor is naturally close. Compare it against expected ground at
    # that pixel, rather than declaring all large inverse-depth values blocked.
    eligible = (floor & valid & (residual <= settings.depth_tolerance)
                & (residual >= -2 * settings.depth_tolerance))
    close_objects = valid & (residual > settings.depth_tolerance)
    side_cost = np.zeros(floor.shape, np.float32)
    if close_objects.any():
        distance = cv2.distanceTransform((~close_objects).astype(np.uint8), cv2.DIST_L2, 5)
        side_cost = np.exp(-distance / max(1, floor.shape[1] * settings.side_margin_fraction))
    # Non-negative costs favour deeper corridors within the same image row,
    # agreement with ground, and clearance from nearby protrusions at the sides.
    cost = (settings.side_weight * side_cost
            + np.clip(np.nan_to_num(residual, nan=1) / settings.depth_tolerance, 0, 1)
            + .5 * np.clip(np.nan_to_num(normalized, nan=1), 0, 1))
    if confidence is not None:
        if confidence.shape != floor.shape:
            raise ValueError("Confidence and floor masks must have matching shapes.")
        cost += .5 * (1 - np.clip(np.nan_to_num(confidence, nan=0), 0, 1))
    return dict(mask=eligible, cost=cost.astype(np.float32), rejected=floor & ~eligible,
                status="FUSED FLOOR + DEPTH", fit_fraction=fraction, settings=asdict(settings))
