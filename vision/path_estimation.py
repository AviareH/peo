"""One local image-space route through visible floor; no metric clearance/control."""

import cv2
import numpy as np


PATH_COLOUR = (255, 0, 255)  # Magenta, distinct from the segmentation colours.
DEFAULT_MARGIN = .01
DEFAULT_ENDPOINT_MARGIN = .01
DEFAULT_MIN_FORWARD = .20
DEFAULT_CORRIDOR_NEAR = .19
DEFAULT_CORRIDOR_FAR = .04
DEFAULT_CORRIDOR_BUFFER = .01


def corridor_radius(y, height, width, corridor_near=DEFAULT_CORRIDOR_NEAR,
                    corridor_far=DEFAULT_CORRIDOR_FAR,
                    corridor_buffer=DEFAULT_CORRIDOR_BUFFER):
    """Image-width half-footprint plus buffer, with a provisional 12%-height horizon."""
    progress = np.clip((np.asarray(y) / max(1, height - 1) - .12) / .88, 0, 1)
    return width * (.5 * (corridor_far + (corridor_near - corridor_far) * progress)
                    + corridor_buffer)


def corridor_mask(shape, points, **options):
    """Swept circular footprint, including all pixels along connecting segments."""
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    if not points:
        return mask
    centre = np.zeros(shape, np.uint8)
    cv2.polylines(centre, [np.asarray(points, np.int32)], False, 255, 1)
    ys, xs = np.nonzero(centre)
    radii = np.ceil(corridor_radius(ys, height, width, **options)).astype(int)
    for x, y, radius in zip(xs, ys, radii):
        cv2.circle(mask, (int(x), int(y)), int(radius), 255, -1)
    return mask


def estimate_path(floor_mask, margin_fraction=DEFAULT_MARGIN, grid_width=160, cost_map=None,
                  endpoint_margin_fraction=DEFAULT_ENDPOINT_MARGIN,
                  minimum_forward_fraction=DEFAULT_MIN_FORWARD, origin=None,
                  corridor_near=DEFAULT_CORRIDOR_NEAR, corridor_far=DEFAULT_CORRIDOR_FAR,
                  corridor_buffer=DEFAULT_CORRIDOR_BUFFER):
    """Find the furthest feasible upward grid route, preferring clearance and gentle turns.

    Both stroke and swept footprint must fit floor at full resolution. Endpoint
    stand-off and minimum reach reject tiny pockets. All clearance is image-space.
    """
    height, width = floor_mask.shape
    if not (0 <= corridor_far <= corridor_near <= .6 and 0 <= corridor_buffer <= .1):
        raise ValueError("Corridor widths must be 0 <= far <= near <= .6; buffer must be 0.. .1.")
    corridor_options = dict(corridor_near=corridor_near, corridor_far=corridor_far,
                            corridor_buffer=corridor_buffer)
    if min(height, width) < 32:
        return []
    gw = min(grid_width, width)
    gh = max(2, round(height * gw / width))
    # Only keep cells entirely inside floor: do not interpolate over small holes.
    floor = cv2.resize((floor_mask != 0).astype(np.float32), (gw, gh),
                       interpolation=cv2.INTER_AREA) >= .999
    extra_cost = np.zeros((gh, gw), np.float32)
    if cost_map is not None:
        if cost_map.shape != floor_mask.shape or not np.isfinite(cost_map).all() or np.any(cost_map < 0):
            raise ValueError("Path costs must be finite, non-negative, and match the floor mask.")
        extra_cost = cv2.resize(cost_map.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    padded = np.pad(floor.astype(np.uint8), 1)
    clearance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    radius = max(1.0, margin_fraction * gw)
    endpoint_radius = max(radius, endpoint_margin_fraction * gw)
    # Require the full perspective footprint around every candidate centre.
    row_radius = corridor_radius(np.arange(gh), gh, gw, **corridor_options)
    required = np.maximum(radius, np.ceil(row_radius) + 1)
    safe = floor & (clearance >= required[:, None])
    start_y = gh - 2 - int(np.ceil(required[-1]))
    start_x = gw // 2
    if origin is not None:
        if not (0 <= origin[0] < 1 and 0 <= origin[1] < 1):
            raise ValueError("Hypothetical path origin must be normalized coordinates in [0, 1).")
        start_x, start_y = int(origin[0] * gw), int(origin[1] * gh)
    if start_y < 1 or not safe[start_y, start_x]:
        return []

    directions = np.arange(-2, 3)
    previous = np.full((gw, len(directions)), np.inf)
    previous[start_x, len(directions) // 2] = 0
    parents = np.full((gh, gw, len(directions)), -1, dtype=np.int8)
    xs = np.arange(gw)
    centre_cost = .15 * np.abs(xs - start_x) / max(1, gw / 2)
    minimum_reach = max(4, int(gh * minimum_forward_fraction))
    endpoints = []
    for y in range(start_y - 1, -1, -1):
        current = np.full((gw, len(directions)), np.inf)
        pixel_cost = 1 + 3 / (clearance[y] + .5) + centre_cost + extra_cost[y]
        for state, dx in enumerate(directions):
            sources = xs - dx
            in_bounds = (sources >= 0) & (sources < gw)
            sources = np.clip(sources, 0, gw - 1)
            # Check both rows along the entire lateral step; never skip a barrier.
            allowed = in_bounds & safe[y] & safe[y + 1, sources]
            if dx:
                allowed &= safe[y, sources] & safe[y + 1]
                for offset in range(1, abs(int(dx))):
                    between = np.clip(sources + int(np.sign(dx)) * offset, 0, gw - 1)
                    allowed &= safe[y, between] & safe[y + 1, between]
            candidates = previous[sources] + .7 * (directions - dx) ** 2
            parent = candidates.argmin(axis=1)
            costs = candidates[xs, parent] + pixel_cost + .1 * abs(dx)
            current[allowed, state] = costs[allowed]
            parents[y, allowed, state] = parent[allowed]
        if not np.isfinite(current).any():
            break
        # A passage may lead beyond an invalid endpoint into a wider region.
        endpoint_ok = (clearance[y] >= max(endpoint_radius, required[y]))[:, None] & np.isfinite(current)
        if start_y - y >= minimum_reach and endpoint_ok.any():
            endpoints.append((y, np.where(endpoint_ok, current, np.inf)))
        previous = current

    # Prefer forward reach, then route cost; verify every accepted route at full resolution.
    for y, costs in reversed(endpoints):
        for index in np.argsort(costs, axis=None, kind="stable"):
            if not np.isfinite(costs.flat[index]):
                break
            endpoint = np.unravel_index(index, costs.shape)
            points = _reconstruct_path(parents, endpoint, y, start_y, directions, floor_mask.shape)
            if _path_fits_floor(floor_mask, points, corridor_options):
                return points
    return []


def _reconstruct_path(parents: np.ndarray, endpoint: tuple[int, int], best_y: int,
                      start_y: int, directions: np.ndarray,
                      image_shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Trace the chosen grid endpoint back to the origin, then map to image pixels."""
    height, width = image_shape
    gh, gw = parents.shape[:2]
    x, state = endpoint
    points = []
    for y in range(best_y, start_y + 1):
        points.append((min(width - 1, int((x + .5) * width / gw)),
                       min(height - 1, int((y + .5) * height / gh))))
        if y < start_y:
            parent = int(parents[y, x, state])
            x -= int(directions[state])
            state = parent
    points.reverse()

    return points


def _path_fits_floor(floor_mask: np.ndarray, points: list[tuple[int, int]],
                     corridor_options: dict) -> bool:
    """Check both the drawn stroke and swept robot footprint at full resolution."""
    height, width = floor_mask.shape
    stroke = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(stroke, [np.asarray(points, np.int32)], False, 255, path_thickness(width))
    if np.any((stroke != 0) & (floor_mask == 0)):
        return False
    footprint = corridor_mask((height, width), points, **corridor_options)
    return not np.any((footprint != 0) & (floor_mask == 0))


def path_thickness(width):
    return max(2, round(width / 240))


def corridor_envelope(shape, points, floor_mask=None, extend_to_base=False):
    """Display envelope, separate from footprint: flare 10% to 80% within connected floor.

    Only origin-connected routes extend to the base; hypothetical starts stay fixed.
    """
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    if not points:
        return mask, [], []
    floor = np.ones(shape, bool) if floor_mask is None else floor_mask != 0
    ordered = np.asarray(points, np.int32)[::-1]
    top, bottom = int(ordered[0, 1]), int(ordered[-1, 1])
    limit = height - 1 if extend_to_base else bottom
    left, right = [], []
    previous = None
    for y in range(top, limit + 1):
        x = int(round(np.interp(y, ordered[:, 1], ordered[:, 0])))
        if not (0 <= x < width and floor[y, x]):
            break
        progress = (y - top) / max(1, height - 1 - top)
        half = int(width * (.05 + .35 * progress))
        lo, hi = max(0, x - half), min(width - 1, x + half)
        blocked = np.flatnonzero(~floor[y, lo:x])
        if len(blocked):
            lo += int(blocked[-1]) + 1
        blocked = np.flatnonzero(~floor[y, x:hi + 1])
        if len(blocked):
            hi = x + int(blocked[0]) - 1
        if previous is not None and (lo > previous[1] or hi < previous[0]):
            break
        mask[y, lo:hi + 1] = 255
        left.append((lo, y))
        right.append((hi, y))
        previous = lo, hi
    return mask, left, right


def draw_corridor(overlay: np.ndarray, points: list[tuple[int, int]], colour: tuple[int, int, int],
                  floor_mask: np.ndarray | None = None, extend_to_base: bool = False) -> None:
    """Draw the broad floor envelope, independently of planner footprint settings."""
    mask, left, right = corridor_envelope(overlay.shape[:2], points, floor_mask, extend_to_base)
    selected = mask != 0
    overlay[selected] = (.93 * overlay[selected] + .07 * np.asarray(colour)).astype(np.uint8)
    # Dots spaced by arc length, with open ends instead of rounded caps.
    edges = overlay.copy()
    spacing = max(6, overlay.shape[1] / 90)
    radius = max(1, round(overlay.shape[1] / 600))
    for side in (left, right):
        distance = spacing
        last = None
        for point in side:
            if last is not None:
                distance += np.hypot(point[0] - last[0], point[1] - last[1])
            if distance >= spacing:
                cv2.circle(edges, point, radius, colour, -1)
                distance %= spacing
            last = point
    # Keep even the boundary dots inside usable floor.
    allowed = np.ones(overlay.shape[:2], bool) if floor_mask is None else floor_mask != 0
    overlay[allowed] = edges[allowed]


def draw_path(overlay: np.ndarray, points: list[tuple[int, int]], frame_width: int,
              floor_mask: np.ndarray | None = None) -> None:
    if points:
        draw_corridor(overlay, points, PATH_COLOUR, floor_mask, extend_to_base=True)
        cv2.polylines(overlay, [np.asarray(points, np.int32)], False, PATH_COLOUR,
                      path_thickness(frame_width), lineType=cv2.LINE_8)
