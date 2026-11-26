"""Image-space detection of a corridor between two blue tape boundaries, with no tracking state."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TapeSettings:
    hsv_low: tuple[int, int, int] = (95, 85, 45)
    hsv_high: tuple[int, int, int] = (135, 255, 255)
    roi_top: float = .35
    near_y: float = .80
    min_area: float = .002
    min_extent: float = .22
    max_width: float = .20
    gap_px: int = 2
    min_lane_width: float = .08  # inner-edge separation / prepared image width

    def __post_init__(self):
        if not (len(self.hsv_low) == len(self.hsv_high) == 3 and
                all(0 <= a <= b <= limit for a, b, limit in
                    zip(self.hsv_low, self.hsv_high, (179, 255, 255)))):
            raise ValueError("Invalid OpenCV HSV range (H 0..179, S/V 0..255)")
        if not (0 <= self.roi_top < self.near_y < 1 and 0 < self.min_area < 1
                and 0 < self.min_extent <= 1 and 0 < self.max_width < 1
                and 0 < self.min_lane_width < 1 and 0 <= self.gap_px <= 8):
            raise ValueError("Invalid tape ROI, near field, area, extent, width or gap")


@dataclass(frozen=True)
class Evidence:
    extent: float = 0.0       # fraction of ROI height
    continuity: float = 0.0   # sampled crossings with observed blue on both ends
    near_support: float = 0.0 # fraction of near-field rows showing both strips

    @property
    def quality(self):
        return round(self.extent * self.continuity * self.near_support, 4)


@dataclass
class TapeRoute:
    status: str
    mask: np.ndarray
    selected: np.ndarray
    evidence: Evidence = Evidence()
    points: tuple[tuple[float, float], ...] = ()  # near to far in prepared pixels
    image_target: tuple[float, float] | None = None
    left_edge: tuple[tuple[int, int], ...] = ()
    right_edge: tuple[tuple[int, int], ...] = ()
    turn_direction: int = 0  # -1 left, +1 right
    approach_end: int | None = None  # last inbound cross-section before the turn
    turn_arm: tuple[tuple[int, int], ...] = ()  # observed pixels in travel order for a turn in place
    short_pair: bool = False


def _boundary_elbow(component):
    """Find the stem at a one-sided far elbow and reject split boundaries."""
    rows, cols = np.nonzero(component)
    component = component[rows.min():rows.max()+1, cols.min():cols.max()+1]
    left = component.argmax(axis=1)
    right = component.shape[1] - 1 - component[:, ::-1].argmax(axis=1)
    widths = right - left + 1
    typical = np.median(widths)
    gaps = widths - component.sum(axis=1) > max(2, .15 * typical)
    broad = np.flatnonzero(widths > 2.5 * typical + 2)
    if not len(broad):
        return bool(gaps.any()), None
    # Compare against the stem just below the arm. Further down, perspective
    # shifts a slanted stem sideways and makes a single arm look two-sided.
    stem = min(len(widths) - 1, broad[-1] + 1)
    if (gaps[stem:].any() or broad[-1] - broad[0] > 2 * typical or
            (broad[0] > typical and len(widths) - stem > typical)):
        return True, None
    widest = int(widths.argmax())
    extends_left = left[widest] < left[stem] - typical
    extends_right = right[widest] > right[stem] + typical
    if extends_left and extends_right:
        return True, None
    if extends_left or extends_right:
        direction = -1 if extends_left else 1
        x = right[stem] if extends_left else left[stem]
        return False, (direction, int(cols.min() + x), int(rows.min() + stem))
    return bool(gaps.any()), None


def _recover_far_tape(hsv, labels, stats, anchors, cfg, frame_height):
    """Keep faint horizontal tape only where it connects to a seeded far tip."""
    hue, saturation, value = cfg.hsv_low
    weak = cv2.inRange(hsv, (hue, min(saturation, max(24, round(.30 * saturation))), value),
                       cfg.hsv_high)
    width = labels.shape[1]
    # A small horizontal gap can appear at a narrow, compressed far corner.
    gap = 2 * (width // 90) + 1 if cfg.gap_px else 1
    joined = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))
    length = max(9, 2 * (width // 110) + 1)
    arms = cv2.morphologyEx(joined, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    recovered = np.zeros_like(weak)
    for (x, y), label in anchors:
        boundary = labels == label
        end = stats[label, cv2.CC_STAT_TOP] + max(4, round(.015 * frame_height)) + 1
        connected = boundary.astype(np.uint8)
        connected[:end] |= (arms[:end] != 0)
        _, components = cv2.connectedComponents(connected, connectivity=8)
        extension = (components == components[y, x]) & ~boundary
        recovered[extension] = 255
    return recovered, cv2.bitwise_and(weak, recovered)


def _seed(labels, start, eligible, min_width, max_tape, *, near=None, clipped=frozenset()):
    """Find the closest pair that brackets the camera axis."""
    axis = labels.shape[1] // 2
    for y in range(len(labels)-1, start-1, -1):
        row = labels[y]
        if row[axis]:
            continue
        left = np.flatnonzero(row[:axis])
        right = np.flatnonzero(row[axis+1:]) + axis + 1
        if not len(left) or not len(right):
            continue
        a, b = int(left[-1]), int(right[0])
        ids = (int(row[a]), int(row[b]))
        if near is not None and y < near and not clipped.intersection(ids):
            continue
        if ids[0] == ids[1]:
            return None, "ambiguous_joined_boundaries"
        if all(i in eligible for i in ids) and b-a >= min_width:
            if max(np.count_nonzero(row == i) for i in ids) <= max_tape:
                return ((a, y), (b, y), ids), "tracked"
    return None, "missing_boundary_pair"


def _clipped_boundaries(labels, stats, eligible, start):
    """Find strips that visibly exit a side of the frame near their lower end."""
    clipped = set()
    for label in eligible:
        _, top, _, height, _ = stats[label]
        end = top + height
        if end > start and (labels[max(start, end-max(3, height//10)):end, [0, -1]] == label).any():
            clipped.add(label)
    return clipped


def _single_seed(labels, stats, near, eligible, max_tape):
    if len(eligible) != 1:
        return None
    label = next(iter(eligible))
    _, top, _, height, _ = stats[label]
    for y in range(top+height-1, near-1, -1):
        xs = np.flatnonzero(labels[y] == label)
        axis = labels.shape[1] // 2
        if len(xs) and len(xs) <= max_tape and (xs[-1] < axis or xs[0] > axis):
            return (int(xs[-1] if xs[-1] < axis else xs[0]), y), label
    return None


def _trace(labels, seed, min_width):
    """Advance between the same two inner edges, rotating cross-sections at bends."""
    left, right, ids = seed
    height, width = labels.shape
    pairs = [(left, right)]
    step = max(2., round(height / 55))
    heading = 0.  # image forward = up, positive heading = clockwise
    turns = np.deg2rad([0, -7.5, 7.5, -15, 15, -30, 30, -45, 45])
    for _ in range(int(2 * (width + height) / step)):
        middle = (np.asarray(left) + right) / 2
        span = np.linalg.norm(np.subtract(right, left))
        angles = heading + turns
        normal = np.column_stack((np.cos(angles), np.sin(angles)))
        forward = np.column_stack((normal[:, 1], -normal[:, 0]))
        origins = middle + step * forward
        radius = np.arange(int(span * .85) + 3)
        hits = []
        valid = np.ones(len(turns), bool)
        for sign, label in zip((-1, 1), ids):
            rays = np.rint(origins[:, None, :] + sign * normal[:, None, :] *
                           radius[None, :, None]).astype(int)
            x, y = rays[:, :, 0], rays[:, :, 1]
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            values = np.where(inside, labels[y.clip(0, height-1), x.clip(0, width-1)], -1)
            first = (values != 0).argmax(axis=1)
            idx = np.arange(len(turns))
            valid &= values[idx, first] == label
            hits.append(rays[idx, first])
        a, b = hits
        centers = (a + b) / 2
        spans = np.linalg.norm(b-a, axis=1)
        movement = centers - middle
        valid &= ((spans >= min_width) & (np.linalg.norm(centers-origins, axis=1) <= step*1.5)
                  & ((movement * forward).sum(axis=1) >= step*.45))
        if not valid.any():
            break
        # Prefer the shortest crossing; penalize abrupt heading changes and
        # off-centre probes. The fixed angle order resolves equal scores.
        score = spans + np.linalg.norm(centers-origins, axis=1) + step * np.abs(turns)
        best = int(np.argmin(np.where(valid, score, np.inf)))
        left, right = tuple(map(int, a[best])), tuple(map(int, b[best]))
        if (left, right) in pairs[-8:]:
            break
        pairs.append((left, right))
        heading = angles[best]
    return np.asarray(pairs)


def _arm_rows(labels, observed, label, end, inner):
    arm = labels[:end+1] == label
    measured = arm & (observed[:end+1] != 0)
    # Closing keeps the strip connected. Use actual tape pixels for the
    # edge wherever they exist in this column.
    arm = np.where(measured.any(axis=0), measured, arm)
    rows = arm.argmax(axis=0) if inner else end - arm[::-1].argmax(axis=0)
    return np.where(arm.any(axis=0), rows, -1)


def _single_turn(route, labels, raw, anchor, label, top, near):
    """Suggest an in-place turn from a clearly visible outside elbow."""
    component = labels == label
    split, elbow = _boundary_elbow(component)
    if split:
        route.status = "ambiguous_split_boundary"
        return route
    if elbow is None:
        return route
    width = labels.shape[1]
    axis = width // 2
    direction, x, y = elbow
    if any(direction * (column-axis) >= -max(2, .04 * width) for column in (anchor[0], x)):
        return route
    rows = _arm_rows(labels, raw, label, y, False)
    end = -1 if direction < 0 else width
    xs = np.arange(x, end, direction)
    missing = np.flatnonzero(rows[xs] < 0)
    xs = xs[:missing[0]] if len(missing) else xs
    if len(xs) < .25 * width or direction * (xs[-1]-axis) < .05 * width:
        return route
    ys = rows[xs]
    observed = raw[ys, xs] != 0
    if observed.mean() < .85 or np.ptp(ys) > .20 * len(xs):
        return route
    route.turn_arm = tuple(zip(xs[observed].tolist(), (ys[observed] + top).tolist()))
    route.selected[top:][component] = 255
    stem_rows = np.flatnonzero(component.any(axis=1))
    near_support = (component[near:] & (raw[near:] != 0)).any(axis=1).mean()
    route.evidence = Evidence(len(stem_rows) / len(labels), float(observed.mean()), float(near_support))
    route.turn_direction = direction
    route.status = "turn_only"
    return route


def _clear_crossing(labels, near, far):
    length = max(abs(near[0] - far[0]), abs(near[1] - far[1])) + 1
    x, y = np.rint(np.linspace(near, far, length)).astype(int).T
    values = labels[y, x]
    middle = length // 2
    before, after = values[:middle], values[middle+1:]
    # Small gaps within a strip can appear on an oblique ray. The crossing
    # still needs a clear centre and each strip must stay on its own side.
    return (values[middle] == 0 and np.all((before == 0) | (before == values[0]))
            and np.all((after == 0) | (after == values[-1])))


def _trace_corner(labels, observed, ids, inbound, elbows, min_width):
    """Sweep across observed arms, holding the inside elbow as the outside edge catches up."""
    if elbows[0] is None or elbows[1] is None or elbows[0][0] != elbows[1][0]:
        return inbound, 0, None
    direction = elbows[0][0]
    inner = 0 if direction < 0 else 1
    outer = 1 - inner
    _, inner_x, inner_top = elbows[inner]
    _, outer_x, outer_top = elbows[outer]
    width = labels.shape[1]
    if direction * (inner_x - outer_x) <= 0 or inner_top <= outer_top:
        return inbound, 0, None

    inside_rows = _arm_rows(labels, observed, ids[inner], inner_top, True)
    outside_rows = _arm_rows(labels, observed, ids[outer], outer_top, False)
    step = max(2, round(len(labels) / 55))
    corner = []
    end_x = -1 if direction < 0 else width
    for x in range(outer_x, end_x, direction * step):
        a = min(x, inner_x) if direction < 0 else max(x, inner_x)
        near, far = (a, int(inside_rows[a])), (x, int(outside_rows[x]))
        if (far[1] < 0 or near[1] <= far[1] or np.hypot(a-x, near[1]-far[1]) < min_width
                or not _clear_crossing(labels, near, far)):
            break
        corner.append((near, far) if inner == 0 else (far, near))
    if len(corner) < 3:
        return inbound, 0, None
    # Keep only the approach that precedes the inside elbow. The new samples
    # start at the bend and run toward the visible end of the paired arms.
    incoming_y = inbound.mean(axis=1)[:, 1]
    after_elbow = np.flatnonzero(incoming_y < inner_top)
    count = int(after_elbow[0]) if len(after_elbow) else len(inbound)
    if not count:
        return inbound, 0, None
    return np.concatenate((inbound[:count], corner)), direction, count - 1


def detect_tape(frame: np.ndarray, cfg: TapeSettings) -> TapeRoute:
    height, width = frame.shape[:2]
    top, near = int(cfg.roi_top * height), int(cfg.near_y * height)
    hsv = cv2.cvtColor(frame[top:], cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, tuple(cfg.hsv_low), tuple(cfg.hsv_high))
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if cfg.gap_px:
        # Odd, centred kernel. Only bridge short, vertically aligned breaks.
        size = 2 * (cfg.gap_px // 2) + 1
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((size, 1), np.uint8),
                                 borderType=cv2.BORDER_REPLICATE)
    mask = np.zeros((height, width), np.uint8)
    mask[top:] = clean
    _, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    route = TapeRoute("no_reliable_tape", mask, np.zeros_like(mask))
    areas = stats[:, cv2.CC_STAT_AREA]
    plausible = areas >= cfg.min_area * clean.size
    labels = np.where(plausible[labels], labels, 0)
    near -= top
    full_extent = cfg.min_extent * len(clean)
    short_extent = max(4, .55 * full_extent)
    candidates = {i for i, (_, _, _, h, _) in enumerate(stats) if i and plausible[i] and h >= short_extent}
    near_ids = {i for i in candidates if stats[i, 1] + stats[i, 3] > near}
    eligible = {i for i in near_ids if stats[i, 3] >= full_extent}
    min_width = max(3, cfg.min_lane_width * width)
    max_tape = cfg.max_width * width
    seed, status = _seed(labels, near, eligible, min_width, max_tape)
    short_pair = False
    if seed is None and status == "missing_boundary_pair":
        start = max(0, near - round(.20 * len(clean)))
        clipped = _clipped_boundaries(labels, stats, candidates, start)
        seed, status = _seed(labels, start, near_ids | clipped, min_width, max_tape,
                             near=near, clipped=clipped)
        short_pair = seed is not None
    single = None
    if seed is None:
        route.status = status if near_ids or status != "missing_boundary_pair" else route.status
        if status != "missing_boundary_pair":
            return route
        single = _single_seed(labels, stats, near, near_ids, max_tape)
        if single is None or single[1] not in eligible:
            return route
    anchors = (single,) if single else tuple(zip(seed[:2], seed[2]))
    recovered, faint_observed = _recover_far_tape(hsv, labels, stats, anchors, cfg, height)
    if recovered.any():
        _, labels = cv2.connectedComponents(((labels != 0) | (recovered != 0)).astype(np.uint8),
                                            connectivity=8)
        mask[top:] |= recovered
        raw |= faint_observed
    ids = tuple(int(labels[y, x]) for (x, y), _ in anchors)
    if single:
        return _single_turn(route, labels, raw, single[0], ids[0], top, near)
    if ids[0] == ids[1]:
        route.status = "ambiguous_joined_boundaries"
        return route
    seed = (*seed[:2], ids)
    elbows = []
    for label in seed[2]:
        component = labels == label
        route.selected[top:][component] = 255
        split, elbow = _boundary_elbow(component)
        if split:
            route.status = "ambiguous_split_boundary"
            return route
        elbows.append(elbow)
    pairs = _trace(labels, seed, min_width)
    pairs, direction, approach_end = _trace_corner(labels, raw, seed[2], pairs, elbows, min_width)
    observed = (raw[pairs[:, :, 1], pairs[:, :, 0]] != 0).all(axis=1)
    centers = pairs.mean(axis=1)
    ys = centers[:, 1]
    extent = float(ys.max() - ys.min() + 1) / len(clean)
    if short_pair:
        y = seed[0][1]
        near_rows = np.arange(max(0, y - max(3, round(.08 * len(clean)))), y + 1)
    else:
        near_rows = np.arange(near, len(clean))
    # Near-field overlap is evaluated on actual observed rows of both strips.
    near_support = np.ones(len(near_rows), bool)
    for label in seed[2]:
        near_support &= ((labels[near_rows] == label) & (raw[near_rows] != 0)).any(axis=1)
    route.evidence = Evidence(extent, float(observed.mean()), float(near_support.mean()))
    if (extent * len(clean) < short_extent or len(pairs) < 3 or route.evidence.continuity < .85
            or route.evidence.near_support < .15):
        route.status = "insufficient_boundary_overlap"
        return route
    pairs[:, :, 1] += top
    route.left_edge = tuple(map(tuple, pairs[:, 0].tolist()))
    route.right_edge = tuple(map(tuple, pairs[:, 1].tolist()))
    route.points = tuple(map(tuple, pairs.mean(axis=1).tolist()))
    target = len(route.points) // 3
    if approach_end is not None:
        target = min(target, approach_end)
    route.image_target = route.points[target]
    route.turn_direction = direction
    route.approach_end = approach_end
    route.short_pair = short_pair or extent < cfg.min_extent
    route.status = "tracked"
    return route
