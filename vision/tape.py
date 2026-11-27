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
    can_estimate_turn: bool = False


def _boundary_elbow(component):
    """Find the stem at a one-sided far elbow and reject split boundaries."""
    rows, cols = np.nonzero(component)
    component = component[rows.min():rows.max()+1, cols.min():cols.max()+1]
    left = component.argmax(axis=1)
    right = component.shape[1] - 1 - component[:, ::-1].argmax(axis=1)
    widths = right - left + 1
    typical = np.median(widths)
    gaps = widths - component.sum(axis=1) > max(2, .15 * typical)
    split_rows = max(3, min(8, round(.30 * typical)))
    broad = widths > 2.5 * typical + 2
    # A slanted far arm can be narrow on each row despite spanning much of
    # the image. Look for a sustained sideways sweep relative to the stem.
    centers = (left + right) / 2
    sweep = np.abs(centers[4:] - centers[:-4]) / 4
    if len(sweep) >= 6:
        fast = sweep > max(3, 2 * np.median(sweep[len(sweep)//2:]))
        coherent = fast[:-2] & fast[1:-1] & fast[2:]
        broad[3:-3] |= coherent & (np.arange(3, len(widths)-3) < .60 * len(widths))
    broad = np.flatnonzero(broad)
    if not len(broad):
        return np.count_nonzero(gaps) >= split_rows, None
    # Compare against the stem at the bend, where perspective has had the
    # least chance to shift the strip sideways.
    stem = min(len(widths) - 1, broad[-1] + 1)
    if (np.count_nonzero(gaps[stem:]) >= split_rows or
            broad[-1] - broad[0] > max(2 * typical, .40 * len(widths)) or
            (broad[0] > typical and len(widths) - stem > typical)):
        return True, None
    extends_left = left[:stem+1].min() < left[stem] - typical
    extends_right = right[:stem+1].max() > right[stem] + typical
    if extends_left and extends_right:
        return True, None
    if extends_left or extends_right:
        direction = -1 if extends_left else 1
        x = right[stem] if extends_left else left[stem]
        return False, (direction, int(cols.min() + x), int(rows.min() + stem))
    return np.count_nonzero(gaps) >= split_rows, None


def _trim_clipped_boundary(component, side, seed_y):
    """Recover a straight inner edge when clutter fans out at the near image border."""
    rows = np.flatnonzero(component.any(axis=1))
    height = rows[-1] - rows[0] + 1
    cutoff = rows[0] + round(.60 * height)
    border = 0 if side == 0 else -1
    if not component[cutoff:, border].any() or cutoff >= seed_y:
        return None

    # Use inward-increasing coordinates on both sides. Fit only the upper,
    # unbranched stem, then check that the visible inner edge keeps that line.
    inward = component if side == 0 else component[:, ::-1]
    edge = np.where(inward.any(axis=1), inward.shape[1] - 1 - inward[:, ::-1].argmax(axis=1), -1)
    stem = rows[(rows >= rows[0] + round(.10 * height)) & (rows < rows[0] + round(.45 * height))]
    if len(stem) < 6 or _boundary_elbow(inward[:cutoff])[0]:
        return None
    tolerance = max(3, float(np.median(inward[stem].sum(axis=1))))
    slope, intercept = np.polyfit(stem, edge[stem], 1)
    expected = slope * np.arange(len(inward)) + intercept
    upper = rows[(rows >= stem[0]) & (rows < cutoff)]
    shared = np.arange(cutoff, seed_y + 1)
    if (slope >= 0 or np.any(np.abs(edge[upper] - expected[upper]) > .65 * tolerance) or
            np.any(np.abs(edge[shared] - expected[shared]) > tolerance)):
        return None

    # Trim only outward from the observed inner edge. A branch into the
    # corridor fails the line check above instead of being edited out.
    trimmed = inward.copy()
    columns = np.arange(inward.shape[1])
    trimmed[cutoff:seed_y+1] &= columns >= edge[shared, None] - 2 * tolerance
    trimmed[seed_y+1:] = False
    if _boundary_elbow(trimmed)[0]:
        return None
    return trimmed if side == 0 else trimmed[:, ::-1]


def _color_masks(hsv, cfg):
    hue, saturation, value = cfg.hsv_low
    raw = cv2.inRange(hsv, cfg.hsv_low, cfg.hsv_high)
    # Chroma is the difference between the brightest and darkest channels.
    # Compare it with the local floor color to suppress broad blue casts.
    chroma = cv2.multiply(hsv[:, :, 1], hsv[:, :, 2], scale=1 / 255)
    size = max(9, 2 * round(hsv.shape[1] / 65) + 1)
    contrast = cv2.morphologyEx(chroma, cv2.MORPH_TOPHAT, np.ones((size, size), np.uint8))
    strong = raw & ((contrast >= 10) | (chroma >= 70)).astype(np.uint8) * 255
    weak = cv2.inRange(hsv, (hue, min(saturation, max(24, round(.30 * saturation))), value),
                       cfg.hsv_high)
    weak &= ((contrast >= 8) & (chroma >= 12)).astype(np.uint8) * 255
    return strong, weak


def _join_gaps(mask, gap_px):
    if not gap_px:
        return mask
    size = 2 * (gap_px // 2) + 1
    # Close each orientation independently. Zero padding keeps a gap at the
    # image edge from being filled without a pixel on both sides.
    return mask | cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((size, 1), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0) | cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((1, size), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _recover_far_tape(weak, labels, stats, anchors, cfg, frame_height):
    """Keep faint horizontal tape only where it connects to a seeded far tip."""
    width = labels.shape[1]
    max_thickness = max(3, round(.018 * frame_height))
    broad = cv2.morphologyEx(weak, cv2.MORPH_OPEN, np.ones((max_thickness + 1, 1), np.uint8))
    thin = weak & ~broad
    joined = _join_gaps(thin, cfg.gap_px)
    length = max(5, 2 * (width // 240) + 1)
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


def _single_seed(labels, stats, start, eligible, max_tape, *, clipped_stem=False):
    if len(eligible) != 1:
        return None
    label = next(iter(eligible))
    _, top, _, height, _ = stats[label]
    for y in range(top+height-1, start-1, -1):
        xs = np.flatnonzero(labels[y] == label)
        axis = labels.shape[1] // 2
        if len(xs) and len(xs) <= max_tape and (xs[-1] < axis or xs[0] > axis):
            on_left = xs[-1] < axis
            if clipped_stem:
                border = 0 if on_left else -1
                if not (labels[max(start, y-max(3, height//10)):y+1, border] == label).any():
                    continue
            return (int(xs[-1] if on_left else xs[0]), y), label
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


def _outside_elbow_arm(labels, raw, anchor, label, elbow):
    if elbow is None:
        return (), 0.
    width = labels.shape[1]
    axis = width // 2
    direction, x, y = elbow
    if any(direction * (column-axis) >= -max(2, .04 * width) for column in (anchor[0], x)):
        return (), 0.
    rows = _arm_rows(labels, raw, label, y, False)
    end = -1 if direction < 0 else width
    xs = np.arange(x, end, direction)
    missing = np.flatnonzero(rows[xs] < 0)
    xs = xs[:missing[0]] if len(missing) else xs
    if len(xs) < .25 * width or direction * (xs[-1]-axis) < .05 * width:
        return (), 0.
    ys = rows[xs]
    observed = raw[ys, xs] != 0
    if observed.mean() < .85 or np.ptp(ys) > .20 * len(xs):
        return (), 0.
    return tuple(zip(xs[observed].tolist(), ys[observed].tolist())), float(observed.mean())


def _side_entering_arm(component, raw, anchor):
    """Follow a single connected strip inwards from the side it enters."""
    height, width = component.shape
    x, y = anchor
    from_left = x < width // 2
    inward = component if from_left else component[:, ::-1]
    end = np.flatnonzero(inward.any(axis=0))[-1] + 1
    near_rows = slice(max(0, y - max(3, round(.07 * height))), y + 1)
    if end < .18 * width or not inward[near_rows, 0].any():
        return (), 0.

    pixels = inward[:, :end]
    top = pixels.argmax(axis=0)
    bottom = height - 1 - pixels[::-1].argmax(axis=0)
    span = bottom - top + 1
    gaps = span - pixels.sum(axis=0)
    if (np.count_nonzero(gaps > max(3, .012 * height)) > max(2, .035 * end) or
            np.median(span) > max(8, .06 * height)):
        return (), 0.

    # Evaluate the near-to-far shape in short sections so stray pixels and
    # thick tape at the frame edge have little influence on its heading.
    centers = (top + bottom) / 2
    rows = np.array([np.median(section) for section in np.array_split(centers, min(8, end))])
    rise = rows[0] - rows[-1]
    if (rise < max(3, .03 * height) or rise > .75 * end or
            np.any(np.diff(rows) > max(3, .02 * height))):
        return (), 0.
    if end >= .85 * width and (rows[0] - rows[1]) - (rows[-2] - rows[-1]) < .02 * end:
        return (), 0.  # A straight crossing that exits both sides has no unique turn.

    observed = (pixels & ((raw if from_left else raw[:, ::-1])[:, :end] != 0))
    visible = observed.any(axis=0)
    support = float(visible.mean())
    if support < .80:
        return (), support
    ys = height - 1 - observed[::-1].argmax(axis=0)
    xs = np.arange(end) if from_left else width - 1 - np.arange(end)
    return tuple(zip(xs[visible].tolist(), ys[visible].tolist())), support


def _single_turn(route, labels, raw, anchor, label, top, near, full_extent):
    component = labels == label
    split, elbow = _boundary_elbow(component)
    side_arm, support = _side_entering_arm(component, raw, anchor)
    stem_rows = np.flatnonzero(component.any(axis=1))
    arm, arm_support = (), 0.
    if len(stem_rows) >= full_extent:
        arm, arm_support = _outside_elbow_arm(labels, raw, anchor, label, elbow)
    if not arm and not side_arm:
        if split:
            route.status = "ambiguous_split_boundary"
        return route
    route.can_estimate_turn = bool(side_arm)
    route.status = "turn_only" if arm else "edge_turn"
    route.turn_direction = elbow[0] if arm else (1 if anchor[0] < labels.shape[1] // 2 else -1)
    route.turn_arm = tuple((x, y + top) for x, y in (arm or side_arm))
    route.selected[top:][component & (raw != 0)] = 255
    y_anchor = anchor[1]
    near_rows = slice(near, None) if arm and y_anchor >= near else slice(
        max(0, y_anchor - max(3, round(.08 * len(labels)))), y_anchor + 1)
    near_support = (component[near_rows] & (raw[near_rows] != 0)).any(axis=1).mean()
    route.evidence = Evidence(len(stem_rows) / len(labels), arm_support if arm else support, float(near_support))
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
    # Follow the outside stem around the inside elbow before sweeping along
    # the arms. Each crossing still has to meet two observed strips.
    incoming_y = inbound.mean(axis=1)[:, 1]
    after_elbow = np.flatnonzero(incoming_y < inner_top)
    count = int(after_elbow[0]) if len(after_elbow) else len(inbound)
    pivot = (inner_x, int(inside_rows[inner_x]))
    if pivot[1] < 0:
        return inbound, 0, None
    sweep = []
    start = int(inbound[max(0, count-1), outer, 1]) - (step if count else 0)
    for y in range(start, outer_top, -step):
        xs = np.flatnonzero(labels[y] == ids[outer])
        if not len(xs):
            return inbound, 0, None
        edge = (int(xs[0] if direction < 0 else xs[-1]), y)
        if np.hypot(edge[0]-pivot[0], edge[1]-pivot[1]) < min_width or not _clear_crossing(labels, pivot, edge):
            return inbound, 0, None
        sweep.append((pivot, edge) if inner == 0 else (edge, pivot))
    return np.asarray(inbound[:count].tolist() + sweep + corner), direction, count - 1 if count else None


def detect_tape(frame: np.ndarray, cfg: TapeSettings) -> TapeRoute:
    height, width = frame.shape[:2]
    top, near = int(cfg.roi_top * height), int(cfg.near_y * height)
    hsv = cv2.cvtColor(frame[top:], cv2.COLOR_BGR2HSV)
    raw, weak = _color_masks(hsv, cfg)
    # A thin strip can have only one pixel across it. Require neighbors
    # without eroding away a line or displacing its inside edge.
    support = cv2.boxFilter((raw != 0).astype(np.uint8), -1, (3, 3), normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
    clean = _join_gaps(np.where(support >= 3, raw, 0).astype(np.uint8), cfg.gap_px)
    mask = np.zeros((height, width), np.uint8)
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
    start = max(0, near - round(.20 * len(clean)))
    clipped = _clipped_boundaries(labels, stats, candidates, start)
    display = np.zeros(len(stats), bool)
    display[list(near_ids | clipped)] = True
    mask[top:] = display[labels].astype(np.uint8) * 255
    seed, status = _seed(labels, near, eligible, min_width, max_tape)
    short_pair = (seed is not None and bool(clipped.intersection(seed[2])) and
                  seed[0][1] - near + 1 < .15 * (len(clean) - near))
    if seed is None and status == "missing_boundary_pair":
        seed, status = _seed(labels, start, near_ids | clipped, min_width, max_tape,
                             near=near, clipped=clipped)
        short_pair = seed is not None
    single = None
    if seed is None:
        route.status = status if near_ids or status != "missing_boundary_pair" else route.status
        if status != "missing_boundary_pair":
            return route
        single = _single_seed(labels, stats, near if near_ids else start, near_ids | clipped,
                              max_tape, clipped_stem=not near_ids)
        if single is None:
            return route
    anchors = (single,) if single else tuple(zip(seed[:2], seed[2]))
    recovered, faint_observed = _recover_far_tape(weak, labels, stats, anchors, cfg, height)
    if recovered.any():
        _, labels = cv2.connectedComponents(((labels != 0) | (recovered != 0)).astype(np.uint8),
                                            connectivity=8)
        mask[top:] |= recovered
        raw |= faint_observed
    ids = tuple(int(labels[y, x]) for (x, y), _ in anchors)
    if single:
        return _single_turn(route, labels, raw, single[0], ids[0], top, near, full_extent)
    if ids[0] == ids[1]:
        route.status = "ambiguous_joined_boundaries"
        return route
    seed = (*seed[:2], ids)
    components = [labels == label for label in ids]
    route.selected[top:][(components[0] | components[1]) & (raw != 0)] = 255
    elbows = []
    for side, component in enumerate(components):
        split, elbow = _boundary_elbow(component)
        if split:
            trimmed = _trim_clipped_boundary(component, side, seed[side][1])
            if trimmed is None:
                route.status = "ambiguous_split_boundary"
                return route
            labels[component & ~trimmed] = 0
            route.selected[top:][component & ~trimmed] = 0
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
    min_overlap = max(4, (.28 if short_pair else .40) * full_extent)
    if (extent * len(clean) < min_overlap or len(pairs) < 3 or route.evidence.continuity < .85
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
