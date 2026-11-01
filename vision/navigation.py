"""Visual navigation suggestions only: no motor or sensor commands."""

from dataclasses import dataclass

import cv2
import numpy as np

from drawing import draw_text

from path_estimation import estimate_path, path_thickness, draw_corridor


POTENTIAL_COLOUR = (255, 210, 0)  # Cyan: hypothetical repositioned origin.


@dataclass
class NavigationSettings:
    backup_occupancy: float = .50
    backup_near_occupancy: float = .35
    unknown_fraction: float = .55
    steering_deadband: float = .06


def alternative_corridors(floor_mask, cost_map=None, path_options=None):
    """One hypothetical corridor per side, with no verified connection from the car."""
    height, width = floor_mask.shape
    floor = floor_mask != 0
    if np.count_nonzero(floor) < 60:
        return []
    clearance = cv2.distanceTransform(np.pad(floor.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    alternatives = []
    for side, starts in (("left", (.18, .32)), ("right", (.68, .82))):
        candidates = []
        for y in (.88, .72, .60):
            for x in starts:
                points = estimate_path(floor, cost_map=cost_map, origin=(x, y), **(path_options or {}))
                if not points:
                    continue
                px, py = np.asarray(points).T
                reach = (points[0][1] - points[-1][1]) / height
                mean_cost = float(cost_map[py, px].mean()) if cost_map is not None else 0.
                score = (reach + .35 * float(clearance[py, px].mean()) / width
                         - .15 * mean_cost - .15 * abs(x - .5))
                candidates.append(dict(points=points, side=side, score=round(score, 5),
                                       origin=[x, y], reachable_from_car=False))
            if candidates:
                break
        if candidates:
            alternatives.append(max(candidates, key=lambda candidate: candidate["score"]))
    return alternatives


def choose_decision(points, alternatives, obstacle_mask, unknown_mask=None, usable=True, settings=None):
    """Explain a frame, keeping repositioning/reversing separate from drive authorization."""
    settings = settings or NavigationSettings()
    height, width = obstacle_mask.shape
    centre = (slice(int(height * .45), height), slice(int(width * .20), int(width * .80)))
    near = (slice(int(height * .65), int(height * .95)), slice(int(width * .35), int(width * .65)))
    occupancy = float((obstacle_mask[centre] != 0).mean())
    near_occupancy = float((obstacle_mask[near] != 0).mean())
    uncertainty = float((unknown_mask[centre] != 0).mean()) if unknown_mask is not None else 0.
    base = dict(obstacle_fraction=round(occupancy, 4), near_obstacle_fraction=round(near_occupancy, 4),
                unknown_fraction=round(uncertainty, 4), motion_authorized=False,
                rear_clearance_required=False)

    def result(action, label, reason, **extra):
        return dict(**base, action=action, label=label, reason=reason, **extra)

    if not usable or uncertainty >= settings.unknown_fraction:
        return result("STOP", "STOP", "View unusable or too uncertain")
    best = max(alternatives, key=lambda candidate: candidate["score"]) if alternatives else None
    if occupancy >= settings.backup_occupancy and near_occupancy >= settings.backup_near_occupancy:
        decision = result("BACK_UP", "BACK UP? CHECK REAR", "Large foreground blockage; rear clearance unverified",
                          target_side=best["side"] if best else None)
        decision["rear_clearance_required"] = True
        return decision
    if points:
        target = points[min(len(points) - 1, max(1, int((len(points) - 1) * .4)))]
        offset = target[0] / width - .5
        if offset < -settings.steering_deadband:
            return result("STEER_LEFT", "STEER LEFT", "Follow the magenta origin-connected path")
        if offset > settings.steering_deadband:
            return result("STEER_RIGHT", "STEER RIGHT", "Follow the magenta origin-connected path")
        return result("FORWARD", "CONTINUE FORWARD", "Follow the magenta origin-connected path")
    if best:
        side = best["side"].upper()
        return result(f"REPOSITION_{side}", f"STEER {side} / REPOSITION",
                      "Cyan corridor is hypothetical; no verified connection from car", target_side=best["side"])
    return result("STOP", "STOP / REASSESS", "No origin path or useful visible alternative")


def plan_navigation(floor_mask, points, obstacle_mask, unknown_mask=None, cost_map=None,
                    path_options=None, usable=True, settings=None):
    alternatives = alternative_corridors(floor_mask, cost_map, path_options) if not points and usable else []
    decision = choose_decision(points, alternatives, obstacle_mask, unknown_mask, usable, settings)
    corridor_settings = {key: value for key, value in (path_options or {}).items()
                         if key in ("corridor_near", "corridor_far", "corridor_buffer")}
    return dict(alternatives=alternatives, decision=decision, corridor_settings=corridor_settings)


def draw_navigation(view, navigation, frame_width, floor_mask=None):
    for candidate in navigation["alternatives"]:
        points = np.asarray(candidate["points"], np.int32)
        draw_corridor(view, candidate["points"], POTENTIAL_COLOUR, floor_mask)
        cv2.polylines(view, [points], False, POTENTIAL_COLOUR, path_thickness(frame_width), cv2.LINE_8)
        cv2.circle(view, tuple(points[0]), 5, POTENTIAL_COLOUR, 1)
    draw_text(view, "Dotted: floor corridor | Magenta: origin | Cyan: reposition", (8, view.shape[0] - 35),
              .45)


def decision_banner(view, navigation):
    decision = navigation["decision"]
    cv2.rectangle(view, (0, 0), (view.shape[1], 62), (0, 0, 0), -1)
    colour = (0, 210, 255) if decision["action"] in ("STOP", "BACK_UP") else (255, 255, 255)
    draw_text(view, f"MOCK: {decision['label']}", (8, 22), .55, colour)
    draw_text(view, decision["reason"], (8, 47), .45)
