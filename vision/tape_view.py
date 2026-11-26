"""Tape diagnostics on the prepared image, plus a separate robot-frame reference."""

import cv2
import numpy as np

from drawing import draw_text


def render_tape(frame, route, proposal, cal, cfg, processing_ms, age_ms, debug, actuate=False):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    if route is not None:
        selected = route.selected != 0
        overlay[selected] = overlay[selected] // 2 + np.array([0, 105, 0], np.uint8)
        if route.points:
            stride = max(1, len(route.points) // 8)
            for a, b in zip(route.left_edge[::stride], route.right_edge[::stride]):
                cv2.line(overlay, a, b, (130, 120, 45), 1, cv2.LINE_AA)
            points = np.rint(route.points).astype(np.int32)
            cv2.polylines(overlay, [points], False, (0, 255, 255), 2)
        if route.turn_arm:
            a, b = route.turn_arm[len(route.turn_arm)//4], route.turn_arm[3*len(route.turn_arm)//4]
            cv2.arrowedLine(overlay, a, b, (255, 0, 255), 2, cv2.LINE_AA, tipLength=.10)
        target = proposal.target_px or route.image_target
        if target:
            cv2.circle(overlay, tuple(map(round, target)), 6, (255, 0, 255), 2)
    for fraction in (cfg.roi_top, cfg.near_y):
        y = int(fraction * height)
        cv2.line(overlay, (0, y), (width-1, y), (150, 150, 150), 1)
    if cal is not None and proposal.calibration == "valid":
        cv2.polylines(overlay, [np.rint(cal.region).astype(np.int32)], True, (220, 160, 30), 1)
        try:
            origin = cal.robot_to_image((0., 0.))
        except ValueError:
            origin = None  # The turn centre can project to infinity or off screen.
        if origin:
            cv2.drawMarker(overlay, tuple(map(round, origin)), (255, 255, 255))
    if debug:
        mask = route.mask if route is not None else np.zeros((height, width), np.uint8)
        overlay = np.hstack((frame, overlay, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
    if overlay.shape[1] < 600:
        scaled_height = max(2, round(overlay.shape[0] * 300 / overlay.shape[1]) * 2)
        overlay = cv2.resize(overlay, (600, scaled_height))
    footer_width = min(overlay.shape[1], 1280)
    footer = np.zeros((100, footer_width, 3), np.uint8)
    state = ("CALIBRATED" if proposal.calibration == "valid" else
             f"CALIBRATION REQUIRED ({proposal.calibration})")
    mode = "LIVE ACTUATION ENABLED" if actuate else "DRY RUN / NOT SENT"
    draw_text(footer, f"{mode}   |   {state}", (8, 18))
    detection = proposal.detection
    if route is not None and route.turn_direction:
        detection += " / left corner" if route.turn_direction < 0 else " / right corner"
    if route is not None and route.short_pair:
        detection += " / short pair"
    draw_text(footer, f"{detection}   |   heuristic quality {proposal.quality_heuristic:.2f}",
              (8, 42))
    movement = f"STOP   {proposal.reason}"
    if proposal.actions:
        turn = proposal.actions[0]
        if len(proposal.actions) == 2:
            movement = f"TURN {turn.angle_rad:+.3f} rad   then DRIVE {proposal.actions[1].distance_m:.3f} m"
        else:
            side = "LEFT" if turn.angle_rad > 0 else "RIGHT"
            movement = f"TURN {side} {abs(turn.angle_rad):.3f} rad   |   in place"
    elif route is not None and route.turn_arm and proposal.reason.startswith("calibration_"):
        side = "LEFT" if route.turn_direction < 0 else "RIGHT"
        movement = f"TURN {side} in place   |   needs calibration"
    draw_text(footer, movement, (8, 65), colour=(140, 230, 255))
    draw_text(footer, f"Processing {processing_ms:.1f} ms   |   local receipt age {age_ms:.1f} ms",
              (8, 89))
    # Top-down inset, useful even when the turn centre is outside the camera view.
    origin = (footer.shape[1]-55, 76)
    cv2.arrowedLine(footer, origin, (origin[0], 22), (155, 155, 155), 1)
    if proposal.target_left_forward_m:
        left, forward = proposal.target_left_forward_m
        scale = 40 / np.hypot(left, forward)
        cv2.arrowedLine(footer, origin, (origin[0]-round(left*scale), origin[1]-round(forward*scale)),
                        (255, 0, 255), 2)
    cv2.circle(footer, origin, 3, (255, 255, 255), -1)
    draw_text(footer, "turn centre", (origin[0]-40, 94), .42)
    if footer_width < overlay.shape[1]:
        footer_height = 2 * round(50 * overlay.shape[1] / footer_width)
        footer = cv2.resize(footer, (overlay.shape[1], footer_height))
    return np.vstack((overlay, footer))
