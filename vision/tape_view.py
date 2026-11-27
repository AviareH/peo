"""Tape diagnostics on the prepared image, plus a separate robot-frame reference."""

import cv2
import numpy as np

from drawing import draw_text


def render_tape(frame, route, proposal, cal, cfg, processing_ms, age_ms, debug, armed=False,
                can_arm=False):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    if route is not None:
        selected = route.selected != 0
        overlay[selected] = overlay[selected] // 2 + np.array([0, 105, 0], np.uint8)
        if route.points:
            stride = max(1, (len(route.points) + 7) // 8)
            for a, b in zip(route.left_edge[::stride], route.right_edge[::stride]):
                cv2.line(overlay, a, b, (130, 120, 45), 1, cv2.LINE_AA)
            if debug:
                for edge in (route.left_edge, route.right_edge):
                    for x, y in edge:
                        if selected[y, x]:
                            cv2.circle(overlay, (x, y), 2, (255, 255, 0), -1, cv2.LINE_AA)
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
    if can_arm:
        mode = "ARMED (S disarm, Q stop)" if armed else "DISARMED (A arm, Q stop)"
    else:
        mode = "LIVE ACTUATION ENABLED" if armed else "DRY RUN / NOT SENT"
    draw_text(footer, f"{mode}   |   {state}", (8, 18))
    detection = proposal.detection
    if route is not None and route.turn_direction:
        detection += " / left corner" if route.turn_direction < 0 else " / right corner"
    if route is not None and route.short_pair:
        detection += " / short pair"
    evidence = f"{detection}   |   heuristic quality {proposal.quality_heuristic:.2f}"
    movement = f"CV: STOP  {proposal.reason}"
    if proposal.actions:
        turn = proposal.actions[0]
        side = "LEFT" if turn.angle_rad > 0 else "RIGHT"
        turning = f"TURN {side} {abs(np.degrees(turn.angle_rad)):.1f} deg"
        if len(proposal.actions) == 2:
            forward = f"FORWARD {proposal.actions[1].distance_m * 1000:.0f} mm"
            movement = f"CV: {turning}, then {forward}" if turn.angle_rad else f"CV: {forward}"
        else:
            movement = f"CV: {turning} in place"
            if proposal.reason == "estimated_edge_turn":
                movement += " (estimated)"
    elif route is not None and route.turn_arm and proposal.reason != "stale_frame":
        side = "LEFT" if route.turn_direction < 0 else "RIGHT"
        evidence = f"CV intent: TURN {side} (image only)   |   quality {proposal.quality_heuristic:.2f}"
    elif (route is not None and route.status == "tracked" and not route.turn_direction
          and route.image_target is not None and proposal.reason != "stale_frame"
          and all(abs(x - width / 2) <= .06 * width for x, _ in route.points)):
        evidence = "CV intent: FORWARD (image only)"
    draw_text(footer, evidence, (8, 42))
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
    elif len(proposal.actions) == 1:
        angle = proposal.actions[0].angle_rad
        heading = (origin[0] - round(48 * np.sin(angle)), origin[1] - round(48 * np.cos(angle)))
        cv2.arrowedLine(footer, origin, heading, (255, 0, 255), 2)
    cv2.circle(footer, origin, 3, (255, 255, 255), -1)
    draw_text(footer, "turn centre", (origin[0]-40, 94), .42)
    if footer_width < overlay.shape[1]:
        footer_height = 2 * round(50 * overlay.shape[1] / footer_width)
        footer = cv2.resize(footer, (overlay.shape[1], footer_height))
    return np.vstack((overlay, footer))
