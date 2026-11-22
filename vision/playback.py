"""Single/debug and four-panel diagnostics, with independent model panels."""

import cv2
import numpy as np

from drawing import draw_text, title_panel

from detector import Settings
from pipeline import FramePlan
from path_estimation import draw_path
from navigation import draw_navigation, decision_banner


def fused_panel(fusion):
    eligible = fusion["mask"]
    # Green = lower cost, yellow/red = higher cost; rejected floor = red.
    cost = np.clip(fusion["cost"] / 5, 0, 1)
    panel = np.zeros((*eligible.shape, 3), np.uint8)
    panel[:, :, 1][eligible] = 220
    panel[:, :, 2][eligible] = (240 * cost[eligible]).astype(np.uint8)
    panel[fusion["rejected"]] = (30, 30, 220)
    return title_panel(panel, f"{fusion['status']} | green preferred / yellow costly / red rejected")


def four_panel(frame, segmentation, depth_panel, fusion, points, timings, navigation=None):
    raw = frame.copy()
    draw_path(raw, points, frame.shape[1], floor_mask=fusion["mask"])
    raw = title_panel(raw, "Raw video + best path" if points else "Raw video | no reliable path")
    if navigation is not None:
        draw_navigation(raw, navigation, frame.shape[1], floor_mask=fusion["mask"])
        decision_banner(raw, navigation)
    draw_text(raw, f"Seg {timings[0]:.0f} ms | Depth {timings[1]:.0f} ms | Path {timings[2]:.0f} ms", (8, frame.shape[0] - 12))
    return np.vstack([np.hstack([raw, segmentation]), np.hstack([depth_panel, fused_panel(fusion)])])


def render_segmentation(frame: np.ndarray, result: dict, method: str, settings: Settings,
                        show_floor_sample: bool = False, debug: bool = False) -> tuple[np.ndarray, tuple]:
    """Render perception before adding route marks to the single-panel view."""
    if method == "semantic":
        from semantic import annotate_semantic
        segmentation, classes, confidence = annotate_semantic(frame, result)
        return segmentation, (classes, confidence)
    from detector import annotate
    segmentation = annotate(frame, result, settings, show_floor_sample)
    debug_panels = ()
    if debug:
        debug_panels = (cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(result["edges"], cv2.COLOR_GRAY2BGR))
    return segmentation, debug_panels


def render_single(segmentation: np.ndarray, plan: FramePlan, debug_panels: tuple,
                  debug: bool = False) -> np.ndarray:
    """Add the existing single/debug route display to segmentation."""
    height, width = segmentation.shape[:2]
    points, navigation, floor_mask = plan.points, plan.navigation, plan.floor_mask
    overlay = segmentation
    draw_path(overlay, points, width, floor_mask=floor_mask)
    draw_navigation(overlay, navigation, width, floor_mask=floor_mask)
    draw_text(overlay, f"MOCK: {navigation['decision']['label']}", (8, height - 58))
    if debug:
        overlay = np.hstack([overlay, *debug_panels])
    return overlay
