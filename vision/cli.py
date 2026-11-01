"""Command-line options and validation for offline replay."""

import argparse
import math
from pathlib import Path

from path_estimation import (DEFAULT_MARGIN, DEFAULT_ENDPOINT_MARGIN, DEFAULT_MIN_FORWARD,
                             DEFAULT_CORRIDOR_NEAR, DEFAULT_CORRIDOR_FAR, DEFAULT_CORRIDOR_BUFFER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline obstacle detection on a local video; optional file picker/preview.")
    parser.add_argument("video", nargs="?", type=Path, help="MOV/MP4 file; omit for file picker")
    parser.add_argument("--method", choices=["classical", "semantic"], default="classical")
    parser.add_argument("--model", choices=["b0", "b2"], default="b0",
                        help="Pretrained SegFormer size for semantic mode (default: b0)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--confidence", type=float, default=.5, help="Semantic minimum class score (0..1)")
    parser.add_argument("--offline", action="store_true", help="Use cached model files only")
    parser.add_argument("--depth", action="store_true", help="Combine depth and floor estimates; show four panels")
    parser.add_argument("--depth-mode", choices=["fused", "compare"], default="fused",
                        help="Use depth for the path (default), or retain segmentation-only routing")
    parser.add_argument("--depth-tolerance", type=float, default=.10,
                        help="Experimental ground-depth tolerance, relative units (default: .10)")
    parser.add_argument("--path-margin", type=float, default=DEFAULT_MARGIN,
                        help="Hard path clearance as a fraction of image width (default: .03)")
    parser.add_argument("--path-end-margin", type=float, default=DEFAULT_ENDPOINT_MARGIN,
                        help="Endpoint stand-off as a fraction of image width (default: .06)")
    parser.add_argument("--path-min-forward", type=float, default=DEFAULT_MIN_FORWARD,
                        help="Minimum forward image height traversed before showing a path (default: .20)")
    parser.add_argument("--corridor-near", type=float, default=DEFAULT_CORRIDOR_NEAR,
                        help="Estimated full footprint width at image bottom, fraction of width (default: .19)")
    parser.add_argument("--corridor-far", type=float, default=DEFAULT_CORRIDOR_FAR,
                        help="Estimated full footprint width near horizon (default: .04)")
    parser.add_argument("--corridor-buffer", type=float, default=DEFAULT_CORRIDOR_BUFFER,
                        help="Extra clearance on each side, fraction of image width (default: .01)")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames for a quick comparison")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--preview", action="store_true", help="Show overlay; Q exits, space pauses")
    parser.add_argument("--debug", action="store_true", help="Save colour/edge masks beside overlay")
    parser.add_argument("--show-floor-sample", action="store_true",
                        help="Show the calibration rectangle in addition to the floor shape")
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="Extra clockwise rotation after decoder orientation")
    parser.add_argument("--start-seconds", type=float, default=0,
                        help="Seek to a frame with clear floor in the sample patch")
    parser.add_argument("--roi-top", type=float, default=0.35)
    parser.add_argument("--floor-patch", type=float, nargs=4, default=[.4, .8, .6, .95],
                        metavar=("X1", "Y1", "X2", "Y2"), help="Floor sample rectangle, fractions 0..1")
    parser.add_argument("--hue-tolerance", type=int, default=18)
    parser.add_argument("--saturation-tolerance", type=int, default=65)
    parser.add_argument("--value-tolerance", type=int, default=65)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--min-area", type=float, default=.003, help="Fraction of ROI area")
    parser.add_argument("--min-edge-support", type=float, default=.08)
    args = parser.parse_args()
    if not (0 <= args.corridor_far <= args.corridor_near <= .6 and 0 <= args.corridor_buffer <= .1):
        parser.error("Corridor widths must be 0 <= far <= near <= .6; buffer must be 0.. .1")
    if not (.005 <= args.path_margin <= args.path_end_margin <= .25
            and .05 <= args.path_min_forward <= .75):
        parser.error("Path margins must be .005 <= margin <= end-margin <= .25; min-forward must be .05 to .75")
    if not .01 <= args.depth_tolerance <= .5:
        parser.error("--depth-tolerance must be between .01 and .5")
    if args.method != "semantic" and args.model != "b0":
        parser.error("--model b2 requires --method semantic")
    if not 0 <= args.confidence <= 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("Confidence must be 0..1 and max-frames must be positive")
    if args.method == "semantic" and args.show_floor_sample:
        parser.error("--show-floor-sample applies only to --method classical")
    x1, y1, x2, y2 = args.floor_patch
    if not (0 <= args.roi_top < 1 and 0 <= x1 < x2 <= 1 and args.roi_top <= y1 < y2 <= 1):
        parser.error("ROI and floor patch must be ordered fractions in 0..1; patch must be within ROI")
    if not (args.max_width >= 64 and math.isfinite(args.start_seconds) and args.start_seconds >= 0
            and 0 < args.min_area < 1 and 0 <= args.min_edge_support <= 1
            and 0 <= args.canny_low < args.canny_high <= 255
            and 0 <= args.hue_tolerance <= 90
            and 0 <= args.saturation_tolerance <= 255 and 0 <= args.value_tolerance <= 255):
        parser.error("Invalid size, time, area, support, or threshold; see --help")
    return args
