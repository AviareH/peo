"""Command-line options and validation for video replay and live capture."""

import argparse
import math
from pathlib import Path

from tape import TapeSettings
from tape_geometry import StepLimits

from path_estimation import (DEFAULT_MARGIN, DEFAULT_ENDPOINT_MARGIN, DEFAULT_MIN_FORWARD,
                             DEFAULT_CORRIDOR_NEAR, DEFAULT_CORRIDOR_FAR, DEFAULT_CORRIDOR_BUFFER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run obstacle detection on a video or live HTTP camera stream.")
    parser.add_argument("video", nargs="?", type=Path, help="MOV/MP4 file; omit for file picker")
    parser.add_argument("--stream", help="HTTP(S) MJPEG URL, e.g. http://192.168.4.1/stream")
    parser.add_argument("--stream-timeout-ms", type=int, default=2000,
                        help="Live connection/read timeout (default: 2000 ms)")
    parser.add_argument("--max-frame-age-ms", type=int, default=2000,
                        help="Maximum local frame age for live input and tape proposals (default: 2000 ms)")
    parser.add_argument("--method", choices=["classical", "semantic", "tape"], default="classical")
    tape = parser.add_argument_group("corridor between two blue tape strips")
    tape.add_argument("--tape-hsv-low", nargs=3, type=int, default=TapeSettings.hsv_low)
    tape.add_argument("--tape-hsv-high", nargs=3, type=int, default=TapeSettings.hsv_high)
    tape.add_argument("--tape-near-y", type=float, default=TapeSettings.near_y)
    tape.add_argument("--tape-min-extent", type=float, default=TapeSettings.min_extent)
    tape.add_argument("--tape-max-width", type=float, default=TapeSettings.max_width)
    tape.add_argument("--tape-min-lane-width", type=float, default=TapeSettings.min_lane_width,
                      help="Minimum visible space between strips, fraction of image width")
    tape.add_argument("--tape-gap-px", type=int, choices=[0, 2, 4, 6, 8], default=TapeSettings.gap_px)
    tape.add_argument("--calibration", type=Path, help="Tape image-to-ground calibration JSON")
    actuation = tape.add_mutually_exclusive_group()
    actuation.add_argument("--actuate", nargs=2, type=float, metavar=("TURN_MS_PER_RAD", "DRIVE_MS_PER_M"),
                           help="Enable timed ESP32 motion for live calibrated tape tracking")
    actuation.add_argument("--actuate-encoders", action="store_true",
                           help="Enable ESP32 motion using configured wheel encoders")
    tape.add_argument("--lookahead-m", type=float, default=StepLimits.lookahead_m)
    tape.add_argument("--max-step-m", type=float,
                      help="Maximum tape step in metres (default: 0.65 in dry run, 0.30 with actuation)")
    tape.add_argument("--min-step-m", type=float, default=StepLimits.min_step_m)
    tape.add_argument("--max-turn-deg", type=float, default=math.degrees(StepLimits.max_turn_rad))
    tape.add_argument("--steering-deadband-deg", type=float, default=math.degrees(StepLimits.deadband_rad))
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
    if args.method == "tape":
        if args.depth or args.show_floor_sample:
            parser.error("--depth and --show-floor-sample are incompatible with --method tape")
        if args.max_step_m is None:
            args.max_step_m = .30 if args.actuate is not None or args.actuate_encoders else StepLimits.max_step_m
        try:
            args.tape_settings = TapeSettings(tuple(args.tape_hsv_low), tuple(args.tape_hsv_high),
                                             args.roi_top, args.tape_near_y, args.min_area,
                                             args.tape_min_extent, args.tape_max_width, args.tape_gap_px,
                                             args.tape_min_lane_width)
            args.step_limits = StepLimits(args.lookahead_m, args.max_step_m, args.min_step_m,
                                         math.radians(args.max_turn_deg),
                                         math.radians(args.steering_deadband_deg))
        except ValueError as exc:
            parser.error(str(exc))
    if args.stream:
        from urllib.parse import urlsplit
        url = urlsplit(args.stream)
        if url.scheme not in ("http", "https") or not url.hostname:
            parser.error("--stream must be an HTTP(S) URL")
        if args.video or args.start_seconds:
            parser.error("--stream cannot be combined with a video file or --start-seconds")
    if args.stream_timeout_ms <= 0 or args.max_frame_age_ms <= 0:
        parser.error("Stream timeout and maximum frame age must be positive")
    if args.actuate is not None or args.actuate_encoders:
        if args.method != "tape" or not args.stream or not args.calibration:
            parser.error("Actuation requires --method tape, --stream and --calibration")
    if args.actuate is not None and not all(math.isfinite(value) and value > 0 for value in args.actuate):
        parser.error("--actuate needs positive, finite timing scales")
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
    if args.method != "tape" and not (0 <= args.roi_top < 1 and 0 <= x1 < x2 <= 1 and args.roi_top <= y1 < y2 <= 1):
        parser.error("ROI and floor patch must be ordered fractions in 0..1; patch must be within ROI")
    if not (args.max_width >= 64 and math.isfinite(args.start_seconds) and args.start_seconds >= 0
            and 0 < args.min_area < 1 and 0 <= args.min_edge_support <= 1
            and 0 <= args.canny_low < args.canny_high <= 255
            and 0 <= args.hue_tolerance <= 90
            and 0 <= args.saturation_tolerance <= 255 and 0 <= args.value_tolerance <= 255):
        parser.error("Invalid size, time, area, support, or threshold; see --help")
    return args
