"""Stable JSONL record construction for offline replay."""

from dataclasses import asdict
import sys

import cv2

from depth import depth_summary
from pipeline import FramePlan, PathSettings


def run_record(args, source, settings, path_settings: PathSettings, navigation_settings,
               detector, depth_estimator, fusion_settings, fps: float,
               processed_size: tuple[int, int]) -> dict:
    return dict(type="run", input=str(source), settings=vars(settings),
                method=args.method,
                path_settings=asdict(path_settings),
                navigation_settings=vars(navigation_settings),
                motion_authorized=False,
                semantic=detector.metadata() if args.method == "semantic" else None,
                depth=({**depth_estimator.metadata(),
                        "affects_path": args.depth_mode == "fused"}
                       if depth_estimator else None),
                fusion=vars(fusion_settings) if depth_estimator else None,
                depth_mode=args.depth_mode if depth_estimator else None,
                layout="quad" if depth_estimator else "single_or_debug",
                fps=fps, rotate=args.rotate, start_seconds=args.start_seconds,
                processed_size=list(processed_size), debug=args.debug,
                show_floor_sample=args.show_floor_sample,
                opencv=cv2.__version__, command=sys.argv)


def frame_record(count: int, position_ms: float, result: dict, plan: FramePlan,
                 method: str, depth_result: dict | None = None, display_limits=None) -> dict:
    fusion = plan.fusion
    record = dict(type="frame", frame=count, source_time_ms=position_ms,
                  processing_ms=round(result["processing_ms"], 3),
                  status=result["status"], boxes=result["boxes"],
                  path=dict(points=plan.points, processing_ms=round(plan.processing_ms, 3)))
    record["navigation"] = plan.navigation
    if method == "semantic":
        record["pixel_fractions"] = {key: round(float(mask.mean()), 5)
                                     for key, mask in result["masks"].items()}
    if depth_result is not None:
        record["fusion"] = dict(status=fusion["status"],
                                fit_fraction=round(fusion["fit_fraction"], 4),
                                eligible_fraction=round(float(fusion["mask"].mean()), 5),
                                rejected_floor_fraction=round(float(fusion["rejected"].mean()), 5))
        record["depth"] = dict(**depth_summary(depth_result["values"]),
                               processing_ms=round(depth_result["processing_ms"], 3),
                               status=depth_result["status"], display_limits=display_limits)
    return record
