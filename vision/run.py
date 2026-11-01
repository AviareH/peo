"""Run offline obstacle detection on a local video; optional file picker/preview."""

import json
import sys

import cv2

from detector import ObstacleDetector, Settings
from cli import parse_args
from reporting import run_record, frame_record
from video_io import (pick_video, prepare, start_capture, output_paths,
                      open_preview, preview_stopped)
from navigation import NavigationSettings
from pipeline import PathSettings, PerceptionMasks, plan_frame
from playback import four_panel, render_segmentation, render_single


def run(args) -> None:
    source = (args.video or pick_video()).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Video does not exist: {source}")
    settings = Settings(**{key: getattr(args, key) for key in Settings.__dataclass_fields__})
    path_settings = PathSettings.from_args(args)
    navigation_settings = NavigationSettings()
    depth_estimator = None
    fusion_settings = None
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        fps, frame = start_capture(capture, args.start_seconds)
        video_path, log_path = output_paths(source, args.output_dir)
        if args.method == "semantic":
            from semantic import SemanticDetector
            detector = SemanticDetector(args.confidence, args.device, args.offline, model_name=args.model)
        else:
            detector = ObstacleDetector(settings)
        if args.depth:
            from depth import DepthEstimator, DepthView
            from fusion import FusionSettings
            depth_estimator = DepthEstimator(
                device=detector.device if args.method == "semantic" else args.device, offline=args.offline)
            depth_view = DepthView()
            fusion_settings = FusionSettings(depth_tolerance=args.depth_tolerance)
        frame = prepare(frame, args)
        height, width = frame.shape[:2]
        output_width = width * (2 if args.depth else (3 if args.debug else 1))
        output_height = height * (2 if args.depth else 1)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (output_width, output_height))
        if not writer.isOpened():
            raise ValueError("Cannot create MP4 output. Check your output folder/codec support.")
        if args.preview:
            open_preview(output_width, output_height)
        count, elapsed_ms, depth_elapsed_ms, path_elapsed_ms = 0, 0.0, 0.0, 0.0
        stopped = False
        with log_path.open("x", encoding="utf-8") as log:
            log.write(json.dumps(run_record(args, source, settings, path_settings,
                                            navigation_settings, detector, depth_estimator,
                                            fusion_settings, fps, (width, height))) + "\n")
            while True:
                result = detector.process(frame)
                depth_result = depth_estimator.process(frame) if depth_estimator else None
                segmentation, debug_panels = render_segmentation(
                    frame, result, args.method, settings, args.show_floor_sample,
                    debug=args.debug and not args.depth)
                masks = PerceptionMasks.from_result(result, args.method)
                plan = plan_frame(masks, path_settings, navigation_settings, depth_result,
                                  args.depth_mode, result.get("confidence"), fusion_settings)
                if depth_result is not None:
                    depth_panel = depth_view.render(depth_result["values"])
                    overlay = four_panel(frame, segmentation, depth_panel, plan.fusion, plan.points,
                                         (result["processing_ms"], depth_result["processing_ms"], plan.processing_ms),
                                         navigation=plan.navigation)
                else:
                    overlay = render_single(segmentation, plan, debug_panels, args.debug)
                path_elapsed_ms += plan.processing_ms
                writer.write(overlay)
                position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                record = frame_record(count, position_ms, result, plan, args.method,
                                      depth_result, depth_view.limits if depth_result is not None else None)
                if depth_result is not None:
                    depth_elapsed_ms += depth_result["processing_ms"]
                log.write(json.dumps(record) + "\n")
                elapsed_ms += result["processing_ms"]
                count += 1
                if count % 30 == 0:
                    print(f"Processed {count} frames...", flush=True)
                if args.max_frames is not None and count >= args.max_frames:
                    break
                if args.preview and preview_stopped(overlay, fps):
                    stopped = True
                    break
                ok, next_frame = capture.read()
                if not ok:
                    break
                frame = prepare(next_frame, args)
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video dimensions changed during decoding.")
            log.write(json.dumps(dict(type="end", frames=count, stopped_by_user=stopped,
                                      frame_limit_reached=args.max_frames is not None and count >= args.max_frames,
                                      note="Read ended; decoder may not distinguish EOF from corrupt/truncated input.")) + "\n")
        print(f"Processed {count} frames; mean detection time {elapsed_ms / count:.1f} ms/frame.")
        if depth_estimator:
            print(f"Mean depth time {depth_elapsed_ms / count:.1f} ms/frame; "
                  f"combined inference {(elapsed_ms + depth_elapsed_ms) / count:.1f} ms/frame.")
        print(f"Mean path/fusion time {path_elapsed_ms / count:.1f} ms/frame.")
        print(f"Video: {video_path}\nDetections: {log_path}")
        print("Offline candidates only; no motion commands. Output has constant FPS and no audio.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
