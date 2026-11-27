"""Tape replay/live loop. The sink sees terminal STOP before resources close."""

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
from drawing import draw_drive_mode

from live_capture import LiveCapture
from tape import detect_tape
from tape_geometry import Calibration
from tape_proposal import MockProposalSink, Proposal, propose
from tape_view import render_tape
from video_io import PREVIEW_TITLE, open_preview, output_paths, prepare, preview_key, start_capture


def write_record(log, **record):
    log.write(json.dumps(record, allow_nan=False) + "\n")
    log.flush()


def run_tape(args, source, sink=None):
    run_id = uuid4().hex
    live = bool(args.stream)
    actuate = sink is None and (args.actuate is not None or args.actuate_encoders)
    armed = actuate and not args.preview
    cal = None
    calibration = "required"
    detail = None
    reason = "processing_failure"
    log = None
    count = 0
    with ExitStack() as resources:
        if actuate:
            from esp32_drive import Esp32DriveSink
            sink = Esp32DriveSink(urlsplit(source).hostname, *(args.actuate or ()))
            resources.callback(sink.close)
        elif sink is None:
            sink = MockProposalSink()
        try:
            if args.calibration:
                try:
                    cal = Calibration.load(args.calibration)
                    calibration = "loaded"
                except ValueError as exc:
                    calibration, detail = "invalid", str(exc)
                    print(detail)
            name = Path("esp32_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")) if live else source
            video_path, log_path = output_paths(name, args.output_dir)
            log = resources.enter_context(log_path.open("x", encoding="utf-8"))
            write_record(log, type="run", run_id=run_id, input=str(source), method="tape",
                         tape_layout="two_boundaries",
                         input_mode="live" if live else "replay", frame_age_reference="laptop_decode",
                         actuation_enabled=actuate, motor_timing_scales=args.actuate if actuate else None,
                         encoder_actuation=actuate and args.actuate_encoders,
                         motion_authorized=armed, tape_settings=asdict(args.tape_settings),
                         step_limits=asdict(args.step_limits), max_frame_age_ms=args.max_frame_age_ms,
                         calibration_file=str(args.calibration) if args.calibration else None,
                         expected_geometry=dict(source_size=cal.source_size, processed_size=cal.processed_size,
                                                rotate=cal.rotate) if cal else None,
                         calibration_error=detail, rotate=args.rotate, max_width=args.max_width,
                         debug=args.debug, start_seconds=args.start_seconds)
            reason = "source_failure"
            capture = (LiveCapture(source, args.stream_timeout_ms, args.max_frame_age_ms)
                       if live else cv2.VideoCapture(str(source)))
            resources.callback(capture.release)
            fps, raw = start_capture(capture, args.start_seconds, live=live, auto_orientation=False)
            writer = None
            while True:
                received = capture.received_at if live else time.monotonic()
                frame_id = capture.frame_id if live else count
                reason = "processing_failure"
                started = time.perf_counter()
                frame = prepare(raw, args)
                source_size = raw.shape[1::-1]
                size = frame.shape[1::-1]
                if cal is not None:
                    calibration = "valid" if cal.matches(source_size, size, args.rotate) else "incompatible"
                base = Proposal(f"{run_id}:{frame_id}", frame_id, received,
                                received + args.max_frame_age_ms / 1000, calibration)
                route = None
                if base.fresh_at(time.monotonic()):
                    route = detect_tape(frame, args.tape_settings)
                    candidate = propose(base, route, cal, args.step_limits)
                else:
                    candidate = base.stop("stale_frame")
                now = time.monotonic()
                processing_ms = (time.perf_counter() - started) * 1000
                if not candidate.fresh_at(now):
                    candidate = candidate.stop("stale_frame")
                drive_mode = capture.drive_mode if live else None
                overlay = render_tape(frame, route, candidate, cal, args.tape_settings,
                                      processing_ms, (now-received)*1000, args.debug, armed,
                                      actuate and args.preview)
                if writer is None:
                    output_size = overlay.shape[1::-1]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, output_size)
                    resources.callback(writer.release)
                    if not writer.isOpened():
                        raise ValueError("Cannot create tape preview MP4")
                    if args.preview:
                        resources.callback(cv2.destroyAllWindows)
                        open_preview(*output_size)
                elif overlay.shape[1::-1] != output_size:
                    raise ValueError("Image dimensions changed during recording")
                if live and capture.error:
                    reason = "stale_frame" if "stale" in capture.error.lower() else "source_failure"
                    raise ValueError(capture.error)
                if live:
                    draw_drive_mode(overlay, drive_mode)
                was_armed = armed
                if args.preview:
                    key = preview_key(overlay, fps, live, allow_pause=not live)
                    if key in (ord("q"), ord("Q"), 27):
                        reason = "user_exit"
                        break
                    if actuate:
                        if key in (ord("a"), ord("A")):
                            armed = True
                        elif key in (ord("s"), ord("S")):
                            armed = False
                # Publication checks the clock again after rendering and writer/window setup.
                if actuate and not armed:
                    stopped = sink.publish(candidate.stop("actuation_disarmed"))
                    published = candidate.stop("stale_frame") if stopped.reason == "stale_frame" else candidate
                else:
                    published = sink.publish(candidate)
                age_ms = (time.monotonic()-received)*1000
                stop_changed = not published.actions and published != candidate
                if stop_changed or armed != was_armed:
                    overlay = render_tape(frame, route, published, cal, args.tape_settings,
                                          processing_ms, age_ms, args.debug, armed, actuate and args.preview)
                    if live:
                        draw_drive_mode(overlay, drive_mode)
                    if args.preview:
                        cv2.imshow(PREVIEW_TITLE, overlay)
                position = capture.get(cv2.CAP_PROP_POS_MSEC)
                write_record(log, type="frame", frame=frame_id,
                             drive_mode=drive_mode,
                             actuation_armed=armed,
                             esp32_control=sink.status if actuate else None,
                             source_time_ms=position if math.isfinite(position) else None,
                             source_size=source_size, processed_size=size,
                             frame_age_ms=round(age_ms, 3), processing_ms=round(processing_ms, 3),
                             evidence=asdict(route.evidence) if route else None,
                             centreline_px=route.points if route else (),
                             turn_direction=route.turn_direction if route else 0,
                             approach_end=route.approach_end if route else None,
                             turn_arm_px=route.turn_arm if route else (),
                             short_pair=route.short_pair if route else False,
                             left_edge_px=route.left_edge if route else (),
                             right_edge_px=route.right_edge if route else (),
                             image_target_px=route.image_target if route else None,
                             proposal=asdict(published))
                writer.write(overlay)
                count += 1
                if args.max_frames and count >= args.max_frames:
                    reason = "frame_limit"
                    break
                reason = "source_failure"
                ok, raw = capture.read()
                if not ok:
                    if live:
                        if "stale" in capture.error.lower():
                            reason = "stale_frame"
                        raise ValueError(capture.error)
                    reason = "end_of_input"
                    break
        except KeyboardInterrupt:
            reason = "interrupted"
            raise
        except Exception as exc:
            detail = str(exc)
            raise
        finally:
            stopped = sink.publish(Proposal(f"{run_id}:end", None, None, None, calibration,
                                            reason=reason))
            if log is not None:
                write_record(log, type="end", frames=count, reason=reason, detail=detail,
                             proposal=asdict(stopped))
    mode = "LIVE ACTUATION AVAILABLE" if actuate else "DRY RUN / NOT SENT"
    print(f"Processed {count} tape frames. {mode}\nVideo: {video_path}\nLog: {log_path}")
