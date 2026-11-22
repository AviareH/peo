# Vsion Prototype

This is a vision prototype I'm playing around with for path estimation and object detection. 99% vibecoded

## Setup

Use Python 3.12 and run from the repository root:

```powershell
py -3.12 -m venv vision/.venv
vision/.venv/Scripts/python.exe -m pip install -r vision/requirements.txt
```

On macOS/Linux, create the environment with `python3.12 -m venv vision/.venv` and use `vision/.venv/bin/python` in place of the Windows interpreter path.

## Run

```powershell
vision/.venv/Scripts/python.exe vision/run.py "path/to/video.mp4" --method semantic --model b2 --depth --preview --output-dir vision/outputs/run1
```

Omit the video path to open the file picker (requires Tkinter). **Space** pauses; **Q / Esc** stops.

- `--model b0`: smaller semantic model (default).
- `--method classical`: OpenCV baseline; omit `--model` and start with clear floor in the lower-centre sample patch.
- `--depth`: four panels: raw/path, segmentation, relative depth, and fused mask/cost.
- `--depth-mode compare`: keep segmentation-only routing in the four-panel view.
- `--offline`: use cached models from `vision/models/`; first use needs a download.
- `--device auto`: CUDA, then MPS, then CPU. Use `cpu`, `cuda`, or `mps` to choose explicitly.
- `--max-frames 30`: short replay. `--help` lists all options and thresholds.

## Live ESP32 camera

Join `PEO-Camera` on the laptop and close browser tabs viewing its stream. From the repository root:

```powershell
vision/.venv/Scripts/python.exe vision/run.py --stream http://192.168.4.1/stream --preview
```

To use cached semantic and depth models with the same feed:

```powershell
vision/.venv/Scripts/python.exe vision/run.py --stream http://192.168.4.1/stream --method semantic --model b0 --depth --offline --preview
```

Live capture continuously drains the decoder on a background thread and keeps one newest frame.
The existing detection, path planning, and overlays process that frame. Q/Esc exits; space pauses the
preview while capture continues. Resuming processes the latest frame. Each live run saves uniquely
named annotated MP4 and JSONL files. MP4 playback uses a nominal 30 FPS, not measured capture timing.

`--stream-timeout-ms` sets connection/read waits (default 2000). `--max-frame-age-ms` limits frame
age before and after processing (default 2000). Timeouts, disconnections, and stale results end the
run; restart the command after reconnecting. Ages measure time since laptop decode, not sensor
exposure or Wi-Fi transit. Slow inference can exceed the age limit; try classical mode or b0 first.
Download any missing model weights while connected to the internet before joining the ESP network.
All decisions remain dry-run; this program does not send commands to the motor queue.

## Saved output

Each run saves `<name>_annotated.mp4` (no audio) and `<name>_detections.jsonl` (settings, timestamps, timings, paths, and decisions). The default folder is `vision/outputs/`. Existing results are never overwritten; use a new output directory for each run.

Magenta marks the origin-connected route; cyan marks hypothetical repositioning routes with no verified connection from the car. Dotted boundaries and faint fill show a broad floor envelope. Routes reach the furthest feasible forward grid endpoint while retaining swept-footprint clearance and endpoint stand-off. Depth and clearance are uncalibrated image-space estimates; decisions are mock suggestions only.

## Tests

```powershell
vision/.venv/Scripts/python.exe -m unittest discover -s vision -p "test_*.py" -v
```

Tests use synthetic inputs and mocked model loading; no download is required.
