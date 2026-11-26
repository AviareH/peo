# Robot vision

Laptop-side camera processing and path estimation for PEO CV. 99% vibecoded

## Connect to the ESP32

1. Power ESP32
2. Connect the laptop to **PEO-Camera**, password **peocamera123**.
3. Close any browser tab viewing the camera stream.
4. Run from the repository root:

```powershell
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream --preview
```

For semantic detection and depth with cached models:

```powershell
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream --method semantic --model b0 --depth --offline --preview
```

Press **Q** to exit. Results save under `vision/outputs`.

## Follow the corridor between two blue tape strips

Tape mode treats the two blue strips as the left and right course boundaries. The desired route is midway between their **inside edges**. The course diagram gives a nominal 90 cm track width. Camera calibration supplies the pixel-to-ground mapping. Tape mode bases each proposal on the visible local lane, including at bends and beside a neighbouring lane.

The debug view shows the prepared camera image, the annotated corridor, and the cleaned mask. Green highlights the two selected strips. Short cross-lines show how the inside edges are paired, yellow traces their midpoints, and magenta marks the target. The footer shows the robot's turning reference with forward pointing up. Tape mode defaults to dry-run.

Run these PowerShell commands from the repository root to open the file picker in `vision/footage`. Choose a recording to start the preview. Each run gets a fresh output directory.

```powershell
$out = "vision/outputs/tape-$(Get-Date -Format yyyyMMdd-HHmmss-fff)"
.\vision\.venv\Scripts\python.exe vision/run.py --method tape --preview --debug --output-dir $out
```

For the ESP32, connect as described above and run

```powershell
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream --method tape --preview --debug
```

Omit `--preview` for headless processing. Q or Esc ends preview, and Space pauses recorded footage. Live tape preview continues consuming the newest frame. Without a matching calibration, the overlay shows **CALIBRATION REQUIRED**. A reliable pair still has an image target one third of the way along the sampled corridor. All proposals in this state have an empty action list.

### Tune detection on recordings

1. Record the actual tape, floor, shadows and lighting with the intended camera settings. Start with `--tape-hsv-low 95 85 45 --tape-hsv-high 135 255 255`. OpenCV hue spans 0–179 and saturation/value span 0–255. Adjust the hue limits to cover the strip, then the lower saturation/value limits while watching which background pixels enter the mask.
2. Set `--roi-top .35` at the top of the usable floor and `--tape-near-y .80` near the robot. Both are fractions of full prepared image height, measured down from the top. The two grey lines mark these settings. A 3×3 opening removes specks. A second pass retains thin, faint horizontal tape connected to a selected strip's far tip. `--tape-gap-px 2` closes short aligned vertical breaks and enables the scaled horizontal closing at far tips. Use 0 to disable gap closing or an even value up to 8.
3. At a near-field row the detector chooses the closest qualifying strip on either side of the image centre. This assumes a forward-facing camera aligned with the robot. A neighbouring lane outside that pair is ignored. Tune `--min-area .003` (fraction of ROI area), `--tape-min-extent .22` (fraction of ROI height), and `--tape-max-width .20` (maximum near-field strip width as a fraction of image width). Set `--tape-min-lane-width .08` to the smallest usable gap between inside edges, also as a fraction of image width.
4. From the near-field pair, the detector samples across the lane and rotates the sampling direction as the boundaries turn. When both strips form elbows in the same direction, it follows the visible arms around the corner and shows the first target on the approach. A short pair can also start above the near-field line if a strip visibly exits the side of the frame. It still needs both edges over a continuous overlapping segment. With one clearly observed outside elbow that turns across the image centre, the preview shows a turn in place. Joined strips, detected forks and insufficient evidence produce STOP.
5. Inspect the JSONL evidence. Extent is the fraction of ROI height covered by the traced centreline. Continuity is the fraction of sampled crossings with original blue pixels at both ends. Near support is the fraction of near-field rows showing both strips. Tracking requires at least 85% continuity and 15% near support. Their product with extent is a heuristic quality score. Each frame is independent.

### Calibrate the mounted camera

Start with [`camera_calibration.json`](camera_calibration.json). It is prefilled for the ESP32's VGA stream at 640 × 480 with `--rotate 0` and the default `--max-width 960`. If you change those settings, copy the `[width, height]` pairs from `source_size` and `processed_size` in a tape run's frame record, and set `rotate` to match the run.

1. Lock the camera mount. Put four marks on a level floor, around the patch where both tape edges should be tracked. For each list, fill the rows in this order: far left, far right, near right, near left as seen by the robot.
2. In `valid_image_quad_px`, enter each mark's `[x, y]` pixel in the prepared camera frame. The top-left pixel is `[0, 0]`. At the default VGA resolution, the left panel of the saved `--debug` video frame shows the prepared image at its original pixel scale.
3. In `ground_quad_left_forward_m`, enter the corresponding `[left_m, forward_m]` from the point on the floor under the robot's turn centre. Left is positive, right is negative, and forward is positive. Leave `camera_offset_left_forward_m` at `[0, 0]` when using this origin.
4. Run tape mode with `--calibration vision/camera_calibration.json` in dry-run. The runner fits the projection from the four pairs. Check the projected position against a separately measured point before using it to drive.

Existing version 1 files can also supply a fitted matrix in `image_to_ground_m`. **The following JSON is synthetic replay/test geometry only. Hardware requires its own measurements.**

```json
{
  "description": "Synthetic 160 x 120 replay example only",
  "version": 1,
  "source_size": [160, 120],
  "processed_size": [160, 120],
  "rotate": 0,
  "valid_image_quad_px": [[0, 42], [159, 42], [159, 119], [0, 119]],
  "image_to_ground_m": [[-0.005, 0, 0.4], [0, -0.005, 0.595], [0, 0, 1]],
  "camera_offset_left_forward_m": [0, 0.05]
}
```

- Lock the camera mount and use a level floor. Tape mode reads the decoder image with automatic orientation disabled, applies the explicit clockwise `--rotate`, then resizes to `--max-width` (default 960) with even dimensions. Read the actual `[width, height]` pairs from a frame record. The homography uses **prepared** pixel coordinates with origin at the top left. The floor ROI masks rows in this coordinate system. The preparation pipeline uses the full image without cropping.
- Measure at least four non-collinear floor points spanning the intended planning area. Record their corresponding pixels in the prepared camera image and their ground positions in metres. Use the camera's floor projection as ground origin, with axes parallel to the robot's left and forward directions. Fit the 3×3 image-to-ground homography (for example with `cv2.findHomography`). Dividing its first two output coordinates by the third gives `[left_m, forward_m]`. Measure the camera floor origin relative to the robot's turn centre and enter that left/forward offset. The planner adds it once. If the measured ground points already use the turn centre as origin, set the offset to `[0, 0]`.
- Store four measured boundary pixels in perimeter order as `valid_image_quad_px`. Use a convex patch where independent measured check points confirm acceptable error, including near its edges. Validate lens distortion, mount angle, actual field of view and ground planarity. A changed mount, zoom, external crop or different camera can have identical pixel dimensions and needs a new calibration. The loader checks dimensions, rotation, finite values, matrix conditioning, polygon bounds and the projection horizon. Invalid or incompatible configurations keep the preview in image space.

### Read a movement proposal

For each cross-section, both inside edges must lie in the measured patch. The planner projects the two edges into metres and averages their ground positions. Starting at the first supported midpoint, the planner walks up to `--lookahead-m .20` along the ground route. It stops at the first exit, visible endpoint or intersection with the `--max-step-m .30` radius. When there is enough measured approach left for a separate step, the first target stays on that approach. A fresh frame can choose a turning target once the bend is closer. Targets must lie ahead of the turn centre. An unreachable first point, turn over `--max-turn-deg 40`, or drive under `--min-step-m .03` yields STOP.

For target `[left, forward]`, the ordered actions are **TURN** by `atan2(left, forward)` radians (positive left), then **DRIVE** by `hypot(left, forward)` metres. The drive uses the straight line from the robot to the target. Inside `--steering-deadband-deg 2` the proposed turn is zero, accepting a small lateral error. Validate these starting distances and angles against the camera visibility and how far the robot actually moves. A straight drive that crosses either measured tape edge produces STOP. Obstacle and chassis clearance still require separate checks.

For a single outside elbow, the direction comes from the visible outgoing arm inside the calibration patch. The proposal turns in place up to `--max-turn-deg` and waits for a fresh frame after completion.

The MP4 and `*_detections.jsonl` are written together. To inspect the first few proposals from the replay above, run

```powershell
Get-Content (Get-ChildItem $out -Filter '*_detections.jsonl').FullName |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Where-Object type -eq 'frame' | Select-Object -First 5 -ExpandProperty proposal |
  ConvertTo-Json -Depth 6
```

Frame records include the paired inside-edge samples (`left_edge_px` and `right_edge_px`), image-space centreline, image target, evidence, processing time and receipt age. Each proposal carries a run-scoped ID, frame ID, local monotonic receipt/expiry times, calibration and detection status, metric target when available, and ordered actions or a STOP reason. Replay frames use a run-local index and the decoder's media time. Live frames use the receiver sequence and can skip IDs as newer images arrive. HTTP MJPEG provides no verified camera exposure time. The age limit (`--max-frame-age-ms`, default 2000) starts at laptop decode, so camera and network buffering before receipt are unmeasured.

In **DRY RUN / NOT SENT**, the mock sink replaces its single slot on each publication, rejects expired movements and receives a final STOP on input end, failure or shutdown. A replay decoder can report the same read failure for EOF and truncated input. JSONL is a history of proposals.

The live overlay and frame records show the ESP32's latest reported drive mode. The preview polls the stream host at the next TCP port (`/mode`, normally port 81) on a background thread. Missing or stale status shows **UNKNOWN**. The mode reflects the board at status read time. Replay has no live board mode.

### Drive from live tape

1. Flash the updated firmware. Connect to **PEO-Camera** and run `python firmware/host/ps4_drive.py` in its own terminal on the same laptop as vision. Centre both sticks to establish the controller session.
2. Verify the mounted camera calibration in dry-run. Set and calibrate the encoder pins and wheel geometry in the firmware. Use the rotation and resolution from the camera calibration. [AS5600 setup](../firmware/README.md#as5600-wheel-encoders)
3. Run live tape tracking with encoder feedback from the repository root.

```powershell
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream `
  --method tape --calibration vision/camera_calibration.json --preview --actuate-encoders
```

4. Press **Share** to enter autonomous. Press it again for manual, or press **Q** in the preview to end CV control. A lost tape pair, incompatible camera calibration, or stale frame causes a stop. Live actuation also limits frame age to 250 ms from laptop decode. The board stops if vision stalls while the PS4 remains connected. The JSONL `esp32_control` record includes the requested move and the board's state. [Firmware controls and protocol](../firmware/README.md#cv-movement-protocol)

For timed trials, supply the measured duration in milliseconds per radian of turn and milliseconds per metre forward. Measure short trial moves at the firmware's drive setting on the intended floor to refine the scales. The log records requested phase durations. Timing varies with the load, battery and floor.

```powershell
$turnMsPerRad = [double](Read-Host 'Turn milliseconds per radian')
$driveMsPerMetre = [double](Read-Host 'Forward milliseconds per metre')
.\vision\.venv\Scripts\python.exe vision/run.py --stream http://192.168.4.1/stream `
  --method tape --calibration vision/camera_calibration.json --preview --actuate $turnMsPerRad $driveMsPerMetre
```
