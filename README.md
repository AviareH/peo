# PEO robot

The ESP32-S3 runs the camera stream and motor control. A Windows laptop runs two processes for autonomous driving. The controller sender handles manual steering and mode switching, and the vision process follows the blue tape corridor.

## Set up a fresh checkout

Install Git and 64-bit Python 3.12 for Windows. Run these commands in PowerShell. If the repository is already cloned, open its root and start at `py`.

```powershell
git clone --branch main https://github.com/AviareH/peo.git
Set-Location peo
py -3.12 -m venv vision\.venv
.\vision\.venv\Scripts\python.exe -m pip install -r vision\requirements.txt pygame platformio
```

All remaining commands run from the repository root. For a different checkout or laptop, install dependencies and build the firmware while you still have internet access.

## Flash and check the board

Connect the ESP32-S3 over USB with propulsion power off. Build and upload the firmware.

```powershell
.\vision\.venv\Scripts\python.exe -m platformio run -d firmware -t upload
```

If auto-detection selects the wrong port, add `--upload-port COM5` and substitute the ESP32's port from Device Manager. The firmware is configured for **82 mm wheels**, a **210 mm drivetrain track width**, and one AS5600 encoder per side. Check encoder status at `http://192.168.4.1:81/encoders` once connected. Both encoders should report valid samples and their counts should increase when the respective wheel is rolled forward. [Firmware and wiring guide](firmware/README.md#as5600-wheel-encoders)

## Test manual and autonomous driving

Connect the 8BitDo to the laptop and join the ESP32's **PEO-Camera** Wi-Fi. Close any browser tab playing the camera stream before starting CV. [Camera connection details](vision/README.md#connect-to-the-esp32)

**Terminal 1 — Controller sender.** First, run the local controller check with the sticks centred.

```powershell
.\vision\.venv\Scripts\python.exe firmware\host\ps4_drive.py --dry-run
```

Move each stick and tap **− (Minus/Select)**. The terminal should show each stick's PWM and switch between `MANUAL` and `AUTONOMOUS`. If Minus does not switch modes, hold it briefly to read its number beside `Buttons`, then add `--mode-button N` to the sender commands, replacing `N` with that index.

Exit the check with **Ctrl+C**, then start the live sender and leave it running.

```powershell
.\vision\.venv\Scripts\python.exe firmware\host\ps4_drive.py
```

**Terminal 2 — CV and encoder control.** Use a camera calibration measured for the mounted camera and current stream geometry. The checked-in file is for the team's setup. Verify it in the disarmed preview and remeasure if the mount or stream changes. [Camera calibration guide](vision/README.md#calibrate-the-mounted-camera)

```powershell
.\vision\.venv\Scripts\python.exe vision\run.py --stream http://192.168.4.1/stream --method tape --calibration vision\camera_calibration.json --preview --debug --actuate-encoders
```

1. Start the first powered test with the wheels raised. Centre both sticks to establish the controller session. In Manual, each stick drives its own side.
2. The CV preview opens **disarmed** and displays its proposed command. Press **A** in the preview to arm CV, then tap **Minus** on the controller to enter Autonomous. Both are required for CV motion.
3. A displayed `FORWARD` with a distance can drive the motors. `FORWARD (image only)` alongside `STOP` means the forward target is outside the measured area or otherwise invalid. A single strip entering from the side can give a small estimated pivot with zero forward travel. The ESP32 reports `DONE` after both wheels reach the encoder targets. CV then waits for a fresh frame before starting the next move.
4. Tap **Minus** again to cancel CV motion and return to Manual. Press **S** in the preview to stop and disarm CV, or **Q** to stop and quit.

For a preview that sends no motor commands, omit `--actuate-encoders`. For an offline replay, replace `--stream http://192.168.4.1/stream` with the path to a recorded video and omit `--actuate-encoders`. CV saves the annotated video and a JSONL log under `vision/outputs`.
