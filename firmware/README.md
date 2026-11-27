# Firmware motor and controller setup

The PS4 controller connects by USB to the computer, not to the ESP32. Run
`host/ps4_drive.py` on the computer to read the left and right stick Y axes and
send their signed PWM requests over UDP. The ESP32 receives commands on UDP
port 4210. Pushing a stick forward drives that side forward; pulling it back
reverses that side.

The UDP receiver and motor PWM task are pinned to core 1; camera streaming is
pinned to core 0. Motor PWM uses 8-bit resolution and ramps by 3 counts every
10 ms (about 0.85 seconds from stopped to full speed). A center dead zone
reduces stick drift, and a 150 ms command timeout requests a stop if the
computer or Wi-Fi link stops sending commands.

## Setup

1. Connect the left motors to one DRV8871 and the right motors to the other.
   Each side's motors connect across that driver's `OUT1` and `OUT2`.
2. Check each motor pair's combined stall current against its breakout's
   current and thermal limits. Connect both drivers' `VM` pins to the motor
   supply and tie both driver grounds, the supply ground and the ESP32 ground
   together. Use a supply within the motors' and breakouts' ratings. The
   DRV8871 chip's recommended `VM` range is 6.5–45 V. [DRV8871 datasheet](https://www.ti.com/lit/ds/symlink/drv8871.pdf)
3. Flash the firmware and connect the computer to the ESP32's `PEO-Camera`
   Wi-Fi access point. The camera stream is available at
   `http://192.168.4.1/stream`.
4. Install the sender dependency with `python -m pip install pygame`, then run
   `python host/ps4_drive.py`. The default destination is `192.168.4.1:4210`.
   Use `--host` and `--port` to override it.

The sender uses SDL's mapped left and right stick Y inputs when available.
Without an SDL mapping, it falls back to raw joystick axes 1 and 4. Use
`--left-axis` or `--right-axis` to override either stick with a raw axis index.
The sender sends zero motor requests when stopped with Ctrl+C. The ESP32
timeout also stops motors if commands cease.

To check the controller from the repo root, run
`python firmware/host/ps4_drive.py --dry-run`. The test runs locally on the
laptop and shows the detected controller, stick PWM, requested mode and raw
axis and button indices. Move one stick at a time, then press and release the
mode button twice to check both mode changes. Use the indices with `--left-axis`,
`--right-axis` or `--mode-button` if the default mapping differs.

## Mode selection and live display

The board boots in **MANUAL**. Centre both sticks to connect. Press
**Share** (Select/Back) to switch to **AUTONOMOUS**. Press it again to return
to manual. The sender uses SDL's mapped Back button. For a controller with
a different mapping, pass `--mode-button N` with its raw pygame button index.
Holding the button makes one change. On return to manual with a displaced
stick, centre both sticks before driving.

The controller sender publishes `PAD,sequence,M,left,right` or
`PAD,sequence,A,0,0` every 20 ms. The sequence is a laptop monotonic
microsecond counter. Retries carry the desired mode. The ESP32 ignores old
sequence numbers and other senders while the pad session is active. Legacy
`left,right` PWM works in manual outside a pad session. A mode change clears
pending drive and zeros PWM at the next motor tick. Autonomous mode currently
waits at zero PWM for a live CV command. Release Share before pressing it again.

After 150 ms without accepted input, the board clears drive and returns to
manual. Reconnect in manual with both sticks centred. If the sender still
requests autonomous after a link loss, press Share to request manual first.

Open `http://192.168.4.1:81/` for the labelled browser stream. Navigating
to `/stream` in a browser also opens this viewer. The viewer and laptop CV
previews read the board's current mode from `http://192.168.4.1:81/mode`
on a separate HTTP worker. A status failure displays **UNKNOWN**. The raw
MJPEG endpoint on port 80 includes an `X-Drive-Mode` part header with each
JPEG. The overlay is drawn by the viewer, keeping camera input clean for CV.
The ESP32 rotates the upside-down camera image by 180° for both clients. Use
`--rotate 0` in CV and measure calibration pixels from the upright stream.
Use one video consumer at a time.

## AS5600 wheel encoders

Set the four GPIOs in [`include/encoder_config.h`](include/encoder_config.h).
Use one SDA/SCL pair per AS5600 because both respond to `0x36`. The left
encoder uses hardware I²C0 and the right encoder uses software I²C. This
firmware build uses hardware I²C1 for the camera. Keep the two encoder buses
separate, share ground, and use 3.3 V I²C levels and pull-ups on each pair.
Leaving all four GPIOs at `-1` disables sampling. [AS5600 datasheet](https://www.flywing-tech.com/blog/wp-content/uploads/2026/03/AS5600_DS000365_5-00.pdf)

The configuration uses the following pins from the supplied ESP32-S3 camera
board pinout. It assumes the board's SD and external JTAG connections are
unused. [ESP32-S3 GPIO reference](https://docs.espressif.com/projects/esp-idf/en/v4.4.8/esp32s3/api-reference/peripherals/gpio.html)

| AS5600 connection | Left encoder | Right encoder |
| --- | --- | --- |
| SDA | GPIO1 | GPIO38 |
| SCL | GPIO14 | GPIO39 |
| Power for a 3.3 V-compatible breakout | 3V3 | 3V3 |
| Ground | GND | GND |

If a breakout lacks I²C pull-ups, add a pull-up to 3V3 on each of its SDA and
SCL lines. Tie any exposed, unstrapped DIR pin to GND.

1. `printCounts` is enabled for setup. Flash and open the serial monitor at
   115200 baud. Over Wi-Fi, join `PEO-Camera` and open
   `http://192.168.4.1:81/encoders` for the same counts and per-side status.
   With motor power off, rotate each wheel forward by hand and refresh the
   page. Its count should increase. Flip its `leftDirection` or
   `rightDirection` sign if it decreases.
2. Set `wheelDiameterMm` to the wheel diameter and `trackWidthMm` to the
   drivetrain track width. The current settings are 82 mm and 210 mm.
   Both sides derive their counts per metre from the wheel circumference and
   4096 counts per revolution. Each encoder must turn once per wheel revolution.
3. Check a straight move over a known distance and a turn through a known angle
   on the intended floor. To measure an effective `trackWidthMm`, convert both
   count changes to millimetres and divide the sum of their magnitudes by the
   absolute turn angle in radians. Set `printCounts = false` to turn off serial
   count logging after setup.

The reader checks magnet status and tracks angle across the 4095-to-0 wrap.
`no_ack` means the AS5600 did not acknowledge an I²C write. Check that side's
power, ground, SDA and SCL. For `bus_held_low`, verify each idle line is pulled
up to 3.3 V. A `magnet_` status means the chip responded. Check the centring
and air gap of that wheel's magnet.

Mount one encoder per drive side and keep shaft travel below half a revolution
between samples. Sampling is nominally every 10 ms. Encoder-controlled motion
requires both samples and the three calibration values. The ESP32 stops each
side when it reaches its count target, turns before driving, and reports
`DONE` after both sides finish. Sensor loss, missing calibration, insufficient
wheel progress, or a phase over three seconds causes `FAULT`. Switch back to
manual before rearming after a fault.

## CV movement protocol

Run the PS4 sender and the [live tape runner](../vision/README.md#drive-from-live-tape)
on the same laptop. The runner uses the camera host and UDP port 4210. It
uses `--actuate-encoders` to send the turn in milliradians and forward distance
in millimetres. For timed trials, `--actuate` converts both values to milliseconds
using the supplied scales.

A positive turn runs the left motors in reverse and the right motors forward.
A negative turn swaps those directions. A turn with zero forward distance
finishes at the end of the turn. When a forward drive follows, the board zeros
PWM for 60 ms between phases. Each phase uses the existing motor ramp toward a target
of 80 out of 255 PWM. Encoder steps are limited to 700 milliradians and 300 mm.
Timed phases are limited to 1500 ms each. In timed mode, `DONE` confirms the
scheduled output ended. Actual movement depends on the floor, battery and load.

The board accepts one move at a time in the current autonomous session. Fresh
frames renew the move under the same ID. After `DONE`, the laptop waits for a
frame received after that reply before planning again. The board zeros PWM on
Share back to manual, a CV stop, 150 ms without PS4 input, or 250 ms without a
renewal of the active move. The two timeouts are independent.

Timed moves use `CV,sequence,session,move,turn_ms,drive_ms`. Encoder moves use
`ENC,sequence,session,move,turn_mrad,drive_mm`. A `CV` status poll sends zero for
the last four fields. With a valid session and new move ID, setting both motion
fields to zero cancels motion. The response is `CV,sequence,session,move,state`. It echoes the
request sequence and reports the current move ID and `MANUAL`, `IDLE`,
`RUNNING`, `DONE`, `STOPPED`, or `FAULT`. A mode change or controller timeout
invalidates the session. Commands must come from the active PS4 sender's IP.

## Motor driver

The two DRV8871s use 3.3 V logic from the ESP32 and 20 kHz PWM. Either input
going high wakes the driver. The assignments are defined in `src/motors.cpp`.
[DRV8871 datasheet](https://www.ti.com/lit/ds/symlink/drv8871.pdf)

| Driver | IN1 | IN2 |
| --- | --- | --- |
| Left DRV8871 | GPIO21 | GPIO47 |
| Right DRV8871 | GPIO41 | GPIO42 |

Confirm the wiring against your breakout labels before powering the motors.
Test with the wheels raised and the propulsion power cutoff within reach.
