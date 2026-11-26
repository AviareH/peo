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

1. Connect the two left motors to one DRV8833 output channel and the two right
   motors to its other channel. The motors on each side run together.
2. Check that each motor pair's stall current is within the DRV8833 board's
   current rating. Power the motors from a suitable motor supply, not the ESP32
   board, and connect the motor-supply ground to ESP32 ground.
3. Flash the firmware and connect the computer to the ESP32's `PEO-Camera`
   Wi-Fi access point. The camera stream is available at
   `http://192.168.4.1/stream`.
4. Install the sender dependency with `python -m pip install pygame`, then run
   `python host/ps4_drive.py`. The default destination is `192.168.4.1:4210`.
   Use `--host` and `--port` to override it.

The sender defaults to pygame joystick axes 1 (left Y) and 4 (right Y), which
are common DualShock 4 mappings. If your computer reports different axis
numbers, pass `--left-axis` and `--right-axis` accordingly. The sender sends
zero motor requests when stopped with Ctrl+C; the ESP32 timeout also stops
motors if commands cease.

## Motor driver

The default DRV8833 GPIO assignments are defined in `src/motors.cpp`. Confirm
that these pins match your ESP32-S3 board and wiring before powering the
motors. Test with the wheels raised and an accessible power cutoff: the
controller commands powered motion, and the ramp does not replace a physical
emergency stop.
