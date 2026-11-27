import argparse
import socket
import time

SEND_INTERVAL_SECONDS = 0.02
DEADZONE = 0.08
PWM_MAX = 255


def axis_to_pwm(value):
    # pygame reports pushed-forward Y as negative; convert it to forward PWM.
    value = -value
    if abs(value) <= DEADZONE:
        return 0
    scaled = (abs(value) - DEADZONE) / (1.0 - DEADZONE)
    return round((1 if value > 0 else -1) * scaled * PWM_MAX)


class PadInput:
    def __init__(self, pressed):
        self.pressed = pressed
        self.autonomous = False

    def update(self, pressed, left, right):
        if pressed and not self.pressed:
            self.autonomous = not self.autonomous
        self.pressed = pressed
        if self.autonomous:
            return "A", 0, 0
        return "M", left, right


def pad_packet(mode, left, right):
    sequence = time.monotonic_ns() // 1000
    return f"PAD,{sequence},{mode},{left},{right}".encode("ascii")


def main():
    parser = argparse.ArgumentParser(
        description="Send gamepad left/right stick tank-drive commands to the ESP32."
    )
    parser.add_argument("--host", default="192.168.4.1", help="ESP32 Wi-Fi IP")
    parser.add_argument("--port", type=int, default=4210, help="ESP32 UDP port")
    parser.add_argument("--left-axis", type=int, help="Raw joystick axis override for left-stick Y")
    parser.add_argument("--right-axis", type=int, help="Raw joystick axis override for right-stick Y")
    parser.add_argument("--mode-button", type=int, help="Raw joystick button override for Share/Select")
    parser.add_argument("--dry-run", action="store_true", help="Test sticks and mode locally")
    args = parser.parse_args()

    import pygame
    from pygame._sdl2 import controller as gamepad

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No USB game controller found")

    controller = pygame.joystick.Joystick(0)
    controller.init()
    axes = range(controller.get_numaxes())
    buttons = range(controller.get_numbuttons())
    gamepad.init()
    mapped = gamepad.Controller(0) if gamepad.is_controller(0) else None
    left_axis = args.left_axis
    right_axis = args.right_axis
    if mapped is None:
        left_axis = 1 if left_axis is None else left_axis
        right_axis = 4 if right_axis is None else right_axis
    if any(index is not None and index not in axes for index in (left_axis, right_axis)):
        message = f"Check --left-axis and --right-axis. This controller has {len(axes)} axes."
        if not args.dry_run:
            parser.error(message)
        print(message)

    if mapped is None and args.mode_button is None:
        message = "Controller has no SDL mapping. Pass --mode-button with its raw button index."
        if not args.dry_run:
            parser.error(message)
        print(message)
    if args.mode_button is not None and args.mode_button not in buttons:
        parser.error("--mode-button is outside this controller's button range")

    def mode_pressed():
        if args.mode_button is not None:
            return controller.get_button(args.mode_button)
        return mapped is not None and mapped.get_button(pygame.CONTROLLER_BUTTON_BACK)

    def stick_pwm(raw_index, mapped_axis):
        if raw_index is not None:
            return axis_to_pwm(controller.get_axis(raw_index)) if raw_index in axes else 0
        value = mapped.get_axis(mapped_axis)
        return axis_to_pwm(value / (32768 if value < 0 else 32767))

    pygame.event.pump()
    pad = PadInput(mode_pressed())
    instance = controller.get_instance_id()
    destination = (args.host, args.port)
    sock = None if args.dry_run else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if args.dry_run:
        print(f"Dry-run controller test for {controller.get_name()}. Ctrl+C exits.")
        left_source = "SDL Left Y" if left_axis is None else f"raw axis {left_axis}"
        right_source = "SDL Right Y" if right_axis is None else f"raw axis {right_axis}"
        print(f"Stick inputs L={left_source}, R={right_source}. Press Share/Select or Minus to test mode.")
    else:
        print("Centre both sticks to start. Share/Select switches modes. Ctrl+C stops.")
    last_report = ""
    next_report = 0.

    try:
        while True:
            events = pygame.event.get()
            if any(e.type == pygame.JOYDEVICEREMOVED and e.instance_id == instance for e in events):
                break
            left = stick_pwm(left_axis, pygame.CONTROLLER_AXIS_LEFTY)
            right = stick_pwm(right_axis, pygame.CONTROLLER_AXIS_RIGHTY)
            # Repeated packets act as a heartbeat for the ESP32 timeout check.
            previous = pad.autonomous
            state = pad.update(mode_pressed(), left, right)
            switched = previous != pad.autonomous
            if sock is None:
                now = time.monotonic()
                if switched or now >= next_report:
                    label = "AUTONOMOUS" if pad.autonomous else "MANUAL"
                    raw_axes = " ".join(f"{i}:{controller.get_axis(i):+.2f}" for i in axes)
                    pressed = ",".join(str(i) for i in buttons if controller.get_button(i)) or "-"
                    report = (f"{label} | Stick PWM L={left:+4} R={right:+4} | "
                              f"Requested PWM L={state[1]:+4} R={state[2]:+4} | "
                              f"Axes {raw_axes} | Buttons {pressed}")
                    if report != last_report:
                        print(report, flush=True)
                        last_report = report
                    next_report = now + .1
            else:
                if switched:
                    print("Requested AUTONOMOUS" if pad.autonomous else "Requested MANUAL; centre both sticks.")
                sock.sendto(pad_packet(*state), destination)
            time.sleep(SEND_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        # Send an explicit stop before closing; the receiver also has a timeout.
        if sock is not None:
            for _ in range(3):
                sock.sendto(pad_packet("M", 0, 0), destination)
                time.sleep(SEND_INTERVAL_SECONDS)
            sock.close()
        controller.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
