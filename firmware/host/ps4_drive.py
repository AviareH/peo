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
        description="Send USB PS4 left/right stick tank-drive commands to the ESP32."
    )
    parser.add_argument("--host", default="192.168.4.1", help="ESP32 Wi-Fi IP")
    parser.add_argument("--port", type=int, default=4210, help="ESP32 UDP port")
    parser.add_argument("--left-axis", type=int, default=1, help="pygame left-stick Y axis")
    parser.add_argument("--right-axis", type=int, default=4, help="pygame right-stick Y axis")
    parser.add_argument("--mode-button", type=int, help="Raw joystick button override for Share/Select")
    args = parser.parse_args()

    import pygame
    from pygame._sdl2 import controller as gamepad

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No USB game controller found")

    controller = pygame.joystick.Joystick(0)
    controller.init()
    if args.mode_button is None:
        gamepad.init()
        if not gamepad.is_controller(0):
            parser.error("Controller has no SDL mapping. Pass --mode-button with its Share button index.")
        mapped = gamepad.Controller(0)
        def mode_pressed():
            return mapped.get_button(pygame.CONTROLLER_BUTTON_BACK)
    else:
        if not 0 <= args.mode_button < controller.get_numbuttons():
            parser.error("--mode-button is outside this controller's button range")
        def mode_pressed():
            return controller.get_button(args.mode_button)

    pygame.event.pump()
    pad = PadInput(mode_pressed())
    instance = controller.get_instance_id()
    destination = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Centre both sticks to start. Share/Select switches modes. Ctrl+C stops.")

    try:
        while True:
            events = pygame.event.get()
            if any(e.type == pygame.JOYDEVICEREMOVED and e.instance_id == instance for e in events):
                break
            left = axis_to_pwm(controller.get_axis(args.left_axis))
            right = axis_to_pwm(controller.get_axis(args.right_axis))
            # Repeated packets act as a heartbeat for the ESP32 timeout check.
            previous = pad.autonomous
            state = pad.update(mode_pressed(), left, right)
            if previous != pad.autonomous:
                print("Requested AUTONOMOUS" if pad.autonomous else "Requested MANUAL; centre both sticks.")
            sock.sendto(pad_packet(*state), destination)
            time.sleep(SEND_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        # Send an explicit stop before closing; the receiver also has a timeout.
        for _ in range(3):
            sock.sendto(pad_packet("M", 0, 0), destination)
            time.sleep(SEND_INTERVAL_SECONDS)
        sock.close()
        controller.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
