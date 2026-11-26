import argparse
import socket
import time

import pygame


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


def main():
    parser = argparse.ArgumentParser(
        description="Send USB PS4 left/right stick tank-drive commands to the ESP32."
    )
    parser.add_argument("--host", default="192.168.4.1", help="ESP32 Wi-Fi IP")
    parser.add_argument("--port", type=int, default=4210, help="ESP32 UDP port")
    parser.add_argument("--left-axis", type=int, default=1, help="pygame left-stick Y axis")
    parser.add_argument("--right-axis", type=int, default=4, help="pygame right-stick Y axis")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No USB game controller found")

    controller = pygame.joystick.Joystick(0)
    controller.init()
    destination = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Sending tank-drive commands; press Ctrl+C to stop.")

    try:
        while True:
            pygame.event.pump()
            if not controller.get_init():
                break
            left = axis_to_pwm(controller.get_axis(args.left_axis))
            right = axis_to_pwm(controller.get_axis(args.right_axis))
            # Repeated packets act as a heartbeat for the ESP32 timeout check.
            sock.sendto(f"{left},{right}".encode("ascii"), destination)
            time.sleep(SEND_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        # Send an explicit stop before closing; the receiver also has a timeout.
        for _ in range(3):
            sock.sendto(b"0,0", destination)
            time.sleep(SEND_INTERVAL_SECONDS)
        sock.close()
        controller.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
