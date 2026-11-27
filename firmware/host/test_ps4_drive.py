from contextlib import redirect_stdout
from io import StringIO
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ps4_drive import PadInput, main, pad_packet


class PadTests(unittest.TestCase):
    def test_share_edges_and_stick_suppression(self):
        pad = PadInput(True)  # A button held during startup needs a release.
        for pressed, expected in [(True, "M"), (False, "M"), (True, "A"),
                                  (True, "A"), (False, "A"), (True, "M")]:
            mode, left, right = pad.update(pressed, 50, -100)
            self.assertEqual(mode, expected)
            self.assertEqual((left, right), (0, 0) if mode == "A" else (50, -100))
            self.assertRegex(pad_packet(mode, left, right),
                             rb"^PAD,[0-9]{1,16},[AM],-?[0-9]+,-?[0-9]+$")

    def test_dry_run_reports_the_sticks_and_mode_without_opening_a_socket(self):
        pygame = Mock()
        pygame.CONTROLLER_AXIS_LEFTY = 1
        pygame.CONTROLLER_AXIS_RIGHTY = 3
        pygame.joystick.get_count.return_value = 1
        joystick = pygame.joystick.Joystick.return_value
        joystick.get_name.return_value = "Test controller"
        joystick.get_instance_id.return_value = 1
        joystick.get_numaxes.return_value = 5
        joystick.get_numbuttons.return_value = 0
        # On this raw layout, axis 4 is a trigger at rest.
        joystick.get_axis.side_effect = lambda i: (0., 0., 0., 0., -1.)[i]
        pygame.event.get.side_effect = [[], [], [], [], [], KeyboardInterrupt]
        gamepad = Mock()
        gamepad.is_controller.return_value = True
        mapped = gamepad.Controller.return_value
        mapped.get_button.side_effect = [False, False, False, True, False, True]
        mapped.get_axis.side_effect = [0, 0, 0, -32768, -32768, 32767, -32768, 32767, 0, 0]
        output = StringIO()
        with (patch.object(sys, "argv", ["ps4_drive.py", "--dry-run"]),
              patch.dict(sys.modules, {"pygame": pygame, "pygame._sdl2": SimpleNamespace(controller=gamepad)}),
              patch("ps4_drive.socket.socket") as socket_class,
              patch("ps4_drive.time.monotonic", side_effect=[1., 2., 3., 4., 5.]),
              patch("ps4_drive.time.sleep"), redirect_stdout(output)):
            main()
        socket_class.assert_not_called()
        reports = [line for line in output.getvalue().splitlines() if "Stick PWM" in line]
        self.assertEqual([line.split(" |")[0] for line in reports],
                         ["MANUAL", "MANUAL", "AUTONOMOUS", "MANUAL"])
        self.assertIn("Stick inputs L=SDL Left Y, R=SDL Right Y", output.getvalue())
        self.assertIn("4:-1.00", reports[0])
        self.assertRegex(reports[0], r"Stick PWM L=\s*\+0 R=\s*\+0")
        self.assertRegex(reports[1], r"Stick PWM L=\s*\+0 R=\+255")
        self.assertIn("Stick PWM L=+255 R=-255", reports[2])
        self.assertRegex(reports[2], r"Requested PWM L=\s*\+0 R=\s*\+0")
