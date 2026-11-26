import unittest

from ps4_drive import PadInput, pad_packet


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
