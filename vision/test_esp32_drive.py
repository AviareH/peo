import unittest
from unittest.mock import patch

from esp32_drive import Esp32DriveSink
from tape_proposal import Drive, Proposal, Turn


class DriveSinkTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.
        self.incoming = []
        socket_patch = patch("esp32_drive.socket.socket")
        self.socket = socket_patch.start().return_value
        self.addCleanup(socket_patch.stop)
        self.socket.recv.side_effect = self.receive
        self.sink = Esp32DriveSink("board", 1000., 2000., lambda: self.now)
        self.addCleanup(self.sink.close)

    def receive(self, size):
        if self.incoming:
            return self.incoming.pop(0)
        raise BlockingIOError

    def publish(self, *, age=.01, angle=.2, distance=.15, stop=False):
        self.now += .04
        actions = (Turn(angle),) if distance is None else (Turn(angle), Drive(distance))
        proposal = Proposal("run:frame", int(self.now * 100), self.now - age, self.now + 1, "valid",
                            status="PROPOSED", actions=actions)
        return self.sink.publish(proposal.stop("tape_lost") if stop else proposal)

    def sent(self):
        return tuple(map(int, self.socket.send.call_args.args[0].decode().split(",")[1:]))

    def reply(self, state, session=100, move=0):
        self.incoming.append(f"CV,{self.sent()[0]},{session},{move},{state}".encode())

    def arm(self):
        self.publish()
        self.reply("MANUAL", session=0)
        self.publish()
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))
        self.reply("IDLE")
        self.publish()
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))
        self.reply("IDLE")

    def test_turn_then_drive_requests_keep_one_id_until_a_post_completion_frame(self):
        self.arm()
        self.publish()
        _, session, move, turn, drive = self.sent()
        self.assertEqual((session, turn, drive), (100, 200, 300))
        self.reply("RUNNING", move=move)
        self.publish(angle=-.1)
        self.assertEqual(self.sent()[1:], (100, move, 200, 300))
        self.reply("DONE", move=move)
        self.publish()
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))
        self.reply("DONE", move=move)
        self.publish(angle=-.1)
        self.assertGreater(self.sent()[2], move)
        self.assertEqual(self.sent()[3:], (-100, 300))

    def test_stale_frame_cancels_and_waits_for_acknowledgement(self):
        self.arm()
        self.publish()
        move = self.sent()[2]
        self.reply("RUNNING", move=move)
        self.assertEqual(self.publish(age=.3).reason, "stale_frame")
        _, session, cancelled, turn, drive = self.sent()
        self.assertEqual((session, turn, drive), (100, 0, 0))
        self.assertGreater(cancelled, move)
        self.reply("RUNNING", move=move)  # The first cancellation did not reach the motor task.
        self.publish()
        self.assertEqual(self.sent()[1:], (100, cancelled, 0, 0))
        self.reply("STOPPED", move=cancelled)
        self.publish()
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))

    def test_unacknowledged_move_expires_and_durations_are_bounded(self):
        self.arm()
        self.assertEqual(self.publish(distance=1.).reason, "motor_duration_out_of_range")
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))
        self.reply("IDLE")
        self.publish()
        move = self.sent()[2]
        self.now += .26
        self.publish()
        self.assertGreater(self.sent()[2], move)
        self.assertEqual(self.sent()[3:], (0, 0))

    def test_switching_to_manual_discards_pending_move(self):
        self.arm()
        self.publish()
        self.reply("MANUAL", session=0)
        self.publish()
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))
        self.assertIsNone(self.sink.status["requested_move"])

    def test_encoder_mode_sends_metric_targets_and_waits_after_fault(self):
        self.sink.close()
        self.sink = Esp32DriveSink("board", clock=lambda: self.now)
        self.addCleanup(self.sink.close)
        self.arm()
        self.publish(angle=-.25, distance=.15)
        self.assertTrue(self.socket.send.call_args.args[0].startswith(b"ENC,"))
        self.assertEqual(self.sent()[3:], (-250, 150))
        self.assertEqual(self.sink.status["requested_move"]["drive_mm"], 150)
        self.reply("FAULT", move=self.sent()[2])
        self.publish()
        self.assertEqual(self.sink.status["state"], "FAULT")
        self.assertIsNone(self.sink.status["requested_move"])
        self.reply("FAULT", move=self.sink.status["move_id"])
        self.publish()
        self.assertEqual(self.sent()[1:], (0, 0, 0, 0))

    def test_encoder_turn_only_is_renewed_and_can_be_cancelled(self):
        self.sink.close()
        self.sink = Esp32DriveSink("board", clock=lambda: self.now)
        self.addCleanup(self.sink.close)
        self.arm()
        self.publish(angle=.25, distance=None)
        self.assertTrue(self.socket.send.call_args.args[0].startswith(b"ENC,"))
        move = self.sent()[2]
        self.assertEqual(self.sent()[3:], (250, 0))
        self.reply("RUNNING", move=move)
        self.publish(angle=.24, distance=None)
        self.assertEqual(self.sent()[1:], (100, move, 250, 0))
        self.reply("RUNNING", move=move)
        self.publish(stop=True)
        self.assertTrue(self.socket.send.call_args.args[0].startswith(b"CV,"))
        self.assertGreater(self.sent()[2], move)
        self.assertEqual(self.sent()[3:], (0, 0))

    def test_estimated_edge_turn_is_sent_as_an_encoder_pivot(self):
        self.sink.close()
        self.sink = Esp32DriveSink("board", clock=lambda: self.now)
        self.addCleanup(self.sink.close)
        self.arm()
        self.now += .04
        estimate = Proposal("run:frame", 1, self.now - .01, self.now + 1, "valid",
                            status="PROPOSED", reason="estimated_edge_turn", actions=(Turn(-.20944),))
        self.assertEqual(self.sink.publish(estimate).reason, "estimated_edge_turn")
        self.assertTrue(self.socket.send.call_args.args[0].startswith(b"ENC,"))
        self.assertEqual(self.sent()[3:], (-209, 0))
