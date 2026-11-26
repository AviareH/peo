"""Dispatch fresh tape proposals to the ESP32's autonomous drive."""

from dataclasses import dataclass, replace
import math
import socket
from time import monotonic


VISION_TIMEOUT_S = .25
MAX_PHASE_MS = 1500
AUTONOMOUS_STATES = ("IDLE", "RUNNING", "DONE", "STOPPED", "FAULT")


@dataclass(frozen=True)
class Move:
    id: int
    turn: int
    drive: int
    start_before: float = 0.

    @property
    def has_motion(self):
        return bool(self.turn or self.drive)


class Esp32DriveSink:
    def __init__(self, host, turn_ms_per_rad=None, drive_ms_per_m=None, clock=monotonic):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._sock.connect((host, 4210))
            self._sock.setblocking(False)
        except OSError:
            self._sock.close()
            raise
        self._clock = clock
        self._encoder = turn_ms_per_rad is None
        self._turn_scale = 1000. if self._encoder else turn_ms_per_rad
        self._drive_scale = 1000. if self._encoder else drive_ms_per_m
        self._sequence = self._reply = 0
        self._session = self._board_move = 0
        self._state = "UNKNOWN"
        self._pending = None
        self._ready_after = self._last_frame = 0.

    @property
    def status(self):
        requested = None
        if self._pending:
            turn_unit, drive_unit = ("turn_mrad", "drive_mm") if self._encoder else ("turn_ms", "drive_ms")
            requested = {"id": self._pending.id, turn_unit: self._pending.turn, drive_unit: self._pending.drive}
        return {"state": self._state if self._status_fresh(self._clock()) else "UNKNOWN",
                "move_id": self._board_move, "requested_move": requested}

    def _status_fresh(self, now):
        return self._reply > 0 and 0 <= now - self._reply / 1_000_000 < VISION_TIMEOUT_S

    def _receive(self, now):
        for _ in range(32):
            try:
                packet = self._sock.recv(64)
            except OSError:
                break
            try:
                tag, echo, session, move, state = packet.decode("ascii").split(",")
                echo, session, move = int(echo), int(session), int(move)
            except ValueError:
                continue
            if (tag != "CV" or not self._reply < echo <= self._sequence or
                    not 0 <= now - echo / 1_000_000 < VISION_TIMEOUT_S):
                continue
            if state == "MANUAL":
                if session != 0 or move != 0:
                    continue
            elif state not in AUTONOMOUS_STATES or session <= 0 or move < 0:
                continue
            self._reply = echo
            if session != self._session:
                self._pending = None
                self._ready_after = now
            elif (move, state) != (self._board_move, self._state) and state != "RUNNING":
                self._ready_after = now
            if self._pending and (move > self._pending.id or
                                  (move == self._pending.id and state != "RUNNING")):
                self._pending = None
                self._ready_after = now
            self._session, self._board_move, self._state = session, move, state

    def _send(self, now, move=None):
        self._sequence = max(self._sequence + 1, int(now * 1_000_000), move.id if move else 0)
        fields = ((self._session, move.id, move.turn, move.drive) if move else (0, 0, 0, 0))
        tag = "ENC" if move and move.has_motion and self._encoder else "CV"
        packet = ",".join(map(str, (tag, self._sequence, *fields))).encode("ascii")
        try:
            self._sock.send(packet)
        except OSError:
            pass  # The board also expires the independently supervised CV stream.

    def _stop_move(self):
        if self._session and (self._pending or self._state == "RUNNING"):
            if self._pending is None or self._pending.has_motion:
                self._pending = Move(max(self._sequence, self._board_move) + 1, 0, 0)
            return self._pending
        return None

    def publish(self, proposal):
        now = self._clock()
        self._receive(now)
        fresh = proposal.frame_id is not None and proposal.fresh_at(now)
        if fresh:
            fresh = now - proposal.received_monotonic_s < VISION_TIMEOUT_S
        if proposal.frame_id is not None and not fresh:
            proposal = proposal.stop("stale_frame")
        move = None
        if proposal.status != "PROPOSED" or proposal.calibration != "valid" or not fresh:
            move = self._stop_move()
        elif not self._status_fresh(now):
            move = self._stop_move()
        elif self._pending:
            if not self._pending.has_motion:
                move = self._pending
            elif self._board_move != self._pending.id and now >= self._pending.start_before:
                move = self._stop_move()
            elif proposal.received_monotonic_s > self._last_frame:
                move = self._pending
        elif (self._state in ("IDLE", "DONE", "STOPPED") and
              proposal.received_monotonic_s > max(self._last_frame, self._ready_after)):
            turn = proposal.actions[0]
            drive = proposal.actions[1] if len(proposal.actions) == 2 else None
            turn_value = turn.angle_rad * self._turn_scale
            drive_value = drive.distance_m * self._drive_scale if drive else 0.
            max_turn, max_drive = (700, 300) if self._encoder else (MAX_PHASE_MS, MAX_PHASE_MS)
            valid_motion = .5 < drive_value <= max_drive if drive else abs(turn_value) > .5
            if (math.isfinite(turn_value) and abs(turn_value) <= max_turn and
                    math.isfinite(drive_value) and valid_motion):
                start_before = min(proposal.expires_monotonic_s,
                                   proposal.received_monotonic_s + VISION_TIMEOUT_S)
                move = Move(max(self._sequence, self._board_move) + 1,
                            round(turn_value), round(drive_value), start_before)
                self._pending = move
            else:
                reason = "encoder_target_out_of_range" if self._encoder else "motor_duration_out_of_range"
                proposal = proposal.stop(reason)
        if fresh:
            self._last_frame = max(self._last_frame, proposal.received_monotonic_s)
        self._send(now, move)
        return replace(proposal, reason="calibrated") if proposal.status == "PROPOSED" else proposal

    def close(self):
        try:
            now = self._clock()
            self._receive(now)
            move = self._stop_move()
            if move:
                self._send(now, move)
        finally:
            self._sock.close()
