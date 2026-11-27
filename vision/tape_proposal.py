"""Dry-run protocol objects and a single replaceable, expiring preview slot."""

from dataclasses import dataclass, field, replace
import math
from time import monotonic

from tape import TapeRoute
from tape_geometry import Calibration, StepLimits, plan_between_edges, plan_boundary_turn


EDGE_TURN_RAD = math.radians(12)


@dataclass(frozen=True)
class Turn:
    angle_rad: float
    kind: str = field(default="TURN", init=False)


@dataclass(frozen=True)
class Drive:
    distance_m: float
    kind: str = field(default="DRIVE", init=False)


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    frame_id: int | None
    received_monotonic_s: float | None
    expires_monotonic_s: float | None
    calibration: str
    detection: str = "unavailable"
    quality_heuristic: float = 0.
    status: str = "STOP"
    reason: str = "no_frame"
    actions: tuple[Turn | Drive, ...] = ()
    target_left_forward_m: tuple[float, float] | None = None
    target_px: tuple[float, float] | None = None
    motion_authorized: bool = field(default=False, init=False)

    def stop(self, reason):
        return replace(self, status="STOP", reason=reason, actions=(),
                       target_px=None, target_left_forward_m=None)

    def fresh_at(self, now):
        return (self.received_monotonic_s is not None and
                self.received_monotonic_s <= now < self.expires_monotonic_s)


def propose(base: Proposal, route: TapeRoute, cal: Calibration | None, limits: StepLimits):
    result = replace(base, detection=route.status, quality_heuristic=route.evidence.quality)
    if route.status not in ("tracked", "turn_only", "edge_turn"):
        return result.stop(route.status)
    if cal is None or result.calibration != "valid":
        return result.stop("calibration_" + result.calibration)
    if route.status == "turn_only":
        try:
            angle = plan_boundary_turn(route.turn_arm, route.turn_direction, cal, limits)
        except ValueError as exc:
            if not route.can_estimate_turn or str(exc) not in (
                    "corner_outside_calibrated_region", "insufficient_measured_turn_arm"):
                return result.stop(str(exc))
        else:
            return replace(result, status="PROPOSED", reason="dry_run", actions=(Turn(angle),))
    if route.status in ("turn_only", "edge_turn"):
        angle = -route.turn_direction * min(EDGE_TURN_RAD, limits.max_turn_rad)
        return replace(result, status="PROPOSED", reason="estimated_edge_turn", actions=(Turn(angle),))
    try:
        step = plan_between_edges(route.left_edge, route.right_edge, cal, limits,
                                  approach_end=route.approach_end)
    except ValueError as exc:
        return result.stop(str(exc))
    return replace(result, status="PROPOSED", reason="dry_run",
                   target_left_forward_m=step.target_m, target_px=step.target_px,
                   actions=(Turn(step.angle_rad), Drive(step.distance_m)))


class MockProposalSink:
    """Preview only. Publication replaces the slot, including on STOP."""

    def __init__(self, clock=monotonic):
        self.clock = clock
        self._latest = None

    def publish(self, proposal: Proposal) -> Proposal:
        if proposal.frame_id is not None and not proposal.fresh_at(self.clock()):
            proposal = proposal.stop("stale_frame")
        self._latest = proposal
        return proposal

    @property
    def latest(self) -> Proposal | None:
        if self._latest is not None and self._latest.actions:
            return self.publish(self._latest)
        return self._latest
