"""Adapt independent perception outputs to floor planning and mock decisions."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from navigation import NavigationSettings, plan_navigation
from path_estimation import (DEFAULT_MARGIN, DEFAULT_ENDPOINT_MARGIN, DEFAULT_MIN_FORWARD,
                             DEFAULT_CORRIDOR_NEAR, DEFAULT_CORRIDOR_FAR,
                             DEFAULT_CORRIDOR_BUFFER, estimate_path)


@dataclass(frozen=True)
class PathSettings:
    """Planner clearance settings; these do not size the displayed floor envelope."""

    margin_fraction: float = DEFAULT_MARGIN
    endpoint_margin_fraction: float = DEFAULT_ENDPOINT_MARGIN
    minimum_forward_fraction: float = DEFAULT_MIN_FORWARD
    corridor_near: float = DEFAULT_CORRIDOR_NEAR
    corridor_far: float = DEFAULT_CORRIDOR_FAR
    corridor_buffer: float = DEFAULT_CORRIDOR_BUFFER

    @classmethod
    def from_args(cls, args) -> "PathSettings":
        return cls(args.path_margin, args.path_end_margin, args.path_min_forward,
                   args.corridor_near, args.corridor_far, args.corridor_buffer)


@dataclass
class PerceptionMasks:
    floor: np.ndarray
    obstacles: np.ndarray
    unknown: np.ndarray | None
    usable: bool

    @classmethod
    def from_result(cls, result: dict, method: str) -> "PerceptionMasks":
        if method == "semantic":
            masks = result["masks"]
            return cls(masks["floor"], masks["wall"] | masks["other"], masks["unknown"],
                       result["status"] != "UNUSABLE FRAME")
        return cls(result["floor_mask"], result["mask"] != 0, None,
                   result["status"] != "UNUSABLE FRAME")


@dataclass
class FramePlan:
    points: list[tuple[int, int]]
    navigation: dict
    floor_mask: np.ndarray
    fusion: dict | None
    processing_ms: float


def plan_frame(masks: PerceptionMasks, settings: PathSettings,
               navigation_settings: NavigationSettings, depth_result: dict | None = None,
               depth_mode: str = "fused", confidence: np.ndarray | None = None,
               fusion_settings=None) -> FramePlan:
    """Plan from segmentation or fused floor, leaving both model outputs unchanged."""
    started = perf_counter()
    floor, obstacles = masks.floor, masks.obstacles
    fusion, cost = None, None
    if depth_result is not None:
        if depth_mode == "fused":
            from fusion import fuse_floor_depth
            fusion = fuse_floor_depth(floor, depth_result["values"], confidence, fusion_settings)
        else:
            fusion = dict(mask=floor != 0, cost=np.zeros(floor.shape, np.float32),
                          rejected=np.zeros(floor.shape, bool),
                          status="SEGMENTATION-ONLY PATH", fit_fraction=0.)
        floor, cost = fusion["mask"], fusion["cost"]
        if fusion["status"] == "FUSED FLOOR + DEPTH":
            obstacles = obstacles | fusion["rejected"]
    options = asdict(settings)
    points = estimate_path(floor, cost_map=cost, **options)
    navigation = plan_navigation(floor, points, obstacles, masks.unknown, cost,
                                 options, masks.usable, navigation_settings)
    return FramePlan(points, navigation, floor, fusion, (perf_counter() - started) * 1000)
