"""Typed settings and validation for the RIOP planner."""

from dataclasses import dataclass, fields
from math import tan, radians
from pathlib import Path
from typing import Tuple

import yaml


@dataclass(frozen=True)
class RIOPConfig:
    max_iterations: int = 2
    refinement_step: float = 0.2
    hidden_dim: int = 128
    max_delta: Tuple[float, ...] = (2., 2., 1., 2., 2., 1., 3., 3., 2.)
    min_improvement: float = 1e-4
    anchor_weight: float = 0.02
    risk_low: float = 0.2
    risk_high: float = 0.6
    risk_collision: float = 1.0
    risk_clearance: float = 0.5
    risk_dynamic: float = 0.3
    risk_temporal: float = 0.2
    safe_distance: float = 1.2
    clearance_sigma: float = 1.0
    max_speed: float = 6.0
    max_acceleration: float = 6.0
    max_jerk: float = 20.0
    samples: int = 20
    segment_time: float = 1.67
    max_depth: float = 20.0
    image_width: int = 160
    image_height: int = 96
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    lambda_risk: float = 1.0
    lambda_iteration: float = 0.5
    lambda_temporal: float = 0.2
    freeze_yopo: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 30

    def __post_init__(self):
        if len(self.max_delta) != 9 or any(x <= 0 for x in self.max_delta):
            raise ValueError("max_delta must contain nine positive values")
        if self.max_iterations < 0 or self.samples < 2 or self.segment_time <= 0:
            raise ValueError("invalid iteration, sample, or segment time setting")
        if not 0 <= self.risk_low < self.risk_high:
            raise ValueError("require 0 <= risk_low < risk_high")
        if self.refinement_step <= 0 or self.safe_distance <= 0 or self.clearance_sigma <= 0:
            raise ValueError("refinement and clearance settings must be positive")
        if min(self.max_speed, self.max_acceleration, self.max_jerk, self.max_depth) <= 0:
            raise ValueError("speed, acceleration, jerk and depth limits must be positive")
        if min(self.image_height, self.image_width) < 2:
            raise ValueError("image dimensions must be at least 2")
        if any(getattr(self, name) < 0 for name in (
                "risk_collision", "risk_clearance", "risk_dynamic", "risk_temporal",
                "anchor_weight", "min_improvement", "lambda_risk", "lambda_iteration", "lambda_temporal")):
            raise ValueError("risk and loss weights must be nonnegative")
        if not 0 < self.horizontal_fov_deg < 180 or not 0 < self.vertical_fov_deg < 180:
            raise ValueError("camera FOV must be between 0 and 180 degrees")

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError("Unknown RIOP settings: " + ", ".join(sorted(unknown)))
        if "max_delta" in data:
            data["max_delta"] = tuple(data["max_delta"])
        return cls(**data)

    @property
    def fx(self):
        return (self.image_width - 1) / (2 * tan(radians(self.horizontal_fov_deg) / 2))

    @property
    def fy(self):
        return (self.image_height - 1) / (2 * tan(radians(self.vertical_fov_deg) / 2))
