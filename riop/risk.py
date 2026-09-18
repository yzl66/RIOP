"""Depth-proxy and dynamics risk for sampled quintic trajectories.

The depth term is a camera-ray proxy, not a 3-D Euclidean ESDF. Unknown or
unobserved rays are penalized. Use an ESDF for validated collision checking.
"""

import torch
import torch.nn.functional as F

from .trajectory import quintic_samples


class RiskEvaluator:
    def __init__(self, config):
        self.config = config

    def __call__(self, depth_m, endstate, obs, previous_direction=None):
        cfg = self.config
        if depth_m.ndim != 4 or depth_m.shape[1] != 1:
            raise ValueError("depth_m must have shape [B,1,H,W]")
        if depth_m.shape[0] != endstate.shape[0] or obs.shape != endstate.shape:
            raise ValueError("batch or observation shape mismatch")
        if depth_m.shape[-2:] != (cfg.image_height, cfg.image_width):
            raise ValueError("depth image size differs from RIOP camera settings")
        positions, velocities, accelerations, jerks = quintic_samples(
            endstate, obs[:, :3], obs[:, 3:6], cfg.segment_time, cfg.samples)

        # YOPO-Simple camera frame: x forward, y left, z up. Pixel u is right.
        x = positions[..., 0]
        x_safe = x.clamp_min(1e-3)
        u = (cfg.image_width - 1) / 2 - cfg.fx * positions[..., 1] / x_safe
        v = (cfg.image_height - 1) / 2 - cfg.fy * positions[..., 2] / x_safe
        grid = torch.stack((2*u/(cfg.image_width-1)-1, 2*v/(cfg.image_height-1)-1), dim=-1)
        grid = grid[:, None, :, :]
        clean_depth = torch.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
        sampled = F.grid_sample(clean_depth, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0, 0, :]
        near_origin = positions.norm(dim=-1) <= 0.05
        known = (x > 0.05) & (x < cfg.max_depth) & (u >= 0) & (u <= cfg.image_width-1) & (v >= 0) & (v <= cfg.image_height-1) & (sampled > 0.05)
        # Positive clearance means the observed surface lies beyond the trajectory point.
        # A missing ray is treated as worse than the most adverse observed ray.
        clearance = torch.where(near_origin, torch.full_like(x, cfg.safe_distance),
                                torch.where(known, sampled - x, torch.full_like(x, -cfg.max_depth)))
        collision = F.relu(cfg.safe_distance - clearance).square().mean(dim=1)
        near = torch.exp(-clearance.clamp_min(0) / cfg.clearance_sigma).mean(dim=1)

        speed = velocities.norm(dim=-1)
        accel = accelerations.norm(dim=-1)
        jerk = jerks.norm(dim=-1)
        dynamic = ((speed / cfg.max_speed).square() +
                   (accel / cfg.max_acceleration).square() +
                   (jerk / cfg.max_jerk).square()).mean(dim=1) / 3
        dynamic = dynamic + (F.relu(speed / cfg.max_speed - 1).square() +
                             F.relu(accel / cfg.max_acceleration - 1).square() +
                             F.relu(jerk / cfg.max_jerk - 1).square()).mean(dim=1)

        if previous_direction is None:
            temporal = torch.zeros_like(collision)
        else:
            current = F.normalize(endstate[:, :3], dim=-1, eps=1e-6)
            previous = F.normalize(previous_direction, dim=-1, eps=1e-6)
            temporal = torch.where(previous_direction.norm(dim=-1) > 1e-6,
                                   1 - (current * previous).sum(dim=-1).clamp(-1, 1),
                                   torch.zeros_like(collision))
        total = (cfg.risk_collision * collision + cfg.risk_clearance * near +
                 cfg.risk_dynamic * dynamic + cfg.risk_temporal * temporal)
        return {"total": total, "collision": collision, "clearance": near,
                "dynamic": dynamic, "temporal": temporal,
                "minimum_ray_clearance": clearance.min(dim=1).values,
                "unknown_fraction": ((~known) & (~near_origin)).float().mean(dim=1)}
