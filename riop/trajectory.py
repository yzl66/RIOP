"""Differentiable quintic trajectory from start P/V/A to YOPO end P/V/A."""

import torch


def quintic_samples(endstate, velocity0, acceleration0, duration, samples):
    """Return position, velocity, acceleration and jerk in body/camera frame.

    endstate is [B,9] ordered [px,py,pz,vx,vy,vz,ax,ay,az].
    Start position is zero; velocity0 and acceleration0 are [B,3].
    """
    if endstate.ndim != 2 or endstate.shape[1] != 9:
        raise ValueError("endstate must have shape [B,9]")
    if velocity0.shape != endstate[:, :3].shape or acceleration0.shape != velocity0.shape:
        raise ValueError("start velocity and acceleration must have shape [B,3]")
    if duration <= 0 or samples < 2:
        raise ValueError("duration must be positive and samples >= 2")
    T = duration
    c0 = torch.zeros_like(velocity0)
    c1 = velocity0
    c2 = acceleration0 / 2
    p1, v1, a1 = endstate[:, :3], endstate[:, 3:6], endstate[:, 6:9]
    dp = p1 - c1 * T - c2 * T * T
    dv = v1 - c1 - 2 * c2 * T
    da = a1 - 2 * c2
    c3 = 10 * dp / T**3 - 4 * dv / T**2 + da / (2 * T)
    c4 = -15 * dp / T**4 + 7 * dv / T**3 - da / T**2
    c5 = 6 * dp / T**5 - 3 * dv / T**4 + da / (2 * T**3)
    t = torch.linspace(T / samples, T, samples, device=endstate.device, dtype=endstate.dtype)[None, :, None]
    c1, c2, c3, c4, c5 = (c[:, None, :] for c in (c1, c2, c3, c4, c5))
    position = c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    velocity = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    acceleration = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    jerk = 6*c3 + 24*c4*t + 60*c5*t**2
    return position, velocity, acceleration, jerk
