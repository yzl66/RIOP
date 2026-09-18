"""Translate YOPO-Simple's grid outputs to RIOP candidate terminal states."""

import torch


def decode_yopo_output(yopo_network, depth_normalized, observation_body):
    """Call upstream YopoNetwork.inference and flatten primitive grid in score order."""
    if observation_body.ndim != 2 or observation_body.shape[1] != 9:
        raise ValueError("observation_body must be [B,9]")
    endstates, scores = yopo_network.inference(depth_normalized, observation_body.clone())
    if endstates.ndim != 4 or endstates.shape[1] != 9 or scores.shape != endstates.shape[:1] + endstates.shape[2:]:
        raise ValueError("Unexpected YOPO-Simple output shape")
    candidates = endstates.flatten(start_dim=2).transpose(1, 2).contiguous()
    return candidates, scores.flatten(start_dim=1)
