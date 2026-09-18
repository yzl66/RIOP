"""Small bounded corrections to the YOPO terminal state."""

import torch
from torch import nn


class RefinementHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(26, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
            nn.Linear(config.hidden_dim, 9),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("max_delta", torch.tensor(config.max_delta, dtype=torch.float32))

    def forward(self, endstate, obs, risk, previous_direction=None):
        batch = endstate.shape[0]
        if previous_direction is None:
            previous_direction = torch.zeros(batch, 3, device=endstate.device, dtype=endstate.dtype)
            has_previous = torch.zeros(batch, 1, device=endstate.device, dtype=endstate.dtype)
        else:
            has_previous = (previous_direction.norm(dim=-1, keepdim=True) > 1e-6).to(endstate.dtype)
        features = torch.cat((endstate, obs,
                              torch.stack([risk[k] for k in ("collision", "clearance", "dynamic", "temporal")], dim=-1),
                              previous_direction, has_previous), dim=-1)
        return torch.tanh(self.net(features)) * self.max_delta.to(endstate.dtype)
