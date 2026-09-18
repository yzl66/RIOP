"""Store the last endpoint direction in world coordinates between ROS frames."""

import torch
import torch.nn.functional as F


class TrajectoryMemory:
    def __init__(self):
        self._direction_world = None

    def reset(self):
        self._direction_world = None

    def get(self, rotation_world_from_body):
        if self._direction_world is None:
            return None
        direction = torch.matmul(rotation_world_from_body.transpose(-1, -2),
                                 self._direction_world[..., None]).squeeze(-1)
        return F.normalize(direction, dim=-1)

    def update(self, endpoint_body, rotation_world_from_body):
        direction = torch.matmul(rotation_world_from_body, endpoint_body[..., :3, None]).squeeze(-1)
        self._direction_world = F.normalize(direction.detach(), dim=-1)
