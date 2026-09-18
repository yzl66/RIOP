"""Per-frame adaptive refinement budget."""

import torch


class IterationController:
    def __init__(self, config):
        self.config = config

    def budgets(self, total_risk):
        one = torch.ones_like(total_risk, dtype=torch.long)
        return torch.where(total_risk < self.config.risk_low, 0,
                           torch.where(total_risk < self.config.risk_high, one,
                                       torch.full_like(one, self.config.max_iterations)))
