"""YOPO candidate selection followed by accepted risk-reducing corrections."""

import torch
from torch import nn

from .controller import IterationController
from .refinement import RefinementHead
from .risk import RiskEvaluator


class RIOPPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.refinement_head = RefinementHead(config)
        self.risk_evaluator = RiskEvaluator(config)
        self.controller = IterationController(config)

    def forward(self, depth_m, obs, candidates, scores, previous_direction=None, training=False):
        """Run RIOP on decoded YOPO end states.

        candidates [B,M,9], scores [B,M], obs [B,9] in camera/body frame.
        During training, all proposals are retained for differentiable losses;
        during inference, only improvements in risk plus anchor cost are accepted.
        """
        if candidates.ndim != 3 or candidates.shape[-1] != 9 or scores.shape != candidates.shape[:2]:
            raise ValueError("candidates must be [B,M,9] and scores [B,M]")
        batch = candidates.shape[0]
        chosen = scores.argmin(dim=1)
        initial = candidates[torch.arange(batch, device=candidates.device), chosen]
        current = initial
        initial_risk = self.risk_evaluator(depth_m, current, obs, previous_direction)
        current_risk = initial_risk
        budget = self.controller.budgets(initial_risk["total"])
        history = [current]
        risks = [current_risk]
        proposals = []
        proposal_risks = []
        accepted = []
        for step in range(self.config.max_iterations):
            eligible = budget > step
            if not training and not bool(eligible.any()):
                break
            delta = self.refinement_head(current, obs, current_risk, previous_direction)
            proposal = current + self.config.refinement_step * delta
            candidate_risk = self.risk_evaluator(depth_m, proposal, obs, previous_direction)
            old_anchor = (current - initial).square().mean(dim=1) * self.config.anchor_weight
            new_anchor = (proposal - initial).square().mean(dim=1) * self.config.anchor_weight
            improvement = candidate_risk["total"] + new_anchor + self.config.min_improvement < current_risk["total"] + old_anchor
            use = eligible if training else eligible & improvement
            current = torch.where(use[:, None], proposal, current)
            current_risk = self.risk_evaluator(depth_m, current, obs, previous_direction)
            history.append(current)
            risks.append(current_risk)
            proposals.append(proposal)
            proposal_risks.append(candidate_risk)
            accepted.append(use)
            if not training and not bool(use.any()):
                break
        return {"trajectory": current, "initial": initial, "history": history,
                "risks": risks, "proposals": proposals, "proposal_risks": proposal_risks,
                "accepted": accepted, "budget": budget, "chosen": chosen,
                "score": scores.gather(1, chosen[:, None]).squeeze(1)}
