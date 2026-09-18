"""Refinement objective; optional YOPO loss can be supplied for joint tuning."""

import torch
import torch.nn.functional as F


def riop_loss(output, config, yopo_loss=None, margin=0.05):
    final_risk = output["risks"][-1]["total"].mean()
    iteration_terms = []
    for old, new in zip(output["risks"][:-1], output["proposal_risks"]):
        iteration_terms.append(F.relu(new["total"] - old["total"] + margin).mean())
    iteration = torch.stack(iteration_terms).mean() if iteration_terms else final_risk.new_zeros(())
    temporal = output["risks"][-1]["temporal"].mean()
    anchor = (output["trajectory"] - output["initial"]).square().mean()
    total = (config.lambda_risk * final_risk + config.lambda_iteration * iteration +
             config.lambda_temporal * temporal + config.anchor_weight * anchor)
    if yopo_loss is not None:
        total = total + yopo_loss
    return {"total": total, "risk": final_risk, "iteration": iteration,
            "temporal": temporal, "anchor": anchor}
