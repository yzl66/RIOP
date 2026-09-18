"""Train RIOP refinement from prepared YOPO-Simple observations.

NPZ keys: depth_m [N,1,H,W], depth_yopo [N,1,H,W], obs [N,9].
Optional previous_direction [N,3] is the prior endpoint direction in the
current camera frame; use zero rows for the first frame of each sequence.
See README for frame and depth conventions. This script requires YOPO-Simple.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from riop import RIOPConfig, RIOPPlanner
from riop.loss import riop_loss
from riop.yopo_adapter import decode_yopo_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yopo-root", type=Path, required=True, help="directory containing YOPO-Simple's policy/ and config/")
    parser.add_argument("--yopo-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/riop.yaml"))
    parser.add_argument("--output", type=Path, default=Path("saved/RIOP_1/refinement.pth"))
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    cfg = RIOPConfig.from_yaml(args.config)
    sys.path.insert(0, str(args.yopo_root.resolve()))
    from policy.yopo_network import YopoNetwork  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yopo = YopoNetwork().to(device)
    yopo.load_state_dict(torch.load(args.yopo_checkpoint, map_location=device, weights_only=True))
    if args.stage == 1:
        yopo.eval()
        for parameter in yopo.parameters():
            parameter.requires_grad_(False)
    elif args.stage == 2:
        for parameter in yopo.image_backbone.parameters():
            parameter.requires_grad_(False)
        yopo.train()
        yopo.image_backbone.eval()
    else:
        yopo.train()

    planner = RIOPPlanner(cfg).to(device)
    arrays = np.load(args.dataset)
    depth_m = torch.from_numpy(arrays["depth_m"].astype(np.float32))
    depth_yopo = torch.from_numpy(arrays["depth_yopo"].astype(np.float32))
    obs = torch.from_numpy(arrays["obs"].astype(np.float32))
    has_temporal = "previous_direction" in arrays.files
    previous = torch.from_numpy(arrays["previous_direction"].astype(np.float32)) if has_temporal else torch.zeros(len(obs), 3)
    if depth_m.shape != depth_yopo.shape or depth_m.shape[1:] != (1, cfg.image_height, cfg.image_width):
        raise ValueError("depth arrays must be [N,1,H,W] matching config")
    if obs.shape != (depth_m.shape[0], 9):
        raise ValueError("obs must be [N,9]")
    if previous.shape != (depth_m.shape[0], 3):
        raise ValueError("previous_direction must be [N,3]")
    dataset = TensorDataset(depth_m, depth_yopo, obs, previous)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    parameters = list(planner.parameters()) + [p for p in yopo.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)

    for epoch in range(1, cfg.epochs + 1):
        planner.train()
        losses = []
        for depth_raw, depth_input, observation, prior in loader:
            depth_raw, depth_input, observation, prior = (x.to(device) for x in (depth_raw, depth_input, observation, prior))
            optimizer.zero_grad(set_to_none=True)
            candidates, scores = decode_yopo_output(yopo, depth_input, observation)
            result = planner(depth_raw, observation, candidates, scores,
                             previous_direction=prior if has_temporal else None, training=True)
            loss = riop_loss(result, cfg)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(loss["total"].item())
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"refinement_head": planner.refinement_head.state_dict(),
                "config": vars(cfg), "stage": args.stage,
                "temporal_trained": has_temporal}, args.output)
    if args.stage != 1:
        torch.save(yopo.state_dict(), args.output.with_name("yopo_finetuned.pth"))
    print("saved", args.output)


if __name__ == "__main__":
    main()
