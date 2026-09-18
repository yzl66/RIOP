import torch

from riop import RIOPConfig, RIOPPlanner
from riop.controller import IterationController
from riop.risk import RiskEvaluator
from riop.trajectory import quintic_samples
from riop.yopo_adapter import decode_yopo_output


def config(**overrides):
    data = dict(image_width=32, image_height=24, samples=20, segment_time=2.0,
                risk_dynamic=0.0, risk_clearance=0.0, risk_temporal=0.0)
    data.update(overrides)
    return RIOPConfig(**data)


def observation():
    return torch.zeros(1, 9)


def endpoint(x):
    return torch.tensor([[x, 0., 0., 0., 0., 0., 0., 0., 0.]])


def test_quintic_reaches_boundary_state():
    end = torch.tensor([[4., 1., 0., 2., 0., 0., 0., 0., 0.]])
    pos, vel, acc, jerk = quintic_samples(end, torch.tensor([[1., 0., 0.]]),
                                           torch.zeros(1, 3), 2.0, 30)
    torch.testing.assert_close(pos[0, -1], end[0, :3], atol=1e-5, rtol=0)
    torch.testing.assert_close(vel[0, -1], end[0, 3:6], atol=1e-5, rtol=0)
    torch.testing.assert_close(acc[0, -1], end[0, 6:9], atol=1e-5, rtol=0)
    assert torch.isfinite(jerk).all()


def test_depth_obstacle_increases_collision_risk():
    cfg = config()
    evaluator = RiskEvaluator(cfg)
    near = torch.full((1, 1, 24, 32), 3.0)
    far = torch.full((1, 1, 24, 32), 12.0)
    risk_near = evaluator(near, endpoint(5.), observation())
    risk_far = evaluator(far, endpoint(5.), observation())
    assert risk_near["collision"].item() > risk_far["collision"].item()
    assert risk_near["unknown_fraction"].item() < 1.0


def test_adaptive_budgets():
    controller = IterationController(config(risk_low=0.2, risk_high=0.6))
    assert controller.budgets(torch.tensor([0.1, 0.4, 0.8])).tolist() == [0, 1, 2]


def test_first_frame_has_no_temporal_penalty():
    cfg = config(risk_temporal=1.0)
    risk = RiskEvaluator(cfg)(torch.full((1, 1, 24, 32), 12.0), endpoint(5.),
                              observation(), previous_direction=torch.zeros(1, 3))
    assert risk["temporal"].item() == 0.0


class FixedDelta(torch.nn.Module):
    def __init__(self, dx):
        super().__init__()
        self.dx = dx

    def forward(self, endstate, obs, risk, previous_direction=None):
        delta = torch.zeros_like(endstate)
        delta[:, 0] = self.dx
        return delta


def test_refinement_accepts_improvement_and_rejects_regression():
    cfg = config(risk_low=0., risk_high=0.01, anchor_weight=0., min_improvement=0.)
    depth = torch.full((1, 1, 24, 32), 4.0)
    candidates = endpoint(5.)[:, None, :]
    scores = torch.zeros(1, 1)
    planner = RIOPPlanner(cfg)
    planner.refinement_head = FixedDelta(-2.)
    improved = planner(depth, observation(), candidates, scores)
    assert improved["trajectory"][0, 0] < 5
    assert improved["risks"][-1]["total"] < improved["risks"][0]["total"]
    planner.refinement_head = FixedDelta(2.)
    rejected = planner(depth, observation(), candidates, scores)
    torch.testing.assert_close(rejected["trajectory"], rejected["initial"])


def test_yopo_adapter_preserves_score_order():
    class MockYOPO:
        def inference(self, depth, obs):
            ends = torch.arange(4*9, dtype=torch.float32).reshape(1, 9, 2, 2)
            return ends, torch.tensor([[[4., 1.], [2., 3.]]])

    candidates, scores = decode_yopo_output(MockYOPO(), torch.zeros(1, 1, 2, 2), observation())
    assert candidates.shape == (1, 4, 9)
    assert scores.tolist() == [[4., 1., 2., 3.]]
    assert candidates[0, 1, 0].item() == 1.
