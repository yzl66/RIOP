"""YOPO-Simple ROS1 drop-in inference node with RIOP terminal-state refinement.

Run with both this repository and upstream YOPO/YOPO on PYTHONPATH. Requires ROS1.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath

from test_yopo_ros import YopoNet  # from upstream YOPO-Simple
from config.config import cfg as yopo_cfg
from policy.primitive import LatticePrimitive
from riop import RIOPConfig, RIOPPlanner
from riop.memory import TrajectoryMemory
from riop.trajectory import quintic_samples


class RIOPNet(YopoNet):
    def __init__(self, settings, yopo_weight, refinement_weight, riop_config):
        self.riop_cfg = RIOPConfig.from_yaml(riop_config)
        yopo_cfg["train"] = False
        if (self.riop_cfg.image_height, self.riop_cfg.image_width) != (yopo_cfg["image_height"], yopo_cfg["image_width"]):
            raise ValueError("RIOP and YOPO image dimensions differ")
        if abs(self.riop_cfg.segment_time - LatticePrimitive.get_instance().segment_time) > 0.02:
            raise ValueError("RIOP segment_time differs from YOPO motion primitive time")
        self.riop = RIOPPlanner(self.riop_cfg)
        state = torch.load(refinement_weight, map_location="cpu", weights_only=True)
        self.riop.refinement_head.load_state_dict(state["refinement_head"])
        self.temporal_trained = bool(state.get("temporal_trained", False))
        self.riop = self.riop.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.memory = TrajectoryMemory()
        self._metric_depth = None
        self._history = None
        self._extra_publishers = None
        settings = dict(settings)
        settings["visualize"] = False  # parent expects all YOPO candidates otherwise
        settings["use_tensorrt"] = 0
        super().__init__(settings, str(yopo_weight))

    def callback_set_goal(self, data):
        self.memory.reset()
        super().callback_set_goal(data)

    def callback_depth(self, data):
        if data.encoding == "32FC1":
            raw = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":
            raw = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError("RIOP expects 32FC1 metres or 16UC1 millimetres")
        if raw.shape != (self.riop_cfg.image_height, self.riop_cfg.image_width):
            raw = cv2.resize(raw, (self.riop_cfg.image_width, self.riop_cfg.image_height), interpolation=cv2.INTER_NEAREST)
        self._metric_depth = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        super().callback_depth(data)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        primitive_count = self.lattice_primitive.traj_num
        pred = endstate_pred.reshape(9, primitive_count).T
        score = score_pred.reshape(primitive_count)
        lattice_ids = np.arange(primitive_count - 1, -1, -1)
        candidates = self.state_transform.pred_to_endstate_cpu(pred, lattice_ids)
        obs_norm = self.process_odom()
        obs = obs_norm.copy()
        obs[:, :3] *= self.lattice_primitive.vel_max
        obs[:, 3:6] *= self.lattice_primitive.acc_max
        obs[:, 6:9] = self.Rotation_wc.T @ (self.goal - self.desire_pos)
        device = self.device
        candidate_t = torch.from_numpy(candidates.astype(np.float32))[None].to(device)
        score_t = torch.from_numpy(score.astype(np.float32))[None].to(device)
        obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)
        depth_t = torch.from_numpy(self._metric_depth)[None, None].to(device)
        rotation_t = torch.from_numpy(self.Rotation_wc.astype(np.float32))[None].to(device)
        previous = self.memory.get(rotation_t) if self.temporal_trained else None
        with torch.inference_mode():
            result = self.riop(depth_t, obs_t, candidate_t, score_t, previous_direction=previous)
        if self.temporal_trained:
            self.memory.update(result["trajectory"], rotation_t)
        self._history = [x.detach().cpu() for x in result["history"]]
        self._obs_body = obs_t.detach().cpu()
        return result["trajectory"].cpu().numpy(), float(result["score"][0].item())

    def visualize_trajectory(self, pred_score, pred_endstate):
        super().visualize_trajectory(pred_score, pred_endstate)
        if self._history is None:
            return
        if self._extra_publishers is None:
            self._extra_publishers = {
                "initial": rospy.Publisher("/riop/initial", RosPath, queue_size=1),
                "iteration_1": rospy.Publisher("/riop/iteration_1", RosPath, queue_size=1),
                "final": rospy.Publisher("/riop/final", RosPath, queue_size=1),
            }
        start = self.desire_pos if self.plan_from_reference else np.array([
            self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        named = {"initial": self._history[0], "final": self._history[-1]}
        if len(self._history) > 1:
            named["iteration_1"] = self._history[1]
        for name, endpoint in named.items():
            if self._extra_publishers[name].get_num_connections() == 0:
                continue
            positions = quintic_samples(endpoint, self._obs_body[:, :3], self._obs_body[:, 3:6],
                                        self.traj_time, self.riop_cfg.samples)[0][0].numpy()
            world = positions @ self.Rotation_wc.T + start
            path = RosPath()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "world"
            for point in world:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, point)
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self._extra_publishers[name].publish(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yopo-weight", type=Path, required=True)
    parser.add_argument("--riop-weight", type=Path, required=True)
    parser.add_argument("--riop-config", type=Path, default=Path("config/riop.yaml"))
    args = parser.parse_args()
    settings = {"use_tensorrt": 0, "goal": [50, 0, 2], "pitch_angle_deg": 0,
                "odom_topic": "/sim/odom", "depth_topic": "/depth_image",
                "ctrl_topic": "/so3_control/pos_cmd", "plan_from_reference": False,
                "verbose": False, "visualize": False}
    RIOPNet(settings, args.yopo_weight, args.riop_weight, args.riop_config)


if __name__ == "__main__":
    main()
