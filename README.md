# RIOP — Risk-aware Iterative Optimization Planner

RIOP 在 [YOPO-Simple](https://github.com/TJU-Aerial-Robotics/YOPO/tree/YOPO-Simple) 预测的运动原语中选出最低分终端状态，再用风险评估和小步迭代修正它。输出仍是位置、速度、加速度组成的 9 维终端状态，可接回 YOPO 的五次多项式与 ROS 控制链路。

> 当前版本是研究原型。深度风险是沿相机射线的近似，不是 3D ESDF 或已验证的避障器。未用真实飞行、实际 YOPO 权重或 ROS 仿真验证。部署前须校准深度单位、相机坐标、视场角、机体尺寸，并在仿真中做碰撞与延迟测试。

## 算法

```text
depth + velocity/acceleration/goal
    → YOPO-Simple end states and scores
    → lowest-score T0
    → sampled quintic path / depth-proxy risk
    → adaptive budget (0, 1, or 2 steps)
    → bounded learned correction ΔT
    → accept only if risk + anchor cost decreases
    → final end state → YOPO polynomial controller
```

风险由碰撞、近障距离、速度/加速度/jerk 和上一帧方向变化组成。`TrajectoryMemory` 把上一帧终点方向保存在世界坐标，在新一帧转换回相机坐标。迭代接受条件另加对初始 YOPO 终端状态的偏移代价，避免网络靠远离原计划来降低深度风险。配置见 [`config/riop.yaml`](config/riop.yaml)。

## 安装

要求 Python 3.8+、PyTorch、NumPy、PyYAML。训练和 ROS 运行还需 YOPO-Simple 对应环境、权重、ROS1 与其自定义消息。

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

YOPO 源码和权重不包含在本仓库。请按[上游安装说明](https://github.com/TJU-Aerial-Robotics/YOPO/tree/YOPO-Simple)单独获取 `YOPO-Simple` 分支。原项目是 MIT 许可证；本仓库未复制其源文件或权重。

## 输入与训练

准备一个 `.npz` 文件，三个键长度一致：

| 键 | 形状 | 约定 |
|---|---|---|
| `depth_m` | `[N,1,H,W]` | 米，未观测/无效像素设为 0 |
| `depth_yopo` | `[N,1,H,W]` | 按 YOPO-Simple 预处理后的 0–1 深度输入 |
| `obs` | `[N,9]` | 相机系 `[vx,vy,vz,ax,ay,az,gx,gy,gz]`，未归一化 |
| `previous_direction`（可选） | `[N,3]` | 上一帧终点方向变换到当前相机系；序列首帧填零 |

`H/W` 和相机视场角要与 `config/riop.yaml` 一致。默认相机系为 x 向前、y 向左、z 向上；投影使用水平 90°、垂直 60° 的默认值。请根据实际标定值修改。`segment_time` 应与 YOPO 的运动原语时长一致；默认 `2 × radio_range / vel_max ≈ 1.67 s`。样本应包含障碍物、空旷区域及不同速度状态。

```bash
python train_riop.py \
  --yopo-root /path/to/YOPO/YOPO \
  --yopo-checkpoint /path/to/YOPO/YOPO/saved/YOPO_1/epoch50.pth \
  --dataset /path/to/train.npz \
  --stage 1 \
  --output saved/RIOP_1/refinement.pth
```

阶段 1 冻结 YOPO；阶段 2 解冻其预测头；阶段 3 联合微调全部网络。后两阶段会另外保存 `yopo_finetuned.pth`。`riop_loss` 接受可选 `yopo_loss`，接入上游损失时可用于联合训练。只有训练数据提供 `previous_direction` 时，ROS 节点才启用时序记忆。当前训练脚本使用深度代理风险的自监督目标，尚无仿真专家标签或真实 ESDF，不能仅凭训练损失判断飞行安全。

## ROS1 仿真接入

先按上游说明启动 Controller 和 Simulator，再让 Python 同时能导入本仓库与上游 `YOPO/YOPO`：

```bash
export PYTHONPATH="/path/to/RIOP:/path/to/YOPO/YOPO:$PYTHONPATH"
python /path/to/RIOP/test_riop_ros.py \
  --yopo-weight /path/to/YOPO/YOPO/saved/YOPO_1/epoch50.pth \
  --riop-weight /path/to/RIOP/saved/RIOP_1/refinement.pth \
  --riop-config /path/to/RIOP/config/riop.yaml
```

ROS 节点复用上游的深度订阅、里程计处理、五次多项式和 `PositionCommand` 发布。额外提供 `/riop/initial`、`/riop/iteration_1` 和 `/riop/final` 三个 `nav_msgs/Path`，在 RViz 中建议设为灰、黄、绿。`/yopo_net/best_traj_visual` 也显示最终轨迹。当前节点只支持原版 PyTorch YOPO 权重，未接 TensorRT。首次运行请确认话题名、相机外参、轨迹时间和控制器配置与上游一致。

## 实验与边界

建议按 YOPO 原版、仅风险评估、风险加修正、再加自适应轮次、最后加时序方向约束做消融。至少记录成功率、碰撞率、最小障碍距离、路径长度、峰值加速度/jerk 和规划延迟。风险代理只比较投影点与单张深度图的前向距离；它不能识别遮挡后的障碍物，也不覆盖动态障碍与传感器不确定性。真实系统应增加 ESDF/地图校验与独立安全监控。

## 来源

本实现依据用户提供的 RIOP 算法链路，并对接 [TJU-Aerial-Robotics/YOPO 的 YOPO-Simple 分支](https://github.com/TJU-Aerial-Robotics/YOPO/tree/YOPO-Simple)。感谢原作者公开 YOPO 的代码与研究。
