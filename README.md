# Uncertainty-Aware Sensor Fusion

面向车辆工程本科生的相机、LiDAR、Radar 标定、时间同步、几何融合和可靠性评估实验平台。项目重点是坐标系、时间戳、观测协方差、缺失模态和 Kalman 跟踪，而不是依赖大型黑盒模型。

## 30 秒运行

    python -m pip install -e .
    python examples/quickstart.py

## 完整实验

    python experiments/run_experiment.py --seed 3 --frames 60
    python -m pytest -q
    python -m ruff check .

实验会生成 `summary.json` 和 `trace.csv`，比较 camera-only 与带 LiDAR/Radar 的融合误差，并在一半帧后模拟 LiDAR 失效。

## 代码结构

- `geometry/`：SE(3) 坐标变换和相机投影；
- `calibration/`：LiDAR 到相机的投影接口；
- `synchronization/`：带乱序检查的时间缓存；
- `fusion/`：带来源和置信度的目标级几何融合；
- `tracking/`：带马氏距离门控的 Kalman 跟踪；
- `reliability/`：协方差权重和传感器健康度；
- `experiments/`：确定性的传感器故障实验。

## 研究问题

- 外参误差和时间延迟为何会让多传感器结果错位？
- 相机、LiDAR 和 Radar 的信息互补性如何量化？
- 只有一个传感器可用时，系统如何降级而不崩溃？
- 置信度是否与实际误差一致？

## 可选扩展

可选接入 CARLA 的 RGB、LiDAR、Radar 数据、ROS 2 Topic 和 nuScenes-mini。核心几何、融合和跟踪代码不需要 GPU 或大型数据集。

## 局限性

当前实验使用合成轨迹和人工注入噪声，结果是仿真研究，不等同于真实道路性能。真实车辆需要重新标定内外参、时钟和传感器协方差。

## License

Apache-2.0

## ROS 2 可选接口

仓库提供独立的 `ros2/sensor_fusion_ros` ament_python 包，将核心融合结果转换为标准
`nav_msgs/OccupancyGrid`、`visualization_msgs/MarkerArray` 和
`diagnostic_msgs/DiagnosticArray`。完整构建命令、话题、参数和无 ROS 2 回退说明见
[`docs/ros2.md`](docs/ros2.md)。ROS 2 与 `rclpy` 不在核心依赖中，普通 Python 环境仍可
直接运行全部实验和测试。

## 高级可靠性感知基准

    python examples/advanced_benchmark.py

高级示例会运行在线外参修正、延迟估计、BEV 占据栅格、多目标轨迹管理和传感器健康度更新。新增代码包括 `calibration/online.py`、`synchronization/latency.py`、`fusion/occupancy.py`、`tracking/manager.py` 和 `reliability/manager.py`。
