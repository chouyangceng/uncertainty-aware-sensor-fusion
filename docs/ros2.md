# ROS 2 接口说明

本项目提供一个可选的 ROS 2 Jazzy/Humble Python 适配器，目录为
`ros2/sensor_fusion_ros`。核心算法不依赖 `rclpy`，因此没有安装 ROS 2 时仍然可以运行
原有的合成实验和全部 Python 测试。

## 构建与运行

### 先安装核心 Python 包

ROS 2 适配器会导入仓库中的核心算法包。对一个全新的工作空间，先在仓库根目录执行：

```bash
cd uncertainty-aware-sensor-fusion
python3 -m pip install -e .
```

这样 `uncertainty_sensor_fusion` 会进入当前 Python 环境；随后再构建下面的
`ament_python` 包。若使用虚拟环境，请在 `source /opt/ros/<distro>/setup.bash`
之后激活同一个虚拟环境，避免 `ros2 run` 和 `pip` 使用不同的 Python。

在已 source ROS 2 环境的工作空间中：

```bash
mkdir -p ~/ros2_ws/src
cp -r ros2/sensor_fusion_ros ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch sensor_fusion_ros fusion.launch.py
```

节点订阅 `std_msgs/msg/Float64MultiArray`，数据按
`[x0, y0, x1, y1, ...]` 解释为当前坐标系下的二维点。默认话题为
`/fusion/points`。输入格式本身**不携带目标 ID**，节点使用带距离门限的确定性贪心
最近邻方法，将当前点与上一帧轨迹关联；点的数组顺序改变不会直接导致 ID 对调。
未匹配点获得单调递增的新 ID，超过 `association_gate` 的点不会强行关联。该轻量方法
适合教学和低密度目标；遮挡、交叉运动或高密度交通场景应替换为匈牙利匹配、运动预测
或传感器原生对象 ID。

它发布以下标准 ROS 消息：

| 话题 | 消息 | 内容 |
| --- | --- | --- |
| `/fusion/occupancy_grid` | `nav_msgs/msg/OccupancyGrid` | Log-odds 栅格，未观测单元为 `-1` |
| `/fusion/tracks` | `visualization_msgs/msg/MarkerArray` | 多目标跟踪结果，可在 RViz 中显示 |
| `/fusion/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 点流健康度和降级状态 |

参数见 `config/fusion.yaml`，包括栅格大小、分辨率、置信度、关联门限、跟踪老化阈值
和话题名。启动时会校验栅格尺寸与分辨率为正、`point_confidence` 严格位于 `(0, 1)`、
`association_gate` 为正；非法参数会让节点立即报错，避免带错误配置运行。
生产传感器驱动可以保留同样的输出接口，只替换节点中的点云解码回调。

节点会记录上一轮已发布的 Marker ID。当轨迹超过 `max_track_age` 被移除时，下一条
`MarkerArray` 会包含相同命名空间和 ID 的 `DELETE` 动作，避免 RViz 留下幽灵目标。

## QoS 假设与调优

当前节点使用 ROS 2 默认的 **KeepLast(depth=10)、Reliable、Volatile** QoS：订阅和
三个发布器均以队列深度 10 创建。该选择适合实验台和本机有线网络，能够保证点流和
诊断消息按序送达，但在高频 LiDAR 或无线链路上可能增加延迟或触发队列丢弃。

- 传感器驱动若使用 `SensorDataQoS`（BestEffort），需要在节点中把订阅器改成兼容
  的 `QoSProfile`，否则会出现“有话题但收不到消息”。
- RViz 通常可用默认 Reliable QoS 订阅 `OccupancyGrid` 和 `MarkerArray`；若只关心
  最新帧，可将显示端设置为 BestEffort/Volatile 以降低延迟。
- `depth=10` 是保守起点，不代表实车最优值。请按传感器频率、网络带宽和允许的端到端
  延迟重新标定，并在 rosbag 回放时记录 QoS 配置。

## 无 ROS 2 环境

桥接转换函数位于 `sensor_fusion_ros/bridge.py`，通过延迟导入实现可选依赖。
没有 ROS 2 时，`bridge.ros2_available()` 返回 `False`，核心包仍可按以下命令验证：

```bash
python -m pytest -q
python -m ruff check .
```

当前 CI 未安装 ROS 2，因此不会虚构 ROS 2 运行时或实车验证结果；在真实系统中应使用
目标发行版和传感器驱动进行端到端测试。
