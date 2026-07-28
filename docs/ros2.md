# ROS 2 接口说明

本项目提供一个可选的 ROS 2 Jazzy/Humble Python 适配器，目录为
`ros2/sensor_fusion_ros`。核心算法不依赖 `rclpy`，因此没有安装 ROS 2 时仍然可以运行
原有的合成实验和全部 Python 测试。

## 构建与运行

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
`/fusion/points`。它发布以下标准 ROS 消息：

| 话题 | 消息 | 内容 |
| --- | --- | --- |
| `/fusion/occupancy_grid` | `nav_msgs/msg/OccupancyGrid` | Log-odds 栅格，未观测单元为 `-1` |
| `/fusion/tracks` | `visualization_msgs/msg/MarkerArray` | 多目标跟踪结果，可在 RViz 中显示 |
| `/fusion/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | 点流健康度和降级状态 |

参数见 `config/fusion.yaml`，包括栅格大小、分辨率、置信度、跟踪老化阈值和话题名。
生产传感器驱动可以保留同样的输出接口，只替换节点中的点云解码回调。

## 无 ROS 2 环境

桥接转换函数位于 `sensor_fusion_ros/bridge.py`，通过延迟导入实现可选依赖。
没有 ROS 2 时，`bridge.ros2_available()` 返回 `False`，核心包仍可按以下命令验证：

```bash
python -m pytest -q
python -m ruff check .
```

当前 CI 未安装 ROS 2，因此不会虚构 ROS 2 运行时或实车验证结果；在真实系统中应使用
目标发行版和传感器驱动进行端到端测试。
