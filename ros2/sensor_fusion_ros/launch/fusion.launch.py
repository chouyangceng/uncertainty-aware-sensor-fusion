from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config = PathJoinSubstitution([FindPackageShare("sensor_fusion_ros"), "config", "fusion.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config", default_value=config, description="Fusion parameter YAML"
            ),
            Node(
                package="sensor_fusion_ros",
                executable="fusion_node",
                name="uncertainty_sensor_fusion",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
