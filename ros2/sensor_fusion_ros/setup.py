from setuptools import find_packages, setup

package_name = "sensor_fusion_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/fusion.yaml"]),
        ("share/" + package_name + "/launch", ["launch/fusion.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    description="ROS 2 adapter for uncertainty-aware vehicle sensor fusion",
    license="Apache-2.0",
    entry_points={"console_scripts": ["fusion_node = sensor_fusion_ros.fusion_node:main"]},
)
