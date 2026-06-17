import os
from glob import glob
from setuptools import find_packages, setup

package_name = "g1_mujoco_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "worlds"), glob("worlds/*.xml")),
        (os.path.join("share", package_name, "worlds", "meshes"), glob("worlds/meshes/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cychoi",
    maintainer_email="cychoi@genon.ai",
    description="MuJoCo simulator bridge for Unitree G1 (nav2).",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "g1_sim_node = g1_mujoco_sim.g1_sim_node:main",
            "self_filter_viz_node = g1_mujoco_sim.self_filter_viz_node:main",
        ],
    },
)
