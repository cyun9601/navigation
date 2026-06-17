# maps

SLAM (slam_toolbox) builds the map online, so no pre-built map is required.

To save a map after exploring:

    ros2 run nav2_map_server map_saver_cli -f ~/g1_nav/src/g1_nav2_bringup/maps/g1_world

Then you can switch to static-map localization (map_server + amcl) instead of SLAM.
