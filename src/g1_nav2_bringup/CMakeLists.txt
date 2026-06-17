cmake_minimum_required(VERSION 3.8)
project(g1_nav2_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch rviz maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
