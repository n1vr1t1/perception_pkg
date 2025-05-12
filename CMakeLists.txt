cmake_minimum_required(VERSION 3.8)
project(perception_pkg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(libstatistics_collector REQUIRED)

include_directories(
  include
  ${TORCH_INCLUDE_DIRS}
)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}; "/home/ubuntu/ros2_ws/src/Robotics/ros2_ws/external/libtorch/libtorch")
find_package(Torch REQUIRED)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

#add_executable(yolo_detection_node src/detection.cc)
#target_include_directories(yolo_detection_node PRIVATE 
#  ${TORCH_INCLUDE_DIRS}
#)
#ament_target_dependencies(yolo_detection_node
#  rclcpp
#  std_msgs
#  sensor_msgs
#  geometry_msgs
#  vision_msgs
#  cv_bridge
#  image_transport
#  OpenCV
#)
#target_link_libraries(yolo_detection_node
#  ${rclcpp_LIBS}
#  ${std_msgs_LIBS}
#  ${sensor_msgs_LIBS}
#  ${geometry_msgs_LIBS}
#  ${vision_msgs_LIBS}
#  ${cv_bridge_LIBS}
#  ${image_transport_LIBS}
#  ${OpenCV_LIBS}
#)

add_executable(pose_from_camera_node src/position.cc)
ament_target_dependencies(pose_from_camera_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  libstatistics_collector
  )
#target_link_libraries(pose_from_camera_node
#  rclcpp
#  std_msgs
#  sensor_msgs
#  geometry_msgs
#  vision_msgs
#  libstatistics_collector
#  )

install(TARGETS
#  yolo_detection_node
  pose_from_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
