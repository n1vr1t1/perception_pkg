cmake_minimum_required(VERSION 3.10)
project(perception_pkg)

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === ROS 2 dependencies ===
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

find_package(Torch REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  include
)

add_executable(yolo_detection_node src/detection.cc)
target_link_libraries(yolo_detection_node
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  rclcpp::rclcpp
)
ament_target_dependencies(yolo_detection_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  cv_bridge
  image_transport
)

install(TARGETS
  yolo_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
