cmake_minimum_required(VERSION 3.10)
project(perception_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.0%2Bcpu.zip")
set(LIBTORCH_DIR ${CMAKE_BINARY_DIR}/libtorch)
set(LIBTORCH_ZIP ${CMAKE_BINARY_DIR}/libtorch.zip)

if(NOT EXISTS ${LIBTORCH_DIR})
  message(STATUS "Downloading libTorch...")
  file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP} SHOW_PROGRESS)
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${LIBTORCH_ZIP}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${LIBTORCH_DIR}")
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
  rclcpp::rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  cv_bridge
  image_transport
)

add_executable(pose_from_camera_node src/position.cc)
ament_target_dependencies(pose_from_camera_node
  rclcpp::rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  vision_msgs
  libstatistics_collector
  )

install(TARGETS
  yolo_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
