# Assume you're using Ubuntu 22.04 with ROS 2 Humble
FROM ros:humble

# Install required ROS 2 and system dependencies
RUN apt update && apt install -y \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    libopencv-dev \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download libTorch (CPU version)
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.0%2Bcpu.zip -O /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /opt && \
    rm /tmp/libtorch.zip

# Set environment variable so CMake can find libTorch
ENV TORCH_DIR=/opt/libtorch

# Set ROS workspace
WORKDIR /root/ros2_ws
