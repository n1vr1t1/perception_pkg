#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/object_hypothesis_with_score.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <string>
#include <memory>
#include <limits>

class PerceptionNode : public rclcpp::Node {
public:
    PerceptionNode() : Node("perception_node") {
        try {
            model = torch::jit::load(model_path);
            model.to(torch::kCPU);
            model.eval();
            RCLCPP_INFO(this->get_logger(), "PyTorch model loaded successfully.");
        } catch (const std::exception &e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to load model: %s", e.what());
            rclcpp::shutdown();
        }

        img_subscription = image_transport::create_subscription(
            this, "/camera/color/image_raw",
            std::bind(&PerceptionNode::image_callback, this, std::placeholders::_1),
            "raw");

        point_cloud_subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/depth/points", 5,
            std::bind(&PerceptionNode::cloud_callback, this, std::placeholders::_1));
        
        publisher = this->create_publisher<vision_msgs::msg::Detection3DArray>("/perception", 5);
        current_point_cloud = nullptr;
        RCLCPP_INFO(this->get_logger(), "PerceptionNode started.");
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        current_point_cloud = msg;
    }

    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        if (!current_point_cloud) {
            RCLCPP_WARN(this->get_logger(), "No point cloud received as yet.");
            return;
        }

        cv::Mat image;
        try {
            image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed.");
            return;
        }

        cv::Mat resized;
        try{
            cv::resize(image, resized, cv::Size(640, 640)); //se if the inputs dime can be decresed
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error resizing image: %s", e.what());
            return;
        }
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        auto tensor = torch::from_blob(resized.data, {1, 224, 224, 3}, torch::kFloat32).permute({0, 3, 1, 2});

        auto output = model.forward({tensor}).toTensor().to(torch::kCPU);

        std::cout << "Output from model: " << output << std::endl;
        if (output.size(0) == 0) {
            RCLCPP_INFO(this->get_logger(), "No detections found.");
            return;
        }
        auto num_detections = output.size(0); //0 or 1?

        vision_msgs::msg::Detection3DArray blocks_array;
        pose_array.header.stamp = msg->header.stamp;
        pose_array.header.frame_id = latest_cloud_->header.frame_id;

        
        for (int i = 0; i < num_detections; ++i) {
            int class_id = output[i][0].item<int>();
            float confidence = output[i][1].item<float>();
            float u = output[i][2].item<float>();
            float v = output[i][3].item<float>();
            float w = output[i][4].item<float>();
            float h = output[i][5].item<float>();

            int img_width = image.cols;
            int img_height = image.rows;

            int u = static_cast<int>(u * img_width);
            int v = static_cast<int>(v * img_height);
            int w = static_cast<int>(w * img_width);
            int h = static_cast<int>(h * img_height);

            if (u < 0 || v < 0 || u >= current_point_cloud->width || v >= current_point_cloud->height) {
                RCLCPP_WARN(this->get_logger(), "Detection (%d) out of bounds", i);
                continue;
            }
            int index = v * current_point_cloud->width + u;

            sensor_msgs::PointCloud2ConstIterator<float> iter_x(*current_point_cloud, "x");
            sensor_msgs::PointCloud2ConstIterator<float> iter_y(*current_point_cloud, "y");
            sensor_msgs::PointCloud2ConstIterator<float> iter_z(*current_point_cloud, "z");

            std::advance(iter_x, index);
            std::advance(iter_y, index);
            std::advance(iter_z, index);

            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;


            if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                vision_msgs::msg::Detection3D detection;
                geometry_msgs::msg::Pose posisiton_3d;

                posisiton_3d.position.x = x;
                posisiton_3d.position.y = y;
                posisiton_3d.position.z = z;
                posisiton_3d.orientation.w = 1.0;

                vision_msgs::msg::ObjectHypothesisWithPose block_hypothesis;
                block_hypothesis.hypothesis.class_id = std::to_string(class_id);
                block_hypothesis.hypothesis.score = confidence;
                block_hypothesis.pose.pose = pose;

                detection.results.push_back(block_hypothesis);
                detection.bbox.center = pose;

                blocks_array.detections.push_back(detection);

                RCLCPP_INFO(this->get_logger(), "3D Block: class=%d conf=%.2f → [%.3f, %.3f, %.3f]",
                            class_id, confidence, x, y, z);
            }else {
                RCLCPP_WARN(this->get_logger(), "Detection %d: Invalid 3D point at (%d, %d)", i, u, v);
            }
        }
        publisher->publish(blocks_array);

    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PerceptionNode>());
    rclcpp::shutdown();
    return 0;
}