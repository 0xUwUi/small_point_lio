/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "small_point_lio_node.hpp"
#include "io/pcd_io.h"
#include "lidar_adapter/custom_mid360_driver.h"
#include "lidar_adapter/livox_custom_msg.h"
#include "lidar_adapter/livox_pointcloud2.h"
#include "lidar_adapter/unitree_lidar.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <mutex>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace small_point_lio {

    struct TransformCache {
        std::mutex mutex;
        bool has_transform{false};
        // Estimator outputs IMU pose (T_WI); world W is the IMU frame at init.
        tf2::Transform base_to_imu;// T_B_I
        tf2::Transform imu_to_base;// T_I_B
        Eigen::Matrix3f base_to_imu_R = Eigen::Matrix3f::Identity();
        Eigen::Vector3f base_to_imu_T = Eigen::Vector3f::Zero();
    };

    SmallPointLioNode::SmallPointLioNode(const rclcpp::NodeOptions &options)
        : Node("small_point_lio", options) {
        std::string lidar_topic = declare_parameter<std::string>("lidar_topic");
        std::string imu_topic = declare_parameter<std::string>("imu_topic");
        std::string lidar_type = declare_parameter<std::string>("lidar_type");
        std::string lidar_frame = declare_parameter<std::string>("lidar_frame");
        bool save_pcd = declare_parameter<bool>("save_pcd");
        small_point_lio = std::make_unique<small_point_lio::SmallPointLio>(*this);
        odometry_publisher = create_publisher<nav_msgs::msg::Odometry>("odometry", 1000);
        pointcloud_publisher = create_publisher<sensor_msgs::msg::PointCloud2>("registered_scan", 1000);
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
        if (save_pcd) {
            pointcloud_mapping = std::make_unique<util::PointcloudMapping>(0.02);
        }
        map_save_trigger = create_service<std_srvs::srv::Trigger>(
                "map_save",
                [this, save_pcd, lidar_frame](const std_srvs::srv::Trigger::Request::SharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res) {
                    if (!save_pcd) {
                        res->success = false;
                        res->message = "pcd save is disabled";
                        RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "pcd save is disabled");
                        return;
                    }
                    res->success = true;
                    RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "waiting for pcd saving ...");
                    auto pointcloud_to_save = std::make_shared<std::vector<Eigen::Vector3f>>();
                    *pointcloud_to_save = pointcloud_mapping->get_points();
                    std::thread([pointcloud_to_save, lidar_frame]() {
                        io::pcd::write_pcd(ROOT_DIR + "/pcd/scan.pcd", *pointcloud_to_save);
                        RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "save pcd success");
                    }).detach();
                });
        // Lidar -> IMU extrinsic (p_I = R * p_L + T), already declared by Parameters.
        const std::vector<double> extrinsic_T = get_parameter("extrinsic_T").as_double_array();
        const std::vector<double> extrinsic_R = get_parameter("extrinsic_R").as_double_array();
        tf2::Transform imu_to_lidar;// T_I_L
        imu_to_lidar.setBasis(tf2::Matrix3x3(
                extrinsic_R[0], extrinsic_R[1], extrinsic_R[2],
                extrinsic_R[3], extrinsic_R[4], extrinsic_R[5],
                extrinsic_R[6], extrinsic_R[7], extrinsic_R[8]));
        imu_to_lidar.setOrigin(tf2::Vector3(extrinsic_T[0], extrinsic_T[1], extrinsic_T[2]));

        auto transform_cache = std::make_shared<TransformCache>();
        auto init_transform_cache = [this, lidar_frame, transform_cache, imu_to_lidar](const builtin_interfaces::msg::Time &time_msg) {
            std::lock_guard<std::mutex> lock(transform_cache->mutex);
            if (transform_cache->has_transform) {
                return true;
            }
            try {
                // lookupTransform(target=lidar, source=base) gives T_L_B
                auto lidar_to_base_msg = tf_buffer->lookupTransform(lidar_frame, "base_footprint", time_msg);
                tf2::Transform lidar_to_base;
                tf2::fromMsg(lidar_to_base_msg.transform, lidar_to_base);
                // T_B_I = T_B_L * T_L_I
                transform_cache->base_to_imu = lidar_to_base.inverse() * imu_to_lidar.inverse();
                transform_cache->imu_to_base = transform_cache->base_to_imu.inverse();
                const tf2::Quaternion q = transform_cache->base_to_imu.getRotation();
                transform_cache->base_to_imu_R = Eigen::Quaternionf(
                        static_cast<float>(q.getW()),
                        static_cast<float>(q.getX()),
                        static_cast<float>(q.getY()),
                        static_cast<float>(q.getZ()))
                        .toRotationMatrix();
                const tf2::Vector3 &t = transform_cache->base_to_imu.getOrigin();
                transform_cache->base_to_imu_T << static_cast<float>(t.getX()),
                        static_cast<float>(t.getY()),
                        static_cast<float>(t.getZ());
                transform_cache->has_transform = true;
                RCLCPP_INFO(rclcpp::get_logger("small_point_lio"), "cached base_footprint <-> %s transform", lidar_frame.c_str());
                return true;
            } catch (tf2::TransformException &ex) {
                RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "Failed to lookup transform from base_footprint to %s: %s", lidar_frame.c_str(), ex.what());
                return false;
            }
        };

        small_point_lio->set_odometry_callback([this, lidar_frame, transform_cache, init_transform_cache](const common::Odometry &odometry) {
            last_odometry = odometry;

            builtin_interfaces::msg::Time time_msg;
            time_msg.sec = std::floor(odometry.timestamp);
            time_msg.nanosec = static_cast<uint32_t>((odometry.timestamp - time_msg.sec) * 1e9);
            
            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped.header.stamp = time_msg;
            transform_stamped.header.frame_id = "odom";
            transform_stamped.child_frame_id = "base_footprint";

            
            if (!init_transform_cache(time_msg)) {
                return;
            }
            
            // Estimator pose is T_WI (IMU in world frame W = IMU frame at init)
            tf2::Transform world_to_imu;
            world_to_imu.setOrigin(tf2::Vector3(odometry.position.x(), odometry.position.y(), odometry.position.z()));
            world_to_imu.setRotation(tf2::Quaternion(odometry.orientation.x(), odometry.orientation.y(), odometry.orientation.z(), odometry.orientation.w()));
            tf2::Transform base_to_imu;
            tf2::Transform imu_to_base;
            {
                std::lock_guard<std::mutex> lock(transform_cache->mutex);
                base_to_imu = transform_cache->base_to_imu;
                imu_to_base = transform_cache->imu_to_base;
            }
            // T_odom_base = T_B_I * T_WI * T_I_B
            const tf2::Transform odom_to_base = base_to_imu * world_to_imu * imu_to_base;
            transform_stamped.transform = tf2::toMsg(odom_to_base);

            nav_msgs::msg::Odometry odometry_msg;
            odometry_msg.header.stamp = time_msg;
            odometry_msg.header.frame_id = "odom";
            odometry_msg.child_frame_id = "base_footprint";

            odometry_msg.pose.pose.position.x = transform_stamped.transform.translation.x;
            odometry_msg.pose.pose.position.y = transform_stamped.transform.translation.y;
            odometry_msg.pose.pose.position.z = transform_stamped.transform.translation.z;
            odometry_msg.pose.pose.orientation.x = transform_stamped.transform.rotation.x;
            odometry_msg.pose.pose.orientation.y = transform_stamped.transform.rotation.y;
            odometry_msg.pose.pose.orientation.z = transform_stamped.transform.rotation.z;
            odometry_msg.pose.pose.orientation.w = transform_stamped.transform.rotation.w;

            // Estimator velocity is expressed in world frame W, angular velocity in IMU body frame.
            // Twist must be expressed in child frame (base_footprint).
            const tf2::Vector3 linear_velocity_world(odometry.velocity.x(), odometry.velocity.y(), odometry.velocity.z());
            const tf2::Vector3 angular_velocity_imu(odometry.angular_velocity.x(), odometry.angular_velocity.y(), odometry.angular_velocity.z());
            const tf2::Vector3 linear_velocity_imu = world_to_imu.getBasis().transpose() * linear_velocity_world;
            const tf2::Vector3 base_origin_in_imu = imu_to_base.getOrigin();
            const tf2::Matrix3x3 &base_to_imu_rotation = base_to_imu.getBasis();
            const tf2::Vector3 linear_velocity_base = base_to_imu_rotation * (linear_velocity_imu + angular_velocity_imu.cross(base_origin_in_imu));
            const tf2::Vector3 angular_velocity_base = base_to_imu_rotation * angular_velocity_imu;
            odometry_msg.twist.twist.linear.x = linear_velocity_base.x();
            odometry_msg.twist.twist.linear.y = linear_velocity_base.y();
            odometry_msg.twist.twist.linear.z = linear_velocity_base.z();
            odometry_msg.twist.twist.angular.x = angular_velocity_base.x();
            odometry_msg.twist.twist.angular.y = angular_velocity_base.y();
            odometry_msg.twist.twist.angular.z = angular_velocity_base.z();


            geometry_msgs::msg::TransformStamped position_only_transform;
            position_only_transform.header = transform_stamped.header;
            position_only_transform.child_frame_id = "base_link_position_only";
            position_only_transform.transform.translation = transform_stamped.transform.translation;
            // Keep the position-only frame aligned with camera_init while following body's position.
            position_only_transform.transform.rotation.x = 0.0;
            position_only_transform.transform.rotation.y = 0.0;
            position_only_transform.transform.rotation.z = 0.0;
            position_only_transform.transform.rotation.w = 1.0;

            tf_broadcaster->sendTransform(transform_stamped);
            tf_broadcaster->sendTransform(position_only_transform);
            odometry_publisher->publish(odometry_msg);
        });
        small_point_lio->set_pointcloud_callback([this, save_pcd, lidar_frame, transform_cache, init_transform_cache](const std::vector<Eigen::Vector3f> &pointcloud) {
            if (pointcloud_publisher->get_subscription_count() > 0) {
                builtin_interfaces::msg::Time time_msg;
                time_msg.sec = std::floor(last_odometry.timestamp);
                time_msg.nanosec = static_cast<uint32_t>((last_odometry.timestamp - time_msg.sec) * 1e9);

                if (!init_transform_cache(time_msg)) {
                    return;
                }
                // Points are in world frame W (IMU at init): p_odom = R_BI * p_W + t_BI
                Eigen::Vector3f base_to_imu_T;
                Eigen::Matrix3f base_to_imu_R;
                {
                    std::lock_guard<std::mutex> lock(transform_cache->mutex);
                    base_to_imu_T = transform_cache->base_to_imu_T;
                    base_to_imu_R = transform_cache->base_to_imu_R;
                }
                
                sensor_msgs::msg::PointCloud2 msg;
                msg.header.stamp = time_msg;
                msg.header.frame_id = "odom";
                msg.width = pointcloud.size();
                msg.height = 1;
                msg.fields.reserve(4);
                sensor_msgs::msg::PointField field;
                field.name = "x";
                field.offset = 0;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                field.name = "y";
                field.offset = 4;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                field.name = "z";
                field.offset = 8;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                field.name = "intensity";
                field.offset = 12;
                field.datatype = sensor_msgs::msg::PointField::FLOAT32;
                field.count = 1;
                msg.fields.push_back(field);
                msg.is_bigendian = false;
                msg.point_step = 16;
                msg.row_step = msg.width * msg.point_step;
                msg.data.resize(msg.row_step * msg.height);
                Eigen::Vector3f transformed_point;
                auto pointer = reinterpret_cast<float *>(msg.data.data());
                for (const auto &point: pointcloud) {
                    transformed_point = base_to_imu_R * point + base_to_imu_T;
                    *pointer = transformed_point.x();
                    ++pointer;
                    *pointer = transformed_point.y();
                    ++pointer;
                    *pointer = transformed_point.z();
                    ++pointer;
                    *pointer = 0;
                    ++pointer;
                }
                msg.is_dense = false;
                pointcloud_publisher->publish(msg);
            }
            if (save_pcd) {
                for (const auto &point: pointcloud) {
                    pointcloud_mapping->add_point(point);
                }
            }
        });
        if (lidar_type == "livox_custom_msg") {
#ifdef HAVE_LIVOX_DRIVER
            lidar_adapter = std::make_unique<LivoxCustomMsgAdapter>();
#else
            RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "livox_custom_msg requested but not available!");
            rclcpp::shutdown();
            return;
#endif
        } else if (lidar_type == "livox_pointcloud2") {
            lidar_adapter = std::make_unique<LivoxPointCloud2Adapter>();
        } else if (lidar_type == "custom_mid360_driver") {
            lidar_adapter = std::make_unique<CustomMid360DriverAdapter>();
        } else if (lidar_type == "unilidar") {
            lidar_adapter = std::make_unique<UnilidarAdapter>();
        } else {
            RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "unknwon lidar type");
            rclcpp::shutdown();
            return;
        }
        lidar_adapter->setup_subscription(this, lidar_topic, [this](const std::vector<common::Point> &pointcloud) {
            small_point_lio->on_point_cloud_callback(pointcloud);
            small_point_lio->handle_once();
        });
        imu_subsciber = create_subscription<sensor_msgs::msg::Imu>(
                imu_topic,
                rclcpp::SensorDataQoS(),
                [this](const sensor_msgs::msg::Imu &msg) {
                    common::ImuMsg imu_msg;
                    imu_msg.angular_velocity = Eigen::Vector3d(msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z);
                    imu_msg.linear_acceleration = Eigen::Vector3d(msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z);
                    imu_msg.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9;
                    small_point_lio->on_imu_callback(imu_msg);
                    small_point_lio->handle_once();
                });
    }

}// namespace small_point_lio

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(small_point_lio::SmallPointLioNode)
