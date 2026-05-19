/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#ifdef HAVE_LIVOX_DRIVER

#include "base_lidar.h"
#include <chrono>
#include <livox_ros_driver2/msg/custom_msg.hpp>

namespace small_point_lio {

    class LivoxCustomMsgAdapter : public LidarAdapterBase {
    private:
        rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr subscription;

    public:
        inline void setup_subscription(rclcpp::Node *node, const std::string &topic, std::function<void(const std::vector<common::Point> &)> callback) override {
            auto clock = node->get_clock();
            subscription = node->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                    topic,
                    rclcpp::SensorDataQoS(),
                    [callback, clock](const livox_ros_driver2::msg::CustomMsg &msg) {
                        static const auto logger = rclcpp::get_logger("small_point_lio");
                        const auto start_time = std::chrono::steady_clock::now();
                        RCLCPP_INFO_THROTTLE(
                                logger, *clock, 1000,
                                "[DEBUG-splio-a4f2] livox callback begin raw_points=%zu timebase=%llu",
                                msg.points.size(), static_cast<unsigned long long>(msg.timebase));
                        std::vector<common::Point> pointcloud;
                        pointcloud.reserve(msg.points.size());
                        common::Point new_point;
                        for (const auto &point: msg.points) {
                            if ((point.tag & 0b00111111) == 0b00000000) {
                                common::Point new_point;
                                new_point.position << point.x, point.y, point.z;
                                new_point.timestamp = static_cast<double>(msg.timebase + point.offset_time) / 1e9;
                                pointcloud.push_back(new_point);
                            }
                        }
                        const auto convert_cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                                             std::chrono::steady_clock::now() - start_time)
                                                             .count() /
                                                     1000.0;
                        RCLCPP_INFO_THROTTLE(
                                logger, *clock, 1000,
                                "[DEBUG-splio-a4f2] livox callback converted valid_points=%zu raw_points=%zu cost_ms=%.3f",
                                pointcloud.size(), msg.points.size(), convert_cost_ms);
                        callback(pointcloud);
                        const auto total_cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                                           std::chrono::steady_clock::now() - start_time)
                                                           .count() /
                                                   1000.0;
                        RCLCPP_INFO_THROTTLE(
                                logger, *clock, 1000,
                                "[DEBUG-splio-a4f2] livox callback end valid_points=%zu total_cost_ms=%.3f",
                                pointcloud.size(), total_cost_ms);
                    });
        }
    };

}// namespace small_point_lio

#endif
