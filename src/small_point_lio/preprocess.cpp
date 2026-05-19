/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "preprocess.h"
#include "parameters.h"
#include <chrono>

namespace small_point_lio {

    void Preprocess::reset() {
        imu_deque.clear();
        point_deque.clear();
        dense_point_deque.clear();
        last_timestamp_lidar = -1;
        last_timestamp_imu = -1;
        last_timestamp_dense_point = -1;
    }

    void Preprocess::on_point_cloud_callback(const std::vector<common::Point> &pointcloud) {
        static const auto logger = rclcpp::get_logger("small_point_lio");
        static rclcpp::Clock clock(RCL_STEADY_TIME);
        const auto start_time = std::chrono::steady_clock::now();
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess begin input=%zu point_queue=%zu dense_queue=%zu imu_queue=%zu",
                pointcloud.size(), point_deque.size(), dense_point_deque.size(), imu_deque.size());
        dense_points.clear();
        dense_points.reserve(pointcloud.size());
        filtered_points.clear();
        filtered_points.reserve(pointcloud.size());
        for (size_t i = 0; i < pointcloud.size(); i++) {
            const auto &point = pointcloud[i];
            if (point.timestamp >= last_timestamp_dense_point) {
                dense_points.push_back(point);
            }
            if (i % parameters->point_filter_num != 0) {
                continue;
            }
            if (point.timestamp < last_timestamp_lidar) {
                continue;
            }
            float dist = point.position.squaredNorm();
            if (dist < parameters->min_distance_squared || dist > parameters->max_distance_squared) {
                continue;
            }
            filtered_points.push_back(point);
        }
        const auto filter_cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - start_time)
                                            .count() /
                                    1000.0;
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess after filter input=%zu dense=%zu filtered=%zu cost_ms=%.3f",
                pointcloud.size(), dense_points.size(), filtered_points.size(), filter_cost_ms);
        if (parameters->space_downsample) {
            RCLCPP_INFO_THROTTLE(
                    logger, clock, 1000,
                    "[DEBUG-splio-a4f2] preprocess before downsample filtered=%zu leaf=%.3f",
                    filtered_points.size(), parameters->space_downsample_leaf_size);
            downsampler.voxelgrid_sampling(filtered_points, processed_pointcloud, parameters->space_downsample_leaf_size);
        } else {
            processed_pointcloud = std::move(filtered_points);
        }
        const auto downsample_cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                                std::chrono::steady_clock::now() - start_time)
                                                .count() /
                                        1000.0;
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess after downsample processed=%zu elapsed_ms=%.3f",
                processed_pointcloud.size(), downsample_cost_ms);
        sort(dense_points.begin(), dense_points.end(),
             [](const auto &x, const auto &y) {
                 return x.timestamp < y.timestamp;
             });
        sort(processed_pointcloud.begin(), processed_pointcloud.end(),
             [](const auto &x, const auto &y) {
                 return x.timestamp < y.timestamp;
             });
        const auto sort_cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - start_time)
                                          .count() /
                                  1000.0;
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess after sort dense=%zu processed=%zu elapsed_ms=%.3f",
                dense_points.size(), processed_pointcloud.size(), sort_cost_ms);
        if (!dense_points.empty()) {
            last_timestamp_dense_point = dense_points.back().timestamp;
            dense_point_deque.insert(dense_point_deque.end(), dense_points.begin(), dense_points.end());
        }
        if (!processed_pointcloud.empty()) {
            last_timestamp_lidar = processed_pointcloud.back().timestamp;
            point_deque.insert(point_deque.end(), processed_pointcloud.begin(), processed_pointcloud.end());
        }
        const auto total_cost_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - start_time)
                                           .count() /
                                   1000.0;
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess end point_queue=%zu dense_queue=%zu imu_queue=%zu total_ms=%.3f",
                point_deque.size(), dense_point_deque.size(), imu_deque.size(), total_cost_ms);
    }

    void Preprocess::on_imu_callback(const common::ImuMsg &imu_msg) {
        static const auto logger = rclcpp::get_logger("small_point_lio");
        static rclcpp::Clock clock(RCL_STEADY_TIME);
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess imu begin stamp=%.9f last=%.9f imu_queue=%zu",
                imu_msg.timestamp, last_timestamp_imu, imu_deque.size());
        if (imu_msg.timestamp < last_timestamp_imu) {
            RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "imu loop back");
            return;
        }
        imu_deque.emplace_back(imu_msg);
        last_timestamp_imu = imu_msg.timestamp;
        RCLCPP_INFO_THROTTLE(
                logger, clock, 1000,
                "[DEBUG-splio-a4f2] preprocess imu end stamp=%.9f imu_queue=%zu",
                imu_msg.timestamp, imu_deque.size());
    }

}// namespace small_point_lio
