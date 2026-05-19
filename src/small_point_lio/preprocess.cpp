/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#include "preprocess.h"
#include "parameters.h"

namespace small_point_lio {

    void Preprocess::reset() {
        imu_deque.clear();
        point_deque.clear();
        last_timestamp_lidar = -1;
        last_timestamp_imu = -1;
    }

    void Preprocess::on_point_cloud_callback(const std::vector<common::Point> &pointcloud) {
        filtered_points.clear();
        filtered_points.reserve(pointcloud.size());
        for (size_t i = 0; i < pointcloud.size(); i++) {
            const auto &point = pointcloud[i];
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
        if (parameters->space_downsample) {
            downsampler.voxelgrid_sampling(filtered_points, processed_pointcloud, parameters->space_downsample_leaf_size);
        } else {
            processed_pointcloud = std::move(filtered_points);
        }
        sort(processed_pointcloud.begin(), processed_pointcloud.end(),
             [](const auto &x, const auto &y) {
                 return x.timestamp < y.timestamp;
             });
        if (!processed_pointcloud.empty()) {
            last_timestamp_lidar = processed_pointcloud.back().timestamp;
            point_deque.insert(point_deque.end(), processed_pointcloud.begin(), processed_pointcloud.end());
        }
    }

    void Preprocess::on_imu_callback(const common::ImuMsg &imu_msg) {
        if (imu_msg.timestamp < last_timestamp_imu) {
            RCLCPP_ERROR(rclcpp::get_logger("small_point_lio"), "imu loop back");
            return;
        }
        imu_deque.emplace_back(imu_msg);
        last_timestamp_imu = imu_msg.timestamp;
    }

}// namespace small_point_lio
