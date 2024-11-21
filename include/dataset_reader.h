/* Modify from LI_Calib: An Open Platform for LiDAR-IMU Calibration 
   https://github.com/APRIL-ZJU/lidar_IMU_calib/tree/master 
*/

/*
 * LI_Calib: An Open Platform for LiDAR-IMU Calibration
 * Copyright (C) 2020 Jiajun Lv
 * Copyright (C) 2020 Kewei Hu
 * Copyright (C) 2020 Jinhong Xu
 * Copyright (C) 2020 LI_Calib Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef DATASET_READER_H
#define DATASET_READER_H

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#include <tf2_msgs/TFMessage.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <fstream>
#include <Eigen/Dense>

class LioDataset {
private:
  std::shared_ptr<rosbag::Bag> bag_;
  std::vector<sensor_msgs::Imu::ConstPtr> imu_data_;
  std::vector<sensor_msgs::PointCloud2::ConstPtr> scan_data_;
  std::vector<nav_msgs::Odometry::ConstPtr> gt_data_;
  std::vector<double> scan_timestamps_;

  double start_time_;
  double end_time_;

  Eigen::Matrix3d R_I_L_;
  Eigen::Vector3d t_I_L_;

public:
  LioDataset() {}

  bool read(const std::string path,
            const std::string imu_topic,
            const std::string lidar_topic,
            const std::string gt_topic,
            const std::string tf_topic,
            const double bag_start = -1.0,
            const double bag_durr = -1.0) {
    bag_.reset(new rosbag::Bag);
    bag_->open(path, rosbag::bagmode::Read);

    rosbag::View view;
    {
      std::vector<std::string> topics;
      topics.push_back(imu_topic);
      topics.push_back(lidar_topic);
      topics.push_back(gt_topic);
      topics.push_back(tf_topic);

      rosbag::View view_full;
      view_full.addQuery(*bag_);
      ros::Time time_init = view_full.getBeginTime();
      time_init += ros::Duration(bag_start);
      ros::Time time_finish = (bag_durr < 0)?
                              view_full.getEndTime() : time_init + ros::Duration(bag_durr);
      view.addQuery(*bag_, rosbag::TopicQuery(topics), time_init, time_finish);
    }

    for (rosbag::MessageInstance const m : view) {
      const std::string &topic = m.getTopic();

      if (lidar_topic == topic) {
        sensor_msgs::PointCloud2::ConstPtr scan_msg =
                m.instantiate<sensor_msgs::PointCloud2>();
        double timestamp = scan_msg->header.stamp.toSec();
        scan_data_.emplace_back(scan_msg);
        scan_timestamps_.emplace_back(timestamp);
      }

      if (imu_topic == topic) {
        sensor_msgs::Imu::ConstPtr  imu_msg = m.instantiate<sensor_msgs::Imu>();
        imu_data_.push_back(imu_msg);
      }

      if (gt_topic == topic) {
        nav_msgs::Odometry::ConstPtr gt_msg = m.instantiate<nav_msgs::Odometry>();
        gt_data_.push_back(gt_msg);
      }

      if (tf_topic == topic) {
        tf2_msgs::TFMessage::ConstPtr tf_msg = m.instantiate<tf2_msgs::TFMessage>();
        if (tf_msg) {
          for (const auto& transform : tf_msg->transforms) {
            if(transform.header.frame_id == "base_link" && 
                transform.child_frame_id == "lidar_top") {
              R_I_L_ = Eigen::Quaterniond(transform.transform.rotation.w, 
                                          transform.transform.rotation.x, 
                                          transform.transform.rotation.y, 
                                          transform.transform.rotation.z).toRotationMatrix();
              t_I_L_ = Eigen::Vector3d(transform.transform.translation.x, 
                                        transform.transform.translation.y, 
                                        transform.transform.translation.z);
            }
          }
        }
      }
    }

    std::cout << "Bag Info: " << std::endl;
    std::cout << "lidar size: " << scan_data_.size() << std::endl;
    std::cout << "imu size: " << imu_data_.size() << std::endl;
    std::cout << "ego_poses size: " << gt_data_.size() << std::endl;
    std::cout << "R_I_L: \n" << R_I_L_ << std::endl << "t_I_L: " << t_I_L_.transpose() << std::endl;

    return scan_data_.size() && imu_data_.size() && gt_data_.size();
  }

  double get_start_time() const {
    return start_time_;
  }

  double get_end_time() const {
    return end_time_;
  }

  const std::vector<double> get_scan_timestamps() const {
    return scan_timestamps_;
  }

  const std::vector<sensor_msgs::Imu::ConstPtr> get_imu_data() const {
    return imu_data_;
  }
  const std::vector<sensor_msgs::PointCloud2::ConstPtr> get_scan_data() const {
    return scan_data_;
  }
  const std::vector<nav_msgs::Odometry::ConstPtr> get_gt_data() const {
    return gt_data_;
  }

  const Eigen::Matrix3d get_R_I_L() const {
    return R_I_L_;
  }

  const Eigen::Vector3d get_t_I_L() const {
    return t_I_L_;
  }
};

#endif // DATASET_READER_H
