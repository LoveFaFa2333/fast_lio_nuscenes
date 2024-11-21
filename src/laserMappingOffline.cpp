// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <filesystem>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <std_srvs/Trigger.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/TransformStamped.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include "dataset_reader.h"

#include <nlohmann/json.hpp>
namespace json = nlohmann;

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

// new
struct PoseWithStamp
{
    PoseWithStamp(const double &timestamp, const Eigen::Quaterniond &q, const Eigen::Vector3d &t) 
        : timestamp(timestamp), q(q), t(t) {}
    double timestamp = 0.0;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
};

std::string nuscenes_path;
std::string nuscenes_version;
std::string pose_save_path;
std::string bag_path;

std::map<double, PoseWithStamp> gt_pose_buf, lio_pose_buf;
std::map<double, PointCloudXYZI> feats_undistort_buff;
std::map<double, PointCloudXYZI> feats_down_body_buff;
std::map<double, json::json> ego_pose_json_map;
Eigen::Quaterniond q_origin;
Eigen::Vector3d t_origin = Eigen::Vector3d::Zero();

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, gt_topic, tf_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type, N_SCANS, time_unit, SCAN_RATE, point_filter_num;
bool feature_enabled;
double blind;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

std::shared_ptr<KD_TREE<PointType>> ikdtree = std::make_shared<KD_TREE<PointType>>();

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
std::shared_ptr<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf = std::make_shared<esekfom::esekf<state_ikfom, 12, input_ikfom>>();
state_ikfom state_point;
std::vector<state_ikfom> state_point_buff;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

json::json read_json_file(const std::string& json_file){
    std::ifstream file(json_file);
    if (!file.is_open()) {
        std::string errMsg = string("Unable to open ") + json_file;
        throw std::runtime_error(errMsg);
    }
    json::json newJson;
    file >> newJson;
    return newJson;
}

void save_as_tum_pose(std::ofstream &fout, const double &timestamp, const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    fout << std::to_string(timestamp) << " " 
         << t[0] << " "
         << t[1] << " "
         << t[2] << " "
         << q.x() << " "
         << q.y() << " "
         << q.z() << " "
         << q.w() << std::endl;
}

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree->acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree->Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void nuscenes_gt_cbk(const nav_msgs::Odometry::ConstPtr& msg) {
    double timestamp = msg->header.stamp.toSec();
    timestamp = std::round(timestamp * 1e6) / 1e6;                          // 保留6位小数多余部分四舍五入

    Eigen::Quaterniond q(msg->pose.pose.orientation.w, 
                         msg->pose.pose.orientation.x, 
                         msg->pose.pose.orientation.y, 
                         msg->pose.pose.orientation.z);
    Eigen::Vector3d t(msg->pose.pose.position.x, 
                      msg->pose.pose.position.y, 
                      msg->pose.pose.position.z);
    PoseWithStamp pose(timestamp, q, t);
    gt_pose_buf.emplace(timestamp, pose);
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    double timestamp = msg->header.stamp.toSec();
    timestamp = std::round(timestamp * 1e6) / 1e6;                          // 保留6位小数多余部分四舍五入

    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (timestamp < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(timestamp);
    last_timestamp_lidar = timestamp;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    double timestamp = msg->header.stamp.toSec();
    timestamp = std::round(timestamp * 1e6) / 1e6;                          // 保留6位小数多余部分四舍五入
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM && lidar_type == NuScenesVELO32)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    // if (last_timestamp_imu < lidar_end_time)
    // {
    //     ROS_ERROR("last_timestamp_imu < lidar_end_time.");
    //     return false;
    // }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree->Add_Points(PointToAdd, true);
    ikdtree->Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "odom";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "base_link";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "odom";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "odom";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "odom";
    odomAftMapped.child_frame_id = "base_link";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);   // ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    
    Eigen::Vector3d t_cur(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z);
    Eigen::Quaterniond q_cur(odomAftMapped.pose.pose.orientation.w, 
                             odomAftMapped.pose.pose.orientation.x, 
                             odomAftMapped.pose.pose.orientation.y, 
                             odomAftMapped.pose.pose.orientation.z);

    pubOdomAftMapped.publish(odomAftMapped);
    // auto P = kf->get_P();
    // for (int i = 0; i < 6; i ++)
    // {
    //     int k = i < 3 ? i + 3 : i - 3;
    //     odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
    //     odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
    //     odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
    //     odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
    //     odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
    //     odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    // }

    // static tf::TransformBroadcaster br;
    // tf::Transform                   transform;
    // tf::Quaternion                  q;
    // transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
    //                                 odomAftMapped.pose.pose.position.y, \
    //                                 odomAftMapped.pose.pose.position.z));
    // q.setW(odomAftMapped.pose.pose.orientation.w);
    // q.setX(odomAftMapped.pose.pose.orientation.x);
    // q.setY(odomAftMapped.pose.pose.orientation.y);
    // q.setZ(odomAftMapped.pose.pose.orientation.z);
    // transform.setRotation( q );
    // br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "odom", "base_link" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "odom";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 5 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

bool align_with_gt(const std::string& name_prefix) {
    std::vector<PoseWithStamp> matched_pose_buff, matched_gt_buff;
    // lio pose对齐世界坐标
    auto first_pose = lio_pose_buf.begin();
    auto iter = gt_pose_buf.find(first_pose->first);
    if(iter == gt_pose_buf.end()) {
        ROS_ERROR_STREAM("Cannot find the gt pose of first lidar frame(ts: " << std::to_string(first_pose->first) << ")");
        return false;
    }

    Eigen::Matrix4d T_first_lio = Eigen::Matrix4d::Identity();
    T_first_lio.block<3,3>(0,0) = first_pose->second.q.toRotationMatrix();
    T_first_lio.block<3,1>(0,3) = first_pose->second.t;

    Eigen::Matrix4d T_gt = Eigen::Matrix4d::Identity();
    T_gt.block<3,3>(0,0) = iter->second.q.toRotationMatrix();
    T_gt.block<3,1>(0,3) = iter->second.t;

    Eigen::Matrix4d T_origin = T_first_lio.inverse() * T_gt;
    std::cout << "origin pose t: " << T_origin.block<3,1>(0,3).transpose() << std::endl;

    // align
    for(auto &pose : lio_pose_buf){
        Eigen::Matrix4d T_cur = Eigen::Matrix4d::Identity(); 
        T_cur.block<3,3>(0,0) = pose.second.q.toRotationMatrix();
        T_cur.block<3,1>(0,3) = pose.second.t;
        T_cur = T_origin * T_cur;
        
        pose.second.q = Eigen::Quaterniond(T_cur.block<3,3>(0,0));
        pose.second.t = T_cur.block<3,1>(0,3);
        pose.second.t[2] = 0.0;
    }

    time_buffer.clear();
    for(auto pose : lio_pose_buf) {
        auto iter = gt_pose_buf.find(pose.first);
        if(iter == gt_pose_buf.end()) {
            ROS_ERROR_STREAM("Cannot find the gt pose of lidar frame(ts: " << std::to_string(pose.first) << ")");
            return false;
        }

        // z = 0.0
        pose.second.t[2] = 0.0;

        time_buffer.push_back(pose.first);
        matched_pose_buff.push_back(pose.second);
        matched_gt_buff.push_back(iter->second);

        // save to statepoint buf used to publish result
        state_ikfom state_point_cur;
        state_point_cur.rot = pose.second.q;
        state_point_cur.pos = pose.second.t;
        state_point_buff.push_back(state_point_cur);
    }

    // save final pose as tum format && evo
    Eigen::Vector3d mean_err_t(0, 0, 0);
    Eigen::Vector3d max_err_t(0, 0, 0);
    Eigen::Vector3d min_err_t(10000000, 10000000, 10000000);    
    Eigen::Vector3d mean_err_euler(0, 0, 0);
    Eigen::Vector3d max_err_euler(0, 0, 0);
    Eigen::Vector3d min_err_euler(10000000, 10000000, 10000000); 
    std::ofstream lio_poses_tum(pose_save_path + name_prefix + "_lio_poses_tum.txt", std::ios::trunc);
    std::ofstream gt_poses_tum(pose_save_path + name_prefix + "_gt_poses_tum.txt", std::ios::trunc);
    std::ofstream eval_res(pose_save_path + name_prefix + "_traj_eval_res.txt", std::ios::trunc);

    auto update_max = [] (Eigen::Vector3d &max, const Eigen::Vector3d &v) {
        max[0] = std::max(max[0], v[0]);
        max[1] = std::max(max[1], v[1]);
        max[2] = std::max(max[2], v[2]);
    };

    auto update_min = [] (Eigen::Vector3d &min, const Eigen::Vector3d &v) {
        min[0] = std::min(min[0], v[0]);
        min[1] = std::min(min[1], v[1]);
        min[2] = std::min(min[2], v[2]);
    };

    int buff_size = (int)matched_gt_buff.size();
    for(int i = 0; i < buff_size; i++) {
        save_as_tum_pose(gt_poses_tum, time_buffer[i], matched_gt_buff[i].q, matched_gt_buff[i].t);
        save_as_tum_pose(lio_poses_tum, time_buffer[i], matched_pose_buff[i].q, matched_pose_buff[i].t);

        Eigen::Vector3d err_t = (matched_gt_buff[i].t - matched_pose_buff[i].t).cwiseAbs();
        Eigen::Vector3d err_euler = (matched_pose_buff[i].q.inverse() * matched_gt_buff[i].q).matrix().eulerAngles(0,1,2).cwiseAbs(); 

        update_max(max_err_t, err_t);
        update_min(min_err_t, err_t);
        update_max(max_err_euler, err_euler);
        update_min(min_err_euler, err_euler);

        mean_err_t += err_t;
        mean_err_euler += err_euler;
    }

    mean_err_t /= buff_size;
    mean_err_euler /= buff_size;

    const double ratio = 180.0 / M_PI;
    std::cout << "Traj. Evaluate Result: " << std::endl
              << "Translation: " << std::endl
              << "\t mean err(m): total: " << mean_err_t.norm() << ", x: " << mean_err_t[0] << ", y: " << mean_err_t[1]  << ", z: " << mean_err_t[2] << std::endl
              << "\t max err(m) : total: " << max_err_t.norm() << " , x: " << max_err_t[0] << ", y: " << max_err_t[1]  << ", z: " << max_err_t[2] << std::endl
              << "\t min err(m) : total: " << min_err_t.norm() << " , x: " << min_err_t[0] << ", y: " << min_err_t[1]  << ", z: " << min_err_t[2] << std::endl
              << "Rotation: " << std::endl
              << "\t mean err(deg): total: " << mean_err_euler.norm() << ", x: " << mean_err_euler[0] << ", y: " << mean_err_euler[1]  << ", z: " << mean_err_euler[2] << std::endl
              << "\t max err(deg) : total: " << max_err_euler.norm() << " , x: " << max_err_euler[0] << ", y: " << max_err_euler[1]  << ", z: " << max_err_euler[2] << std::endl
              << "\t min err(deg) : total: " << min_err_euler.norm() << " , x: " << min_err_euler[0] << ", y: " << min_err_euler[1]  << ", z: " << min_err_euler[2] << std::endl;

    eval_res  << "Traj. Evaluate Result: " << std::endl
              << "Translation: " << std::endl
              << "\t mean err(m): total: " << mean_err_t.norm() << ", x: " << mean_err_t[0] << ", y: " << mean_err_t[1]  << ", z: " << mean_err_t[2] << std::endl
              << "\t max err(m) : total: " << max_err_t.norm() << " , x: " << max_err_t[0] << ", y: " << max_err_t[1]  << ", z: " << max_err_t[2] << std::endl
              << "\t min err(m) : total: " << min_err_t.norm() << " , x: " << min_err_t[0] << ", y: " << min_err_t[1]  << ", z: " << min_err_t[2] << std::endl
              << "Rotation: " << std::endl
              << "\t mean err(deg): total: " << mean_err_euler.norm() << ", x: " << mean_err_euler[0] << ", y: " << mean_err_euler[1]  << ", z: " << mean_err_euler[2] << std::endl
              << "\t max err(deg) : total: " << max_err_euler.norm() << " , x: " << max_err_euler[0] << ", y: " << max_err_euler[1]  << ", z: " << max_err_euler[2] << std::endl
              << "\t min err(deg) : total: " << min_err_euler.norm() << " , x: " << min_err_euler[0] << ", y: " << min_err_euler[1]  << ", z: " << min_err_euler[2] << std::endl;

    lio_poses_tum.close();
    gt_poses_tum.close();
    eval_res.close();

    // use fast_lio to replace gt
    int miss_counter = 0;
    for(int j = 0; j < time_buffer.size(); j++) {
        double ts_sec = std::round(time_buffer[j] * 1e6) / 1e6;                          // 保留6位小数多余部分四舍五入
        auto lio_pose = matched_pose_buff[j];
        auto iter = ego_pose_json_map.find(ts_sec);
        if(iter != ego_pose_json_map.end()) {
            iter->second["translation"][0] = lio_pose.t[0];
            iter->second["translation"][1] = lio_pose.t[1];
            iter->second["translation"][2] = lio_pose.t[2];

            iter->second["rotation"][0] = lio_pose.q.w();
            iter->second["rotation"][1] = lio_pose.q.x();
            iter->second["rotation"][2] = lio_pose.q.y();
            iter->second["rotation"][3] = lio_pose.q.z();
        } else {
            miss_counter++;
        }
    }
    std::cout << "Missed matching between lio pose and gt number: " << miss_counter<< std::endl;

    return true;
}

void reset_fast_lio() {
    // cloud reset
    _featsArray.reset(new PointCloudXYZI());
    featsFromMap.reset(new PointCloudXYZI());
    feats_undistort.reset(new PointCloudXYZI());
    feats_down_body.reset(new PointCloudXYZI());
    feats_down_world.reset(new PointCloudXYZI());
    laserCloudOri.reset(new PointCloudXYZI(100000, 1));
    corr_normvect.reset(new PointCloudXYZI(100000, 1));

    // reset variables
    lidar_pushed = false; 
    flg_first_scan = true; 
    flg_EKF_inited = false;
    pointSearchInd_surf.clear(); 
    cub_needrm.clear();
    Nearest_Points.clear(); 
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="odom";
    path.poses.clear();
    last_timestamp_lidar = 0; 
    last_timestamp_imu = -1.0;
    lidar_end_time = 0.0;
    first_lidar_time = 0.0;

    // reset kf
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.reset(new esekfom::esekf<state_ikfom, 12, input_ikfom>());
    kf->init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    // reset imu processor
    p_imu.reset(new ImuProcess());
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;

    // reset lidar processor
    p_pre.reset(new Preprocess());
    p_pre->blind = blind;
    p_pre->N_SCANS = N_SCANS;
    p_pre->time_unit = time_unit;
    p_pre->SCAN_RATE = SCAN_RATE;
    p_pre->point_filter_num = point_filter_num;
    p_pre->feature_enabled = feature_enabled;
    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

    // sensors data buff clear
    lidar_buffer.clear();
    time_buffer.clear();
    imu_buffer.clear();
    lio_pose_buf.clear();
    gt_pose_buf.clear();
    state_point_buff.clear();
    feats_undistort_buff.clear();
    feats_down_body_buff.clear();

    // reset ikdtree
    ikdtree.reset(new KD_TREE<PointType>());
}

int main(int argc, char** argv)
{   
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<string>("nuscenes_path",nuscenes_path, "");
    nh.param<string>("nuscenes_version",nuscenes_version, "v1.0-mini");

    bag_path = nuscenes_path + "bag/";
    pose_save_path = nuscenes_path + "fast_lio_poses/";

    if (!std::filesystem::exists(bag_path) || !std::filesystem::is_directory(bag_path)) {
        ROS_ERROR_STREAM("Bag path does not exist: " << bag_path);
        return -1;
    }

    if (!std::filesystem::exists(pose_save_path)) {
        if (std::filesystem::create_directories(pose_save_path)) {
            ROS_INFO("Create fast_lio_poses dir.");
        } else {
           ROS_ERROR_STREAM("Failed to create fast_lio_poses dir: " << pose_save_path);
           return -1;
        }
    } 

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/gt_topic", gt_topic,"/odom");
    nh.param<string>("common/tf_topic", tf_topic,"/tf");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<double>("preprocess/blind", blind, 0.01);
    nh.param<int>("preprocess/scan_line", N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", time_unit, US);
    nh.param<int>("preprocess/scan_rate", SCAN_RATE, 10);
    nh.param<int>("point_filter_num", point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", feature_enabled, false);
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    /*** ROS publisher initialization ***/
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    // read bag files
    std::cout << "Bags list: " << std::endl;
    std::shared_ptr<LioDataset> bag_reader = std::make_shared<LioDataset>();
    std::vector<std::string> bag_files;
    for (const auto& entry : std::filesystem::directory_iterator(bag_path)) {
        if (std::filesystem::is_regular_file(entry.status())) { 
            std::string file_name = entry.path().filename().string();
            bag_files.push_back(file_name);
            std::cout << file_name << std::endl;
        }
    }
    if(bag_files.empty()) {
        ROS_ERROR("No bag exist!");
        return -1;
    }

    // read ego_pose json file for replacing
    std::string ori_ego_pose_file = nuscenes_path + nuscenes_version + "/ego_pose.json";
    json::json ego_pose_jsons = read_json_file(ori_ego_pose_file);
    for (const auto& ego_pose_json : ego_pose_jsons) {
        uint64_t ts_usec = ego_pose_json["timestamp"];    // us
        double ts_sec = ts_usec / 1e6;
        ego_pose_json_map.emplace(ts_sec, ego_pose_json);
    }

    // start to run lio from bags
    for(auto const bag_file : bag_files) {
        std::cout << "--------------------------- Start Run fast_lio from " << bag_file << " ------------------------- " << std::endl;
        // read bag
        bag_reader.reset(new LioDataset());
        if(!bag_reader->read(bag_path + bag_file, imu_topic, lid_topic, gt_topic, tf_topic)) {
            ROS_ERROR_STREAM("Failed to read bag: " << bag_file << " , skip to next bag !");
            continue;
        }
        ROS_INFO("Bag reading succeed.");
        
        // reset fast_lio
        int effect_feat_num = 0, frame_num = 0;
        double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
        bool flg_EKF_converged, EKF_stop_flg = 0;
        reset_fast_lio();
        ROS_INFO("Fast_LIO reset.");
        
        // read extrinsic
        Lidar_R_wrt_IMU = bag_reader->get_R_I_L();
        Lidar_T_wrt_IMU = bag_reader->get_t_I_L();
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        ROS_INFO("Extrinsic between imu and lidar set.");

        // lidar point cloud process
        const std::vector<sensor_msgs::PointCloud2::ConstPtr> lidar_msg_buf = bag_reader->get_scan_data();
        for(auto &msg : lidar_msg_buf) {
            standard_pcl_cbk(msg);
        }

        // imu process
        const std::vector<sensor_msgs::Imu::ConstPtr> imu_msg_buf = bag_reader->get_imu_data();
        for(auto &msg : imu_msg_buf) {
            imu_cbk(msg);
        }

        // gt process
        const std::vector<nav_msgs::Odometry::ConstPtr> gt_msg_buf = bag_reader->get_gt_data();
        for(auto &msg : gt_msg_buf) {
            nuscenes_gt_cbk(msg);
        }
        ROS_INFO("sensors data processing finished.");

        // start to run lio
        ROS_INFO("Start to run lio mapping...");
        signal(SIGINT, SigHandle);
        while (!lidar_buffer.empty() && !imu_buffer.empty() && !gt_pose_buf.empty())
        {   
            if (flg_exit) break;
            if(sync_packages(Measures)) 
            {   
                // cout << "Measures.lidar_beg_time: " << to_string(Measures.lidar_beg_time) << ", imu number: " << Measures.imu.size() << endl;
                if (flg_first_scan)
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    p_imu->first_lidar_time = first_lidar_time;
                    flg_first_scan = false;
                    // continue;
                }

                double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

                match_time = 0;
                kdtree_search_time = 0.0;
                solve_time = 0;
                solve_const_H_time = 0;
                svd_time   = 0;
                t0 = omp_get_wtime();

                if (Measures.lidar->empty())
                {
                    ROS_WARN("Measures.lidar is empty, skip this scan!\n");
                    continue;
                }

                p_imu->Process(Measures, *kf, feats_undistort);
                state_point = kf->get_x();
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

                if (feats_undistort->empty() || (feats_undistort == NULL))
                {
                    ROS_WARN("feats_undistort is empty, skip this scan!\n");
                    continue;
                }

                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                                false : true;
                /*** Segment the map in lidar FOV ***/
                lasermap_fov_segment();

                /*** downsample the feature points in a scan ***/
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                feats_undistort_buff.emplace(lidar_end_time, *feats_undistort);
                feats_down_body_buff.emplace(lidar_end_time, *feats_down_body);

                t1 = omp_get_wtime();
                feats_down_size = feats_down_body->points.size();
                /*** initialize the map kdtree ***/
                if(ikdtree->Root_Node == nullptr)
                {
                    if(feats_down_size > 5)
                    {
                        ikdtree->set_downsample_param(filter_size_map_min);
                        feats_down_world->resize(feats_down_size);
                        for(int i = 0; i < feats_down_size; i++)
                        {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }
                        ikdtree->Build(feats_down_world->points);
                    }
                    // continue;
                }

                int featsFromMapNum = ikdtree->validnum();
                kdtree_size_st = ikdtree->size();
                
                // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

                /*** ICP and iterated Kalman filter update ***/
                if (feats_down_size < 5)
                {
                    ROS_WARN("feats_down_size < 5, skip this scan!\n");
                    continue;
                }
                
                normvec->resize(feats_down_size);
                feats_down_world->resize(feats_down_size);

                V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);

                pointSearchInd_surf.resize(feats_down_size);
                Nearest_Points.resize(feats_down_size);
                int  rematch_num = 0;
                bool nearest_search_en = true; //

                t2 = omp_get_wtime();
                
                /*** iterated state estimation ***/
                double t_update_start = omp_get_wtime();
                double solve_H_time = 0;
                kf->update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
                state_point = kf->get_x();
                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                geoQuat.x = state_point.rot.coeffs()[0];
                geoQuat.y = state_point.rot.coeffs()[1];
                geoQuat.z = state_point.rot.coeffs()[2];
                geoQuat.w = state_point.rot.coeffs()[3];
                Eigen::Vector3d t_cur(state_point.pos);
                Eigen::Quaterniond q_cur(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z);
                // save lio pose
                lio_pose_buf.emplace(lidar_end_time, PoseWithStamp(lidar_end_time, q_cur, t_cur));
                
                double t_update_end = omp_get_wtime();

                /*** add the feature points to map kdtree ***/
                t3 = omp_get_wtime();
                map_incremental();
                t5 = omp_get_wtime();
            }
        }

        /* save lio and gt path of current bag */
        ROS_INFO("Start to aligned lio and gt path.");
        if(!align_with_gt(bag_file)) {
            ROS_ERROR_STREAM("Failed to aligned lio and gt path, skip...");
            continue;
        }

        // publish result
        for(int i = 0; i < state_point_buff.size(); i++) {
            state_point = state_point_buff[i];
            state_point.offset_T_L_I = Lidar_T_wrt_IMU;
            state_point.offset_R_L_I = Lidar_R_wrt_IMU;
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            feats_undistort.reset(new PointCloudXYZI(feats_undistort_buff[time_buffer[i]]));
            feats_down_body.reset(new PointCloudXYZI(feats_down_body_buff[time_buffer[i]]));

            // publisher 
            publish_odometry(pubOdomAftMapped);
            publish_path(pubPath);
            publish_frame_world(pubLaserCloudFull);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            usleep(10000);
        }
    }

    // create ego_pose_fast_lio.json file
    std::string filename = nuscenes_path + nuscenes_version + "/ego_pose_fast_lio.json" ;
    std::ofstream json_file(filename, ios::trunc);

    json::json out_json;
    for (const auto& pose : ego_pose_json_map) {
        out_json.emplace_back(pose.second);
    }
    json_file << out_json.dump(4);  // 4 表示缩进级别
    json_file.close();
    std::cout << "ego_pose_fast_lio.json saved to: "  << filename << std::endl;
    std::cout << "--------------------------- Finished All bags mapping & localization! ------------------------- " << std::endl;
    return 0;
}
