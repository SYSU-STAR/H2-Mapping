#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "estimator.h"
#include "parameters.h"
#include "feature_tracker.h"
#include "utility/visualization.h"

Estimator estimator;
bool start_pub_odom = false;
int pub_num = 0;
std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
queue<cv::Mat> gray_img_buf;
queue<cv::Mat> depth_img_buf;

ros::Publisher pub_restart;
ros::Publisher pub_match;
ros::Publisher pub_odom;
ros::Subscriber sub_img;
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,sensor_msgs::Image> syncPolicy;
std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_color;
std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> sub_depth;
std::shared_ptr<message_filters::Synchronizer<syncPolicy>> synchronizer;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d P_rt;
Eigen::Quaterniond Q_rt;
Eigen::Vector3d V_rt;
Eigen::Vector3d W_rt;
Eigen::Vector3d Ba_rt;
Eigen::Vector3d Bg_rt;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;
double last_rt_imu_t = 0;

std::fstream imu_pose_op_file;

struct Measurement
{
  std::vector<sensor_msgs::ImuConstPtr> imu_vec;
  sensor_msgs::PointCloudConstPtr feature;
  cv::Mat gray_img;
  // cv::Mat depth_img;

  Measurement(const std::vector<sensor_msgs::ImuConstPtr>& _imu_vec,
              const sensor_msgs::PointCloudConstPtr& _feature,
              const cv::Mat& _gray_img)
  : imu_vec(_imu_vec), feature(_feature), gray_img(_gray_img) {}
};

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
  double t = imu_msg->header.stamp.toSec();
  if (init_imu) {
    latest_time = t;
    init_imu = 0;
    P_rt.setZero();
    V_rt.setZero();
    Q_rt.setIdentity();
    Ba_rt.setZero();
    Bg_rt.setZero();
    acc_0 = Eigen::Vector3d(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    gyr_0 = Eigen::Vector3d(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    return;
  }
  double dt = t - latest_time;
  latest_time = t;

  double dx = imu_msg->linear_acceleration.x;
  double dy = imu_msg->linear_acceleration.y;
  double dz = imu_msg->linear_acceleration.z;
  Eigen::Vector3d acc_1{dx, dy, dz};

  double rx = imu_msg->angular_velocity.x;
  double ry = imu_msg->angular_velocity.y;
  double rz = imu_msg->angular_velocity.z;
  Eigen::Vector3d gyr_1{rx, ry, rz};

  Eigen::Vector3d un_acc_0 = Q_rt * (acc_0 - Ba_rt) - estimator.g;
  Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - Bg_rt;
  Q_rt = Q_rt * Utility::deltaQ(un_gyr * dt);
  Eigen::Vector3d un_acc_1 = Q_rt * (acc_1 - Ba_rt) - estimator.g;
  Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
  P_rt = P_rt + dt * V_rt + 0.5 * dt * dt * un_acc;
  V_rt = V_rt + dt * un_acc;
  W_rt = gyr_1 - Bg_rt;

  acc_0 = acc_1;
  gyr_0 = gyr_1;
}

void update()
{
  TicToc t_predict;
  latest_time = current_time;
  P_rt = estimator.Ps[WINDOW_SIZE];
  Q_rt = estimator.Rs[WINDOW_SIZE];
  V_rt = estimator.Vs[WINDOW_SIZE];
  Ba_rt = estimator.Bas[WINDOW_SIZE];
  Bg_rt = estimator.Bgs[WINDOW_SIZE];
  acc_0 = estimator.acc_0;
  gyr_0 = estimator.gyr_0;

  queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
  for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
    predict(tmp_imu_buf.front());

}

void saveFeatureMono(const cv::Mat& gray_img, const std_msgs::Header& header) 
{
  pub_count++;
  sensor_msgs::PointCloudPtr feature_msg(new sensor_msgs::PointCloud);
  sensor_msgs::ChannelFloat32 id_of_point;
  sensor_msgs::ChannelFloat32 u_of_point;
  sensor_msgs::ChannelFloat32 v_of_point;
  sensor_msgs::ChannelFloat32 velocity_x_of_point;
  sensor_msgs::ChannelFloat32 velocity_y_of_point;

  feature_msg->header = header;
  feature_msg->header.frame_id = "world";

  vector<set<int>> hash_ids(NUM_OF_CAM);
  for (int i = 0; i < NUM_OF_CAM; i++) {
    auto &un_pts = trackerData[i].cur_un_pts;
    auto &cur_pts = trackerData[i].cur_pts;
    auto &ids = trackerData[i].ids;
    auto &pts_velocity = trackerData[i].pts_velocity;
    for (unsigned int j = 0; j < ids.size(); j++) {
      if (trackerData[i].track_cnt[j] > 1) {
        int p_id = ids[j];
        hash_ids[i].insert(p_id);
        geometry_msgs::Point32 p;
        p.x = un_pts[j].x;
        p.y = un_pts[j].y;
        p.z = 1;

        feature_msg->points.push_back(p);
        id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
        u_of_point.values.push_back(cur_pts[j].x);
        v_of_point.values.push_back(cur_pts[j].y);
        velocity_x_of_point.values.push_back(pts_velocity[j].x);
        velocity_y_of_point.values.push_back(pts_velocity[j].y);
      }
    }
  }
  feature_msg->channels.push_back(id_of_point);
  feature_msg->channels.push_back(u_of_point);
  feature_msg->channels.push_back(v_of_point);
  feature_msg->channels.push_back(velocity_x_of_point);
  feature_msg->channels.push_back(velocity_y_of_point);
  ROS_DEBUG("publish %f, at %f", feature_msg->header.stamp.toSec(), ros::Time::now().toSec());
  // skip the first image; since no optical speed on frist image
  if (!init_pub) {
    init_pub = 1;
  }
  else {
    m_buf.lock();
    feature_buf.push(feature_msg);
    gray_img_buf.push(gray_img);
    m_buf.unlock();
    con.notify_one();
  }
}

void saveFeatureRGBD(const cv::Mat& gray_img, const cv::Mat& depth_img, const std_msgs::Header& header) 
{
  pub_count++;
  sensor_msgs::PointCloudPtr feature_msg(new sensor_msgs::PointCloud);
  sensor_msgs::ChannelFloat32 id_of_point;
  sensor_msgs::ChannelFloat32 u_of_point;
  sensor_msgs::ChannelFloat32 v_of_point;
  sensor_msgs::ChannelFloat32 velocity_x_of_point;
  sensor_msgs::ChannelFloat32 velocity_y_of_point;
  sensor_msgs::ChannelFloat32 depth_of_point;

  feature_msg->header = header;
  feature_msg->header.frame_id = "world";

  vector<set<int>> hash_ids(NUM_OF_CAM);
  for (int i = 0; i < NUM_OF_CAM; i++) {
    auto &un_pts = trackerData[i].cur_un_pts;
    auto &cur_pts = trackerData[i].cur_pts;
    auto &ids = trackerData[i].ids;
    auto &pts_velocity = trackerData[i].pts_velocity;
    for (unsigned int j = 0; j < ids.size(); j++) {
      if (trackerData[i].track_cnt[j] > 1) {
        int p_id = ids[j];
        hash_ids[i].insert(p_id);
        geometry_msgs::Point32 p;
        p.x = un_pts[j].x;
        p.y = un_pts[j].y;
        p.z = 1;

        feature_msg->points.push_back(p);
        id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
        u_of_point.values.push_back(cur_pts[j].x);
        v_of_point.values.push_back(cur_pts[j].y);
        velocity_x_of_point.values.push_back(pts_velocity[j].x);
        velocity_y_of_point.values.push_back(pts_velocity[j].y);
        depth_of_point.values.push_back((float)depth_img.at<unsigned short>(round(cur_pts[j].y), round(cur_pts[j].x))/1000.0f);
      }
    }
  }
  feature_msg->channels.push_back(id_of_point);
  feature_msg->channels.push_back(u_of_point);
  feature_msg->channels.push_back(v_of_point);
  feature_msg->channels.push_back(velocity_x_of_point);
  feature_msg->channels.push_back(velocity_y_of_point);
  feature_msg->channels.push_back(depth_of_point);
  ROS_DEBUG("publish %f, at %f", feature_msg->header.stamp.toSec(), ros::Time::now().toSec());
  // skip the first image; since no optical speed on frist image
  if (!init_pub) {
    init_pub = 1;
  }
  else {
    m_buf.lock();
    feature_buf.push(feature_msg);
    gray_img_buf.push(gray_img);
    // depth_img_buf.push(depth_img);
    m_buf.unlock();
    con.notify_one();
  }
}


// std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
std::vector<Measurement>
getMeasurements()
{
  // std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
  std::vector<Measurement> measurements;
  while (true) {
    // printf("getMeasurements\n");
    if (imu_buf.empty() || feature_buf.empty()) {
      return measurements;
    }
      
    if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td)) {
      sum_of_wait++;
      // ROS_INFO("Waiting imu");
      return measurements;
    }

    if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td)) {
      ROS_WARN("throw img, only should happen at the beginning");
      feature_buf.pop();
      gray_img_buf.pop();
      continue;
    }
    sensor_msgs::PointCloudConstPtr feature_msg = feature_buf.front();
    cv::Mat gray_img = gray_img_buf.front();
    feature_buf.pop();
    gray_img_buf.pop();

    std::vector<sensor_msgs::ImuConstPtr> imu_vec;
    while (imu_buf.front()->header.stamp.toSec() < feature_msg->header.stamp.toSec() + estimator.td) {
      imu_vec.emplace_back(imu_buf.front());
      imu_buf.pop();
    }
    imu_vec.emplace_back(imu_buf.front());

    if (imu_vec.empty())
      ROS_WARN("no imu between two image");

    // ROS_WARN("Push buffer");
    // measurements.emplace_back(imu_vec, feature_msg);
    measurements.emplace_back(Measurement(imu_vec, feature_msg, gray_img));
  }
  return measurements;
}

void imuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
  // printf("[imuCallback] time %lf\n", imu_msg->header.stamp.toSec());
  if (imu_msg->header.stamp.toSec() <= last_imu_t) {
    ROS_WARN("imu message in disorder!");
    return;
  }
  // printf("[imuCallback] delta time %lf\n", imu_msg->header.stamp.toSec() - last_imu_t);
  // printf("[imuCallback] rt delta time %lf\n", ros::Time::now().toSec() - last_rt_imu_t);
  last_imu_t = imu_msg->header.stamp.toSec();
  last_rt_imu_t = ros::Time::now().toSec();
  
  m_buf.lock();
  imu_buf.push(imu_msg);
  m_buf.unlock();
  con.notify_one();

  {
    std::lock_guard<std::mutex> lg(m_state);
    predict(imu_msg);
    std_msgs::Header header = imu_msg->header;
    header.frame_id = "world";
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) {
      pubLatestOdometry(P_rt, Q_rt, V_rt, W_rt, header);
      imu_pose_op_file << std::setprecision(20) << (uint64_t)std::round(imu_msg->header.stamp.toSec() * 1e6)
                       << std::setprecision(7) << " " << P_rt.x() << " " << P_rt.y() << " " << P_rt.z()
                       << " " << Q_rt.x() << " " << Q_rt.y() << " " << Q_rt.z() << " " << Q_rt.w()
                       << " " << V_rt.x() << " " << V_rt.y() << " " << V_rt.z() << std::endl;
      pub_num++;
      // std::cout << "XYZ: " << P_rt.transpose() << " YPR: " << Utility::R2ypr(Q_rt.toRotationMatrix()).transpose() << std::endl;

      if(pub_num > 200)
        start_pub_odom = true;
    } 
  }
}

void rgbCallback(const sensor_msgs::ImageConstPtr &img_msg)
{
  if(first_image_flag) {
    first_image_flag = false;
    first_image_time = img_msg->header.stamp.toSec();
    last_image_time = img_msg->header.stamp.toSec();
    return;
  }
  // detect unstable camera stream
  if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time) {
    ROS_WARN("image discontinue! reset the feature tracker!");
    first_image_flag = true; 
    last_image_time = 0;
    pub_count = 1;

    m_buf.lock();
    while(!feature_buf.empty())
      feature_buf.pop();
    while(!imu_buf.empty())
      imu_buf.pop();
    while(!gray_img_buf.empty())
      gray_img_buf.pop();
    while(!depth_img_buf.empty())
      depth_img_buf.pop();
    m_buf.unlock();

    m_estimator.lock();
    estimator.clearState();
    estimator.setParameter();
    m_estimator.unlock();
    current_time = -1;
    last_imu_t = 0;
    // std_msgs::Bool restart_flag;
    // restart_flag.data = true;
    // pub_restart.publish(restart_flag);
    return;
  }
  last_image_time = img_msg->header.stamp.toSec();
  // frequency control
  if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ) {
    PUB_THIS_FRAME = true;
    // reset the frequency control
    if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ) {
      first_image_time = img_msg->header.stamp.toSec();
      pub_count = 0;
    }
  }
  else
    PUB_THIS_FRAME = false;

  cv_bridge::CvImageConstPtr gray_img_ptr;
  if (img_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = img_msg->header;
    img.height = img_msg->height;
    img.width = img_msg->width;
    img.is_bigendian = img_msg->is_bigendian;
    img.step = img_msg->step;
    img.data = img_msg->data;
    img.encoding = "mono8";
    gray_img_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  }
  else
    gray_img_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

  TicToc t_r;
  for (int i = 0; i < NUM_OF_CAM; i++) {
    trackerData[i].trackMono(gray_img_ptr->image, img_msg->header.stamp.toSec());
  }

  for (unsigned int i = 0;; i++) {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      if (j != 1 || !STEREO_TRACK)
        completed |= trackerData[j].updateID(i);
    if (!completed)
      break;
  }

  if (PUB_THIS_FRAME) {
    saveFeatureMono(gray_img_ptr->image, img_msg->header);
    cv::Mat show_img = gray_img_ptr->image;
    if (SHOW_TRACK) {
      gray_img_ptr = cv_bridge::cvtColor(gray_img_ptr, sensor_msgs::image_encodings::BGR8);
      cv::Mat stereo_img = gray_img_ptr->image;
      for (int i = 0; i < NUM_OF_CAM; i++) {
        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
        cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);
        for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) {
          double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
          cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), std::round(COL/300) + 1);
        }
      }
      pub_match.publish(gray_img_ptr->toImageMsg());
    }
  }
  // ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

void rgbdCallback(const sensor_msgs::ImageConstPtr &color_msg, const sensor_msgs::ImageConstPtr &depth_msg)
{
  // printf("rgbdCallback %lf\n", color_msg->header.stamp.toSec());
  if(first_image_flag) {
    first_image_flag = false;
    first_image_time = color_msg->header.stamp.toSec();
    last_image_time = color_msg->header.stamp.toSec();
    ROS_WARN("First_flag");
    return;
  }
  // detect unstable camera stream
  if (color_msg->header.stamp.toSec() - last_image_time > 1.0 || color_msg->header.stamp.toSec() < last_image_time) {
    ROS_WARN("image discontinue! reset the feature tracker!");
    first_image_flag = true;
    last_image_time = 0;
    pub_count = 1;
    // std_msgs::Bool restart_flag;
    // restart_flag.data = true;

    m_buf.lock();
    while(!feature_buf.empty())
      feature_buf.pop();
    while(!imu_buf.empty())
      imu_buf.pop();
    while(!gray_img_buf.empty())
      gray_img_buf.pop();
    while(!depth_img_buf.empty())
      depth_img_buf.pop();
    m_buf.unlock();

    m_estimator.lock();
    estimator.clearState();
    estimator.setParameter();
    m_estimator.unlock();
    current_time = -1;
    last_imu_t = 0;

    // pub_restart.publish(restart_flag);
    return;
  }
  last_image_time = color_msg->header.stamp.toSec();
  // frequency control
  if (round(1.0 * pub_count / (color_msg->header.stamp.toSec() - first_image_time)) <= FREQ) {
    PUB_THIS_FRAME = true;
    // reset the frequency control
    if (abs(1.0 * pub_count / (color_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ) {
      first_image_time = color_msg->header.stamp.toSec();
      pub_count = 0;
    }
  }
  else
    PUB_THIS_FRAME = false;

  // encodings in ros: http://docs.ros.org/diamondback/api/sensor_msgs/html/image__encodings_8cpp_source.html
  //color has encoding RGB8
  cv_bridge::CvImageConstPtr gray_img_ptr;
  if (color_msg->encoding == "8UC1") {
    sensor_msgs::Image img;
    img.header = color_msg->header;
    img.height = color_msg->height;
    img.width = color_msg->width;
    img.is_bigendian = color_msg->is_bigendian;
    img.step = color_msg->step;
    img.data = color_msg->data;
    img.encoding = "mono8";
    gray_img_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
  }
  else
    gray_img_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::MONO8);

  //depth has encoding TYPE_16UC1
  cv_bridge::CvImageConstPtr depth_img_ptr;
  sensor_msgs::Image img;
  img.header = depth_msg->header;
  img.height = depth_msg->height;
  img.width = depth_msg->width;
  img.is_bigendian = depth_msg->is_bigendian;
  img.step = depth_msg->step;
  img.data = depth_msg->data;
  img.encoding = sensor_msgs::image_encodings::MONO16;
  depth_img_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO16);

  TicToc t_r;
  for (int i = 0; i < NUM_OF_CAM; i++) {
    trackerData[i].trackMono(gray_img_ptr->image, color_msg->header.stamp.toSec());
  }
  // ROS_INFO("feature tracker latency: %lf\n", t_r.toc());
  for (unsigned int i = 0;; i++) {
    bool completed = false;
    for (int j = 0; j < NUM_OF_CAM; j++)
      if (j != 1 || !STEREO_TRACK)
        completed |= trackerData[j].updateID(i);
    if (!completed)
      break;
  }

  // {
  //   std::lock_guard<std::mutex> lg(m_state);
  //   if(estimator.solver_flag == Estimator::NON_LINEAR && start_pub_odom) {
  //       Eigen::Matrix4d Twc_rt{Eigen::Matrix4d::Identity()}, Tic{Eigen::Matrix4d::Identity()};
  //       Eigen::Matrix4d Twb_rt{Eigen::Matrix4d::Identity()};
  //       Twb_rt.block<3, 3>(0, 0) = Q_rt.toRotationMatrix();
  //       Twb_rt.block<3, 1>(0, 3) = P_rt;
  //       Tic.block<3, 3>(0, 0) = RIC[0];
  //       Tic.block<3, 1>(0, 3) = TIC[0];
  //       Twc_rt = Twb_rt * Tic;
  //       Eigen::Matrix4d Tcb_odometry;
  //       Tcb_odometry << -0.0158859644997, -0.0191193244826, 0.99969099604, 0.141183173271,
  //                       -0.999836576065, -0.00832461922041, -0.0160474881863, 0.0472184818348,
  //                       0.00862886401387, -0.999782552431, -0.0189839553219, -0.10256232772,
  //                       0, 0, 0, 1;
  //       Eigen::Matrix4d T_final = Twc_rt * Tcb_odometry;
  //       Eigen::Quaterniond Q_final(T_final.block<3, 3>(0, 0));
  //       Eigen::Vector3d P_final = T_final.block<3, 1>(0, 3);
  //       nav_msgs::Odometry final_odom_msg;
  //       final_odom_msg.header = color_msg->header;
  //       final_odom_msg.pose.pose.orientation.w = Q_final.w();
  //       final_odom_msg.pose.pose.orientation.x = Q_final.x();
  //       final_odom_msg.pose.pose.orientation.y = Q_final.y();
  //       final_odom_msg.pose.pose.orientation.z = Q_final.z();
  //       final_odom_msg.pose.pose.position.x = P_final.x();
  //       final_odom_msg.pose.pose.position.y = P_final.y();
  //       final_odom_msg.pose.pose.position.z = P_final.z();
  //       pub_odom.publish(final_odom_msg);    
  //   }

  // }
  // Eigen::Matrix4d Twc_rt{Eigen::Matrix4d::Identity()}, Tic{Eigen::Matrix4d::Identity()};
  // Eigen::Matrix4d Twb_rt{Eigen::Matrix4d::Identity()};
  // Twb_rt.block<3, 3>(0, 0) = Q_rt.toRotationMatrix();
  // Twb_rt.block<3, 1>(0, 3) = P_rt;
  // Tic.block<3, 3>(0, 0) = RIC[0];
  // Tic.block<3, 1>(0, 3) = TIC[0];
  // Twc_rt = Twb_rt * Tic;
  // Eigen::Matrix4d Tcb_odometry;
  // Tcb_odometry << -0.0158859644997, -0.0191193244826, 0.99969099604, 0.141183173271,
  //                 -0.999836576065, -0.00832461922041, -0.0160474881863, 0.0472184818348,
  //                 0.00862886401387, -0.999782552431, -0.0189839553219, -0.10256232772,
  //                 0, 0, 0, 1;
  // Eigen::Matrix4d T_final = Twc_rt * Tcb_odometry;
  // Eigen::Quaterniond Q_final(T_final.block<3, 3>(0, 0));
  // Eigen::Vector3d P_final = T_final.block<3, 1>(0, 3);
  // nav_msgs::Odometry final_odom_msg;
  // final_odom_msg.header = color_msg->header;
  // final_odom_msg.pose.pose.orientation.w = Q_final.w();
  // final_odom_msg.pose.pose.orientation.x = Q_final.x();
  // final_odom_msg.pose.pose.orientation.y = Q_final.y();
  // final_odom_msg.pose.pose.orientation.z = Q_final.z();
  // final_odom_msg.pose.pose.position.x = P_final.x();
  // final_odom_msg.pose.pose.position.y = P_final.y();
  // final_odom_msg.pose.pose.position.z = P_final.z();
  // pub_odom.publish(final_odom_msg);

  if (PUB_THIS_FRAME) {
    saveFeatureRGBD(gray_img_ptr->image, depth_img_ptr->image, color_msg->header);
    // cv::Mat show_img = gray_img_ptr->image;
    // if (SHOW_TRACK) {
    //   gray_img_ptr = cv_bridge::cvtColor(gray_img_ptr, sensor_msgs::image_encodings::BGR8);
    //   cv::Mat stereo_img = gray_img_ptr->image;
    //   for (int i = 0; i < NUM_OF_CAM; i++) {
    //     cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
    //     cv::cvtColor(show_img, tmp_img, cv::COLOR_GRAY2RGB);
    //     for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) {
    //       double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
    //       cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), std::round(COL/300) + 1);
    //     }
    //   }
    //   pub_match.publish(gray_img_ptr->toImageMsg());
    // }
  }
  // ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}


void featureCallback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
  if (!init_feature) {
    //skip the first detected feature, which doesn't contain optical flow speed
    init_feature = 1;
    return;
  }
  m_buf.lock();
  feature_buf.push(feature_msg);
  m_buf.unlock();
  con.notify_one();
}

void restartCallback(const std_msgs::BoolConstPtr &restart_msg)
{
  if (restart_msg->data == true)
  {
    ROS_WARN("restart the estimator!");
    m_buf.lock();
    while(!feature_buf.empty()) {
      feature_buf.pop();
      gray_img_buf.pop();
    }
    while(!imu_buf.empty()) {
      imu_buf.pop();
    }
    m_buf.unlock();
    m_estimator.lock();
    estimator.clearState();
    estimator.setParameter();
    m_estimator.unlock();
    current_time = -1;
    last_imu_t = 0;
  }
  return;
}

void relocalizationCallback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
  m_buf.lock();
  relo_buf.push(points_msg);
  m_buf.unlock();
}

cv::Mat buildTrackImage(const Estimator& estimator, cv::Mat& gray_img)
{
  cv::cvtColor(gray_img, gray_img, cv::COLOR_GRAY2BGR);

  for(auto lm: estimator.f_manager.feature) {
    if(lm.endFrame() < WINDOW_SIZE-1)
      continue;
    
    const Eigen::Vector2d ob = lm.feature_per_frame.back().uv;
    cv::Scalar color;
    switch (lm.status)
    {
    case FeaturePerId::Status::GOOD:
      color = CV_COLOR_GREEN;
      break;
    case FeaturePerId::Status::NORMAL:
      color = CV_COLOR_SKYBLUE;
      break;
    case FeaturePerId::Status::TRIANGULATE_WEAK:
      color = CV_COLOR_YELLOW;
      break;
    case FeaturePerId::Status::TRIANGULATE_BAD:
      color = CV_COLOR_RED;
      break;
    case FeaturePerId::Status::OUTLIER:
      color = CV_COLOR_DARKRED;
      break;
    default:
      break;
    }
    cv::circle(gray_img, cv::Point2f(ob.x(), ob.y()), (int)(ROW/100), color, -1);
  }
  return gray_img;
}

// thread: visual-inertial odometry
void vioLoop()
{
  while (true) {
    // std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
    std::vector<Measurement> measurements;
    std::unique_lock<std::mutex> lk(m_buf);
    con.wait(lk, [&]{
      return (measurements = getMeasurements()).size() != 0;
    });
    lk.unlock();
    m_estimator.lock();
    for (auto &measurement : measurements) {
      // auto feature_msg = measurement.second;
      auto &feature_msg = measurement.feature;
      double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
      for (auto &imu_msg : measurement.imu_vec) {
        double t = imu_msg->header.stamp.toSec();
        double img_t = feature_msg->header.stamp.toSec() + estimator.td;
        if (t <= img_t) { 
          if (current_time < 0)
            current_time = t;
          double dt = t - current_time;
          ROS_ASSERT(dt >= 0);
          current_time = t;
          dx = imu_msg->linear_acceleration.x;
          dy = imu_msg->linear_acceleration.y;
          dz = imu_msg->linear_acceleration.z;
          rx = imu_msg->angular_velocity.x;
          ry = imu_msg->angular_velocity.y;
          rz = imu_msg->angular_velocity.z;
          estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        }
        else {
          double dt_1 = img_t - current_time;
          double dt_2 = t - img_t;
          current_time = img_t;
          ROS_ASSERT(dt_1 >= 0);
          ROS_ASSERT(dt_2 >= 0);
          ROS_ASSERT(dt_1 + dt_2 > 0);
          double w1 = dt_2 / (dt_1 + dt_2);
          double w2 = dt_1 / (dt_1 + dt_2);
          dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
          dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
          dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
          rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
          ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
          rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
          estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
        }
      }
      // set relocalization frame
      sensor_msgs::PointCloudConstPtr relo_msg = NULL;
      while (!relo_buf.empty()) {
        relo_msg = relo_buf.front();
        relo_buf.pop();
      }
      if (relo_msg != NULL) {
        vector<Vector3d> match_points;
        double frame_stamp = relo_msg->header.stamp.toSec();
        for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
          Vector3d u_v_id;
          u_v_id.x() = relo_msg->points[i].x;
          u_v_id.y() = relo_msg->points[i].y;
          u_v_id.z() = relo_msg->points[i].z;
          match_points.push_back(u_v_id);
        }
        Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
        Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
        Matrix3d relo_r = relo_q.toRotationMatrix();
        int frame_index;
        frame_index = relo_msg->channels[0].values[7];
        estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
      }

      ROS_DEBUG("processing vision data with stamp %f \n", feature_msg->header.stamp.toSec());

      TicToc t_s;
      map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> feature;
      for (unsigned int i = 0; i < feature_msg->points.size(); i++) {
        int v = feature_msg->channels[0].values[i] + 0.5;
        int feature_id = v / NUM_OF_CAM;
        int camera_id = v % NUM_OF_CAM;
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[1].values[i];
        double p_v = feature_msg->channels[2].values[i];
        double velocity_x = feature_msg->channels[3].values[i];
        double velocity_y = feature_msg->channels[4].values[i];
        ROS_ASSERT(z == 1);
        if(USE_RGBD) {
          double depth = feature_msg->channels[5].values[i];
          Eigen::Matrix<double, 8, 1> ob;
          ob << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
          feature[feature_id].emplace_back(camera_id,  ob);
        }
        else {
          Eigen::Matrix<double, 8, 1> ob;
          ob << x, y, z, p_u, p_v, velocity_x, velocity_y, 0.0;
          feature[feature_id].emplace_back(camera_id,  ob);
        }
      }
      TicToc t_vio;
      estimator.processImage(feature, Q_rt.toRotationMatrix(), P_rt, feature_msg->header);
      // ROS_INFO("vio latency: %lf\n", t_vio.toc());

      cv::Mat track_img = buildTrackImage(estimator, measurement.gray_img);
      
      double whole_t = t_s.toc();
      printStatistics(estimator, whole_t);
      std_msgs::Header header = feature_msg->header;
      header.frame_id = "world";

      pubOdometry(estimator, header);
      pubTrackImage(track_img, header);
      pubKeyPoses(estimator, header);
      pubCameraPose(estimator, header);
      pubPointCloud(estimator, header);
      pubTF(estimator, header);
      pubKeyframe(estimator);
      if (relo_msg != NULL)
        pubRelocalization(estimator);
    }
    m_estimator.unlock();
    m_buf.lock();
    m_state.lock();
    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
      update();
    m_state.unlock();
    m_buf.unlock();
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vins_estimator");
  ros::NodeHandle n("~");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  readParameters(n);
  estimator.setParameter();

  imu_pose_op_file.open("/home/summervibe/critical_vio_traj.txt", std::ios::out);

  for (int i = 0; i < NUM_OF_CAM; i++)
    trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

  if(FISHEYE) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
      trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
      if(!trackerData[i].fisheye_mask.data) {
        ROS_INFO("load mask fail");
        ROS_BREAK();
      }
      else
        ROS_INFO("load mask success");
    }
  }

  ROS_WARN("Waiting for image and imu...");

  registerPub(n);
  ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 200, imuCallback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2, restartCallback);
  ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2, relocalizationCallback);

  pub_match = n.advertise<sensor_msgs::Image>("feature_img", 1000);
  pub_odom = n.advertise<nav_msgs::Odometry>("/player/odom", 1000);
  if(USE_RGBD) {
    sub_color = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(n, IMAGE_TOPIC, 1);
    sub_depth = std::make_shared<message_filters::Subscriber<sensor_msgs::Image>>(n, DEPTH_TOPIC, 1);
    synchronizer = std::make_shared<message_filters::Synchronizer<syncPolicy>>(syncPolicy(10), *sub_color, *sub_depth);
    synchronizer->registerCallback(boost::bind(&rgbdCallback, _1, _2));
  }
  else {
    sub_img = n.subscribe(IMAGE_TOPIC, 100, rgbCallback);
  }

  std::thread vio_thread{vioLoop};
  ros::spin();
  return 0;
}
