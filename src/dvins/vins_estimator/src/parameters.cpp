#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::string IMAGE_TOPIC;
std::string DEPTH_TOPIC;
std::string IMU_TOPIC;
std::string IMU_RAW_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;

int MAX_CNT;
int MIN_DIST;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
double FX, FY;
double FOCAL_LENGTH = 460.0;
int FISHEYE;
int USE_FAST = 1;
int USE_RGBD = 0;
int MIN_FAST_RESP = 10;
bool PUB_THIS_FRAME;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
double REPROJ_THRESHOLD;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TRAJECTORY_FILE;
double TD, TR;
double MAX_DEPTH, MIN_DEPTH, DEPTH_COV;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
  T ans;
  if (n.getParam(name, ans)) {
    ROS_INFO_STREAM("Loaded " << name << ": " << ans);
  }
  else {
    ROS_ERROR_STREAM("Failed to load " << name);
    n.shutdown();
  }
  return ans;
}

void readParameters(ros::NodeHandle &n)
{
  std::string config_file;
  config_file = readParam<std::string>(n, "config_file");
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if(!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings" << std::endl;
  }

  std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");
  fsSettings["imu_topic"] >> IMU_TOPIC;
  fsSettings["imu_raw_topic"] >> IMU_RAW_TOPIC;
  fsSettings["image_topic"] >> IMAGE_TOPIC;

  cv::FileNode node = fsSettings["projection_parameters"];
  FX = static_cast<double>(node["fx"]);
  FY = static_cast<double>(node["fy"]);
  FOCAL_LENGTH = (FX + FY) * 0.5;

  MAX_CNT = fsSettings["max_cnt"];
  MIN_DIST = fsSettings["min_dist"];
  ROW = fsSettings["image_height"];
  COL = fsSettings["image_width"];
  FREQ = fsSettings["freq"];
  F_THRESHOLD = fsSettings["F_threshold"];
  SHOW_TRACK = fsSettings["show_track"];
  EQUALIZE = fsSettings["equalize"];
  FISHEYE = fsSettings["fisheye"];
  USE_FAST = fsSettings["use_fast"];
  USE_RGBD = fsSettings["use_rgbd"];

  if (FISHEYE == 1) {
    FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
  }
    
  if(USE_FAST) {
    fsSettings["min_fast_resp"] >> MIN_FAST_RESP;
  }

  CAM_NAMES.push_back(config_file);

  STEREO_TRACK = false;
  PUB_THIS_FRAME = false;

  if (FREQ == 0)
    FREQ = 100;

  SOLVER_TIME = fsSettings["max_solver_time"];
  NUM_ITERATIONS = fsSettings["max_num_iterations"];
  MIN_PARALLAX = fsSettings["keyframe_parallax"];
  MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;
  REPROJ_THRESHOLD = fsSettings["reproj_threshold"];

  std::string OUTPUT_PATH;
  fsSettings["output_path"] >> OUTPUT_PATH;
  VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
  IMU_TRAJECTORY_FILE = OUTPUT_PATH + "/imu_trajectory.txt";
  std::cout << "result path " << VINS_RESULT_PATH << std::endl;

  // create folder if not exists
  FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str());

  std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
  fout.close();

  ACC_N = fsSettings["acc_n"];
  ACC_W = fsSettings["acc_w"];
  GYR_N = fsSettings["gyr_n"];
  GYR_W = fsSettings["gyr_w"];
  G.z() = fsSettings["g_norm"];

  if(USE_RGBD) {
    MAX_DEPTH = fsSettings["max_depth"];
    MIN_DEPTH = fsSettings["min_depth"];
    DEPTH_COV = fsSettings["depth_cov"];
    fsSettings["depth_topic"] >> DEPTH_TOPIC;
  }
  
  ROS_INFO("ROW: %d COL: %d ", ROW, COL);

  ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
  if (ESTIMATE_EXTRINSIC == 2) {
    ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
    RIC.push_back(Eigen::Matrix3d::Identity());
    TIC.push_back(Eigen::Vector3d::Zero());
    EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
  }
  else {
    if ( ESTIMATE_EXTRINSIC == 1) {
      ROS_WARN(" Optimize extrinsic param around initial guess!");
      EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
    }
    if (ESTIMATE_EXTRINSIC == 0)
      ROS_WARN(" fix extrinsic param ");

    cv::Mat cv_R, cv_T;
    fsSettings["extrinsicRotation"] >> cv_R;
    fsSettings["extrinsicTranslation"] >> cv_T;
    Eigen::Matrix3d eigen_R;
    Eigen::Vector3d eigen_T;
    cv::cv2eigen(cv_R, eigen_R);
    cv::cv2eigen(cv_T, eigen_T);
    Eigen::Quaterniond Q(eigen_R);
    eigen_R = Q.normalized();
    RIC.push_back(eigen_R);
    TIC.push_back(eigen_T);
    ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
    ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
  } 

  INIT_DEPTH = 5.0;
  BIAS_ACC_THRESHOLD = 0.1;
  BIAS_GYR_THRESHOLD = 0.1;

  TD = fsSettings["td"];
  ESTIMATE_TD = fsSettings["estimate_td"];
  if (ESTIMATE_TD)
    ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
  else
    ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

  ROLLING_SHUTTER = fsSettings["rolling_shutter"];
  if (ROLLING_SHUTTER) {
    TR = fsSettings["rolling_shutter_tr"];
    ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
  }
  else {
    TR = 0;
  }
  
  fsSettings.release();
}
