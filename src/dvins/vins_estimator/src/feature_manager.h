#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include <ros/console.h>
#include <ros/assert.h>
#include "parameters.h"

class FeaturePerFrame {
 public:
  FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, double td) {
    point.x() = _point(0);
    point.y() = _point(1);
    point.z() = _point(2);
    uv.x() = _point(3);
    uv.y() = _point(4);
    velocity.x() = _point(5); 
    velocity.y() = _point(6); 
    z = _point(7);
    cur_td = td;
    is_used = true;
    parallax = 0;
  }
  double cur_td;
  Eigen::Vector3d point;
  Eigen::Vector2d uv;
  Eigen::Vector2d velocity;
  double z;
  bool is_used;
  double parallax;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
};

class FeaturePerId
{
 public:
  enum Status {
    NOT_INIT = 0,
    NORMAL = 1,
    GOOD = 2,
    TRIANGULATE_BAD = 3,
    TRIANGULATE_WEAK = 4,
    OUTLIER = 5,
  };  

  const int feature_id;
  int start_frame;
  std::vector<FeaturePerFrame> feature_per_frame;
  std::vector<double> vec_reproj_err;
  std::vector<double> vec_depth_err;
  std::vector<bool> vec_inlier_flag;

  int used_num;
  bool is_outlier;
  bool is_margin;
  double estimated_depth;
  int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

  Eigen::Vector3d gt_p;
  Status status;

  FeaturePerId(int _feature_id, int _start_frame)
      : feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), is_outlier(false), is_margin(false), 
        estimated_depth(-1.0), solve_flag(0), status(NOT_INIT) {}

  int endFrame();
};

class FeatureManager
{
 public:
  FeatureManager(Eigen::Matrix3d _Rs[]);

  void setRic(Eigen::Matrix3d _ric[]);

  void clearState();

  int getFeatureCount();

  bool addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td);
  void debugShow();
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> getCorrespondingWithDepth(int frame_count_l, int frame_count_r);

  //void updateDepth(const VectorXd &x);
  void setDepth(const Eigen::VectorXd &x);
  void removeFailures();
  void clearDepth(const Eigen::VectorXd &x);
  Eigen::VectorXd getDepthVector();

  void triangulateTwoFrame(const Eigen::Vector3d& Pi, const Eigen::Matrix3d& Ri,
                            const Eigen::Vector3d& Pj, const Eigen::Matrix3d& Rj,
                            const Eigen::Vector3d& tic, const Eigen::Matrix3d& Ric,
                            const Eigen::Vector3d& fi, const Eigen::Vector3d& fj,
                            FeaturePerId& lm);

  void triangulateMultiFrame(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                FeaturePerId& lm);

  void triangulateMultiFrameSet(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                FeaturePerId& lm, const std::vector<int>& inliers);

  void checkReprojError(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                              Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                              FeaturePerId& lm, std::vector<int>& inliers);

  void checkReprojErrorWithDepth(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                  Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                  FeaturePerId& lm, std::vector<int>& inliers);

  void triangulateRansac(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                         Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                         FeaturePerId& lm, std::vector<int>& inliers);

  void triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

  void triangulateWithDepth(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

  void evaluateQuality(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]);

  void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
  void removeBack();
  void removeFront(int frame_count);
  void removeOutlier();
  std::list<FeaturePerId> feature;
  int last_track_num;
  int normal_feature_sum;

 private:
  double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
  const Eigen::Matrix3d *Rs;
  Eigen::Matrix3d ric[NUM_OF_CAM];
};

#endif