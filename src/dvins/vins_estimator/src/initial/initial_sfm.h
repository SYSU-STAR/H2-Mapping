#pragma once 
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>


#include "../factor/projection_xyz_factor.h"
#include "../factor/depth_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../utility/utility.h"

using namespace Eigen;
using namespace std;

struct SFMFeature
{
  bool state;
  int id;
  vector<pair<int,Vector2d>> ob_vec;
  vector<pair<int,double>> depth_vec;
  vector<bool> flags;
  double position[3];
  double depth;
};

class GlobalSFM
{
public:
	GlobalSFM();
	bool construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);

private:
	bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i, vector<SFMFeature> &sfm_f);

	void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
							Vector2d &point0, Vector2d &point1, Vector3d &point_3d);
	void triangulateTwoFrames(int frame_i, Eigen::Matrix<double, 3, 4> &pose_i, 
                            int frame_j, Eigen::Matrix<double, 3, 4> &pose_j,
                            vector<SFMFeature> &sfm_f);
  void triangulateTwoFramesWithDepth(int frame_i, Eigen::Matrix<double, 3, 4> &pose_i,
                                    int frame_j, Eigen::Matrix<double, 3, 4> &pose_j,
                                    vector<SFMFeature> &sfm_f);

  void evaluateError(double para_pose[][6], const vector<SFMFeature> &sfm_f);

  std::vector<std::pair<std::pair<int, int>, Eigen::Vector2d>> projection_edges;
  std::vector<std::pair<std::pair<int, int>, double>> depth_edges;
	int feature_num;
};