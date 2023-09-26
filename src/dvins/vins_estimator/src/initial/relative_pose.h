#pragma once

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
#include "../parameters.h" 
#include <eigen3/Eigen/Dense>

using namespace Eigen;

#include <ros/console.h>
class MotionEstimator
{
 public:

  bool solveFundamentalMat(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);

  bool solveRelativePose(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &Rotation, Vector3d &Translation);

 private:
  double testTriangulation(const vector<cv::Point2f> &l,
                            const vector<cv::Point2f> &r,
                            cv::Mat_<double> R, cv::Mat_<double> t);
  void decomposeE(cv::Mat E,
                  cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                  cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};


