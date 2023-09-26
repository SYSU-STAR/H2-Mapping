#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../parameters.h"
#include "integration_base.h"

#include <ceres/ceres.h>
// #include <se3.hpp>
// #include <so3.hpp>

class DepthHostFactor : public ceres::SizedCostFunction<1, 1>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DepthHostFactor(const double _depth, const double cov): depth(_depth) {
    sqrt_info = 1 / (cov * depth);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double *residuals,
                        double **jacobians) const;

private:
  double depth, sqrt_info;
};

class DepthXYZFactor : public ceres::SizedCostFunction<1, 6, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DepthXYZFactor(const double _depth, const double _cov)
    : depth(_depth) 
  {
    sqrt_info = 1 / (DEPTH_COV * depth);
  }

  virtual bool Evaluate(double const* const* parameters,
                        double *residuals,
                        double **jacobians) const;
  void Check(double **parameters);

private:
  double depth, sqrt_info;
};

class DepthProjectionFactor : public ceres::SizedCostFunction<1, 7, 7, 7, 1>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DepthProjectionFactor(const Eigen::Vector3d &_pts_i, const double _depth_j, const double cov) {
    pts_i = _pts_i;
    depth_j = _depth_j;
    sqrt_info_j = 1 / (cov * depth_j);
  }
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void Check(double **parameters);

  Eigen::Vector3d pts_i;
  double depth_j, sqrt_info_j;
};