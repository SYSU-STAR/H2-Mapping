#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionXYZFactor : public ceres::SizedCostFunction<2, 6, 3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ProjectionXYZFactor(const Eigen::Vector2d &_uv): uv(_uv) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  Eigen::Vector2d uv;
  static Eigen::Matrix2d sqrt_info;
};
