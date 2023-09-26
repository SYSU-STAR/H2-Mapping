#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
  Eigen::Map<const Eigen::Vector3d> dp(delta);
  Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = (_q * dq).normalized();

  return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
  J.topRows<6>().setIdentity();
  J.bottomRows<1>().setZero();
  return true;
}


bool PoseLocalLeftMultiplyParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Vector3d> _r(x + 3);
  Eigen::Matrix3d const _R = ExpSO3(_r);

  Eigen::Map<const Eigen::Vector3d> dp(delta);
  Eigen::Map<const Eigen::Vector3d> dr(delta + 3);
  Eigen::Matrix3d const dR = ExpSO3(dr);

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Vector3d> r(x_plus_delta + 3);

  p = _p + dp;
  r = LogSO3(dR * _R);

  return true;
}

bool PoseLocalLeftMultiplyParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
  Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
  J.setIdentity();
  return true;
}
