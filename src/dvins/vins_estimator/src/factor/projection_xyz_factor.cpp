#include "projection_xyz_factor.h"

Eigen::Matrix2d ProjectionXYZFactor::sqrt_info;

bool ProjectionXYZFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  Eigen::Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Matrix3d Rcw = ExpSO3(Eigen::Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));
  Eigen::Vector3d Pw(parameters[1][0], parameters[1][1], parameters[1][2]);

  Eigen::Vector3d Pc = Rcw * Pw + tcw;
  double x = Pc.x(), y = Pc.y(), z = Pc.z(), z2 = z*z;
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual = sqrt_info * (Pc.head<2>() / z - uv);

  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_dpc;
  duv_dpc << 1/z,   0, -x/z2, 
                0, 1/z, -y/z2;

  if(jacobians) {
    if(jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
      jacobian_pose.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dpc_dpw;
      dpc_dpw.leftCols<3>().setIdentity();
      dpc_dpw.rightCols<3>() = -Utility::skewSymmetric(Rcw * Pw);
      jacobian_pose = sqrt_info * duv_dpc * dpc_dpw;
    }
    if(jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_point(jacobians[1]);
      jacobian_point = sqrt_info * duv_dpc * Rcw;
    }
  }

  return true;
}
