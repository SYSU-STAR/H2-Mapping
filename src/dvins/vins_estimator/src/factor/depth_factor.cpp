#include "depth_factor.h"


bool DepthHostFactor::Evaluate(double const* const* parameters, double *residuals, double **jacobians) const
{
  double lambda = parameters[0][0];
  residuals[0] = sqrt_info * (1 / lambda - depth);  
  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> jacobian_lambda(jacobians[0]);
      jacobian_lambda << - sqrt_info / (lambda * lambda);

      // if(jacobian_lambda.maxCoeff() > 1e8 || jacobian_lambda.minCoeff() < -1e8) {
      //   printf("[Numeric Instability]\n");
      //   printf("depth: %lf, lambda: %lf, sqrt_info: %lf\n", depth, lambda, sqrt_info);
      //   assert(1==0);
      // }
    }
  }
  return true;
}

bool DepthProjectionFactor::Evaluate(double const* const* parameters, double *residuals, double **jacobians) const
{
  Eigen::Vector3d Pwbi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qwbi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Pwbj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qwbj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

  Eigen::Vector3d Pbc(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond Qbc(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  double inv_dep_i = parameters[3][0];

  Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
  Eigen::Vector3d pts_imu_i = Qbc * pts_camera_i + Pbc;
  Eigen::Vector3d pts_w = Qwbi * pts_imu_i + Pwbi;
  Eigen::Vector3d pts_imu_j = Qwbj.inverse() * (pts_w - Pwbj);
  Eigen::Vector3d pts_camera_j = Qbc.inverse() * (pts_imu_j - Pbc);
  residuals[0] = sqrt_info_j * (pts_camera_j.z() - depth_j);  


  if (jacobians) {

    Eigen::Matrix3d Rwbi = Qwbi.toRotationMatrix();
    Eigen::Matrix3d Rwbj = Qwbj.toRotationMatrix();
    Eigen::Matrix3d Rbc = Qbc.toRotationMatrix();

    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
      jacobian_pose_i.setZero();
      Eigen::Matrix<double, 3, 6> jacobian_i;
      jacobian_i.leftCols<3>() = Rbc.transpose() * Rwbj.transpose();
      jacobian_i.rightCols<3>() = Rbc.transpose() * Rwbj.transpose() * Rwbi * -Utility::skewSymmetric(pts_imu_i);
      jacobian_pose_i.leftCols<6>() = sqrt_info_j * jacobian_i.row(2);
    }
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
      jacobian_pose_j.setZero();
      Eigen::Matrix<double, 3, 6> jacobian_j;
      jacobian_j.leftCols<3>() = Rbc.transpose() * -Rwbj.transpose();
      jacobian_j.rightCols<3>() = Rbc.transpose() * Utility::skewSymmetric(pts_imu_j);
      jacobian_pose_j.leftCols<6>() = sqrt_info_j * jacobian_j.row(2);
    }
    if (jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
      jacobian_ex_pose.setZero();
      Eigen::Matrix<double, 3, 6> jacobian_ex;
      Eigen::Matrix3d tmp_r = Rbc.transpose() * Rwbj.transpose() * Rwbi * Rbc;
      jacobian_ex.leftCols<3>() = Rbc.transpose() * (Rwbj.transpose() * Rwbi - Eigen::Matrix3d::Identity());
      
      jacobian_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                Utility::skewSymmetric(Rbc.transpose() * (Rwbj.transpose() * (Rwbi * Pbc + Pwbi - Pwbj) - Pbc));
      jacobian_ex_pose.leftCols<6>() = sqrt_info_j * jacobian_ex.row(2);
    }
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> jacobian_feature(jacobians[3]);
      jacobian_feature = sqrt_info_j * (Rbc.transpose() * Rwbj.transpose() * Rwbi * Rbc * pts_i * -1.0 / (inv_dep_i * inv_dep_i)).row(2);
    }
  }
  return true;
}

void DepthProjectionFactor::Check(double **parameters)
{
  Eigen::Vector3d Pwbi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qwbi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Vector3d Pwbj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qwbj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

  Eigen::Vector3d Pbc(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond Qbc(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

  double inv_dep_i = parameters[3][0];

  Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
  Eigen::Vector3d pts_imu_i = Qbc * pts_camera_i + Pbc;
  Eigen::Vector3d pts_w = Qwbi * pts_imu_i + Pwbi;
  Eigen::Vector3d pts_imu_j = Qwbj.inverse() * (pts_w - Pwbj);
  Eigen::Vector3d pts_camera_j = Qbc.inverse() * (pts_imu_j - Pbc);
  Eigen::Matrix<double, 1, 1> residual;
  residual << sqrt_info_j * (pts_camera_j.z() - depth_j); 

  Eigen::Matrix3d Rwbi = Qwbi.toRotationMatrix();
  Eigen::Matrix3d Rwbj = Qwbj.toRotationMatrix();
  Eigen::Matrix3d Rbc = Qbc.toRotationMatrix();

  Eigen::Matrix<double, 1, 7> jacobian_pose_i;
  jacobian_pose_i.setZero();
  Eigen::Matrix<double, 3, 6> jacobian_i;
  jacobian_i.leftCols<3>() = Rbc.transpose() * Rwbj.transpose();
  jacobian_i.rightCols<3>() = Rbc.transpose() * Rwbj.transpose() * Rwbi * -Utility::skewSymmetric(pts_imu_i);
  jacobian_pose_i.leftCols<6>() = sqrt_info_j * jacobian_i.row(2);

  Eigen::Matrix<double, 1, 7> jacobian_pose_j;
  jacobian_pose_j.setZero();
  Eigen::Matrix<double, 3, 6> jacobian_j;
  jacobian_j.leftCols<3>() = Rbc.transpose() * -Rwbj.transpose();
  jacobian_j.rightCols<3>() = Rbc.transpose() * Utility::skewSymmetric(pts_imu_j);
  jacobian_pose_j.leftCols<6>() = sqrt_info_j * jacobian_j.row(2);

  Eigen::Matrix<double, 1, 7> jacobian_ex_pose;
  jacobian_ex_pose.setZero();
  Eigen::Matrix<double, 3, 6> jacobian_ex;
  Eigen::Matrix3d tmp_r = Rbc.transpose() * Rwbj.transpose() * Rwbi * Rbc;
  jacobian_ex.leftCols<3>() = Rbc.transpose() * (Rwbj.transpose() * Rwbi - Eigen::Matrix3d::Identity());
  
  jacobian_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                            Utility::skewSymmetric(Rbc.transpose() * (Rwbj.transpose() * (Rwbi * Pbc + Pwbi - Pwbj) - Pbc));
  jacobian_ex_pose.leftCols<6>() = sqrt_info_j * jacobian_ex.row(2);

  Eigen::Matrix<double, 1, 1> jacobian_feature;
  jacobian_feature = sqrt_info_j * (Rbc.transpose() * Rwbj.transpose() * Rwbi * Rbc * pts_i * -1.0 / (inv_dep_i * inv_dep_i)).row(2);

  const double norm = 1e-3;
  // turb pose i
  {   
    Eigen::Vector3d turb_p = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d turb_r = Eigen::Vector3d::Random() * norm;
    Eigen::Quaterniond Qwbi_trubed = Qwbi * Utility::deltaQ(turb_r);
    Eigen::Vector3d Pwbi_trubed = Pwbi + turb_p;

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = Qbc * pts_camera_i + Pbc;
    Eigen::Vector3d pts_w_turbed = Qwbi_trubed * pts_imu_i + Pwbi_trubed;
    Eigen::Vector3d pts_imu_j_turbed = Qwbj.inverse() * (pts_w_turbed - Pwbj);
    Eigen::Vector3d pts_camera_j_turbed = Qbc.inverse() * (pts_imu_j_turbed - Pbc);
    Eigen::Matrix<double, 1, 1> residual_turbed;
    residual_turbed << sqrt_info_j * (pts_camera_j_turbed.z() - depth_j); 


    Eigen::Matrix<double, 7, 1> state_diff;
    state_diff << turb_p, turb_r, 0;
    std::cout << "turb i" << std::endl;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual) << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_pose_i * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_pose_i << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }

  // turb pose j
  {   
    Eigen::Vector3d turb_p = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d turb_r = Eigen::Vector3d::Random() * norm;
    Eigen::Quaterniond Qwbj_trubed = Qwbj * Utility::deltaQ(turb_r);
    Eigen::Vector3d Pwbj_trubed = Pwbj + turb_p;

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = Qbc * pts_camera_i + Pbc;
    Eigen::Vector3d pts_w = Qwbi * pts_imu_i + Pwbi;
    Eigen::Vector3d pts_imu_j_turbed = Qwbj_trubed.inverse() * (pts_w - Pwbj_trubed);
    Eigen::Vector3d pts_camera_j_turbed = Qbc.inverse() * (pts_imu_j_turbed - Pbc);
    Eigen::Matrix<double, 1, 1> residual_turbed;
    residual_turbed << sqrt_info_j * (pts_camera_j_turbed.z() - depth_j); 


    Eigen::Matrix<double, 7, 1> state_diff;
    state_diff << turb_p, turb_r, 0;
    std::cout << "turb j" << std::endl;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual) << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_pose_j * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_pose_j << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }

  // turb invdepth
  {   
    double inv_depth_i_trubed = inv_dep_i + norm;

    Eigen::Vector3d pts_camera_i_trubed = pts_i / inv_depth_i_trubed;
    Eigen::Vector3d pts_imu_i_turbed = Qbc * pts_camera_i_trubed + Pbc;
    Eigen::Vector3d pts_w_trubed = Qwbi * pts_imu_i_turbed + Pwbi;
    Eigen::Vector3d pts_imu_j_turbed = Qwbj.inverse() * (pts_w_trubed - Pwbj);
    Eigen::Vector3d pts_camera_j_turbed = Qbc.inverse() * (pts_imu_j_turbed - Pbc);
    Eigen::Matrix<double, 1, 1> residual_turbed;
    residual_turbed << sqrt_info_j * (pts_camera_j_turbed.z() - depth_j); 


    Eigen::Matrix<double, 1, 1> state_diff;
    state_diff << norm;
    std::cout << "turb j" << std::endl;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual) << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_feature * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_feature << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }
}
                
bool DepthXYZFactor::Evaluate(double const* const* parameters,
                              double *residuals,
                              double **jacobians) const
{
  Eigen::Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Matrix3d Rcw = ExpSO3(Eigen::Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));
  Eigen::Vector3d Pw(parameters[1][0], parameters[1][1], parameters[1][2]);

  Eigen::Vector3d Pc = Rcw * Pw + tcw;
  residuals[0] = sqrt_info * (Pc.z() - depth);

  if(jacobians) {
    if(jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
      jacobian_pose.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dpc_dpw;
      dpc_dpw.leftCols<3>().setIdentity();
      dpc_dpw.rightCols<3>() = -Utility::skewSymmetric(Rcw * Pw);
      jacobian_pose = sqrt_info * dpc_dpw.row(2);
    }
    if(jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_point(jacobians[1]);
      jacobian_point = sqrt_info * Rcw.matrix().row(2);
    }
  }
  return true;
}

void DepthXYZFactor::Check(double **parameters)
{
  Eigen::Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Matrix3d Rcw = ExpSO3(Eigen::Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));
  Eigen::Vector3d Pw(parameters[1][0], parameters[1][1], parameters[1][2]);

  Eigen::Vector3d Pc = Rcw * Pw + tcw;
  double residual = Pc.z() - depth;

  Eigen::Matrix<double, 1, 6> jacobian_pose;
  jacobian_pose.setZero();
  Eigen::Matrix<double, 3, 6> dpc_dpw;
  dpc_dpw.leftCols<3>().setIdentity();
  dpc_dpw.rightCols<3>() = -Utility::skewSymmetric(Rcw * Pw);
  jacobian_pose = dpc_dpw.row(2);

  Eigen::Matrix<double, 1, 3, Eigen::RowMajor> jacobian_point;
  jacobian_point =  Rcw.matrix().row(2);

  const double norm = 1e-3;

  //turb pose
  {
    Eigen::Vector3d tcw_noise = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d rcw_noise = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d tcw_trubed = tcw + tcw_noise;
    Eigen::Matrix3d Rcw_turbed = ExpSO3(rcw_noise) * Rcw;
    Eigen::Vector3d Pc_turbed = Rcw_turbed * Pw + tcw_trubed;
    double residual_turbed = (Pc_turbed.z() - depth);
    Eigen::Matrix<double, 6, 1> state_diff;
    state_diff << tcw_noise, rcw_noise;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual) << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_pose * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_pose << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }
  std::cout << "\n";
  //turb point
  {
    Eigen::Vector3d point_noise = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d Pw_trubed = Pw + point_noise;
    Eigen::Vector3d Pc_turbed = Rcw * Pw_trubed + tcw;
    double residual_turbed = (Pc_turbed.z() - depth);
    Eigen::Matrix<double, 3, 1> state_diff;
    state_diff = point_noise;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual) << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_point * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_point << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }

  
}