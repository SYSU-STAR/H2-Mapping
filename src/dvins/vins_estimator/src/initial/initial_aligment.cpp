#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
  Matrix3d A;
  Vector3d b;
  Vector3d delta_bg;
  A.setZero();
  b.setZero();
  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
    frame_j = next(frame_i);
    MatrixXd Ai(3, 3);
    VectorXd bi(3);
    Ai.setZero();
    bi.setZero();
    Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
    Ai = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
    bi = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
    A += Ai.transpose() * Ai;
    b += Ai.transpose() * bi;
  }
  delta_bg = A.ldlt().solve(b);
  ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

  for (int i = 0; i <= WINDOW_SIZE; i++)
    Bgs[i] += delta_bg;

  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++) {
    frame_j = next(frame_i);
    frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
  }
}


MatrixXd TangentBasis(Vector3d &g0)
{
  Vector3d b, c;
  Vector3d a = g0.normalized();
  Vector3d tmp(0, 0, 1);
  if(a == tmp)
    tmp << 1, 0, 0;

  b = (tmp - a * (a.transpose() * tmp)).normalized();
  c = a.cross(b);
  MatrixXd bc(3, 2);
  bc.block<3, 1>(0, 0) = b;
  bc.block<3, 1>(0, 1) = c;
  return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
  Vector3d g0 = g.normalized() * G.norm();
  Vector3d lx, ly;
  //VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for(int k = 0; k < 4; k++)
  {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
      frame_j = next(frame_i);

      MatrixXd Ai(6, 9);
      Ai.setZero();
      VectorXd bi(6);
      bi.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;

      Ai.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      Ai.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
      Ai.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
      bi.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

      Ai.block<3, 3>(3, 0) = -Matrix3d::Identity();
      Ai.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
      Ai.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      bi.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Identity();

      MatrixXd r_A = Ai.transpose() * cov_inv * Ai;
      VectorXd r_b = Ai.transpose() * cov_inv * bi;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
      b.tail<3>() += r_b.tail<3>();

      A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
      A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 3);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
    //double s = x(n_state - 1);
  }   
  g = g0;
}

void RefineGravityWithDepth(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
  Vector3d g0 = g.normalized() * G.norm();
  Vector3d lx, ly;
  //VectorXd x;
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 2;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  for(int k = 0; k < 4; k++) {
    MatrixXd lxly(3, 2);
    lxly = TangentBasis(g0);
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
      frame_j = next(frame_i);

      MatrixXd Ai(6, 8);
      Ai.setZero();
      VectorXd bi(6);
      bi.setZero();

      double dt = frame_j->second.pre_integration->sum_dt;
      Ai.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
      Ai.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
      bi.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0 - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);

      Ai.block<3, 3>(3, 0) = -Matrix3d::Identity();
      Ai.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
      Ai.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
      bi.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


      Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Identity();
      MatrixXd r_A = Ai.transpose() * cov_inv * Ai;
      VectorXd r_b = Ai.transpose() * cov_inv * bi;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
      b.tail<2>() += r_b.tail<2>();

      A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
      A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
    }
    // A = A * 1000.0;
    // b = b * 1000.0;
    x = A.ldlt().solve(b);
    VectorXd dg = x.segment<2>(n_state - 2);
    g0 = (g0 + lxly * dg).normalized() * G.norm();
    //double s = x(n_state - 1);
  }
  g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 3 + 1;

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd Ai(6, 10);
    Ai.setZero();
    VectorXd bi(6);
    bi.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    Ai.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    Ai.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    Ai.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
    bi.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];

    Ai.block<3, 3>(3, 0) = -Matrix3d::Identity();
    Ai.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    Ai.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
    bi.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

    MatrixXd r_A = Ai.transpose() * Ai;
    VectorXd r_b = Ai.transpose() * bi;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
    b.tail<4>() += r_b.tail<4>();

    A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
    A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
  }
  A = A * 1000.0;
  b = b * 1000.0;
  x = A.ldlt().solve(b);
  double s = x(n_state - 1) / 100.0;
  ROS_INFO("estimated scale: %f", s);
  g = x.segment<3>(n_state - 4);
  ROS_INFO_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if(fabs(g.norm() - G.norm()) > 1.0) {
    return false;
  }

  RefineGravity(all_image_frame, g, x);
  s = (x.tail<1>())(0) / 100.0;
  (x.tail<1>())(0) = s;
  ROS_INFO_STREAM(" refine     " << g.norm() << " " << g.transpose());
  if(s < 0.1)
    return false;   
  else
    return true;
}

bool LinearAlignmentWithDepth(map<double, ImageFrame> &all_image_frame, Vector3d* Bas, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
  int all_frame_count = all_image_frame.size();
  int n_state = all_frame_count * 3 + 3;//no scale now

  MatrixXd A{n_state, n_state};
  A.setZero();
  VectorXd b{n_state};
  b.setZero();

  map<double, ImageFrame>::iterator frame_i;
  map<double, ImageFrame>::iterator frame_j;
  int i = 0;
  for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
    frame_j = next(frame_i);

    MatrixXd Ai(6, 9);//no scale now
    Ai.setZero();
    VectorXd bi(6);
    bi.setZero();

    double dt = frame_j->second.pre_integration->sum_dt;

    Ai.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
    Ai.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
    bi.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T);
    Ai.block<3, 3>(3, 0) = -Matrix3d::Identity();
    Ai.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
    Ai.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
    bi.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;

    Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Identity();
    MatrixXd r_A = Ai.transpose() * cov_inv * Ai;
    VectorXd r_b = Ai.transpose() * cov_inv * bi;

    A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
    b.segment<6>(i * 3) += r_b.head<6>();

    A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
    b.tail<3>() += r_b.tail<3>();

    A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
    A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
}
  // A = A * 1000.0;
  // b = b * 1000.0;
  x = A.ldlt().solve(b);
  g = x.segment<3>(n_state - 3);

  ROS_INFO_STREAM(" result g     " << g.norm() << " " << g.transpose());
  if(fabs(g.norm() - G.norm()) > 1.0) {
    return false;
  }

  RefineGravityWithDepth(all_image_frame, g, x);
  ROS_INFO_STREAM(" refine     " << g.norm() << " " << g.transpose());
  return true;
}

bool alignVI(map<double, ImageFrame> &all_image_frame, Vector3d* Bas, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
  solveGyroscopeBias(all_image_frame, Bgs);
  if(USE_RGBD) {
    if(LinearAlignmentWithDepth(all_image_frame, Bas, Bgs, g, x))
      return true;
    else 
      return false;
  }
  else {
    if(LinearAlignment(all_image_frame, g, x))
      return true;
    else 
      return false;
  }
}
