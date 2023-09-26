#include "feature_manager.h"

int FeaturePerId::endFrame()
{
  return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Eigen::Matrix3d _Rs[])
    : Rs(_Rs)
{
  for (int i = 0; i < NUM_OF_CAM; i++)
    ric[i].setIdentity();
}

void FeatureManager::setRic(Eigen::Matrix3d _ric[]) {
  for (int i = 0; i < NUM_OF_CAM; i++) {
    ric[i] = _ric[i];
  }
}

void FeatureManager::clearState()
{
  feature.clear();
}

int FeatureManager::getFeatureCount()
{
  int cnt = 0;
  for (auto &it : feature) {
    it.used_num = it.feature_per_frame.size();
    if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2) {
      cnt++;
    }
  }
  return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td)
{
  ROS_DEBUG("input feature: %d", (int)image.size());
  ROS_DEBUG("num of feature: %d", getFeatureCount());
  double parallax_sum = 0;
  int parallax_num = 0;
  last_track_num = 0;
  for (auto &id_pts : image) {
    FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

    int feature_id = id_pts.first;
    auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
    {
      return it.feature_id == feature_id;
    });

    if (it == feature.end()) {
      feature.push_back(FeaturePerId(feature_id, frame_count));
      feature.back().feature_per_frame.push_back(f_per_fra);
    }
    else if (it->feature_id == feature_id) {
      it->feature_per_frame.push_back(f_per_fra);
      last_track_num++;
    }
  }

  if (frame_count < 2 || last_track_num < 20)
    return true;

  for (auto &it_per_id : feature) {
    if (it_per_id.start_frame <= frame_count - 2 &&
        it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) {
      parallax_sum += compensatedParallax2(it_per_id, frame_count);
      parallax_num++;
    }
  }

  if (parallax_num == 0) {
    return true;
  }
  else {
    ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
    ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
    return parallax_sum / parallax_num >= MIN_PARALLAX;
  }
}

void FeatureManager::debugShow()
{
  ROS_DEBUG("debug show");
  for (auto &it : feature) {
    ROS_ASSERT(it.feature_per_frame.size() != 0);
    ROS_ASSERT(it.start_frame >= 0);
    ROS_ASSERT(it.used_num >= 0);

    ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
    int sum = 0;
    for (auto &j : it.feature_per_frame) {
      ROS_DEBUG("%d,", int(j.is_used));
      sum += j.is_used;
      printf("(%lf,%lf) ",j.point(0), j.point(1));
    }
    ROS_ASSERT(it.used_num == sum);
  }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
  for (auto &it : feature) {
    if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
      Eigen::Vector3d a = Eigen::Vector3d::Zero(), b = Eigen::Vector3d::Zero();
      int idx_l = frame_count_l - it.start_frame;
      int idx_r = frame_count_r - it.start_frame;

      a = it.feature_per_frame[idx_l].point;
      b = it.feature_per_frame[idx_r].point;
      
      corres.push_back(std::make_pair(a, b));
    }
  }
  return corres;
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> FeatureManager::getCorrespondingWithDepth(int frame_count_l, int frame_count_r)
{
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> corres;
	for (auto &it : feature) {
		if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
			Eigen::Vector3d pwa = Eigen::Vector3d::Zero(), pnb = Eigen::Vector3d::Zero();
			int idx_l = frame_count_l - it.start_frame;
			int idx_r = frame_count_r - it.start_frame;

			double depth_a = it.feature_per_frame[idx_l].z;
			if (depth_a < MIN_DEPTH || depth_a > 10)//max and min measurement
        continue;
			// double depth_b = it.feature_per_frame[idx_r].z;
      // if (depth_b < MIN_DEPTH || depth_b > 10)//max and min measurement
      //   continue;
			pwa = it.feature_per_frame[idx_l].point * depth_a;
			pnb = it.feature_per_frame[idx_r].point;
			corres.push_back(std::make_pair(pwa, pnb));
		}
	}
	return corres;
}

void FeatureManager::setDepth(const Eigen::VectorXd &x)
{
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    it_per_id.estimated_depth = 1.0 / x(++feature_index);
    if (it_per_id.estimated_depth < 0) {
      it_per_id.solve_flag = 2;
    }
    else
      it_per_id.solve_flag = 1;
  }
}

void FeatureManager::removeFailures()
{
  for (auto it = feature.begin(), it_next = feature.begin();it != feature.end(); it = it_next) {
    it_next++;
    if (it->solve_flag == 2)
      feature.erase(it);
  }
}

void FeatureManager::clearDepth(const Eigen::VectorXd &x)
{
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    it_per_id.estimated_depth = 1.0 / x(++feature_index);
  }
}

Eigen::VectorXd FeatureManager::getDepthVector() {
  Eigen::VectorXd dep_vec(getFeatureCount());
  int feature_index = -1;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;
    dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
  }
  return dep_vec;
}

// void FeatureManager::triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
// {
//   for (auto &it_per_id : feature) {
//     it_per_id.used_num = it_per_id.feature_per_frame.size();
//     if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
//       continue;

//     if (it_per_id.estimated_depth > 0)
//       continue;

//     int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

//     ROS_ASSERT(NUM_OF_CAM == 1);
//     Eigen::MatrixXd A(2 * it_per_id.feature_per_frame.size(), 4);
//     int svd_idx = 0;

//     Eigen::Matrix<double, 3, 4> P0;
//     Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
//     Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
//     P0.leftCols<3>() = Eigen::Matrix3d::Identity();
//     P0.rightCols<1>() = Eigen::Vector3d::Zero();

//     for (auto &it_per_frame : it_per_id.feature_per_frame) {
//       imu_j++;

//       Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
//       Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
//       Eigen::Vector3d t = R0.transpose() * (t1 - t0);
//       Eigen::Matrix3d R = R0.transpose() * R1;
//       Eigen::Matrix<double, 3, 4> P;
//       P.leftCols<3>() = R.transpose();
//       P.rightCols<1>() = -R.transpose() * t;
//       Eigen::Vector3d f = it_per_frame.point.normalized();
//       A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
//       A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

//       if (imu_i == imu_j)
//         continue;
//     }
//     ROS_ASSERT(svd_idx == A.rows());
//     Eigen::Vector4d V = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
//     double svd_method = V[2] / V[3];
//     it_per_id.estimated_depth = svd_method;
//     if (it_per_id.estimated_depth < 0.1) {
//       it_per_id.estimated_depth = INIT_DEPTH;
//     }
//   }
// }

void FeatureManager::triangulateTwoFrame(const Eigen::Vector3d& Pi, const Eigen::Matrix3d& Ri,
                                           const Eigen::Vector3d& Pj, const Eigen::Matrix3d& Rj,
                                           const Eigen::Vector3d& tic, const Eigen::Matrix3d& Ric,
                                           const Eigen::Vector3d& fi, const Eigen::Vector3d& fj,
                                           FeaturePerId& lm) 
{
  Eigen::Matrix<double, 3, 4> Ti, Tj;
  Eigen::Matrix3d Rcwi = (Ri * Ric).transpose();
  Eigen::Matrix3d Rcwj = (Rj * Ric).transpose();
  Eigen::Vector3d tcwi = -Rcwi * (Pi + Ri * tic);
  Eigen::Vector3d tcwj = -Rcwj * (Pj + Rj * tic);
  Ti.leftCols<3>() = Rcwi;
  Ti.rightCols<1>() = tcwi;
  Tj.leftCols<3>() = Rcwj;
  Tj.rightCols<1>() = tcwj;

  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  
	A.row(0) = fi[0] * Ti.row(2) - Ti.row(0);
	A.row(1) = fi[1] * Ti.row(2) - Ti.row(1);
	A.row(2) = fj[0] * Tj.row(2) - Tj.row(0);
	A.row(3) = fj[1] * Tj.row(2) - Tj.row(1);

  Eigen::Vector4d pwh;
	pwh = A.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

  Eigen::Vector3d pw;
	pw(0) = pwh(0) / pwh(3);
	pw(1) = pwh(1) / pwh(3);
	pw(2) = pwh(2) / pwh(3);

  lm.estimated_depth = (Rcwi * pw + tcwi).z();
}

void FeatureManager::triangulateMultiFrame(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                              Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                              FeaturePerId& lm)
{
  int imu_i = lm.start_frame, imu_j = imu_i - 1;

  Eigen::MatrixXd A(2 * lm.feature_per_frame.size(), 4);
  int svd_idx = 0;
  Eigen::Matrix<double, 3, 4> P0;
  Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
  Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
  P0.leftCols<3>() = Eigen::Matrix3d::Identity();
  P0.rightCols<1>() = Eigen::Vector3d::Zero();

  for(auto &ob : lm.feature_per_frame) {
    imu_j++;
    Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
    Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
    Eigen::Vector3d t = R0.transpose() * (t1 - t0);
    Eigen::Matrix3d R = R0.transpose() * R1;
    Eigen::Matrix<double, 3, 4> P;
    P.leftCols<3>() = R.transpose();
    P.rightCols<1>() = -R.transpose() * t;
    Eigen::Vector3d f = ob.point.normalized();
    A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
    A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
  }

  Eigen::Vector4d V = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
  lm.estimated_depth = V[2] / V[3];
}

void FeatureManager::triangulateMultiFrameSet(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                                Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                                FeaturePerId& lm, const std::vector<int>& inliers)
{
  int imu_i = lm.start_frame, imu_j = imu_i - 1;

  Eigen::MatrixXd A(2 * inliers.size(), 4);
  int svd_idx = 0;
  Eigen::Matrix<double, 3, 4> P0;
  Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
  Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
  P0.leftCols<3>() = Eigen::Matrix3d::Identity();
  P0.rightCols<1>() = Eigen::Vector3d::Zero();

  for(size_t i = 0; i < inliers.size(); ++i) {
    imu_j = lm.start_frame + inliers[i];
    Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
    Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
    Eigen::Vector3d t = R0.transpose() * (t1 - t0);
    Eigen::Matrix3d R = R0.transpose() * R1;
    Eigen::Matrix<double, 3, 4> P;
    P.leftCols<3>() = R.transpose();
    P.rightCols<1>() = -R.transpose() * t;
    Eigen::Vector3d f = lm.feature_per_frame[inliers[i]].point.normalized();
    A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
    A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);
  }

  Eigen::Vector4d V = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
  lm.estimated_depth = V[2] / V[3];
}

void FeatureManager::checkReprojError(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                            Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                            FeaturePerId& lm, std::vector<int>& inliers) {

  int imu_i = lm.start_frame, imu_j = imu_i;    
  FeaturePerFrame const& hob = lm.feature_per_frame.front();
  Eigen::Matrix3d Rwch = Rs[imu_i] * ric[0];
  Eigen::Vector3d twch = Ps[imu_i] + Rs[imu_i] * tic[0];
  Eigen::Vector3d fh = hob.point;
  Eigen::Vector3d pw = Rwch * fh * lm.estimated_depth + twch;

  inliers.clear();
  inliers.push_back(0);
  lm.vec_reproj_err.clear();
  lm.vec_inlier_flag.clear();
  lm.vec_reproj_err.push_back(0);
  lm.vec_inlier_flag.push_back(true);

  for(size_t i = 1; i < lm.feature_per_frame.size(); ++i) {
    imu_j++;
    Eigen::Vector3d twci = Ps[imu_j] + Rs[imu_j] * tic[0];
    Eigen::Matrix3d Rwci = Rs[imu_j] * ric[0];
    Eigen::Vector3d pci = Rwci.transpose() * (pw - twci);
    pci /= pci.z();
    Eigen::Vector3d f = lm.feature_per_frame[i].point;
    double err = (pci.head<2>() - f.head<2>()).norm() * FOCAL_LENGTH;
    lm.vec_reproj_err.push_back(err);
    if(err < REPROJ_THRESHOLD) {
      inliers.push_back(i);
      lm.vec_inlier_flag.push_back(true);
    }
    else {
      lm.vec_inlier_flag.push_back(false);
    }
  }
  assert(lm.vec_reproj_err.size() == lm.vec_inlier_flag.size());
}


void FeatureManager::checkReprojErrorWithDepth(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                                Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                                FeaturePerId& lm, std::vector<int>& inliers) 
{

  int imu_i = lm.start_frame, imu_j = imu_i;    
  FeaturePerFrame const& hob = lm.feature_per_frame.front();
  Eigen::Matrix3d Rwch = Rs[imu_i] * ric[0];
  Eigen::Vector3d twch = Ps[imu_i] + Rs[imu_i] * tic[0];
  Eigen::Vector3d fh = hob.point;
  Eigen::Vector3d pw = Rwch * fh * lm.estimated_depth + twch;

  inliers.clear();
  inliers.push_back(0);
  lm.vec_reproj_err.clear();
  lm.vec_depth_err.clear();
  lm.vec_inlier_flag.clear();
  lm.vec_reproj_err.push_back(0);
  lm.vec_depth_err.push_back(0);
  lm.vec_inlier_flag.push_back(true);

  for(size_t i = 1; i < lm.feature_per_frame.size(); ++i) {
    imu_j++;
    Eigen::Vector3d twci = Ps[imu_j] + Rs[imu_j] * tic[0];
    Eigen::Matrix3d Rwci = Rs[imu_j] * ric[0];
    Eigen::Vector3d pci = Rwci.transpose() * (pw - twci);
    const double proj_depth = pci.z();
    pci /= proj_depth;
    Eigen::Vector3d f = lm.feature_per_frame[i].point;
    double reproj_err = (pci.head<2>() - f.head<2>()).norm() * FOCAL_LENGTH;
    lm.vec_reproj_err.push_back(reproj_err);
    double depth_err;
    if (lm.feature_per_frame[i].z > MIN_DEPTH && lm.feature_per_frame[i].z < MAX_DEPTH) {
      depth_err = std::abs(proj_depth - lm.feature_per_frame[i].z) / lm.feature_per_frame[i].z;
    }
    else {
      depth_err = 0;
    }
    if(reproj_err < REPROJ_THRESHOLD || depth_err < 0.04) {
      inliers.push_back(i);
      lm.vec_inlier_flag.push_back(true);
    }
    else {
      lm.vec_inlier_flag.push_back(false);
    }
  }
  assert(lm.vec_reproj_err.size() == lm.vec_inlier_flag.size());
}

void FeatureManager::triangulateRansac(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], 
                                       Eigen::Vector3d tic[], Eigen::Matrix3d ric[], 
                                       FeaturePerId& lm, std::vector<int>& inliers) 
{
  int imu_i = lm.start_frame, imu_j = imu_i;
  FeaturePerFrame const& hob = lm.feature_per_frame.front();

  inliers.clear();
  std::vector<int> inliers_tmp;
  for(size_t i = 1; i < lm.feature_per_frame.size(); ++i) {
    imu_j = lm.start_frame + i;
    FeaturePerFrame const& ob = lm.feature_per_frame[i];
    triangulateTwoFrame(Ps[imu_i], Rs[imu_i], Ps[imu_j], Rs[imu_j], tic[0], ric[0], hob.point, ob.point, lm);
    checkReprojError(Ps, Rs, tic, ric, lm, inliers_tmp);
    if(inliers_tmp.size() > inliers.size()) {
      inliers = inliers_tmp;
    }
    inliers_tmp.clear();
  }
  if(inliers.size() <= 1)
    return;

  triangulateMultiFrameSet(Ps, Rs, tic, ric, lm, inliers);
  checkReprojError(Ps, Rs, tic, ric, lm, inliers);

  // std::cout << "inlier/view: " << inliers.size() << "/" << lm.feature_per_frame.size() << " ave project err: "; 
  // for(size_t i = 1; i < lm.vec_reproj_err.size(); ++i) {
  //   std::cout << lm.vec_reproj_err[i] << " ";
  // }
  // std::cout << std::endl;
  // for(size_t i = 1; i < lm.vec_inlier_flag.size(); ++i) {
  //   std::cout << lm.vec_inlier_flag[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << " triangulate ransac: " << lm.estimated_depth << std::endl;
  
}

void FeatureManager::triangulate(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
  // std::cout << "there are " << feature.size() << " landmarks" << std::endl;
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) {
      continue;
    }

    if(it_per_id.used_num == 2 && it_per_id.start_frame < WINDOW_SIZE - 2) {
      it_per_id.solve_flag = 2;
      continue;
    }
      

    if (it_per_id.status == FeaturePerId::Status::GOOD || it_per_id.status == FeaturePerId::Status::NORMAL) {
      // std::cout << "success" << std::endl;
      std::vector<int> inliers;
      checkReprojError(Ps, Rs, tic, ric, it_per_id, inliers);
      // std::cout << "inlier/view: " << inliers.size() << "/" << it_per_id.feature_per_frame.size() << " ave project err: "; 
      // for(size_t i = 1; i < it_per_id.vec_reproj_err.size(); ++i) {
      //   std::cout << it_per_id.vec_reproj_err[i] << " ";
      // }
      // std::cout << std::endl;
      // for(size_t i = 1; i < it_per_id.vec_inlier_flag.size(); ++i) {
      //   std::cout << it_per_id.vec_inlier_flag[i] << " ";
      // }
      // std::cout << std::endl;
      // std::cout << " triangulate ransac: " << it_per_id.estimated_depth << std::endl;
      continue;
    }
    std::vector<int> inliers;  
    triangulateRansac(Ps, Rs, tic, ric, it_per_id, inliers);
    // triangulateMultiFrame(Ps, Rs, tic, ric, it_per_id);
    // std::cout << "triangulate depth: " << it_per_id.estimated_depth << std::endl;
    if (it_per_id.estimated_depth < 0.1) {
      triangulateMultiFrame(Ps, Rs, tic, ric, it_per_id);
      if (it_per_id.estimated_depth < 0.1) {
        it_per_id.estimated_depth = -1.0f;
        it_per_id.solve_flag = 2;
      }
    }
  }
  // ROS_ASSERT(1==0);
}

void FeatureManager::triangulateWithDepth(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[])
{
  for (auto &it_per_id : feature) {
    it_per_id.used_num = it_per_id.feature_per_frame.size();
    if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
      continue;

    if (it_per_id.estimated_depth > 0) {
      std::vector<int> inliers;
      checkReprojErrorWithDepth(Ps, Rs, tic, ric, it_per_id, inliers);
      continue;
    }

    bool flag = false;
    int start_frame = it_per_id.start_frame;
    Eigen::Matrix3d Rwch = Rs[start_frame] * ric[0];
    Eigen::Vector3d twch = Ps[start_frame] + Rs[start_frame] * tic[0];
    for(int i = 0; i < (int)it_per_id.feature_per_frame.size(); i++) {
      double host_depth = it_per_id.feature_per_frame[i].z;
      if(host_depth > MIN_DEPTH && host_depth < MAX_DEPTH) {
        Eigen::Matrix3d Rwci = Rs[start_frame + i] * ric[0];
        Eigen::Vector3d twci = Ps[start_frame + i] + Rs[start_frame + i] * tic[0];
        Eigen::Vector3d pw = Rwci * it_per_id.feature_per_frame[i].point * host_depth + twci;
        Eigen::Vector3d pch = Rwch.transpose() * (pw - twch);
        it_per_id.estimated_depth = pch.z();
        std::vector<int> inliers;
        checkReprojErrorWithDepth(Ps, Rs, tic, ric, it_per_id, inliers);
        flag = true;
        break;
      }
    }

    if(!flag) {
      std::vector<int> inliers;  
      triangulateRansac(Ps, Rs, tic, ric, it_per_id, inliers);
    }
  }
}

void FeatureManager::evaluateQuality(Eigen::Vector3d Ps[], Eigen::Matrix3d Rs[], Eigen::Vector3d tic[], Eigen::Matrix3d ric[]) 
{
  int good_cnt = 0, normal_cnt = 0, weak_cnt = 0, not_init_cnt = 0;
  for (auto &lm : feature) {
    int inlier_size = 0;
    for (auto f: lm.vec_inlier_flag) {
      if(f) {
        inlier_size++;
      }
    }

    double inlier_ratio = (double)inlier_size / lm.feature_per_frame.size();
    if(inlier_ratio < 0.4) {
      // lm.status = FeaturePerId::Status::OUTLIER;
      lm.status = FeaturePerId::Status::TRIANGULATE_BAD;
      continue;
    }
    if(lm.estimated_depth < 0.1) {
      lm.status = FeaturePerId::Status::OUTLIER;
      continue;
    }

    double max_angle = 0.0;
    int imu_i = lm.start_frame, imu_j = imu_i;
    FeaturePerFrame const& hob = lm.feature_per_frame.front();    
    Eigen::Matrix3d Rwch = Rs[imu_i] * ric[0];
    Eigen::Vector3d twch = Ps[imu_i] + Rs[imu_i] * tic[0];
    Eigen::Vector3d fh = hob.point;
    Eigen::Vector3d pw = Rwch * fh * lm.estimated_depth + twch;

    for(size_t i = 1; i < lm.feature_per_frame.size(); ++i) {
      imu_j = lm.start_frame + i;
      Eigen::Vector3d twcj = Ps[imu_j] + Rs[imu_j] * tic[0];
      const Eigen::Vector3d v1 = (twch - pw).normalized();
      const Eigen::Vector3d v2 = (twcj - pw).normalized();
      double angle = std::acos(v1.dot(v2));
      if(angle > max_angle) {
        max_angle = angle;
      }
    }

    // std::cout << "max_angle: " << max_angle << std::endl;

    if(inlier_size <= 2) {
      lm.status = FeaturePerId::Status::NOT_INIT;
      not_init_cnt++;
    }
    else {
      if(max_angle < M_PI/360.0) {
        lm.status = FeaturePerId::Status::TRIANGULATE_WEAK;
        weak_cnt++;
      }
      else {
        if(inlier_size > 4) {
          if(max_angle > M_PI/36.0) {
            lm.status = FeaturePerId::Status::GOOD;
            good_cnt++;
          }
          else if(inlier_size > 6 && max_angle > M_PI/60.0) {
            lm.status = FeaturePerId::Status::GOOD;
            good_cnt++;
          }
          else {
            lm.status = FeaturePerId::Status::NORMAL;
            normal_cnt++;
          }
        }
      }
    }
  }
  normal_feature_sum = good_cnt + normal_cnt;
  // printf("[Evaluate] Good: %d, Normal: %d, Weak: %d, Not Init: %d\n", good_cnt, normal_cnt, weak_cnt, not_init_cnt);
}

void FeatureManager::removeOutlier()
{
  int i = -1;
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;
    i += it->used_num != 0;
    if (it->used_num != 0 && it->is_outlier == true && it->endFrame() < WINDOW_SIZE) {
      feature.erase(it);
    }
  }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else {
      Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() < 2) {
        feature.erase(it);
        continue;
      }
      else {
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
        Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
        double dep_j = pts_j(2);
        if (dep_j > 0) {
          it->estimated_depth = dep_j;
        }
        else {
          it->estimated_depth = INIT_DEPTH;
          it->solve_flag = 2;
        }
      }
    }
  }
}

void FeatureManager::removeBack()
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
  {
    it_next++;

    if (it->start_frame != 0)
      it->start_frame--;
    else
    {
      it->feature_per_frame.erase(it->feature_per_frame.begin());
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

void FeatureManager::removeFront(int frame_count)
{
  for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next) {
    it_next++;

    if (it->start_frame == frame_count) {
      it->start_frame--;
    }
    else {
      int j = WINDOW_SIZE - 1 - it->start_frame;
      if (it->endFrame() < frame_count - 1)
        continue;
      it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
      if (it->feature_per_frame.size() == 0)
        feature.erase(it);
    }
  }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
  //check the second last frame is keyframe or not
  //parallax betwwen seconde last frame and third last frame
  const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
  const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

  double ans = 0;
  Eigen::Vector3d p_j = frame_j.point;

  double u_j = p_j(0);
  double v_j = p_j(1);

  Eigen::Vector3d p_i = frame_i.point;
  Eigen::Vector3d p_i_comp;

  p_i_comp = p_i;
  double dep_i = p_i(2);
  assert(dep_i == 1.0);
  double u_i = p_i(0) / dep_i;
  double v_i = p_i(1) / dep_i;
  double du = u_i - u_j, dv = v_i - v_j;

  double dep_i_comp = p_i_comp(2);
  double u_i_comp = p_i_comp(0) / dep_i_comp;
  double v_i_comp = p_i_comp(1) / dep_i_comp;
  double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

  ans = std::max(ans, std::sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

  return ans;
}