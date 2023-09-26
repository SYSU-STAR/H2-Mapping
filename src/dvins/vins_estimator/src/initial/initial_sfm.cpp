#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &pose_i, Eigen::Matrix<double, 3, 4> &pose_j,
						Vector2d &point_i, Vector2d &point_j, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point_i[0] * pose_i.row(2) - pose_i.row(0);
	design_matrix.row(1) = point_i[1] * pose_i.row(2) - pose_i.row(1);
	design_matrix.row(2) = point_j[0] * pose_j.row(2) - pose_j.row(0);
	design_matrix.row(3) = point_j[1] * pose_j.row(2) - pose_j.row(1);
	Vector4d point_4d = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = point_4d(0) / point_4d(3);
	point_3d(1) = point_4d(1) / point_4d(3);
	point_3d(2) = point_4d(2) / point_4d(3);
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &Rinit, Vector3d &Pinit, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts2ds;
	vector<cv::Point3f> pts3ds;
	for (int j = 0; j < feature_num; j++) {
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].ob_vec.size(); k++) {
			if (sfm_f[j].ob_vec[k].first == i) {
				Vector2d img_pts = sfm_f[j].ob_vec[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts2ds.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts3ds.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts2ds.size()) < 15) {
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts2ds.size()) < 10)
			return false;
	}
	cv::Mat R, rvec, tvec, D, tmp_r;
	cv::eigen2cv(Rinit, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(Pinit, tvec);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts3ds, pts2ds, K, D, rvec, tvec, 1);
	if(!pnp_succ) {
		return false;
	}
	cv::Rodrigues(rvec, R);
	MatrixXd R_pnp;
	cv::cv2eigen(R, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(tvec, T_pnp);
	Rinit = R_pnp;
	Pinit = T_pnp;
	return true;

}

void GlobalSFM::triangulateTwoFrames(int frame_i, Eigen::Matrix<double, 3, 4> &pose_i, 
									 int frame_j, Eigen::Matrix<double, 3, 4> &pose_j,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame_i != frame_j);
	for (int j = 0; j < feature_num; j++) {
		if (sfm_f[j].state == true)
			continue;
		bool ob_i = false, ob_j = false;
		Vector2d point_i;
		Vector2d point_j;
		for (int k = 0; k < (int)sfm_f[j].ob_vec.size(); k++) {
			if (sfm_f[j].ob_vec[k].first == frame_i) {
				point_i = sfm_f[j].ob_vec[k].second;
				ob_i = true;
			}
			if (sfm_f[j].ob_vec[k].first == frame_j) {
				point_j = sfm_f[j].ob_vec[k].second;
				ob_j = true;
			}
		}
		if (ob_i && ob_j) {
			Vector3d point_3d;
			triangulatePoint(pose_i, pose_j, point_i, point_j, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_j << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

void GlobalSFM::triangulateTwoFramesWithDepth(int frame_i, Eigen::Matrix<double, 3, 4> &pose_i,
                                              int frame_j, Eigen::Matrix<double, 3, 4> &pose_j,
                                              vector<SFMFeature> &sfm_f)
{
	assert(frame_i != frame_j);
	Matrix3d Ri = pose_i.block<3, 3>(0, 0);
	Matrix3d Rj = pose_j.block<3, 3>(0, 0);
	Vector3d ti = pose_i.block<3, 1>(0, 3);
	Vector3d tj = pose_j.block<3, 1>(0, 3);
	for (int id = 0; id < feature_num; id++) {
		if (sfm_f[id].state == true)
			continue;
		bool ob_i = false, ob_j = false;
    bool depth_i = false, depth_j = false;
		Vector3d point_i, point_j;
		Vector2d point_n_i, point_n_j;
		for (int k = 0; k < (int)sfm_f[id].ob_vec.size(); k++) {
      double depth = sfm_f[id].depth_vec[k].second;
			if (sfm_f[id].ob_vec[k].first == frame_i) {
				point_i = Vector3d(sfm_f[id].ob_vec[k].second.x() * depth, 
                          sfm_f[id].ob_vec[k].second.y() * depth,
                          depth);
        point_n_i = sfm_f[id].ob_vec[k].second;
				ob_i = true;
        if (depth > MIN_DEPTH && depth < MAX_DEPTH) {
          depth_i = true;
        }
			}
			if (sfm_f[id].ob_vec[k].first == frame_j) {
        point_j = Vector3d(sfm_f[id].ob_vec[k].second.x() * depth, 
                          sfm_f[id].ob_vec[k].second.y() * depth,
                          depth);
				point_n_j = sfm_f[id].ob_vec[k].second;
        ob_j = true;
        if (depth > MIN_DEPTH && depth < MAX_DEPTH) {
          depth_j = true;
        }
			}
		}
    const double max_err = REPROJ_THRESHOLD/FOCAL_LENGTH;
		if (ob_i && ob_j) {
      Vector3d point_3d;
      if(depth_i && !depth_j) {
        Vector3d point_in_j;
        point_3d = Ri.transpose() * (point_i - ti);
        point_in_j = Rj * point_3d + tj;
        point_in_j /= point_in_j.z();
        double err_j = (point_n_j - point_in_j.head<2>()).norm();
        if (err_j < max_err){
          sfm_f[id].state = true;
          sfm_f[id].position[0] = point_3d(0);
          sfm_f[id].position[1] = point_3d(1);
          sfm_f[id].position[2] = point_3d(2);
        }
      }
      else if(!depth_i && depth_j) {
        Vector3d point_in_i;
        point_3d = Rj.transpose()* (point_j - tj);
        point_in_i = Ri * point_3d + ti;
        point_in_i /= point_in_i.z();
        double err_i = (point_n_i - point_in_i.head<2>()).norm();
        if (err_i < max_err){
          sfm_f[id].state = true;
          sfm_f[id].position[0] = point_3d(0);
          sfm_f[id].position[1] = point_3d(1);
          sfm_f[id].position[2] = point_3d(2);
        }
      }
      else if(depth_i && depth_j) {
        Vector3d point_in_i, point_in_j;
        Eigen::Vector3d point_3d_i = Ri.transpose()* (point_i - ti);
        Eigen::Vector3d point_3d_j = Rj.transpose()* (point_j - tj);
        point_in_i = Ri * point_3d_j + ti;
        point_in_j = Rj * point_3d_i + tj;
        point_in_i /= point_in_i.z();
        point_in_j /= point_in_j.z();
        const double err_i = (point_n_i - point_in_i.head<2>()).norm();
        const double err_j = (point_n_j - point_in_j.head<2>()).norm();
        if (err_i < max_err && err_j < max_err){
          sfm_f[id].state = true;
          sfm_f[id].position[0] = (point_3d_i(0) + point_3d_j(0)) * 0.5;
          sfm_f[id].position[1] = (point_3d_i(1) + point_3d_j(1)) * 0.5;
          sfm_f[id].position[2] = (point_3d_i(2) + point_3d_j(2)) * 0.5;
        }
        else if(err_i < max_err && err_j > max_err) {
          sfm_f[id].state = true;
          sfm_f[id].position[0] = point_3d_j(0);
          sfm_f[id].position[1] = point_3d_j(1);
          sfm_f[id].position[2] = point_3d_j(2);
        }
        else if(err_i > max_err && err_j < max_err) {
          sfm_f[id].state = true;
          sfm_f[id].position[0] = point_3d_i(0);
          sfm_f[id].position[1] = point_3d_i(1);
          sfm_f[id].position[2] = point_3d_i(2);
        }
        else {
          sfm_f[id].state = false;
        }
      }
      else {
        triangulatePoint(pose_i, pose_j, point_n_i, point_n_j, point_3d);
        sfm_f[id].state = true;
        sfm_f[id].position[0] = point_3d(0);
        sfm_f[id].position[1] = point_3d(1);
        sfm_f[id].position[2] = point_3d(2);
      }
		}
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
                          const Matrix3d Rrel, const Vector3d trel,
                          vector<SFMFeature> &sfm_f, map<int, Vector3d> &landmarks)
{
	feature_num = sfm_f.size();
	// intial two view
  q[l].setIdentity();
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(Rrel);
	T[frame_num - 1] = trel;

	//rotate to cam frame
	Matrix3d Rcws[frame_num];
	Vector3d tcws[frame_num];
  Vector3d rcws[frame_num];

  double para_pose[frame_num][6];
	Eigen::Matrix<double, 3, 4> Tcws[frame_num];
	Rcws[l] = q[l].toRotationMatrix().transpose();
	tcws[l] = -1 * (q[l].inverse() * T[l]);
  rcws[l] = LogSO3(Rcws[l]);
	Tcws[l].block<3, 3>(0, 0) = Rcws[l];
	Tcws[l].block<3, 1>(0, 3) = tcws[l];

  Rcws[frame_num - 1] = q[frame_num - 1].toRotationMatrix().transpose();
	tcws[frame_num - 1] = -1 * (Rcws[frame_num - 1] * T[frame_num - 1]);
  rcws[frame_num - 1] = LogSO3(Rcws[frame_num - 1]);
	Tcws[frame_num - 1].block<3, 3>(0, 0) = Rcws[frame_num - 1];
	Tcws[frame_num - 1].block<3, 1>(0, 3) = tcws[frame_num - 1];

	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++) {
		// solve pnp
		if (i > l) {
			Matrix3d Rinit = Rcws[i - 1];
			Vector3d Pinit = tcws[i - 1];
			if(!solveFrameByPnP(Rinit, Pinit, i, sfm_f))
				return false;
			Rcws[i] = Rinit;
			tcws[i] = Pinit;
      rcws[i] = LogSO3(Rcws[i]);
			Tcws[i].block<3, 3>(0, 0) = Rcws[i];
			Tcws[i].block<3, 1>(0, 3) = tcws[i];
		}

		// triangulate point based on the solve pnp result
    if(USE_RGBD) {
      triangulateTwoFramesWithDepth(i, Tcws[i], frame_num - 1, Tcws[frame_num - 1], sfm_f);
    }
    else {
      triangulateTwoFrames(i, Tcws[i], frame_num - 1, Tcws[frame_num - 1], sfm_f);
    }
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++) {
    if(USE_RGBD) {
      triangulateTwoFramesWithDepth(l, Tcws[l], i, Tcws[i], sfm_f);
    }
    else {
      triangulateTwoFrames(l, Tcws[l], i, Tcws[i], sfm_f);
    }
  }
		
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--) {
		//solve pnp
		Matrix3d Rinit = Rcws[i + 1];
		Vector3d Pinit = tcws[i + 1];
		if(!solveFrameByPnP(Rinit, Pinit, i, sfm_f))
			return false;
		Rcws[i] = Rinit;
		tcws[i] = Pinit;
    rcws[i] = LogSO3(Rcws[i]);
		Tcws[i].block<3, 3>(0, 0) = Rcws[i];
		Tcws[i].block<3, 1>(0, 3) = tcws[i];
		//triangulate
		triangulateTwoFrames(i, Tcws[i], l, Tcws[l], sfm_f);
	}
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++) {
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].ob_vec.size() >= 2) {
			Vector2d point_i, point_j;
			int frame_0 = sfm_f[j].ob_vec[0].first;
			point_i = sfm_f[j].ob_vec[0].second;
			int frame_1 = sfm_f[j].ob_vec.back().first;
			point_j = sfm_f[j].ob_vec.back().second;
			Vector3d point_3d;
			triangulatePoint(Tcws[frame_0], Tcws[frame_1], point_i, point_j, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
		}		
	}

	//full BA
	ceres::Problem problem;
  ceres::LossFunction *loss_func = new ceres::CauchyLoss(1.0);
  ceres::LocalParameterization* local_parameterization = new PoseLocalLeftMultiplyParameterization();
	for (int i = 0; i < frame_num; i++) {
    para_pose[i][0] = tcws[i].x();
    para_pose[i][1] = tcws[i].y();
    para_pose[i][2] = tcws[i].z();
    para_pose[i][3] = rcws[i].x();
    para_pose[i][4] = rcws[i].y();
    para_pose[i][5] = rcws[i].z();
		problem.AddParameterBlock(para_pose[i], 6, local_parameterization);
		if (i == l) {
      problem.SetParameterBlockConstant(para_pose[i]);
		}
	}

	for (int i = 0; i < feature_num; i++) {
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].ob_vec.size()); j++) {
			int l = sfm_f[i].ob_vec[j].first;
      Eigen::Vector3d tcw(para_pose[l][0], para_pose[l][1], para_pose[l][2]);
      Eigen::Matrix3d Rcw = ExpSO3(Eigen::Vector3d(para_pose[l][3], para_pose[l][4], para_pose[l][5]));
      Eigen::Vector3d Pw(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
      Eigen::Vector3d Pc = Rcw * Pw + tcw;
      double reproj_err = (Pc.head<2>() / Pc.z() - sfm_f[i].ob_vec[j].second).norm() * FOCAL_LENGTH;
      if(reproj_err < REPROJ_THRESHOLD) {
        ceres::CostFunction* project_factor = new ProjectionXYZFactor(sfm_f[i].ob_vec[j].second);
        problem.AddResidualBlock(project_factor, loss_func, para_pose[l], sfm_f[i].position);	 
        projection_edges.push_back(std::make_pair(std::make_pair(l, i), sfm_f[i].ob_vec[j].second));
        sfm_f[i].flags.push_back(true);
      }
      else {
        sfm_f[i].flags.push_back(false);
      }
		}

    if(USE_RGBD) {
      for(int j = 0; j < int(sfm_f[i].depth_vec.size()); j++) {
        int l = sfm_f[i].depth_vec[j].first;
        double depth = sfm_f[i].depth_vec[j].second;
        if(depth < MIN_DEPTH || depth > MAX_DEPTH)
          continue;
        ceres::CostFunction* depth_factor = new DepthXYZFactor(depth, DEPTH_COV);
        problem.AddResidualBlock(depth_factor, loss_func, para_pose[l], sfm_f[i].position);
        depth_edges.push_back(std::make_pair(std::make_pair(l, i), depth)); 
      }
    }
	}

  evaluateError(para_pose, sfm_f);
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	// options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.05;
  options.function_tolerance = 1e-3;
  options.gradient_tolerance = 1e-3;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
  if (!summary.termination_type == ceres::CONVERGENCE) {
		return false;
	}
  evaluateError(para_pose, sfm_f);
  std::cout << summary.BriefReport() << "\n";
	for (int i = 0; i < frame_num; i++) {
    Eigen::Vector3d tcw(para_pose[i][0], para_pose[i][1], para_pose[i][2]);
    Eigen::Vector3d rcw(para_pose[i][3], para_pose[i][4], para_pose[i][5]);
    q[i] = Eigen::Quaterniond(ExpSO3(rcw)).inverse();
		T[i] = -1 * (q[i] * tcw);
	}

	for (int i = 0; i < (int)sfm_f.size(); i++) {
		if(sfm_f[i].state)
			landmarks[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
  // assert(1==0);
	return true;
}

void GlobalSFM::evaluateError(double para_pose[][6], const vector<SFMFeature> &sfm_f)
{
	double reproj_err = 0.0;
	int reproj_cnt = 0;
	for(auto e: projection_edges) {
		int frame_id = e.first.first;
		int point_id = e.first.second;
		Eigen::Vector2d image_ob = e.second;

		Eigen::Vector3d tcw(para_pose[frame_id][0], para_pose[frame_id][1], para_pose[frame_id][2]);
    Eigen::Matrix3d Rcw = ExpSO3(Eigen::Vector3d(para_pose[frame_id][3], para_pose[frame_id][4], para_pose[frame_id][5]));
		Eigen::Vector3d Pw(sfm_f[point_id].position[0], sfm_f[point_id].position[1], sfm_f[point_id].position[2]);

		Eigen::Vector3d Pc = Rcw * Pw + tcw;
		reproj_err += (Pc.head<2>() / Pc.z() - image_ob).norm();
		reproj_cnt++;
	}
	if(reproj_cnt != 0)
		reproj_err /= reproj_cnt;

	printf("[GlobalSFM::evaluateError] ATE P: %lf\n", reproj_err);

  if(USE_RGBD) {
    double depth_error = 0.0;
    int depth_cnt = 0;
    for(auto e: depth_edges) {
      int frame_id = e.first.first;
      int point_id = e.first.second;
      double depth_ob = e.second;

      Eigen::Vector3d tcw(para_pose[frame_id][0], para_pose[frame_id][1], para_pose[frame_id][2]);
      Eigen::Matrix3d Rcw = ExpSO3(Eigen::Vector3d(para_pose[frame_id][3], para_pose[frame_id][4], para_pose[frame_id][5]));
      Eigen::Vector3d Pw(sfm_f[point_id].position[0], sfm_f[point_id].position[1], sfm_f[point_id].position[2]);

      Eigen::Vector3d Pc = Rcw * Pw + tcw;
      depth_error += std::fabs(Pc.z() - depth_ob);
      // std::cout << "depth_error: " << std::fabs(Pc.z() - depth_ob) << std::endl;
      depth_cnt++;
    }
    if(depth_cnt != 0)
      depth_error /= depth_cnt;
    printf("[GlobalSFM::evaluateError] ATE D: %lf\n", depth_error);
  }

}