#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

// if your ros version == noetic, you should remove this part
namespace cv {
// class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
// {
//  private:
//   std::function<void(const Range&)> m_functor;
//  public:
//   ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor) :
//     m_functor(functor)
//   {}    
//   virtual void operator() (const cv::Range& range) const
//   {
//     m_functor(range);
//   }
// };

// inline void parallel_for_(const cv::Range& range, std::function<void(const cv::Range&)> functor, double nstripes=-1.)
// {
//   parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
// }
}
// endif

bool inBorder(const cv::Point2f &pt)
{
  const int BORDER_SIZE = 1;
  int img_x = cvRound(pt.x);
  int img_y = cvRound(pt.y);
  return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
  int j = 0;
  for (int i = 0; i < int(v.size()); i++)
    if (status[i])
      v[j++] = v[i];
  v.resize(j);
}


FeatureTracker::FeatureTracker() {
  fast_det = cv::FastFeatureDetector::create(MIN_FAST_RESP);
}

void FeatureTracker::setMask()
{
  if(FISHEYE)
    mask = fisheye_mask.clone();
  else
    mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
  
  // prefer to keep features that are tracked for long time
  vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

  for (unsigned int i = 0; i < forw_pts.size(); i++)
    cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

  sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) {
    return a.first > b.first;
  });

  forw_pts.clear();
  ids.clear();
  track_cnt.clear();

  for (auto &it : cnt_pts_id) {
    if (mask.at<uchar>(it.second.first) == 255) {
      forw_pts.push_back(it.second.first);
      ids.push_back(it.second.second);
      track_cnt.push_back(it.first);
      cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
    }
  }
}

void FeatureTracker::addPoints()
{
  for (auto &p : n_pts) {
    forw_pts.push_back(p);
    ids.push_back(-1);
    track_cnt.push_back(1);
  }
}

void FeatureTracker::detectGridFAST(const cv::Mat &img, const int ncellsize, 
        const std::vector<cv::Point2f> &vcurkps, std::vector<cv::Point2f>& kps)
{    
  kps.clear();
  if(img.empty()) {
    return;
  }

  size_t ncols = img.cols;
  size_t nrows = img.rows;
  size_t nhcells = nrows / ncellsize;
  size_t nwcells = ncols / ncellsize;
  size_t nbcells = nhcells * nwcells;
  kps.reserve(nbcells);

  std::vector<std::vector<bool>> voccupcells(
    nhcells+1, 
    std::vector<bool>(nwcells+1, false)
  );

  cv::Mat mask = cv::Mat::ones(img.rows, img.cols, CV_32F);

  for(const auto &px : vcurkps) {
    voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
    cv::circle(mask, px, ncellsize, cv::Scalar(0), -1);
  }

  size_t nboccup = 0;
  size_t nbempty = 0;
  std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells);
  auto cvrange = cv::Range(0, nbcells);

  parallel_for_(cvrange, [&](const cv::Range& range) {
    for( int i = range.start ; i < range.end ; i++ ) {
      size_t r = floor(i / nwcells);
      size_t c = i % nwcells;

      if( voccupcells[r][c] ) {
        nboccup++;
        continue;
      }

      nbempty++;
      size_t x = c*ncellsize;
      size_t y = r*ncellsize;
      cv::Rect hroi(x,y,ncellsize,ncellsize);

      if(x+ncellsize < ncols-1 && y+ncellsize < nrows-1) {
        std::vector<cv::KeyPoint> vkps;
        fast_det->detect(img(hroi), vkps, mask(hroi));
        if(vkps.empty()) {
          continue;
        } else {
          std::sort(vkps.begin(), vkps.end(), [&](cv::KeyPoint first, cv::KeyPoint second) {
            return first.response > second.response;
          });
        }
        if(vkps.at(0).response >= fast_det->getThreshold()) {
          cv::Point2f pxpt = vkps.at(0).pt;
          pxpt.x += x;
          pxpt.y += y;
          cv::circle(mask, pxpt, ncellsize, cv::Scalar(0), -1);
          vvdetectedpx.at(i).push_back(pxpt);
        }
      }
    }
  });

  for( const auto &vpx : vvdetectedpx ) {
    if(!vpx.empty()) {
      kps.insert(kps.end(), vpx.begin(), vpx.end());
    }
  }

  size_t nbkps = kps.size();

  // Update FAST th.
  int nfast_th = fast_det->getThreshold();
  if(nbkps < 0.5 * nbempty && nbempty > 10) {
    nfast_th *= 0.5;
    nfast_th = std::max(MIN_FAST_RESP, nfast_th);
    fast_det->setThreshold(nfast_th);
  } else if (nbkps == nbempty) {
    nfast_th *= 2.0;
    fast_det->setThreshold(nfast_th);
  }

  // Compute Corners with Sub-Pixel Accuracy
  if(!kps.empty()) {
    /// Set the need parameters to find the refined corners
    cv::Size winSize = cv::Size(3,3);
    cv::Size zeroZone = cv::Size(-1,-1);
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
    cv::cornerSubPix(img, kps, winSize, zeroZone, criteria);
  }
}

void FeatureTracker::trackMono(const cv::Mat &_img, double _cur_time)
{
  cv::Mat img;
  TicToc t_r;
  cur_time = _cur_time;

  if (EQUALIZE) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    TicToc t_c;
    clahe->apply(_img, img);
    ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
  }
  else
    img = _img;

  if (forw_img.empty()) {
    prev_img = cur_img = forw_img = img;
  }
  else {
    forw_img = img;
  }
  forw_pts.clear();
  if (cur_pts.size() > 0) {
    TicToc t_o;
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

    for (int i = 0; i < int(forw_pts.size()); i++)
      if (status[i] && !inBorder(forw_pts[i]))
        status[i] = 0;

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());

    // std::cout << "track feature: " << track_cnt.size() << std::endl;

    std::vector<cv::Point2f> back_pts;
    cv::calcOpticalFlowPyrLK(forw_img, cur_img, forw_pts, back_pts, status, err, cv::Size(21, 21), 3);

    for(int i = 0; i < int(forw_pts.size()); i++) {
      if(!status[i] || cv::norm(cur_pts[i] - back_pts[i]) > 0.5) {
        status[i] = false;
      }
    }

    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(ids, status);
    reduceVector(cur_un_pts, status);
    reduceVector(track_cnt, status);
  }

  for (auto &n : track_cnt)
    n++;

  if (PUB_THIS_FRAME) {
    rejectWithF();
    ROS_DEBUG("set mask begins");
    TicToc t_m;
    setMask();
    ROS_DEBUG("set mask costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
    if (n_max_cnt > 0) {
      if(mask.empty())
        cout << "mask is empty " << endl;
      if (mask.type() != CV_8UC1)
        cout << "mask type wrong " << endl;
      if (mask.size() != forw_img.size())
        cout << "wrong size " << endl;
      
      if(USE_FAST)
        detectGridFAST(forw_img, MIN_DIST, forw_pts, n_pts);
      else
        cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
      // std::cout << "detect new feature: " << n_pts.size() << std::endl;
    } 
    else
      n_pts.clear();
    ROS_DEBUG("detect feature costs: %fms", t_t.toc());

    ROS_DEBUG("add feature begins");
    TicToc t_a;
    addPoints();
    ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
  }
  prev_img = cur_img;
  prev_pts = cur_pts;
  prev_un_pts = cur_un_pts;
  cur_img = forw_img;
  cur_pts = forw_pts;
  undistortedPoints();
  prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
  if (forw_pts.size() >= 8)
  {
    ROS_DEBUG("FM ransac begins");
    TicToc t_f;
    vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      Eigen::Vector3d tmp_p;
      m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

      m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
      tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
      tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
      un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }

    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
    int size_a = cur_pts.size();
    reduceVector(prev_pts, status);
    reduceVector(cur_pts, status);
    reduceVector(forw_pts, status);
    reduceVector(cur_un_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
    ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
    ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
  }
}

bool FeatureTracker::updateID(unsigned int i)
{
  if (i < ids.size())
  {
    if (ids[i] == -1)
      ids[i] = n_id++;
    return true;
  }
  else
    return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
  ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
  m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
  cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
  vector<Eigen::Vector2d> distortedp, undistortedp;
  for (int i = 0; i < COL; i++)
    for (int j = 0; j < ROW; j++)
    {
      Eigen::Vector2d a(i, j);
      Eigen::Vector3d b;
      m_camera->liftProjective(a, b);
      distortedp.push_back(a);
      undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
      //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
    }
  for (int i = 0; i < int(undistortedp.size()); i++)
  {
    cv::Mat pp(3, 1, CV_32FC1);
    pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
    pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
    pp.at<float>(2, 0) = 1.0;
    //cout << trackerData[0].K << endl;
    //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
    //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
    if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
    {
      undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
    }
    else
    {
      //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
    }
  }
  cv::imshow(name, undistortedImg);
  cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
  cur_un_pts.clear();
  cur_un_pts_map.clear();
  //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
  for (unsigned int i = 0; i < cur_pts.size(); i++)
  {
    Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
    Eigen::Vector3d b;
    m_camera->liftProjective(a, b);
    cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
    //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
  }
  // caculate points velocity
  if (!prev_un_pts_map.empty())
  {
    double dt = cur_time - prev_time;
    pts_velocity.clear();
    for (unsigned int i = 0; i < cur_un_pts.size(); i++)
    {
      if (ids[i] != -1)
      {
        std::map<int, cv::Point2f>::iterator it;
        it = prev_un_pts_map.find(ids[i]);
        if (it != prev_un_pts_map.end())
        {
          double v_x = (cur_un_pts[i].x - it->second.x) / dt;
          double v_y = (cur_un_pts[i].y - it->second.y) / dt;
          pts_velocity.push_back(cv::Point2f(v_x, v_y));
        }
        else
          pts_velocity.push_back(cv::Point2f(0, 0));
      }
      else
      {
        pts_velocity.push_back(cv::Point2f(0, 0));
      }
    }
  }
  else
  {
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
      pts_velocity.push_back(cv::Point2f(0, 0));
    }
  }
  prev_un_pts_map = cur_un_pts_map;
}
