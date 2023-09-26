#pragma once
#include <sys/types.h>
#include <sys/stat.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <cstring>
#include <eigen3/Eigen/Dense>

#define CV_COLOR_BLACK      cv::Scalar(0,0,0)          // 纯黑
#define CV_COLOR_WHITE      cv::Scalar(255,255,255)    // 纯白
#define CV_COLOR_RED        cv::Scalar(0,0,255)        // 纯红
#define CV_COLOR_GREEN      cv::Scalar(0,255,0)        // 纯绿
#define CV_COLOR_BLUE       cv::Scalar(255,0,0)        // 纯蓝

#define CV_COLOR_DARKGRAY   cv::Scalar(169,169,169)    // 深灰色
#define CV_COLOR_DARKRED    cv::Scalar(0,0,169)        // 深红色
#define CV_COLOR_ORANGERED  cv::Scalar(0,69,255)       // 橙红色

#define CV_COLOR_CHOCOLATE  cv::Scalar(30,105,210)     // 巧克力色
#define CV_COLOR_GOLD       cv::Scalar(10,215,255)     // 金色
#define CV_COLOR_YELLOW     cv::Scalar(0,255,255)      // 纯黄色

#define CV_COLOR_OLIVE      cv::Scalar(0,128,128)      // 橄榄色
#define CV_COLOR_LIGHTGREEN cv::Scalar(144,238,144)    // 浅绿色
#define CV_COLOR_DARKCYAN   cv::Scalar(139,139,0)      // 深青色
#define CV_COLOR_CYAN       cv::Scalar(255,255,0)      // 青色

#define CV_COLOR_SKYBLUE    cv::Scalar(235,206,135)    // 天蓝色 
#define CV_COLOR_INDIGO     cv::Scalar(130,0,75)       // 藏青色
#define CV_COLOR_PURPLE     cv::Scalar(128,0,128)      // 紫色

#define CV_COLOR_PINK       cv::Scalar(203,192,255)    // 粉色
#define CV_COLOR_DEEPPINK   cv::Scalar(147,20,255)     // 深粉色
#define CV_COLOR_VIOLET     cv::Scalar(238,130,238)    // 紫罗兰

class Utility
{
  public:
    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    template <typename Derived>
    static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q)
    {
        //printf("a: %f %f %f %f", q.w(), q.x(), q.y(), q.z());
        //Eigen::Quaternion<typename Derived::Scalar> p(-q.w(), -q.x(), -q.y(), -q.z());
        //printf("b: %f %f %f %f", p.w(), p.x(), p.y(), p.z());
        //return q.template w() >= (typename Derived::Scalar)(0.0) ? q : Eigen::Quaternion<typename Derived::Scalar>(-q.w(), -q.x(), -q.y(), -q.z());
        return q;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
    {
        Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
        ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) = qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
        return ans;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
    {
        Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
        Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
        ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
        ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) = pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
        return ans;
    }

    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
            sin(y), cos(y), 0,
            0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
            0., 1., 0.,
            -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
            0., cos(r), -sin(r),
            0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g);

    template <size_t N>
    struct uint_
    {
    };

    template <size_t N, typename Lambda, typename IterT>
    void unroller(const Lambda &f, const IterT &iter, uint_<N>)
    {
        unroller(f, iter, uint_<N - 1>());
        f(iter + N);
    }

    template <typename Lambda, typename IterT>
    void unroller(const Lambda &f, const IterT &iter, uint_<0>)
    {
        f(iter);
    }

    template <typename T>
    static T normalizeAngle(const T& angle_degrees) {
      T two_pi(2.0 * 180);
      if (angle_degrees > 0)
      return angle_degrees -
          two_pi * std::floor((angle_degrees + T(180)) / two_pi);
      else
        return angle_degrees +
            two_pi * std::floor((-angle_degrees + T(180)) / two_pi);
    };
};

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ExpSO3(const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;
  Scalar_t theta_norm;
  Scalar_t theta_sq = theta.squaredNorm();
  Scalar_t imag_factor;
  Scalar_t real_factor;
  if (theta_sq < 1e-10) {
    Scalar_t theta_po4 = theta_sq * theta_sq;
    imag_factor = Scalar_t(0.5) - Scalar_t(1.0 / 48.0) * theta_sq +
                  Scalar_t(1.0 / 3840.0) * theta_po4;
    real_factor = Scalar_t(1) - Scalar_t(1.0 / 8.0) * theta_sq +
                  Scalar_t(1.0 / 384.0) * theta_po4;
  } else {
    theta_norm = std::sqrt(theta_sq);
    Scalar_t half_theta = Scalar_t(0.5) * (theta_norm);
    Scalar_t sin_half_theta = std::sin(half_theta);
    imag_factor = sin_half_theta / (theta_norm);
    real_factor = std::cos(half_theta);
  }
  Eigen::Quaternion<Scalar_t> q(real_factor, imag_factor * theta.x(), imag_factor * theta.y(), imag_factor * theta.z());
  return q.toRotationMatrix();
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 1> LogSO3(const Eigen::MatrixBase<Derived> &R) {
  typedef typename Derived::Scalar Scalar_t;
  auto q = Eigen::Quaternion<Scalar_t>(R);
  Scalar_t squared_n = q.vec().squaredNorm();
  Scalar_t w = q.w();
  Scalar_t two_atan_nbyw_by_n;
  if (squared_n < 1e-10) {
    Scalar_t squared_w = w * w;
    two_atan_nbyw_by_n = Scalar_t(2) / w - Scalar_t(2.0 / 3.0) * (squared_n) / (w * squared_w);
  } else {
    Scalar_t n = std::sqrt(squared_n);
    Scalar_t atan_nbyw = (w < Scalar_t(0)) ? Scalar_t(std::atan2(-n, -w)) : Scalar_t(std::atan2(n, w));
    two_atan_nbyw_by_n = Scalar_t(2) * atan_nbyw / n;
  }
  return two_atan_nbyw_by_n * q.vec();
}


class FileSystemHelper
{
  public:

    /******************************************************************************
     * Recursively create directory if `path` not exists.
     * Return 0 if success.
     *****************************************************************************/
    static int createDirectoryIfNotExists(const char *path)
    {
        struct stat info;
        int statRC = stat(path, &info);
        if( statRC != 0 )
        {
            if (errno == ENOENT)  
            {
                printf("%s not exists, trying to create it \n", path);
                if (! createDirectoryIfNotExists(dirname(strdupa(path))))
                {
                    if (mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
                    {
                        fprintf(stderr, "Failed to create folder %s \n", path);
                        return 1;
                    }
                    else
                        return 0;
                }
                else 
                    return 1;
            } // directory not exists
            if (errno == ENOTDIR) 
            { 
                fprintf(stderr, "%s is not a directory path \n", path);
                return 1; 
            } // something in path prefix is not a dir
            return 1;
        }
        return ( info.st_mode & S_IFDIR ) ? 0 : 1;
    }
};
