#ifndef VO_NONO_GEOMETRY_H
#define VO_NONO_GEOMETRY_H

#include <array>
#include <opencv2/calib3d.hpp>

namespace vo_nono {
class Geometry {
public:
    static cv::Mat get_proj_mat(const cv::Mat &camera_intrin,
                                const cv::Mat &Rcw, const cv::Mat &tcw) {
        assert(Rcw.type() == CV_32F);
        assert(tcw.type() == CV_32F);
        cv::Mat proj = cv::Mat::zeros(3, 4, CV_32F);
        Rcw.copyTo(proj.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(proj.rowRange(0, 3).col(3));
        proj = camera_intrin * proj;
        return proj;
    }

    static cv::Mat hm3d_to_euclid3d(const cv::Mat &pt) {
        assert(pt.type() == CV_32F);
        cv::Mat res = cv::Mat::zeros(3, 1, CV_32F);
        pt.rowRange(0, 3).copyTo(res);
        res /= pt.at<float>(3);
        return res;
    }

    static cv::Mat euclid3d_to_hm3d(const cv::Mat &pt) {
        assert(pt.type() == CV_32F);
        cv::Mat res = cv::Mat::zeros(4, 1, CV_32F);
        pt.copyTo(res.rowRange(0, 3));
        res.at<float>(3, 0) = 1.0f;
        return res;
    }

    static cv::Point2f hm2d_to_euclid2d(const cv::Mat &pt) {
        assert(pt.type() == CV_32F);
        cv::Mat normalized = pt / pt.at<float>(2);
        cv::Point2f res(normalized.at<float>(0), normalized.at<float>(1));
        return res;
    }

    static cv::Point2f project_hm3d(const cv::Mat &proj_mat,
                                    const cv::Mat &pt) {
        return hm2d_to_euclid2d(proj_mat * pt);
    }

    static cv::Point2f project_euclid3d(const cv::Mat &proj_mat,
                                        const cv::Mat &pt) {
        cv::Mat res = proj_mat * euclid3d_to_hm3d(pt);
        return hm2d_to_euclid2d(res);
    }

    // project but check whether the point is in front of the point(z-value is positive)
    static bool project_euclid3d_in_front(const cv::Mat &cam_intrinsic,
                                         const cv::Mat &Rcw, const cv::Mat &tcw,
                                         const cv::Mat &coord,
                                         cv::Point2f &pixel) {
        cv::Mat coord_trans = Geometry::transform_coord(Rcw, tcw, coord);
        if (coord_trans.at<float>(2) <= 0) { return false; }
        pixel = Geometry::hm2d_to_euclid2d(cam_intrinsic * coord_trans);
        return true;
    }

    static double reprojection_err2(const cv::Mat &proj_mat,
                                    const cv::Mat &coord,
                                    const cv::Point2f &pixel) {
        cv::Point2f projected_pixel =
                hm2d_to_euclid2d(proj_mat * euclid3d_to_hm3d(coord));
        auto dx = double(projected_pixel.x - pixel.x);
        auto dy = double(projected_pixel.y - pixel.y);
        return dx * dx + dy * dy;
    }

    static cv::Mat transform_coord(const cv::Mat &Rcw, const cv::Mat &tcw,
                                   const cv::Mat &pt_world) {
        return Rcw * pt_world + tcw;
    }

    static void relative_pose(const cv::Mat &Rcw1, const cv::Mat &tcw1,
                              const cv::Mat &Rcw2, const cv::Mat &tcw2,
                              cv::Mat &R21, cv::Mat &t21) {
        t21 = tcw2 - tcw1;
        R21 = Rcw2 * Rcw1.t();
    }

    static void angle_axis_to_rotation_mat(const double angle_axis[3],
                                           double result[3][3]) {
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat R;
        for (int i = 0; i < 3; ++i) { rvec.at<double>(i) = angle_axis[i]; }
        cv::Rodrigues(rvec, R);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) { result[i][j] = R.at<double>(i, j); }
        }
    }

    static void angle_axis_to_rotation_mat(const double angle_axis[3],
                                           cv::Mat &rotation) {
        double result[3][3];
        angle_axis_to_rotation_mat(angle_axis, result);
        rotation = cv::Mat::zeros(3, 3, CV_32F);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rotation.at<float>(i, j) = (float) result[i][j];
            }
        }
    }

    static std::array<double, 3> rotation_mat_to_angle_axis(const cv::Mat &R);
};
}// namespace vo_nono

#endif//VO_NONO_GEOMETRY_H
