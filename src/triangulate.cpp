#include "vo_nono/keypoint/triangulate.h"

#include <opencv2/calib3d.hpp>

#include "vo_nono/util/geometry.h"

namespace vo_nono {
void Triangulator::triangulate(const cv::Mat& proj1, const cv::Mat& proj2,
                               const std::vector<cv::Point2f>& img_pt1,
                               const std::vector<cv::Point2f>& img_pt2,
                               std::vector<cv::Mat>& result) {
    assert(img_pt1.size() == img_pt2.size());
    if (img_pt1.empty()) { return; }
    cv::Mat tri_res;
    cv::triangulatePoints(proj1, proj2, img_pt1, img_pt2, tri_res);
    result.reserve(tri_res.cols);
    for (int i = 0; i < tri_res.cols; ++i) {
        result.push_back(Geometry::hm3d_to_euclid3d(tri_res.col(i)));
    }
}

int Triangulator::filter_triangulate(const cv::Mat& Rcw1, const cv::Mat& tcw1,
                                     const cv::Mat& Rcw2, const cv::Mat& tcw2,
                                     const std::vector<cv::Mat>& tri_res,
                                     std::vector<bool>& is_inlier,
                                     double grad_th) {
    int cnt_inlier = 0;
    is_inlier.resize(tri_res.size(), true);
    for (size_t i = 0; i < tri_res.size(); ++i) {
        const cv::Mat pt = tri_res[i];
        if (!std::isfinite(pt.at<float>(0, 0)) ||
            !std::isfinite(pt.at<float>(1, 0)) ||
            !std::isfinite(pt.at<float>(2, 0))) {
            is_inlier[i] = false;
            continue;
        }
        cv::Mat coord_c1 = Geometry::transform_coord(Rcw1, tcw1, pt);
        cv::Mat coord_c2 = Geometry::transform_coord(Rcw2, tcw2, pt);
        // depth must be positive
        if (coord_c1.at<float>(2, 0) < EPS || coord_c2.at<float>(2, 0) < EPS) {
            is_inlier[i] = false;
            continue;
        }
        // compute parallax
        cv::Mat op1 = pt + tcw1;// (coord - (-tcw1) = coord + tcw1)
        op1 /= cv::norm(op1);
        cv::Mat op2 = pt + tcw2;
        op2 /= cv::norm(op2);
        double sin_theta3 = cv::norm(op1.cross(op2));
        if (grad_th * sin_theta3 * sin_theta3 < 1.0) {
            is_inlier[i] = false;
            continue;
        }
        cnt_inlier += 1;
    }
    return cnt_inlier;
}

int Triangulator::triangulate_and_filter_frames(
        const Frame* frame1, const Frame* frame2,
        const cv::Mat& cam_intrinsic_mat,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Mat>& tri_result, std::vector<bool>& is_inlier,
        double grad_th) {
    std::vector<cv::Point2f> img_pt1, img_pt2;
    cv::Mat proj1 = Geometry::get_proj_mat(cam_intrinsic_mat, frame1->get_Rcw(),
                                           frame1->get_Tcw());
    cv::Mat proj2 = Geometry::get_proj_mat(cam_intrinsic_mat, frame2->get_Rcw(),
                                           frame2->get_Tcw());
    for (auto& match : matches) {
        img_pt1.push_back(frame1->feature_points[match.queryIdx].keypoint.pt);
        img_pt2.push_back(frame2->feature_points[match.trainIdx].keypoint.pt);
    }
    triangulate(proj1, proj2, img_pt1, img_pt2, tri_result);
    return filter_triangulate(frame1->get_Rcw(), frame1->get_Tcw(),
                              frame2->get_Rcw(), frame2->get_Tcw(), tri_result,
                              is_inlier, grad_th);
}
}// namespace vo_nono