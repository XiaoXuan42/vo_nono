#include "vo_nono/keypoint/triangulate.h"

#include <opencv2/calib3d.hpp>

#include "vo_nono/util/geometry.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
// triangulate one points given two observation
cv::Mat Triangulator::triangulate(const cv::Mat& proj1, const cv::Mat& proj2,
                                  const cv::Point2f& pixel1,
                                  const cv::Point2f& pixel2) {
    std::vector<cv::Mat> proj{proj1, proj2};
    std::vector<cv::Point2f> pixels{pixel1, pixel2};
    return triangulate(proj, pixels);
}

// triangulate one points given multiple observations
cv::Mat Triangulator::triangulate(const std::vector<cv::Mat>& proj,
                                  const std::vector<cv::Point2f>& pixels) {
    assert(proj.size() == pixels.size());
    std::vector<float> weight(pixels.size(), 1.0f);
    cv::Mat A = cv::Mat::zeros(2 * (int) proj.size(), 4, CV_32F);
    float max_change = 0.0f;
    int iter_cnt = 0;
    cv::Mat res, old_res;
    do {
        max_change = 0.0f;
        for (int i = 0; i < int(proj.size()); i++) {
            cv::Mat r1 = proj[i].row(0) - pixels[i].x * proj[i].row(2);
            cv::Mat r2 = proj[i].row(1) - pixels[i].y * proj[i].row(2);
            r1 *= weight[i];
            r2 *= weight[i];
            r1.copyTo(A.row(i * 2));
            r2.copyTo(A.row(i * 2 + 1));
        }
        cv::Mat U, w, Vt;
        cv::SVDecomp(A, w, U, Vt);
        res = Vt.row(3).t();
        if (iter_cnt == 0) {
            old_res = res.clone();
        }
        for (int i = 0; i < int(proj.size()); ++i) {
            cv::Mat l3 = (proj[i].row(2) * res);
            float new_w = l3.at<float>(0);
            max_change = std::max(max_change, std::abs(new_w - weight[i]));
            weight[i] = new_w;
        }
        iter_cnt += 1;
    } while (max_change > 0.1 && iter_cnt < 1);
    if (iter_cnt >= 10) {
        return Geometry::hm3d_to_euclid3d(old_res);
    } else {
        return Geometry::hm3d_to_euclid3d(res);
    }
}

// triangulate multiple points given two frame
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

bool Triangulator::is_triangulate_inlier(
        const cv::Mat& Rcw1, const cv::Mat& tcw1, const cv::Mat& Rcw2,
        const cv::Mat& tcw2, const cv::Mat& tri_res, double grad_th) {
    if (!std::isfinite(tri_res.at<float>(0, 0)) ||
        !std::isfinite(tri_res.at<float>(1, 0)) ||
        !std::isfinite(tri_res.at<float>(2, 0))) {
        return false;
    }
    cv::Mat coord_c1 = Geometry::transform_coord(Rcw1, tcw1, tri_res);
    cv::Mat coord_c2 = Geometry::transform_coord(Rcw2, tcw2, tri_res);
    // depth must be positive
    if (coord_c1.at<float>(2, 0) < EPS || coord_c2.at<float>(2, 0) < EPS) {
        return false;
    }
    // compute parallax
    cv::Mat op1 = tri_res + tcw1;// (coord - (-tcw1) = coord + tcw1)
    op1 /= cv::norm(op1);
    cv::Mat op2 = tri_res + tcw2;
    op2 /= cv::norm(op2);
    double sin_theta3 = cv::norm(op1.cross(op2));
    if (grad_th * sin_theta3 * sin_theta3 < 1.0) { return false; }
    return true;
}

int Triangulator::filter_triangulate(const cv::Mat& Rcw1, const cv::Mat& tcw1,
                                     const cv::Mat& Rcw2, const cv::Mat& tcw2,
                                     const std::vector<cv::Mat>& tri_res,
                                     std::vector<bool>& is_inlier,
                                     double grad_th) {
    is_inlier.resize(tri_res.size());
    for (size_t i = 0; i < tri_res.size(); ++i) {
        is_inlier[i] = is_triangulate_inlier(Rcw1, tcw1, Rcw2, tcw2, tri_res[i],
                                             grad_th);
    }
    return cnt_inliers_from_mask(is_inlier);
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
        img_pt1.push_back(frame1->feature_points[match.queryIdx]->keypoint.pt);
        img_pt2.push_back(frame2->feature_points[match.trainIdx]->keypoint.pt);
    }
    triangulate(proj1, proj2, img_pt1, img_pt2, tri_result);
    return filter_triangulate(frame1->get_Rcw(), frame1->get_Tcw(),
                              frame2->get_Rcw(), frame2->get_Tcw(), tri_result,
                              is_inlier, grad_th);
}
}// namespace vo_nono