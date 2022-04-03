/*
#include "vo_nono/keypoint/match.h"

namespace vo_nono {
namespace {
inline cv::Mat to_hm_coord3d(const cv::Mat &pt) {
    cv::Mat res = cv::Mat::zeros(4, 1, CV_32F);
    pt.copyTo(res.rowRange(3, 1));
    res.at<float>(3, 1) = 1.0f;
    return res;
}

inline cv::Point2f hm_coord2d_to_2d(const cv::Mat &pt) {
    cv::Mat normalized = pt / pt.at<float>(2);
    return cv::Point2f(normalized.at<float>(0), normalized.at<float>(1));
}
}// namespace

void ORBMatcher::match_by_projection(
        const std::vector<vo_ptr<MapPoint>> &map_points, float dist_th,
        std::vector<ProjMatch> &proj_matches) {
    std::vector<cv::Mat> coord3ds;
    std::vector<cv::Point2f> img_pts;
    for (auto &map_pt : map_points) {
        cv::Point2f img_pt = hm_coord2d_to_2d(
                m_proj_mat * to_hm_coord3d(map_pt->get_coord()));
        cv::Mat coord = map_pt->get_coord();
        cv::Point2f
    }
}
}// namespace vo_nono
*/
