#ifndef VO_NONO_FEATURE_H
#define VO_NONO_FEATURE_H

#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "vo_nono/camera.h"
#include "vo_nono/frame.h"
#include "vo_nono/types.h"
#include "vo_nono/util.h"

namespace vo_nono {
static inline void filter_match_pts(const std::vector<cv::Point2f> &pts1,
                                    const std::vector<cv::Point2f> &pts2,
                                    std::vector<unsigned char> &mask,
                                    double ransac_th = 1.0) {
    cv::findFundamentalMat(pts1, pts2, mask, cv::FM_RANSAC, ransac_th, 0.99);
    assert(mask.size() == pts1.size());
}

inline void filter_match_key_pts(const std::vector<cv::KeyPoint> &kpts1,
                                 const std::vector<cv::KeyPoint> &kpts2,
                                 std::vector<unsigned char> &mask,
                                 double ransac_th = 1.0) {
    assert(kpts1.size() == kpts2.size());
    auto ang_diff_index = [](double diff_ang) {
        if (diff_ang < 0) { diff_ang += 360; }
        return (int) (diff_ang / 3.6);
    };
    Histogram<double> histo(101, ang_diff_index);
    std::vector<cv::Point2f> pt21;
    std::vector<cv::Point2f> pt22;
    std::vector<double> ang_diff;
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        double diff = kpts1[i].angle - kpts2[i].angle;
        pt21.push_back(kpts1[i].pt);
        pt22.push_back(kpts2[i].pt);
        histo.insert_element(diff);
        ang_diff.push_back(diff);
    }
    filter_match_pts(pt21, pt22, mask, ransac_th);
    assert(mask.size() == kpts1.size());
    histo.cal_topK(3);
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        if (!histo.is_topK(ang_diff[i])) { mask[i] = 0; }
    }
}

struct ReprojRes {
    vo_id_t frame_id;
    vo_id_t map_point_id;
    int point_index;
    double desc_dis;
};

// book: image point index of right frame -> map point id and descriptor distance
void reproj_points_from_frame(const vo_ptr<Frame> &left_frame,
                              const vo_ptr<Frame> &right_frame,
                              const Camera &camera,
                              std::map<int, ReprojRes> &book);
}// namespace vo_nono

#endif//VO_NONO_FEATURE_H
