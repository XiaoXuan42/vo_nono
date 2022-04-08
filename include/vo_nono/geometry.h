#ifndef VO_NONO_GEOMETRY_H
#define VO_NONO_GEOMETRY_H

namespace vo_nono {
inline cv::Mat euclid3d_to_hm3d(const cv::Mat &pt) {
    assert(pt.type() == CV_32F);
    cv::Mat res = cv::Mat::zeros(4, 1, CV_32F);
    pt.copyTo(res.rowRange(0, 3));
    res.at<float>(3, 0) = 1.0f;
    return res;
}

inline cv::Point2f hm2d_to_euclid2d(const cv::Mat &pt) {
    cv::Mat normalized = pt / pt.at<float>(2);
    cv::Point2f res(normalized.at<float>(0), normalized.at<float>(1));
    return res;
}
}// namespace vo_nono

#endif//VO_NONO_GEOMETRY_H
