#ifndef VO_NONO_UTIL_H
#define VO_NONO_UTIL_H

#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace vo_nono {
template<typename T, typename U>
[[nodiscard]] inline std::vector<T> filter_by_mask(
        const std::vector<T> &targets, const std::vector<U> &mask) {
    assert(targets.size() == mask.size());
    std::vector<T> res;
    res.reserve(targets.size());
    for (int i = 0; i < (int) targets.size(); ++i) {
        if (mask[i]) { res.push_back(targets[i]); }
    }
    return res;
}
template<typename T>
[[nodiscard]] inline int cnt_inliers_from_mask(const std::vector<T> &mask) {
    int cnt = 0;
    for (int i = 0; i < (int) mask.size(); ++i) {
        if (mask[i]) { cnt += 1; }
    }
    return cnt;
}

[[maybe_unused]] inline void show_matches(
        const vo_ptr<Frame> &frame1, const vo_ptr<Frame> &frame2,
        const std::vector<cv::DMatch> &matches) {
    std::string title = std::to_string(frame1->get_id()) + " match " +
                        std::to_string(frame2->get_id());
    cv::Mat out_img;
    cv::drawMatches(frame1->image, frame1->get_keypoints(), frame2->image,
                    frame2->get_keypoints(), matches, out_img);
    cv::imshow(title, out_img);
    cv::waitKey(0);
}

[[maybe_unused]] inline void show_matches(
        const vo_ptr<Frame> &frame1, const vo_ptr<Frame> &frame2,
        const std::vector<ProjMatch> &proj_matches) {
    std::string title = std::to_string(frame1->get_id()) +
                        " projection match " + std::to_string(frame2->get_id());
    std::vector<cv::DMatch> matches;
    for (auto &proj_match : proj_matches) {
        auto feature_set = proj_match.p_map_pt->get_feature_points();
        for (auto &feat : feature_set) {
            if (feat->frame->get_id() == frame1->get_id()) {
                matches.emplace_back(feat->index, proj_match.index, 2.0);
            }
        }
    }
    cv::Mat out_img;
    cv::drawMatches(frame1->image, frame1->get_keypoints(), frame2->image,
                    frame2->get_keypoints(), matches, out_img);
    cv::imshow(title, out_img);
    cv::waitKey(0);
}

[[maybe_unused]] inline void show_coordinate(const vo_ptr<Frame> &frame) {
    cv::Mat img = frame->image.clone();
    cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    int cnt = 0;
    for (int i = 0; i < int(frame->feature_points.size()); ++i) {
        auto pt = frame->feature_points[i]->keypoint.pt;
        if (frame->is_index_set(i)) {
            cnt += 1;
            if (cnt > 100) { break; }
            cv::Mat coord = frame->get_map_pt(i)->get_coord();
            std::string annotate = "(" + std::to_string(coord.at<float>(0)) +
                                   ", " + std::to_string(coord.at<float>(1)) +
                                   ", " + std::to_string(coord.at<float>(2)) +
                                   ")";
            cv::circle(img, pt, 5, CV_RGB(0.0, 0.0, 255.0));
            cv::putText(img, annotate, pt, cv::FONT_HERSHEY_PLAIN, 1,
                        CV_RGB(255.0, 0.0, 0.0));
        }
    }
    cv::imshow("frame " + std::to_string(frame->get_id()), img);
    cv::waitKey(0);
}
}// namespace vo_nono

#endif