#ifndef VO_NONO_DEBUG_H
#define VO_NONO_DEBUG_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace vo_nono {

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
            if (cnt > 50) { break; }
            cv::Mat coord = frame->get_map_pt(i)->get_coord();
            std::string annotate = std::to_string(coord.at<float>(2));
//            std::string annotate = "(" + std::to_string(coord.at<float>(0)) +
//                                   ", " + std::to_string(coord.at<float>(1)) +
//                                   ", " + std::to_string(coord.at<float>(2)) +
//                                   ")";
            cv::circle(img, pt, 5, CV_RGB(0.0, 0.0, 255.0));
            cv::putText(img, annotate, pt, cv::FONT_HERSHEY_PLAIN, 1,
                        CV_RGB(255.0, 0.0, 0.0));
        }
    }
    cv::imshow("frame " + std::to_string(frame->get_id()), img);
    cv::waitKey(0);
}
}
#endif//VO_NONO_DEBUG_H
