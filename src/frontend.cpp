#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/keypoint/triangulate.h"
#include "vo_nono/pnp.h"
#include "vo_nono/point.h"
#include "vo_nono/util/constants.h"
#include "vo_nono/util/macro.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
namespace {
[[maybe_unused]] void show_matches(const vo_ptr<Frame> &frame1,
                                   const vo_ptr<Frame> &frame2,
                                   const std::vector<cv::DMatch> &matches) {
    std::string title = std::to_string(frame1->get_id()) + " match " +
                        std::to_string(frame2->get_id());
    cv::Mat out_img;
    cv::drawMatches(frame1->image, frame1->get_keypoints(), frame2->image,
                    frame2->get_keypoints(), matches, out_img);
    cv::imshow(title, out_img);
    cv::waitKey(0);
}

[[maybe_unused]] void show_matches(const vo_ptr<Frame> &frame1,
                                   const vo_ptr<Frame> &frame2,
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

[[maybe_unused]] void show_coordinate(const vo_ptr<Frame> &frame) {
    cv::Mat img = frame->image.clone();
    cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    int cnt = 0;
    for (int i = 0; i < int(frame->feature_points.size()); ++i) {
        auto pt = frame->feature_points[i]->keypoint.pt;
        if (frame->is_index_set(i)) {
            cnt += 1;
            if (cnt > 10) { break; }
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
}// namespace

// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts, int nfeatures) {
    cv::Ptr orb_detector = cv::ORB::create(nfeatures, 1.2f);
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
}

void Frontend::get_image(const cv::Mat &image, double t) {
    reset_state();

    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptor;
    detect_and_compute(image, kpts, descriptor, CNT_KEY_PTS);
    matcher_ =
            std::make_unique<ORBMatcher>(ORBMatcher(kpts, descriptor, camera_));
    curframe_ =
            std::make_shared<Frame>(Frame::create_frame(descriptor, kpts, t));
    curframe_->image = image;

    std::unique_lock<std::mutex> lock(map_->map_global_mutex);
    bool b_succ = false;
    if (state_ == State::Start) {
        state_ = State::Initializing;
        keyframe_ = curframe_;
        b_succ = true;
    } else if (state_ == State::Initializing) {
        int init_state = initialize(image);
        if (init_state == 0) {
            state_ = State::Tracking;
            map_->initialize(keyframe_, curframe_, init_matches_);
            b_succ = true;
        } else if (init_state == -1) {
            // not enough matches
            keyframe_ = curframe_;
        }
    } else if (state_ == State::Tracking) {
        if (tracking(image, t)) {
            FrameMessage message(curframe_, track_matches_, b_new_keyframe_);
            map_->insert_frame(message);
            if (b_new_keyframe_) {
                keyframe_ = curframe_;
                b_new_keyframe_ = false;
                points_seen_.clear();
            }
            b_succ = true;
        }
    } else {
        unimplemented();
    }

    if (b_succ) {
        motion_pred_.inform_pose(curframe_->get_Rcw(), curframe_->get_Tcw(), t);
    }
}

int Frontend::initialize(const cv::Mat &image) {
    keyframe_matches_ =
            matcher_->match_descriptor_bf(keyframe_->get_descriptors());
    init_matches_ = ORBMatcher::filter_match_by_dis(keyframe_matches_, 8, 15,
                                                    CNT_INIT_MATCHES);
    init_matches_ = ORBMatcher::filter_match_by_rotation_consistency(
            init_matches_, keyframe_->get_keypoints(),
            curframe_->get_keypoints(), 3);

    if (init_matches_.size() < 10) { return -1; }
    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : init_matches_) {
        matched_pt1.push_back(
                keyframe_->feature_points[match.queryIdx]->keypoint.pt);
        matched_pt2.push_back(
                curframe_->feature_points[match.trainIdx]->keypoint.pt);
    }
    std::vector<unsigned char> mask;
    cv::Mat Ess;
    Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                               camera_.get_intrinsic_mat(), cv::RANSAC, 0.999,
                               2.0, 1000, mask);
    // filter outliers
    init_matches_ = filter_by_mask(init_matches_, mask);
    if (init_matches_.size() < 50) { return -1; }
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : init_matches_) {
        matched_pt1.push_back(
                keyframe_->feature_points[match.queryIdx]->keypoint.pt);
        matched_pt2.push_back(
                curframe_->feature_points[match.trainIdx]->keypoint.pt);
    }

    cv::Mat Rcw, tcw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, camera_.get_intrinsic_mat(),
                    Rcw, tcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    curframe_->set_Rcw(Rcw);
    curframe_->set_Tcw(tcw);

    // triangulate points
    std::vector<cv::Mat> triangulate_result;
    std::vector<bool> inliers;
    int cnt_new_pt = Triangulator::triangulate_and_filter_frames(
            keyframe_.get(), curframe_.get(), camera_.get_intrinsic_mat(),
            init_matches_, triangulate_result, inliers, 1000);
    if (cnt_new_pt < 40) { return -2; }

    double scale = 3;
    for (int i = 0; i < int(triangulate_result.size()); ++i) {
        if (inliers[i]) {
            assert(triangulate_result[i].at<float>(2) > 0);
            scale += triangulate_result[i].at<float>(2);
        }
    }
    scale /= cnt_new_pt;
    tcw /= scale;
    curframe_->set_Tcw(tcw);

    // triangulated points is not scaled because it's no longer needed.
    assert(init_matches_.size() == inliers.size());
    init_matches_ = filter_by_mask(init_matches_, inliers);

    log_debug_line("Initialize with R: " << std::endl
                                         << Rcw << std::endl
                                         << "T: " << std::endl
                                         << tcw << std::endl
                                         << cnt_new_pt << " new map points.");
    return 0;
}

bool Frontend::tracking(const cv::Mat &image, double t) {
    cv::Mat motion_Rcw, motion_Tcw;
    motion_pred_.predict_pose(t, motion_Rcw, motion_Tcw);
    curframe_->set_Rcw(motion_Rcw);
    curframe_->set_Tcw(motion_Tcw);

    keyframe_matches_ =
            matcher_->match_descriptor_bf(keyframe_->get_descriptors());
    track_matches_ = ORBMatcher::filter_match_by_dis(keyframe_matches_, 8, 30,
                                                     CNT_MATCHES);
    track_matches_ = ORBMatcher::filter_match_by_rotation_consistency(
            track_matches_, keyframe_->get_keypoints(),
            curframe_->get_keypoints(), 3);

    int cnt_match = track_by_match(keyframe_, track_matches_, 6);
    int cnt_proj_match = 0;
    if (cnt_match < CNT_MATCH_MIN_MATCHES) {
        cnt_proj_match = track_by_projection_local_map();
    } else {
        b_match_good_ = true;
    }
    if (std::max(cnt_proj_match, cnt_match) >= CNT_TRACKING_MIN_MATCHES) {
        b_track_good_ = true;
    }

    if (!b_match_good_ && b_track_good_) {
        if (double(cnt_match) < 0.2 * double(keyframe_->get_cnt_map_pt())) {
            b_new_keyframe_ = true;
        }
    }

    log_debug_line("Track good: " << b_track_good_);
    log_debug_line("Match good: " << b_match_good_);
    log_debug_line("Match " << cnt_match << ". Project " << cnt_proj_match
                            << ". Set " << curframe_->get_cnt_map_pt()
                            << " map points.");
    log_debug_line(curframe_->get_cnt_map_pt()
                   << ":\n"
                   << curframe_->get_Rcw() << std::endl
                   << curframe_->get_Tcw() << std::endl);
    log_debug_line("Keyframe " << keyframe_->get_id() << " has "
                               << keyframe_->get_cnt_map_pt()
                               << " map points.");
    return b_track_good_;
}

int Frontend::track_by_match(const vo_ptr<Frame> &ref_frame,
                             const std::vector<cv::DMatch> &matches,
                             float ransac_th) {
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::DMatch> old_matches;
    std::unordered_map<int, int> origin_index;

    for (int i = 0; i < int(matches.size()); ++i) {
        if (ref_frame->is_index_set(matches[i].queryIdx)) {
            int cur_old_index = int(matches.size());
            origin_index[cur_old_index] = i;
            old_matches.push_back(matches[i]);
            pt_coords.push_back(
                    ref_frame->get_map_pt(matches[i].queryIdx)->get_coord());
            img_pts.push_back(curframe_->feature_points[matches[i].trainIdx]
                                      ->keypoint.pt);
        }
    }
    log_debug_line("Match with frame " << ref_frame->get_id() << " "
                                       << old_matches.size() << " old match. "
                                       << matches.size() - old_matches.size()
                                       << " new match.");

    if (old_matches.size() < CNT_MATCH_MIN_MATCHES) {
        return int(old_matches.size());
    }

    int cnt_inlier = 0;
    std::vector<bool> inliers;
    std::vector<cv::Matx31f> inlier_coords;
    std::vector<cv::Point2f> inlier_img_pts;
    cv::Mat Rcw = curframe_->get_Rcw(), tcw = curframe_->get_Tcw();
    PnP::pnp_ransac(pt_coords, img_pts, camera_, 100, ransac_th, Rcw, tcw,
                    inliers);
    assert(inliers.size() == pt_coords.size());
    assert(old_matches.size() == pt_coords.size());
    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i] && !curframe_->is_index_set(old_matches[i].trainIdx)) {
            cnt_inlier += 1;
        } else {
            inliers[i] = false;
        }
    }
    if (cnt_inlier < 10) { return int(cnt_inlier); }

    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i]) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
        }
    }
    std::vector<bool> inliers2 = PnP::pnp_by_optimize(
            inlier_coords, inlier_img_pts, camera_, Rcw, tcw);
    curframe_->set_Rcw(Rcw);
    curframe_->set_Tcw(tcw);
    return cnt_inliers_from_mask(inliers2);
}

int Frontend::track_by_projection(const std::vector<vo_ptr<MapPoint>> &points,
                                  float r_th, float ransac_th) {
    int cnt_proj_match = 0;
    std::unordered_set<vo_id_t> map_pt_set;
    std::vector<ProjMatch> proj_matches;
    std::vector<cv::Matx31f> pt_coords, inlier_coords;
    std::vector<cv::Point2f> img_pts, inlier_img_pts;
    std::vector<int> inlier_proj_index;
    std::vector<bool> is_inliers, is_inliers2;
    cv::Mat Rcw = curframe_->get_Rcw(), tcw = curframe_->get_Tcw();
    matcher_->set_estimate_pose(Rcw, tcw);

    proj_matches = matcher_->match_by_projection(points, r_th);
    if (proj_matches.size() < CNT_TRACKING_MIN_MATCHES) {
        return int(proj_matches.size());
    }

    for (auto &proj_match : proj_matches) {
        pt_coords.push_back(proj_match.coord3d);
        img_pts.push_back(proj_match.img_pt);
    }
    PnP::pnp_ransac(pt_coords, img_pts, camera_, 100, ransac_th, Rcw, tcw,
                    is_inliers);

    cnt_proj_match = cnt_inliers_from_mask(is_inliers);
    assert(proj_matches.size() == is_inliers.size());
    if (cnt_proj_match < CNT_TRACKING_MIN_MATCHES) { return cnt_proj_match; }
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i] && !curframe_->is_index_set(proj_matches[i].index)) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            inlier_proj_index.push_back(i);
        }
    }
    PnP::pnp_by_optimize(inlier_coords, inlier_img_pts, camera_, Rcw, tcw);

    cv::Mat proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(), Rcw, tcw);
    is_inliers2.resize(inlier_coords.size());
    for (int i = 0; i < int(inlier_coords.size()); ++i) {
        cv::Mat mat_coord = cv::Mat(inlier_coords[i]);
        double err2 = Geometry::reprojection_err2(proj_mat, mat_coord,
                                                  inlier_img_pts[i]);
        if (err2 < chi2_2_5) {
            is_inliers2[i] = true;
        } else {
            is_inliers2[i] = false;
        }
    }

    cnt_proj_match = 0;
    for (int i = 0; i < int(is_inliers2.size()); ++i) {
        if (is_inliers2[i]) {
            int proj_index = inlier_proj_index[i];
            curframe_->set_map_pt(proj_matches[proj_index].index,
                                  proj_matches[proj_index].p_map_pt);
            cnt_proj_match += 1;
        }
    }
    curframe_->set_Rcw(Rcw);
    curframe_->set_Tcw(tcw);
    log_debug_line("Pose estimate using " << is_inliers2.size()
                                          << " projection with "
                                          << cnt_proj_match << " map points.");
    return cnt_proj_match;
}

int Frontend::track_by_projection_frame(const vo_ptr<Frame> &ref_frame) {
    std::vector<vo_ptr<MapPoint>> points = ref_frame->get_all_map_pts();
    int proj_res = track_by_projection(points, 30, 2);
    return proj_res;
}

int Frontend::track_by_projection_local_map() {
    std::vector<vo_ptr<MapPoint>> local_pts = map_->get_local_map_points();
    return track_by_projection(local_pts, 20, 15);
}

}// namespace vo_nono