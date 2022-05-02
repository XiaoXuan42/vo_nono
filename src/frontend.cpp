#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
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
// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts, int nfeatures) {
    cv::Ptr orb_detector = cv::ORB::create(nfeatures, 1.2f);
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
    // undistort keypoints
    std::vector<cv::Point2f> pts, res_pts;
    for (auto &kpt : kpts) { pts.push_back(kpt.pt); }
    cv::undistortPoints(pts, res_pts, camera_.get_intrinsic_mat(),
                        camera_.get_dist_coeff(), cv::noArray(),
                        camera_.get_intrinsic_mat());
    for (int i = 0; i < int(kpts.size()); ++i) { kpts[i].pt = res_pts[i]; }
}

void Frontend::get_image(const cv::Mat &image, double t) {
    reset_state();

    std::vector<cv::KeyPoint> kpts;
    cv::Mat descriptor;
    detect_and_compute(image, kpts, descriptor, CNT_KEYPTS);
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
            map_->insert_frame(keyframe_);
            insert_local_frame(curframe_);
            b_succ = true;
        } else if (init_state == -1) {
            // not enough matches
            keyframe_ = curframe_;
        }
    } else if (state_ == State::Tracking) {
        if (tracking(image, t)) {
            insert_local_frame(curframe_);
            new_keyframe();
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
                                                    CNT_MATCHES);
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

    for (auto &tri : triangulate_result) { tri /= scale; }
    init_matches_ = filter_by_mask(init_matches_, inliers);
    triangulate_result = filter_by_mask(triangulate_result, inliers);
    _set_keyframe(keyframe_);
    _update_points_location(triangulate_result, init_matches_);
    _associate_points(init_matches_, std::numeric_limits<double>::infinity());
    return 0;
}

bool Frontend::tracking(const cv::Mat &image, double t) {
    cv::Mat motion_Rcw, motion_Tcw;
    motion_pred_.predict_pose(t, motion_Rcw, motion_Tcw);
    curframe_->set_Rcw(motion_Rcw);
    curframe_->set_Tcw(motion_Tcw);

    keyframe_matches_ =
            matcher_->match_descriptor_bf(keyframe_->get_descriptors());
    tracking_with_keyframe();
    if (cnt_inlier_direct_match_ < CNT_TRACK_MIN_MATCHES) {
        //auto local_points = map_->get_local_map_points();
        //cnt_inlier_proj_match_ = track_by_projection(local_points, 20, 10);
    } else {
        b_match_good_ = true;
    }
    if (std::max(cnt_inlier_proj_match_, cnt_inlier_direct_match_) >=
        CNT_TRACK_MIN_MATCHES) {
        b_track_good_ = true;
    }
    if (b_track_good_) { triangulate_and_set(direct_matches_); }

    log_debug_line("Track good: " << b_track_good_);
    log_debug_line("Match good: " << b_match_good_);
    log_debug_line("Match " << cnt_inlier_direct_match_ << ". Project "
                            << cnt_inlier_proj_match_);
    log_debug_line(curframe_->get_cnt_map_pt()
                   << ":\n"
                   << curframe_->get_Rcw() << std::endl
                   << curframe_->get_Tcw() << std::endl);
    log_debug_line("Keyframe " << keyframe_->get_id() << " has "
                               << keyframe_->get_cnt_map_pt()
                               << " map points.");
    return b_track_good_;
}

void Frontend::tracking_with_keyframe() {
    direct_matches_ = ORBMatcher::filter_match_by_dis(keyframe_matches_, 8, 64,
                                                      CNT_MATCHES);
    direct_matches_ = ORBMatcher::filter_match_by_rotation_consistency(
            direct_matches_, keyframe_->get_keypoints(),
            curframe_->get_keypoints(), 3);
    cnt_inlier_direct_match_ = track_by_match(keyframe_, direct_matches_, 6);
}

void Frontend::relocalization() {
    // case1: not enough (true) matches between keyframe and curframe.
}

int Frontend::track_by_match(const vo_ptr<Frame> &ref_frame,
                             const std::vector<cv::DMatch> &matches,
                             float ransac_th) {
    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::DMatch> old_matches;

    for (int i = 0; i < int(matches.size()); ++i) {
        if (ref_frame->is_index_set(matches[i].queryIdx)) {
            old_matches.push_back(matches[i]);
            pt_coords.push_back(
                    ref_frame->get_map_pt(matches[i].queryIdx)->get_coord());
            img_pts.push_back(curframe_->get_pixel_pt(matches[i].trainIdx));
        }
    }
    if (old_matches.size() < 10) { return int(old_matches.size()); }

    log_debug_line("Total match: " << matches.size());
    log_debug_line("Old match: " << old_matches.size());
    int cnt_inlier = 0;
    std::vector<bool> inliers;
    std::vector<cv::Matx31f> inlier_coords;
    std::vector<cv::Point2f> inlier_img_pts;
    cv::Mat Rcw = curframe_->get_Rcw(), tcw = curframe_->get_Tcw();
    PnP::pnp_ransac(pt_coords, img_pts, camera_, 100, ransac_th, Rcw, tcw,
                    inliers);

    log_debug_line("Old match after pnp: " << old_matches.size());
    for (int i = 0; i < (int) old_matches.size(); ++i) {
        if (inliers[i] && !curframe_->is_index_set(old_matches[i].trainIdx)) {
            cnt_inlier += 1;
        } else {
            inliers[i] = false;
        }
    }
    if (cnt_inlier < 10) { return int(cnt_inlier); }

    for (int i = 0; i < (int) old_matches.size(); ++i) {
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
    for (auto &proj_match : proj_matches) {
        pt_coords.push_back(proj_match.coord3d);
        img_pts.push_back(proj_match.img_pt);
    }
    PnP::pnp_ransac(pt_coords, img_pts, camera_, 100, ransac_th, Rcw, tcw,
                    is_inliers);

    assert(proj_matches.size() == is_inliers.size());
    for (int i = 0; i < int(is_inliers.size()); ++i) {
        if (is_inliers[i] && !curframe_->is_index_set(proj_matches[i].index)) {
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            inlier_proj_index.push_back(i);
            cnt_proj_match += 1;
        }
    }
    if (cnt_proj_match < 10) { return cnt_proj_match; }
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

void Frontend::new_keyframe() {
    bool is_keyframe = false;
    if (is_keyframe) { _set_keyframe(curframe_); }
    //    if (!b_match_good_ && b_track_good_) {
    //        if (double(cnt_inlier_direct_match_) <
    //            0.2 * double(keyframe_->get_cnt_map_pt())) {
    //            //b_new_keyframe_ = true;
    //        }
    //    }
    //    if (direct_matches_.size() < CNT_MATCHES &&
    //        double(direct_matches_.size()) <
    //                0.8 * double(keyframe_->get_cnt_map_pt())) {
    //        b_new_keyframe_ = true;
    //    }
}

void Frontend::triangulate_and_set(const std::vector<cv::DMatch> &matches) {
    std::vector<cv::DMatch> valid_matches = filter_match(matches, 0.01);
    std::vector<cv::DMatch> tri_matches;
    std::vector<bool> tri_inliers;
    std::vector<cv::Mat> tri_results;
    Triangulator::triangulate_and_filter_frames(
            keyframe_.get(), curframe_.get(), camera_.get_intrinsic_mat(),
            valid_matches, tri_results, tri_inliers, 1000);
    tri_matches = filter_by_mask(valid_matches, tri_inliers);
    tri_results = filter_by_mask(tri_results, tri_inliers);
    _update_points_location(tri_results, tri_matches);
    _associate_points(valid_matches, 0.1);
}

std::vector<cv::DMatch> Frontend::filter_match(
        const std::vector<cv::DMatch> &matches, double epi_th) {
    std::vector<cv::DMatch> valid_match;
    std::vector<bool> mask;
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat R21, t21;
    Geometry::relative_pose(keyframe_->get_Rcw(), keyframe_->get_Tcw(),
                            curframe_->get_Rcw(), curframe_->get_Tcw(), R21,
                            t21);
    cv::Mat ess = Epipolar::compute_essential(R21, t21);
    for (auto &match : matches) {
        pts1.push_back(keyframe_->feature_points[match.queryIdx]->keypoint.pt);
        pts2.push_back(curframe_->feature_points[match.trainIdx]->keypoint.pt);
    }
    ORBMatcher::filter_match_by_ess(ess, camera_.get_intrinsic_mat(), pts1,
                                    pts2, epi_th, mask);
    valid_match = filter_by_mask(matches, mask);
    return valid_match;
}

void Frontend::_update_points_location(const std::vector<cv::Mat> &tri_res,
                                       const std::vector<cv::DMatch> &matches) {
    assert(tri_res.size() == matches.size());
    cv::Mat cur_proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(),
                                   curframe_->get_Rcw(), curframe_->get_Tcw());
    for (int i = 0; i < int(matches.size()); ++i) {
        int keyframe_index = matches[i].queryIdx;
        int curframe_index = matches[i].trainIdx;
        if (local_map_.own_points[keyframe_index]) {
            auto &filter = local_map_.filters[keyframe_index];
            filter.filter(keyframe_->get_Tcw(), keyframe_->get_Rcw(),
                          curframe_->get_Tcw(), tri_res[i]);
            _add_observation(curframe_->get_Rcw(), curframe_->get_Tcw(),
                             curframe_->get_pixel_pt(curframe_index),
                             keyframe_index);
            cv::Mat new_coord = _get_local_map_point_coord(keyframe_index);
            if (keyframe_->is_index_set(keyframe_index)) {
                keyframe_->get_map_pt(keyframe_index)->set_coord(new_coord);
            }
        }
    }
}

void Frontend::_associate_points(const std::vector<cv::DMatch> &matches,
                                 double rel_th) {
    cv::Mat proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(),
                                   curframe_->get_Rcw(), curframe_->get_Tcw());
    for (auto &match : matches) {
        int keyframe_index = match.queryIdx;
        int curframe_index = match.trainIdx;
        if (!keyframe_->is_index_set(keyframe_index)) {
            if (local_map_.filters[keyframe_index].relative_error_less(
                        rel_th)) {
                cv::Mat coord = _get_local_map_point_coord(keyframe_index);
                auto new_pt =
                        std::make_shared<MapPoint>(MapPoint::create_map_point(
                                coord,
                                keyframe_->feature_points[keyframe_index]
                                        ->descriptor,
                                keyframe_->feature_points[keyframe_index]
                                        ->keypoint.octave));
                new_pt->associate_feature_point(
                        keyframe_->feature_points[keyframe_index]);
                keyframe_->set_map_pt(keyframe_index, new_pt);
            }
        }
        if (keyframe_->is_index_set(keyframe_index) &&
            !curframe_->is_index_set(curframe_index)) {
            auto pt = keyframe_->get_map_pt(keyframe_index);
            double err2 = Geometry::reprojection_err2(
                    proj_mat, pt->get_coord(),
                    curframe_->get_pixel_pt(curframe_index));
            if (err2 < chi2_2_5) { curframe_->set_map_pt(curframe_index, pt); }
        }
    }
}

void Frontend::_set_keyframe(const vo_ptr<Frame> &keyframe) {
    local_map_.local_frames.clear();
    local_map_.tri_mats.clear();
    local_map_.tri_mats.resize(keyframe->feature_points.size());
    local_map_.filters.clear();
    local_map_.filters.resize(keyframe->feature_points.size());
    local_map_.own_points = std::vector(keyframe->feature_points.size(), false);

    double basic_var = std::min(camera_.fx(), camera_.fy());
    basic_var = 1.0 / (basic_var * basic_var);
    for (int i = 0; i < (int) keyframe->feature_points.size(); ++i) {
        if (!keyframe->is_index_set(i)) {
            local_map_.own_points[i] = true;
            local_map_.filters[i].set_information(
                    camera_, keyframe->feature_points[i]->keypoint.pt,
                    basic_var);
            _add_observation(keyframe->get_Rcw(), keyframe->get_Tcw(),
                             keyframe->get_pixel_pt(i), i);
        } else {
            keyframe->get_map_pt(i)->associate_feature_point(
                    keyframe->feature_points[i]);
        }
    }

    local_map_.local_frames.push_back(keyframe);
    keyframe_ = keyframe;
}

cv::Mat Frontend::_get_local_map_point_coord(int index) {
    cv::Mat mat = local_map_.tri_mats[index].clone();
    cv::Mat U, Vt, w;
    cv::SVDecomp(mat, w, U, Vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    cv::Mat coord = Vt.row(3).t();
    return Geometry::hm3d_to_euclid3d(coord);
}

void Frontend::_add_observation(const cv::Mat &Rcw, const cv::Mat &tcw,
                                const cv::Point2f &pixel, int index) {
    cv::Mat proj_mat =
            Geometry::get_proj_mat(camera_.get_intrinsic_mat(), Rcw, tcw);
    int origin_row = local_map_.tri_mats[index].rows;
    cv::Mat updated_tri_mat = cv::Mat::zeros(origin_row + 2, 4, CV_32F);
    if (origin_row > 0) {
        local_map_.tri_mats[index].copyTo(
                updated_tri_mat.rowRange(0, origin_row));
    }
    cv::Mat r1 = proj_mat.row(1) - pixel.y * proj_mat.row(2);
    cv::Mat r2 = proj_mat.row(0) - pixel.x * proj_mat.row(2);
    r1.copyTo(updated_tri_mat.row(origin_row));
    r2.copyTo(updated_tri_mat.row(origin_row + 1));
    local_map_.tri_mats[index] = updated_tri_mat;
}
}// namespace vo_nono