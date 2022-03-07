#include "vo_nono/frontend.h"

#include <algorithm>
#include <map>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <vector>

#include "vo_nono/point.h"
#include "vo_nono/util.h"

#ifndef NDEBUG
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#endif

namespace vo_nono {
// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts) {
    static cv::Ptr<cv::ORB> orb_detector;
    if (!orb_detector) { orb_detector = cv::ORB::create(); }
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
}

std::vector<cv::DMatch> Frontend::match_descriptor(const cv::Mat &dscpt1,
                                                   const cv::Mat &dscpt2) {
    static const int MAXIMAL_MATCH_COUNT = 100;

    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    matcher.match(dscpt1, dscpt2, matches);

    if (matches.size() > MAXIMAL_MATCH_COUNT) {
        std::sort(matches.begin(), matches.end(),
                  [](cv::DMatch &match1, cv::DMatch &match2) {
                      return match1.distance < match2.distance;
                  });
        matches.resize(MAXIMAL_MATCH_COUNT);
    }
    return matches;
}

std::vector<cv::DMatch> Frontend::filter_matches(
        const std::vector<cv::DMatch> &matches,
        const std::vector<cv::KeyPoint> &kpt1,
        const std::vector<cv::KeyPoint> &kpt2, const double ransac_th) {
    std::vector<cv::KeyPoint> match_kp1, match_kp2;
    for (auto &match : matches) {
        match_kp1.push_back(kpt1[match.queryIdx]);
        match_kp2.push_back(kpt2[match.trainIdx]);
    }

    std::vector<unsigned char> mask;
    filter_match_key_pts(match_kp1, match_kp2, mask, ransac_th);
    return filter_by_mask(matches, mask);
}

void Frontend::filter_triangulate_points(
        const cv::Mat &tri, const cv::Mat &Rcw1, const cv::Mat &tcw1,
        const cv::Mat &Rcw2, const cv::Mat &tcw2,
        const std::vector<cv::Point2f> &kpts1,
        const std::vector<cv::Point2f> &kpts2, std::vector<bool> &inliers,
        float thresh_square) {
    assert(tri.cols == (int) kpts1.size());
    assert(tri.cols == (int) kpts2.size());
    const int total_pts = tri.cols;
    inliers.resize(total_pts);
    cv::Mat proj2 = get_proj_mat(m_camera.get_intrinsic_mat(), Rcw2, tcw2);
    for (int i = 0; i < total_pts; ++i) {
        cv::Mat hm_coord = tri.col(i).clone();
        hm_coord /= hm_coord.at<float>(3, 0);
        cv::Mat coord = hm_coord.rowRange(0, 3);
        // filter out infinite point
        if (!std::isfinite(coord.at<float>(0, 0)) ||
            !std::isfinite(coord.at<float>(1, 0)) ||
            !std::isfinite(coord.at<float>(2, 0))) {
            inliers[i] = false;
            continue;
        }
        // depth must be positive
        cv::Mat coord_c1 = Rcw1 * coord + tcw1;
        cv::Mat coord_c2 = Rcw2 * coord + tcw2;
        if (coord_c1.at<float>(2, 0) < EPS || coord_c1.at<float>(2, 0) < EPS) {
            inliers[i] = false;
            continue;
        }

        // compute parallax
        cv::Mat op1 = coord + tcw1;// (coord - (-tcw1) = coord + tcw1)
        cv::Mat op2 = coord + tcw2;
        double cos_val = op1.dot(op2) / (cv::norm(op1) * cv::norm(op2));
        if (cos_val > 0.99998) {
            inliers[i] = false;
            continue;
        }

        // re-projection error
        cv::Mat reproj_pt = proj2 * hm_coord;
        reproj_pt /= reproj_pt.at<float>(2, 0);
        float dx = reproj_pt.at<float>(0, 0) - kpts2[i].x,
              dy = reproj_pt.at<float>(1, 0) - kpts2[i].y;
        float diff_square = dx * dx + dy * dy;
        if (diff_square > thresh_square) {
            inliers[i] = false;
            continue;
        }

        inliers[i] = true;
    }
}

void Frontend::get_image(const cv::Mat &image, double t) {
    m_last_frame = m_cur_frame;
    if (m_state == State::Start) {
        assert(!m_keyframe);
        cv::Mat dscpts;
        std::vector<cv::KeyPoint> kpts;
        detect_and_compute(image, kpts, dscpts);
        m_keyframe = std::make_shared<Frame>(
                Frame::create_frame(dscpts, std::move(kpts), t));
        m_cur_frame = m_keyframe;

        m_map->insert_key_frame(m_keyframe);
        m_state = State::Initializing;
    } else if (m_state == State::Initializing) {
        assert(m_keyframe);
        initialize(image, t);

        m_map->insert_key_frame(m_cur_frame);
        m_state = State::Tracking;
    } else if (m_state == State::Tracking) {
        assert(m_keyframe);
        tracking(image, t);
        m_map->insert_key_frame(m_cur_frame);
    } else {
        unimplemented();
    }
    assert(m_cur_frame);
    m_cur_frame->img = image;
    m_motion_pred.inform_pose(m_cur_frame->get_Rcw(), m_cur_frame->get_Tcw(),
                              m_cur_frame->get_time());
}

void Frontend::initialize(const cv::Mat &image, double t) {
    std::vector<cv::KeyPoint> kpts;
    cv::Mat dscpts;
    detect_and_compute(image, kpts, dscpts);
    std::vector<cv::DMatch> matches =
            match_descriptor(m_keyframe->get_dscpts(), dscpts);

    const std::vector<cv::KeyPoint> &prev_kpts = m_keyframe->get_kpts();
    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : matches) {
        matched_pt1.push_back(prev_kpts[match.queryIdx].pt);
        matched_pt2.push_back(kpts[match.trainIdx].pt);
    }

    // todo: less than 8 matched points?
    // todo: normalize scale?
    // todo: findEssentialMat hyper parameters
    std::vector<unsigned char> mask;
    cv::Mat Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                                       m_camera.get_intrinsic_mat(), cv::RANSAC,
                                       0.999, 0.1, mask);
    // filter outliers
    matches = filter_by_mask(matches, mask);
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : matches) {
        matched_pt1.push_back(prev_kpts[match.queryIdx].pt);
        matched_pt2.push_back(kpts[match.trainIdx].pt);
    }

    cv::Mat Rcw, t_cw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, Rcw, t_cw);
    Rcw.convertTo(Rcw, CV_32F);
    t_cw.convertTo(t_cw, CV_32F);
    m_cur_frame = std::make_shared<Frame>(
            Frame::create_frame(dscpts, std::move(kpts), t, Rcw, t_cw));

    // triangulate points
    cv::Mat tri_res;
    cv::Mat proj_mat1 =
            get_proj_mat(m_camera.get_intrinsic_mat(), m_keyframe->get_Rcw(),
                         m_keyframe->get_Tcw());
    cv::Mat proj_mat2 = get_proj_mat(m_camera.get_intrinsic_mat(), Rcw, t_cw);
    cv::triangulatePoints(proj_mat1, proj_mat2, matched_pt1, matched_pt2,
                          tri_res);
    std::vector<bool> inliers;
    filter_triangulate_points(tri_res, m_keyframe->get_Rcw(),
                              m_keyframe->get_Tcw(), m_cur_frame->get_Rcw(),
                              m_cur_frame->get_Tcw(), matched_pt1, matched_pt2,
                              inliers, 0.2);

    log_debug_line("Initialize with R: " << std::endl
                                         << Rcw << std::endl
                                         << "T: " << std::endl
                                         << t_cw << std::endl
                                         << "Number of map points: "
                                         << tri_res.cols);
    _finish_tracking(tri_res, matches, inliers);
}

void Frontend::tracking(const cv::Mat &image, double t) {
    std::vector<cv::KeyPoint> kpts;
    cv::Mat dscpts;
    TIME_IT(detect_and_compute(image, kpts, dscpts),
            "Time cost for feature extraction: ");
    m_cur_frame = std::make_shared<Frame>(Frame::create_frame(dscpts, kpts, t));
    m_cur_frame->img = image;

    // todo: retracking(both model fails)
    // todo: camera dist coeff?
    int motion_res = track_with_motion(20);
    if (motion_res == 2) {
        log_debug_line("Frame " << m_cur_frame->get_id() << ": motion model.");
    } else if (motion_res == 1) {
        track_with_keyframe(true);
        log_debug_line("Frame " << m_cur_frame->get_id()
                                << ": keyframe model.");
    } else {
        track_with_keyframe(false);
        log_debug_line("Frame " << m_cur_frame->get_id()
                                << ": keyframe model.");
    }
    log_debug_line(m_cur_frame->get_id()
                   << ":\n"
                   << m_cur_frame->get_Rcw() << std::endl
                   << m_cur_frame->get_Tcw() << std::endl);
}

bool Frontend::track_with_keyframe(bool b_estimate_valid) {
    std::vector<cv::DMatch> matches = match_descriptor(
            m_keyframe->get_dscpts(), m_cur_frame->get_dscpts());
    matches = filter_matches(matches, m_keyframe->get_kpts(),
                             m_cur_frame->get_kpts());

    std::vector<cv::DMatch> new_point_match;
    std::vector<cv::Point2f> new_point1, new_point2;
    std::vector<cv::Point2f> known_img_pt2;
    std::vector<cv::Matx31f> known_pt_coords;
    const std::vector<cv::KeyPoint> &prev_kpts = m_keyframe->get_kpts();
    const std::vector<cv::KeyPoint> &cur_kpts = m_cur_frame->get_kpts();
    for (auto &match : matches) {
        if (m_keyframe->is_pt_set(match.queryIdx)) {
            cv::Mat pt_coord = m_keyframe->get_pt_3dcoord(match.queryIdx);
            vo_id_t pt_id = m_keyframe->get_pt_id(match.queryIdx);
            known_pt_coords.push_back(pt_coord);
            known_img_pt2.push_back(cur_kpts[match.trainIdx].pt);
            m_cur_frame->set_pt(match.trainIdx, pt_id, pt_coord.at<float>(0, 0),
                                pt_coord.at<float>(0, 1),
                                pt_coord.at<float>(0, 2));
        } else {
            new_point_match.push_back(match);
            new_point1.push_back(prev_kpts[match.queryIdx].pt);
            new_point2.push_back(cur_kpts[match.trainIdx].pt);
        }
    }

    // recover pose of current frame
    cv::Mat rcw_vec, t_cw, Rcw;
    if (b_estimate_valid) {
        cv::Rodrigues(m_cur_frame->get_Rcw(), rcw_vec);
    } else {
        cv::Rodrigues(m_last_frame->get_Rcw(), rcw_vec);
    }
    rcw_vec.convertTo(rcw_vec, CV_64F);
    t_cw = m_cur_frame->get_Tcw();
    t_cw.convertTo(t_cw, CV_64F);
    log_debug_line("Track reference frame with " << known_pt_coords.size()
                                                 << " match points.");
    cv::solvePnPRansac(known_pt_coords, known_img_pt2,
                       m_camera.get_intrinsic_mat(), std::vector<float>(),
                       rcw_vec, t_cw, true, 100, 1);
    cv::Rodrigues(rcw_vec, Rcw);
    Rcw.convertTo(Rcw, CV_32F);
    t_cw.convertTo(t_cw, CV_32F);
    m_cur_frame->set_pose(Rcw, t_cw);

    // triangulate new points
    if (!new_point1.empty()) {
        cv::Mat tri_res(4, (int) new_point1.size(), CV_32F);
        cv::Mat proj_mat1 =
                get_proj_mat(m_camera.get_intrinsic_mat(),
                             m_keyframe->get_Rcw(), m_keyframe->get_Tcw());
        cv::Mat proj_mat2 =
                get_proj_mat(m_camera.get_intrinsic_mat(), Rcw, t_cw);
        cv::triangulatePoints(proj_mat1, proj_mat2, new_point1, new_point2,
                              tri_res);
        std::vector<bool> inliers;
        filter_triangulate_points(tri_res, m_keyframe->get_Rcw(),
                                  m_keyframe->get_Tcw(), Rcw, t_cw, new_point1,
                                  new_point2, inliers);
        _finish_tracking(tri_res, new_point_match, inliers);
    }
    m_cur_frame->set_pose(Rcw, t_cw);
    // _try_switch_keyframe(new_point1.size(), known_img_pt2.size());
    return true;
}

int Frontend::track_with_motion(const size_t cnt_pt_th) {
    if (!m_motion_pred.is_available() || !m_last_frame) { return 0; }
    cv::Mat pred_Rcw, pred_tcw;
    m_motion_pred.predict_pose(m_cur_frame->get_time(), pred_Rcw, pred_tcw);
    m_cur_frame->set_pose(pred_Rcw, pred_tcw);

    std::map<int, ReprojRes> book;
    std::vector<int> img_pt_index;
    std::vector<cv::Matx31f> map_coords;
    std::vector<vo_id_t> map_point_ids;
    std::vector<cv::Point2f> img_pts;
    reproj_points_from_frame(m_last_frame, m_cur_frame, m_camera, book);
    for (auto &pair : book) {
        img_pt_index.push_back(pair.first);
        img_pts.push_back(m_cur_frame->get_pt_keypt(pair.first).pt);
        map_coords.push_back(
                m_last_frame->get_pt_3dcoord(pair.second.point_index));
        map_point_ids.push_back(pair.second.map_point_id);
    }

    std::vector<int> pnp_inliers;
    cv::Mat rvec, Rcw, tcw = pred_tcw;
    cv::Rodrigues(pred_Rcw, rvec);
    rvec.convertTo(rvec, CV_64F);
    tcw.convertTo(tcw, CV_64F);
    cv::solvePnPRansac(map_coords, img_pts, m_camera.get_intrinsic_mat(),
                       std::vector<float>(), rvec, tcw, true, 100, 2, 0.99,
                       pnp_inliers);
    cv::Rodrigues(rvec, Rcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    m_cur_frame->set_pose(Rcw, tcw);
    if (pnp_inliers.size() < cnt_pt_th) {
        log_debug_line("Motion model failed because of pnp ("
                       << pnp_inliers.size() << ") inliers found.");
        log_debug_line("Motion model prediction:\n" << tcw);
        return 1;
    }

    // set map points for new frame
    assert(img_pt_index.size() == map_point_ids.size());
    assert(img_pt_index.size() == map_coords.size());
    for (int i = 0; i < (int) img_pt_index.size(); ++i) {
        m_cur_frame->set_pt(img_pt_index[i], map_point_ids[i],
                            map_coords[i](0, 0), map_coords[i](1, 0),
                            map_coords[i](2, 0));
    }
    log_debug_line("Motion model with " << img_pt_index.size() << " points.");
    return 2;
}

void Frontend::_finish_tracking(const cv::Mat &new_tri_res,
                                const std::vector<cv::DMatch> &matches,
                                const std::vector<bool> &inliers) {
    assert(new_tri_res.cols == (int) matches.size());
    assert(inliers.size() == matches.size());
    std::vector<vo_uptr<MapPoint>> new_points;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!inliers[i]) { continue; }
        cv::Mat cur_point = new_tri_res.col((int) i);
        if (!float_eq_zero(cur_point.at<float>(3))) {
            // not infinite point
            cur_point /= cur_point.at<float>(3);
            float x = cur_point.at<float>(0);
            float y = cur_point.at<float>(1);
            float z = cur_point.at<float>(2);

            vo_uptr<MapPoint> cur_map_pt = std::make_unique<MapPoint>(
                    MapPoint::create_map_point(x, y, z));
            m_keyframe->set_pt(matches[i].queryIdx, cur_map_pt->get_id(), x, y,
                               z);
            m_cur_frame->set_pt(matches[i].trainIdx, cur_map_pt->get_id(), x, y,
                                z);
            new_points.push_back(std::move(cur_map_pt));
        }
    }
    insert_map_points(new_points);
}

void Frontend::_try_switch_keyframe(size_t new_pt, size_t old_pt) {
    assert(m_cur_frame);
    assert(m_keyframe);

    bool should_change = false;
    const size_t already_set = m_keyframe->get_set_cnt();
    const size_t total_kpts = m_keyframe->get_kpts().size();

    if (already_set * 4 >= total_kpts * 5) {
        log_debug_line("change because set enough points");
        should_change = true;
    }
    if (should_change) {
        log_debug_line("New keyframe id: " << m_cur_frame->get_id());
        m_keyframe = m_cur_frame;
    }
}
}// namespace vo_nono