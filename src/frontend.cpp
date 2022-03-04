#include "vo_nono/frontend.h"

#include <algorithm>
#include <opencv2/calib3d.hpp>
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

cv::Mat Frontend::get_proj_mat(const cv::Mat &Rcw, const cv::Mat &t_cw) {
    assert(Rcw.type() == CV_32F);
    assert(t_cw.type() == CV_32F);
    cv::Mat proj = cv::Mat::zeros(3, 4, CV_32F);
    Rcw.copyTo(proj.rowRange(0, 3).colRange(0, 3));
    t_cw.copyTo(proj.rowRange(0, 3).col(3));
    proj = m_camera.get_intrinsic_mat() * proj;
    return proj;
}

void Frontend::get_image(const cv::Mat &image, double t) {
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
}

void Frontend::initialize(const cv::Mat &image, double time) {
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
    // todo: filter outliers?
    // todo: findEssentialMat hyper parameters
    cv::Mat Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                                       m_camera.get_intrinsic_mat(), cv::RANSAC,
                                       0.999, 0.05);
    cv::Mat Rcw, t_cw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, Rcw, t_cw);
    Rcw.convertTo(Rcw, CV_32F);
    t_cw.convertTo(t_cw, CV_32F);
    m_cur_frame = std::make_shared<Frame>(
            Frame::create_frame(dscpts, std::move(kpts), time, Rcw, t_cw));

    // triangulate points
    cv::Mat tri_res;
    cv::Mat proj_mat1 =
            get_proj_mat(m_keyframe->get_Rcw(), m_keyframe->get_Tcw());
    cv::Mat proj_mat2 = get_proj_mat(Rcw, t_cw);
    cv::triangulatePoints(proj_mat1, proj_mat2, matched_pt1, matched_pt2,
                          tri_res);

    log_debug_line("Initialize with R: " << std::endl
                                         << Rcw << std::endl
                                         << "T: " << std::endl
                                         << t_cw << std::endl
                                         << "Number of map points: "
                                         << tri_res.cols);
    _finish_tracking(tri_res, matches);
}

void Frontend::tracking(const cv::Mat &image, double time) {
    std::vector<cv::KeyPoint> kpts;
    cv::Mat dscpts;
    detect_and_compute(image, kpts, dscpts);
    std::vector<cv::DMatch> matches =
            match_descriptor(m_keyframe->get_dscpts(), dscpts);
    m_cur_frame =
            std::make_shared<Frame>(Frame::create_frame(dscpts, kpts, time));

    std::vector<cv::DMatch> new_point_match;
    std::vector<cv::Point2f> new_point1, new_point2;
    std::vector<cv::Point2f> known_img_pt2;
    std::vector<cv::Matx31f> known_pt_coords;
    const std::vector<cv::KeyPoint> &prev_kpts = m_keyframe->get_kpts();
    const std::vector<cv::KeyPoint> &cur_kpts = m_cur_frame->get_kpts();
    for (auto &match : matches) {
        if (m_keyframe->is_pt_set(match.queryIdx)) {
            cv::Matx31f pt_coord = m_keyframe->get_pt_coord(match.queryIdx);
            vo_id_t pt_id = m_keyframe->get_pt_id(match.queryIdx);
            known_pt_coords.push_back(pt_coord);
            known_img_pt2.push_back(cur_kpts[match.trainIdx].pt);
            m_cur_frame->set_pt(match.trainIdx, pt_id, pt_coord(0, 0),
                                pt_coord(0, 1), pt_coord(0, 2));
        } else {
            new_point_match.push_back(match);
            new_point1.push_back(prev_kpts[match.queryIdx].pt);
            new_point2.push_back(cur_kpts[match.trainIdx].pt);
        }
    }

    log_debug_line("Number of new points: " << new_point1.size());
    log_debug_line("Number of old points: " << known_img_pt2.size());

    // todo: retracking(known_point_match is not enough)
    // todo: camera dist coeff?

    // recover pose of current frame
    cv::Mat rcw_vec, t_cw, Rcw;
    cv::solvePnPRansac(known_pt_coords, known_img_pt2,
                       m_camera.get_intrinsic_mat(), std::vector<float>(),
                       rcw_vec, t_cw, false, 100, 1);
    cv::Rodrigues(rcw_vec, Rcw);
    Rcw.convertTo(Rcw, CV_32F);
    t_cw.convertTo(t_cw, CV_32F);
    m_cur_frame->set_pose(Rcw, t_cw);

    log_debug_line(m_cur_frame->get_id() << ":\n"
                                         << Rcw << std::endl
                                         << t_cw << std::endl);

    /*
#ifndef NDEBUG
    if (t_cw.at<float>(2, 0) > 100.0f || t_cw.at<float>(2, 0) < -100.0f) {
        cv::Mat matched_img;
        cv::drawMatches(m_keyframe->img, m_keyframe->get_kpts(), image,
                        m_cur_frame->get_kpts(), matches, matched_img);
        std::string title = "image " + std::to_string(m_keyframe->get_id()) +
                            " vs " + std::to_string(m_cur_frame->get_id());
        cv::imshow(title, matched_img);
        cv::waitKey(0);
    };
#endif
     */

    // triangulate new points
    if (!new_point1.empty()) {
        cv::Mat tri_res(4, (int) new_point1.size(), CV_32F);
        cv::Mat proj_mat1 =
                get_proj_mat(m_keyframe->get_Rcw(), m_keyframe->get_Tcw());
        cv::Mat proj_mat2 = get_proj_mat(Rcw, t_cw);
        cv::triangulatePoints(proj_mat1, proj_mat2, new_point1, new_point2,
                              tri_res);
        _finish_tracking(tri_res, new_point_match);
    }
    m_cur_frame->set_pose(Rcw, t_cw);
    _try_switch_keyframe(new_point1.size(), known_img_pt2.size());
}

void Frontend::_finish_tracking(const cv::Mat &new_tri_res,
                                const std::vector<cv::DMatch> &matches) {
    assert(new_tri_res.cols == (int) matches.size());
    std::vector<vo_uptr<MapPoint>> new_points;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Mat cur_point = new_tri_res.col((int) i);
        if (!float_eq_zero(cur_point.at<float>(3))) {
            // not infinite point
            cur_point /= cur_point.at<float>(3);
            float x = cur_point.at<float>(0);
            float y = cur_point.at<float>(1);
            float z = cur_point.at<float>(2);

            if (z > EPS) {
                vo_uptr<MapPoint> cur_map_pt = std::make_unique<MapPoint>(
                        MapPoint::create_map_point(x, y, z));
                m_keyframe->set_pt(matches[i].queryIdx, cur_map_pt->get_id(), x,
                                   y, z);
                m_cur_frame->set_pt(matches[i].trainIdx, cur_map_pt->get_id(),
                                    x, y, z);
                new_points.push_back(std::move(cur_map_pt));
            }
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