#include "vo_nono/frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <utility>
#include <vector>

#include "vo_nono/pnp.h"
#include "vo_nono/point.h"
#include "vo_nono/util.h"

namespace vo_nono {
namespace {
bool hm3d_to_euclid(cv::Mat &hm_coord, int col) {
    assert(hm_coord.type() == CV_32F);
    assert(hm_coord.rows == 4);
    float scale = hm_coord.at<float>(3, col);
    if (!std::isfinite(scale)) { return false; }
    hm_coord.col(col) /= scale;
    if (!std::isfinite(hm_coord.at<float>(0, col)) ||
        !std::isfinite(hm_coord.at<float>(1, col)) ||
        !std::isfinite(hm_coord.at<float>(2, col))) {
        return false;
    }
    return true;
}

void filter_match_with_kpts(const std::vector<cv::KeyPoint> &kpts1,
                            const std::vector<cv::KeyPoint> &kpts2,
                            std::vector<unsigned char> &mask, const int topK) {
    assert(kpts1.size() == kpts2.size());
    auto ang_diff_index = [](double diff_ang) {
        if (diff_ang < 0) { diff_ang += 360; }
        return (int) (diff_ang / 3.6);
    };
    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(kpts1.size());
    pts2.reserve(kpts2.size());
    Histogram<double> histo(101, ang_diff_index);
    std::vector<double> ang_diff;
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        double diff = kpts1[i].angle - kpts2[i].angle;
        histo.insert_element(diff);
        ang_diff.push_back(diff);

        pts1.push_back(kpts1[i].pt);
        pts2.push_back(kpts2[i].pt);
    }
    mask = std::vector<unsigned char>(pts1.size(), 1);
    histo.cal_topK(topK);
    for (int i = 0; i < (int) kpts1.size(); ++i) {
        if (!histo.is_topK(ang_diff[i])) { mask[i] = 0; }
    }
}

[[maybe_unused]] void show_matches(const vo_ptr<Frame> &left_frame,
                                   const vo_ptr<Frame> &right_frame,
                                   const std::vector<cv::DMatch> &matches) {
    cv::Mat outimg;
    std::string title = std::to_string(left_frame->get_id()) + " match " +
                        std::to_string(right_frame->get_id());
    cv::drawMatches(left_frame->img, left_frame->get_kpts(), right_frame->img,
                    right_frame->get_kpts(), matches, outimg);
    cv::imshow(title, outimg);
    cv::waitKey(0);
}
}// namespace

// detect feature points and compute descriptors
void Frontend::detect_and_compute(const cv::Mat &image,
                                  std::vector<cv::KeyPoint> &kpts,
                                  cv::Mat &dscpts, int nfeatures) {
    cv::Ptr orb_detector = cv::ORB::create(nfeatures);
    orb_detector->detectAndCompute(image, cv::Mat(), kpts, dscpts);
}

std::vector<cv::DMatch> Frontend::match_descriptor(const cv::Mat &dscpt1,
                                                   const cv::Mat &dscpt2,
                                                   float soft_dis_th,
                                                   float hard_dis_th,
                                                   int expect_cnt) {
    assert(hard_dis_th >= soft_dis_th);
    std::vector<cv::DMatch> matches;
    auto matcher = cv::BFMatcher(cv::NORM_HAMMING, true);
    matcher.match(dscpt1, dscpt2, matches);

    std::vector<bool> mask(matches.size(), true);
    int bin[256];
    memset(bin, 0, sizeof(bin));
    for (size_t i = 0; i < mask.size(); ++i) {
        bin[(int) std::floor(matches[i].distance)] += 1;
    }
    int total_cnt = 0;
    int soft_th_int = (int) soft_dis_th;
    int hard_th_int = (int) hard_dis_th;
    int final_th = soft_th_int;
    for (int i = 0; i < (int) mask.size(); ++i) {
        total_cnt += bin[i];
        if (i >= soft_th_int && total_cnt >= expect_cnt) {
            final_th = int(i);
            break;
        } else if (i >= hard_th_int) {
            final_th = hard_th_int;
            break;
        }
    }
    for (size_t i = 0; i < mask.size(); ++i) {
        if (int(matches[i].distance) > final_th) { mask[i] = false; }
    }

    matches = filter_by_mask(matches, mask);
    return matches;
}

std::vector<cv::DMatch> Frontend::filter_matches(
        const std::vector<cv::DMatch> &matches,
        const std::vector<cv::KeyPoint> &kpt1,
        const std::vector<cv::KeyPoint> &kpt2, int topK) {
    std::vector<cv::KeyPoint> match_kp1, match_kp2;
    for (auto &match : matches) {
        match_kp1.push_back(kpt1[match.queryIdx]);
        match_kp2.push_back(kpt2[match.trainIdx]);
    }

    std::vector<unsigned char> mask;
    filter_match_with_kpts(match_kp1, match_kp2, mask, topK);
    return filter_by_mask(matches, mask);
}

int Frontend::filter_triangulate_points(cv::Mat &tri, const cv::Mat &Rcw1,
                                        const cv::Mat &tcw1,
                                        const cv::Mat &Rcw2,
                                        const cv::Mat &tcw2,
                                        const std::vector<cv::Point2f> &pts1,
                                        const std::vector<cv::Point2f> &pts2,
                                        std::vector<bool> &inliers,
                                        double ang_cos_th) {
    assert(tri.cols == (int) pts1.size());
    assert(tri.cols == (int) pts2.size());
    int cnt_inlier = 0;
    const int total_pts = tri.cols;
    inliers.resize(total_pts);
    for (int i = 0; i < total_pts; ++i) {
        if (!hm3d_to_euclid(tri, i)) {
            inliers[i] = false;
            continue;
        }
        cv::Mat coord = tri.col(i).rowRange(0, 3);
        // depth must be positive
        cv::Mat coord_c1 = Rcw1 * coord + tcw1;
        cv::Mat coord_c2 = Rcw2 * coord + tcw2;
        if (coord_c1.at<float>(2, 0) < EPS || coord_c2.at<float>(2, 0) < EPS) {
            inliers[i] = false;
            continue;
        }
        // compute parallax
        cv::Mat op1 = coord + tcw1;// (coord - (-tcw1) = coord + tcw1)
        cv::Mat op2 = coord + tcw2;
        double cos_val = op1.dot(op2) / (cv::norm(op1) * cv::norm(op2));
        if (cos_val > ang_cos_th) {
            inliers[i] = false;
            continue;
        }
        inliers[i] = true;
        cnt_inlier += 1;
    }
    return cnt_inlier;
}

void Frontend::get_image(const cv::Mat &image, double t) {
    bool ok = false;
    m_last_frame = m_cur_frame;

    std::vector<cv::KeyPoint> kpts;
    cv::Mat dscpts;
    TIME_IT(detect_and_compute(image, kpts, dscpts, CNT_KEY_PTS),
            "Detect keypoint cost ");
    m_cur_frame = std::make_shared<Frame>(
            Frame::create_frame(dscpts, kpts, m_camera, t));
    m_cur_frame->img = image;

    if (m_state == State::Start) {
        assert(!m_keyframe);
        m_keyframe = m_cur_frame;
        m_state = State::Initializing;
        ok = true;
    } else if (m_state == State::Initializing) {
        assert(m_keyframe);
        int init_state = initialize(image, t);

        if (init_state == 0) {
            m_state = State::Tracking;
            ok = true;
        } else if (init_state == -1) {
            m_keyframe = m_cur_frame;
            ok = false;
        } else if (init_state == -2) {
            ok = false;
        } else {
            unimplemented();
        }
    } else if (m_state == State::Tracking) {
        assert(m_keyframe);
        if (tracking(image, t)) {
            m_map->insert_key_frame(m_cur_frame);
            ok = true;
        } else {
            m_cur_frame = m_last_frame;
        }
    } else {
        unimplemented();
    }
    assert(m_cur_frame);
    if (ok) {
        m_motion_pred.inform_pose(m_cur_frame->get_Rcw(),
                                  m_cur_frame->get_Tcw(),
                                  m_cur_frame->get_time());
    }
}

int Frontend::initialize(const cv::Mat &image, double t) {
    std::vector<cv::DMatch> matches = match_descriptor(
            m_keyframe->get_descs(), m_cur_frame->get_descs(), 8, 15, 500);

    const std::vector<cv::KeyPoint> &prev_kpts = m_keyframe->get_kpts();
    std::vector<cv::Point2f> matched_pt1, matched_pt2;
    for (auto &match : matches) {
        matched_pt1.push_back(prev_kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
    }

    // todo: less than 8 matched points?
    // todo: findEssentialMat hyper parameters
    std::vector<unsigned char> mask;
    cv::Mat Ess;
    TIME_IT(Ess = cv::findEssentialMat(matched_pt1, matched_pt2,
                                       m_camera.get_intrinsic_mat(), cv::RANSAC,
                                       0.999, 0.5, mask),
            "Find essential mat cost ");
    // filter outliers
    matches = filter_by_mask(matches, mask);
    if (matches.size() < 50) { return -1; }
    matched_pt1.clear();
    matched_pt2.clear();
    for (auto &match : matches) {
        matched_pt1.push_back(prev_kpts[match.queryIdx].pt);
        matched_pt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
    }

    cv::Mat Rcw, tcw;
    cv::recoverPose(Ess, matched_pt1, matched_pt2, Rcw, tcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);

    // triangulate points
    cv::Mat tri_res;
    cv::Mat proj_mat1 =
            get_proj_mat(m_camera.get_intrinsic_mat(), m_keyframe->get_Rcw(),
                         m_keyframe->get_Tcw());
    cv::Mat proj_mat2 = get_proj_mat(m_camera.get_intrinsic_mat(), Rcw, tcw);
    cv::triangulatePoints(proj_mat1, proj_mat2, matched_pt1, matched_pt2,
                          tri_res);
    std::vector<bool> inliers;
    int cnt_new_pt = filter_triangulate_points(
            tri_res, m_keyframe->get_Rcw(), m_keyframe->get_Tcw(), Rcw, tcw,
            matched_pt1, matched_pt2, inliers);
    if (cnt_new_pt < 40) { return -2; }

    double scale = 3;
    for (int i = 0; i < tri_res.cols; ++i) {
        if (inliers[i]) {
            assert(tri_res.at<float>(2, i) > 0);
            scale += tri_res.at<float>(2, i);
        }
    }
    scale /= cnt_new_pt;
    tcw /= scale;
    m_cur_frame->set_pose(Rcw, tcw);
    for (int i = 0; i < tri_res.cols; ++i) {
        tri_res.rowRange(0, 3).col(i) /= scale;
    }

    log_debug_line("Initialize with R: " << std::endl
                                         << Rcw << std::endl
                                         << "T: " << std::endl
                                         << tcw << std::endl
                                         << cnt_new_pt << " new map points.");
    set_new_map_points(m_keyframe, tri_res, matches, inliers);
    select_new_keyframe(m_keyframe);
    return 0;
}

bool Frontend::tracking(const cv::Mat &image, double t) {
    // todo: relocalization(both model fails)
    // todo: camera dist coeff?
    cv::Mat motion_Rcw, motion_Tcw;
    m_motion_pred.predict_pose(m_cur_frame->get_time(), motion_Rcw, motion_Tcw);
    m_cur_frame->set_pose(motion_Rcw, motion_Tcw);
    TIME_IT(track_with_match(m_keyframe),
            "Track with match cost ");

    log_debug_line(m_cur_frame->get_id()
                   << ":\n"
                   << m_cur_frame->get_Rcw() << std::endl
                   << m_cur_frame->get_Tcw() << std::endl);
    log_debug_line("Frame has " << m_cur_frame->get_set_cnt()
                                << " points set.");
    return true;
}

int Frontend::track_with_match(const vo_ptr<Frame> &o_frame) {
    std::vector<cv::DMatch> matches;
    TIME_IT(matches = match_descriptor(o_frame->get_descs(),
                                       m_cur_frame->get_descs(), 8, 30, 200),
            "ORB match cost ");
    std::vector<cv::KeyPoint> match_kpt1, match_kpt2;
    match_kpt1.reserve(matches.size());
    match_kpt2.reserve(matches.size());
    for (auto &match : matches) {
        match_kpt1.push_back(o_frame->get_kpt_by_index(match.queryIdx));
        match_kpt2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx));
    }
    std::vector<unsigned char> mask;
    filter_match_with_kpts(match_kpt1, match_kpt2, mask, 3);
    matches = filter_by_mask(matches, mask);
    match_kpt1 = filter_by_mask(match_kpt1, mask);
    match_kpt2 = filter_by_mask(match_kpt2, mask);
    log_debug_line(matches.size() << " matches after filter.");

    std::vector<cv::Matx31f> pt_coords;
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::Point2f> new_img_pt1, new_img_pt2;
    std::vector<cv::DMatch> new_match, old_match;
    for (auto &match : matches) {
        if (o_frame->is_pt_set(match.queryIdx)) {
            old_match.push_back(match);
            pt_coords.push_back(
                    o_frame->get_map_pt(match.queryIdx)->get_coord());
            img_pts.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
        } else {
            new_match.push_back(match);
            new_img_pt1.push_back(o_frame->get_kpt_by_index(match.queryIdx).pt);
            new_img_pt2.push_back(
                    m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
        }
    }
    log_debug_line(old_match.size() << " old match.");

    std::vector<bool> inliers;
    std::vector<cv::Matx31f> inlier_coords;
    std::vector<cv::Point2f> inlier_img_pts;
    cv::Mat Rcw = m_cur_frame->get_Rcw().clone(),
            tcw = m_cur_frame->get_Tcw().clone();
    pnp_ransac(pt_coords, img_pts, m_camera, 100, 1, Rcw, tcw, inliers,
               PNP_RANSAC::VO_NONO_PNP_RANSAC);
    assert(inliers.size() == pt_coords.size());
    size_t cnt_inlier = 0;
    for (int i = 0; i < (int) pt_coords.size(); ++i) {
        if (inliers[i]) {
            cnt_inlier += 1;
            inlier_coords.push_back(pt_coords[i]);
            inlier_img_pts.push_back(img_pts[i]);
            m_cur_frame->set_map_pt(old_match[i].trainIdx,
                                    o_frame->get_map_pt(old_match[i].queryIdx));
        }
    }
    log_debug_line(cnt_inlier << " inliers after pnp ransac");
    pnp_optimize_proj_err(inlier_coords, inlier_img_pts, m_camera, Rcw, tcw);
    m_cur_frame->set_pose(Rcw, tcw);

    int cnt_new_map_pt = 0;
    if (!new_img_pt1.empty() && o_frame->get_id() + 1 < m_cur_frame->get_id()) {
        cv::Mat tri_res;
        cv::Mat proj_mat1 =
                get_proj_mat(m_camera.get_intrinsic_mat(), o_frame->get_Rcw(),
                             o_frame->get_Tcw());
        cv::Mat proj_mat2 =
                get_proj_mat(m_camera.get_intrinsic_mat(),
                             m_cur_frame->get_Rcw(), m_cur_frame->get_Tcw());
        cv::triangulatePoints(proj_mat1, proj_mat2, new_img_pt1, new_img_pt2,
                              tri_res);
        std::vector<bool> tri_inliers;
        filter_triangulate_points(tri_res, o_frame->get_Rcw(),
                                  o_frame->get_Tcw(), m_cur_frame->get_Rcw(),
                                  m_cur_frame->get_Tcw(), new_img_pt1,
                                  new_img_pt2, tri_inliers);
        log_debug_line("Triangulate with transition " << cv::norm(
                               m_cur_frame->get_Tcw() - o_frame->get_Tcw()));
        cnt_new_map_pt =
                set_new_map_points(o_frame, tri_res, new_match, tri_inliers);
    }
    log_debug_line(cnt_new_map_pt << " new map points.");
    return cnt_new_map_pt;
}

int Frontend::set_new_map_points(const vo_ptr<Frame> &ref_frame,
                                 const cv::Mat &new_tri_res,
                                 const std::vector<cv::DMatch> &matches,
                                 const std::vector<bool> &inliers) {
    assert(new_tri_res.cols == (int) matches.size());
    assert(inliers.size() == matches.size());
    std::vector<vo_ptr<MapPoint>> new_points;
    int total_new_pt = 0;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (!inliers[i]) { continue; }
        cv::Mat cur_point = new_tri_res.col((int) i);
        float z_val = cur_point.at<float>(3);
        if (!float_eq_zero(z_val) && std::isfinite(z_val)) {
            // not infinite point
            cur_point /= z_val;
            float x = cur_point.at<float>(0);
            float y = cur_point.at<float>(1);
            float z = cur_point.at<float>(2);

            vo_ptr<MapPoint> cur_map_pt = std::make_shared<MapPoint>(
                    MapPoint::create_map_point(x, y, z,
                                               m_cur_frame->get_desc_by_index(
                                                       matches[i].trainIdx)));
            ref_frame->set_map_pt(matches[i].queryIdx, cur_map_pt);
            m_cur_frame->set_map_pt(matches[i].trainIdx, cur_map_pt);
            new_points.push_back(cur_map_pt);
            total_new_pt += 1;
        }
    }
    insert_map_points(new_points);
    return total_new_pt;
}

void Frontend::select_new_keyframe(const vo_ptr<Frame> &new_keyframe) {
    m_keyframe = new_keyframe;
    log_debug_line("New key frame: " << m_keyframe->get_id());
}
}// namespace vo_nono