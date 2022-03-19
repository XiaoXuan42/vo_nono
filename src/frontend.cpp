#include "vo_nono/frontend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <opencv2/calib3d.hpp>
#include <utility>
#include <vector>

#include "vo_nono/pnp.h"
#include "vo_nono/point.h"
#include "vo_nono/util.h"

#ifndef NDEBUG
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

//#define SHOW_IMGAGE 1
#define START_IMAGE 95
#endif

namespace vo_nono {
struct ReprojInfo {
    std::shared_ptr<MapPoint> map_pt;
    double dis{};

    ReprojInfo() = default;
    ReprojInfo(std::shared_ptr<MapPoint> o_map_pt, double o_dis)
        : map_pt(std::move(o_map_pt)),
          dis(o_dis) {}
};

class ReprojRes {
public:
    [[nodiscard]] inline size_t size() const { return m_index_to_info.size(); }

    inline void clear() {
        m_index_to_info.clear();
        m_id_to_index.clear();
    }

    inline bool insert_info_check(int index,
                                  const std::shared_ptr<MapPoint> &map_pt,
                                  double dis) {
        self_check();
        vo_id_t id = map_pt->get_id();
        if (m_id_to_index.count(id)) {
            int old_index = m_id_to_index[id];
            assert(m_index_to_info.count(old_index));
            if (m_index_to_info.at(old_index).dis < dis) { return false; }
            m_index_to_info.erase(old_index);
            m_id_to_index.erase(id);
        }
        if (m_index_to_info.count(index)) {
            vo_id_t old_id = m_index_to_info.at(index).map_pt->get_id();
            assert(m_id_to_index.count(old_id));
            if (m_index_to_info.at(index).dis < dis) { return false; }
            m_id_to_index.erase(old_id);
            m_index_to_info.erase(index);
        }
        m_index_to_info[index] = ReprojInfo(map_pt, dis);
        m_id_to_index[id] = index;
        self_check();
        return true;
    }

    [[nodiscard]] inline bool is_better_match(int index, double dis) const {
        if (m_index_to_info.count(index)) {
            return m_index_to_info.at(index).dis > dis;
        }
        return true;
    }

    inline void erase(int index) {
        if (m_index_to_info.count(index)) {
            vo_id_t id = m_index_to_info.at(index).map_pt->get_id();
            m_index_to_info.erase(index);
            assert(m_id_to_index.count(id));
            m_id_to_index.erase(id);
        }
        self_check();
    }

    inline void self_check() const {
        assert(m_index_to_info.size() == m_id_to_index.size());
        for (auto &pair : m_index_to_info) {
            vo_id_t id = pair.second.map_pt->get_id();
            assert(m_id_to_index.at(id) == pair.first);
        }
        for (auto &pair : m_id_to_index) {
            assert(m_index_to_info.at(pair.second).map_pt->get_id() ==
                   pair.first);
        }
    }

    [[nodiscard]] auto begin() const { return m_index_to_info.begin(); }
    [[nodiscard]] auto end() const { return m_index_to_info.end(); }

private:
    std::map<int, ReprojInfo> m_index_to_info;
    std::map<vo_id_t, int> m_id_to_index;
};

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

cv::Point2f reproj_point(const cv::Mat &proj_mat, const cv::Mat &coord_3d) {
    cv::Mat hm_coord = cv::Mat(4, 1, CV_32F);
    coord_3d.copyTo(hm_coord.rowRange(0, 3));
    hm_coord.at<float>(3, 0) = 1.0f;
    cv::Mat proj_img = proj_mat * hm_coord;
    proj_img /= proj_img.at<float>(2, 0);
    return cv::Point2f(proj_img.at<float>(0, 0), proj_img.at<float>(1, 0));
}

void reproj_points_from_frame(const vo_ptr<Frame> &left_frame,
                              const vo_ptr<Frame> &right_frame,
                              const Camera &camera, ReprojRes &proj_res) {
    std::vector<int> match_left_id;
    std::vector<cv::KeyPoint> left_img_kpts;
    std::vector<cv::Mat> left_img_descs;
    std::vector<cv::Point2f> proj_right_img_pts;
    cv::Mat proj_mat =
            get_proj_mat(camera.get_intrinsic_mat(), right_frame->get_Rcw(),
                         right_frame->get_Tcw());
    for (int i = 0; i < (int) left_frame->get_kpts().size(); ++i) {
        if (left_frame->is_pt_set(i)) {
            cv::Point2f img_pt = reproj_point(
                    proj_mat, left_frame->get_map_pt(i)->get_coord());
            float img_x = img_pt.x, img_y = img_pt.y;
            if (!std::isfinite(img_x) || img_x < 0 ||
                img_x > camera.get_width() || !std::isfinite(img_y) ||
                img_y < 0 || img_y > camera.get_height()) {
                continue;
            }
            proj_right_img_pts.emplace_back(img_x, img_y);
            left_img_descs.push_back(left_frame->get_desc_by_index(i));
            left_img_kpts.push_back(left_frame->get_kpt_by_index(i));
            match_left_id.push_back(i);
        }
    }
    if (match_left_id.size() < 10) {
        log_debug_line("Original frame has less than 10 map points.");
        return;
    }

    std::vector<cv::KeyPoint> right_img_kpts(match_left_id.size());
    std::vector<bool> inliers(match_left_id.size());
    std::vector<double> desc_dis(match_left_id.size());
    std::vector<int> match_right_id(match_left_id.size());
    std::map<int, int> local_book;
    assert(proj_right_img_pts.size() == left_img_kpts.size());
    assert(left_img_kpts.size() == left_img_descs.size());
    assert(match_left_id.size() == left_img_kpts.size());
    std::vector<float> radius_ths = {6.0f, 12.0f, 18.0f};

    for (auto r_th : radius_ths) {
        size_t not_found_cnt = 0, matched = 0;
        local_book.clear();

        for (int i = 0; i < (int) match_left_id.size(); ++i) {
            inliers[i] = false;
            double cur_dis;
            int match_id = right_frame->local_match(left_img_descs[i],
                                                    proj_right_img_pts[i],
                                                    cur_dis, r_th, 1.0);
            if (match_id < 0) {
                not_found_cnt += 1;
            } else {
                if (local_book.count(match_id)) {
                    double prev_dis = desc_dis[local_book[match_id]];
                    if (prev_dis <= cur_dis) {
                        continue;
                    } else {
                        inliers[local_book[match_id]] = false;
                    }
                }
                matched += 1;
                inliers[i] = true;
                desc_dis[i] = cur_dis;
                match_right_id[i] = match_id;
                right_img_kpts[i] = right_frame->get_kpt_by_index(match_id);
                local_book[match_id] = i;
            }
        }
        log_debug_line("Radius: " << r_th
                                  << " total left pts: " << match_left_id.size()
                                  << " matched " << matched << " pts"
                                  << ", not found " << not_found_cnt);
        if (matched >= 40 ||
            (matched + 10 >= match_left_id.size() && matched >= 20)) {
            break;
        }
    }

    // filter matches
    match_left_id = filter_by_mask(match_left_id, inliers);
    match_right_id = filter_by_mask(match_right_id, inliers);
    right_img_kpts = filter_by_mask(right_img_kpts, inliers);
    left_img_kpts = filter_by_mask(left_img_kpts, inliers);
    desc_dis = filter_by_mask(desc_dis, inliers);
    assert(match_left_id.size() == match_right_id.size());

#ifdef SHOW_IMGAGE
    if (left_frame->get_id() >= START_IMAGE) {
        std::vector<cv::DMatch> matches;
        for (int i = 0; i < (int) match_right_id.size(); ++i) {
            matches.emplace_back(i, i, 0);
        }
        cv::Mat outimg;
        cv::drawMatches(left_frame->img, left_img_kpts, right_frame->img,
                        right_img_kpts, matches, outimg);
        std::string title = std::to_string(left_frame->get_id()) + " vs " +
                            std::to_string(right_frame->get_id());
        cv::imshow(title, outimg);
        cv::waitKey(0);
    }
#endif

    if (match_left_id.size() < 10) {
        log_debug_line("Not enough matched in reprojection.");
        return;
    }
    std::vector<unsigned char> inliers2;
    filter_match_with_kpts(left_img_kpts, right_img_kpts, inliers2, 4);
    left_img_kpts = filter_by_mask(left_img_kpts, inliers2);
    right_img_kpts = filter_by_mask(right_img_kpts, inliers2);
    match_left_id = filter_by_mask(match_left_id, inliers2);
    match_right_id = filter_by_mask(match_right_id, inliers2);
    desc_dis = filter_by_mask(desc_dis, inliers2);
    assert(left_img_kpts.size() == right_img_kpts.size());
    assert(match_left_id.size() == match_right_id.size());
    assert(left_img_kpts.size() == match_right_id.size());

    // update proj_res
    for (int i = 0; i < (int) match_right_id.size(); ++i) {
        int img_pt_index = match_right_id[i];
        proj_res.insert_info_check(img_pt_index,
                                   left_frame->get_map_pt(match_left_id[i]),
                                   desc_dis[i]);
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
    /*
    ReprojRes proj_res;
    m_cur_frame->set_pose(m_last_frame->get_Rcw(), m_last_frame->get_Tcw());
    TIME_IT(reproj_with_motion(proj_res), "reproj with motion cost ");

    log_debug_line("Motion model predict:\n"
                   << m_cur_frame->get_Rcw() << "\n"
                   << m_cur_frame->get_Tcw());

    if (proj_res.size() <= 10) {
        log_debug_line("Motion model totally abandoned.");
        proj_res.clear();
        m_cur_frame->set_pose(m_last_frame->get_Rcw(), m_last_frame->get_Tcw());
    }
     */
    /*
    log_debug_line(m_local_points.size() << " local points.");
    TIME_IT(reproj_with_local_points(proj_res),
            "reproj with local points cost ");
    int cnt_proj_succ = (int) proj_res.size();
    log_debug_line("Projected " << cnt_proj_succ << " points.");

    if (proj_res.size() < 10) {
        std::string title = "image " + std::to_string(m_cur_frame->get_id());
        cv::imshow(title, m_cur_frame->img);
        cv::waitKey(0);
        log_debug_line("Track missing. Discard this frame.");
        return false;
    }
    log_debug_line("Pose estimate with " << proj_res.size() << " points.");
    TIME_IT(reproj_pose_estimate(proj_res, 1), "reproj pose estimate cost ");
    log_debug_line("Pose estimated with " << proj_res.size() << " left.");
    int cnt_new_pts = triangulate(m_keyframe, proj_res);
    log_debug_line("Generate " << cnt_new_pts << " new map points.");

    for (auto &pair : proj_res) {
        if (!m_cur_frame->is_pt_set(pair.first)) {
            m_cur_frame->set_map_pt(pair.first, pair.second.map_pt);
        }
    }
    */
    int cnt_new_pts = 0;
    TIME_IT(cnt_new_pts = track_with_match(m_keyframe),
            "Track with match cost ");
    if (cnt_new_pts <= 10 ||
        m_cur_frame->get_id() > m_keyframe->get_id() + 20) {
        if (m_last_frame->get_kpt_cnt() > m_cur_frame->get_kpt_cnt()) {
            select_new_keyframe(m_last_frame);
        } else {
            select_new_keyframe(m_cur_frame);
        }
    }

    log_debug_line(m_cur_frame->get_id()
                   << ":\n"
                   << m_cur_frame->get_Rcw() << std::endl
                   << m_cur_frame->get_Tcw() << std::endl);
    log_debug_line("Frame has " << m_cur_frame->get_set_cnt()
                                << " points set.");
    return true;
}

void Frontend::reproj_with_motion(ReprojRes &proj_res) {
    if (!m_motion_pred.is_available() || !m_last_frame) { return; }
    cv::Mat pred_Rcw, pred_tcw;
    m_motion_pred.predict_pose(m_cur_frame->get_time(), pred_Rcw, pred_tcw);
    m_cur_frame->set_pose(pred_Rcw, pred_tcw);

    reproj_points_from_frame(m_last_frame, m_cur_frame, m_camera, proj_res);
}

void Frontend::reproj_pose_estimate(ReprojRes &proj_res, float reproj_th) {
    size_t proj_pt_cnt = proj_res.size();
    std::vector<cv::Point2f> img_pts;
    std::vector<cv::Matx31f> map_pt_coords;
    img_pts.reserve(proj_pt_cnt);
    map_pt_coords.reserve(proj_pt_cnt);

    std::vector<int> seq_to_index;
    for (auto &pair : proj_res) {
        img_pts.push_back(m_cur_frame->get_kpts()[pair.first].pt);
        map_pt_coords.push_back(pair.second.map_pt->get_coord());
        seq_to_index.push_back(pair.first);
    }
    cv::Mat init_Rcw = m_cur_frame->get_Rcw().clone(),
            init_tcw = m_cur_frame->get_Tcw().clone();
    cv::Mat rvec, tcw, Rcw;
    cv::Rodrigues(init_Rcw, rvec);
    rvec.convertTo(rvec, CV_64F);
    tcw = init_tcw;
    tcw.convertTo(tcw, CV_64F);

    std::vector<int> inliers;
    cv::solvePnPRansac(map_pt_coords, img_pts, m_camera.get_intrinsic_mat(),
                       std::vector<double>(), rvec, tcw, true, 100, reproj_th,
                       0.99, inliers);

    size_t cur_inlier = 0;
    for (int i = 0; i < (int) seq_to_index.size(); ++i) {
        if (cur_inlier >= inliers.size() || i != inliers[cur_inlier]) {
            proj_res.erase(seq_to_index[i]);
        }
        while (cur_inlier < inliers.size() && inliers[cur_inlier] <= i) {
            cur_inlier += 1;
        }
    }

    cv::Rodrigues(rvec, Rcw);
    Rcw.convertTo(Rcw, CV_32F);
    tcw.convertTo(tcw, CV_32F);
    m_cur_frame->set_pose(Rcw, tcw);
}

void Frontend::reproj_with_local_points(ReprojRes &proj_res) {
    cv::Mat proj_mat =
            get_proj_mat(m_camera.get_intrinsic_mat(), m_cur_frame->get_Rcw(),
                         m_cur_frame->get_Tcw());
    size_t not_in_view = 0, too_far = 0, not_evident = 0;
    for (auto &pair : m_local_points) {
        vo_ptr<MapPoint> map_pt = pair.second.second;
        cv::Point2f pt = reproj_point(proj_mat, map_pt->get_coord());
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || pt.x < 0 ||
            pt.x > m_camera.get_width() || pt.y < 0 ||
            pt.y > m_camera.get_height()) {
            not_in_view += 1;
            continue;
        }

        double dis;
        int index = m_cur_frame->local_match(map_pt->get_desc(), pt, dis, 12.0f,
                                             0.8);
        if (index > 0) {
            proj_res.insert_info_check(index, map_pt, dis);
        } else if (index == -1) {
            too_far += 1;
        } else if (index == -2) {
            not_evident += 1;
        }
    }
    log_debug_line("Not in view: " << not_in_view << ". Too far: " << too_far
                                   << " Not evident: " << not_evident);
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
    if (!new_img_pt1.empty()) {
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

int Frontend::triangulate(const vo_ptr<Frame> &ref_frame, ReprojRes &proj_res) {
    int total_new_pts = 0;
    std::vector<cv::DMatch> matches;
    TIME_IT(matches = match_descriptor(ref_frame->get_descs(),
                                       m_cur_frame->get_descs(), 8, 15, 100),
            "Match cost ");
    std::vector<unsigned char> mask;
    std::vector<cv::KeyPoint> kpts1, kpts2;
    kpts1.reserve(matches.size());
    kpts2.reserve(matches.size());
    for (auto &match : matches) {
        kpts1.push_back(ref_frame->get_kpt_by_index(match.queryIdx));
        kpts2.push_back(m_cur_frame->get_kpt_by_index(match.trainIdx));
    }
    filter_match_with_kpts(kpts1, kpts2, mask, 3);
    for (size_t i = 0; i < mask.size(); ++i) {
        if (!mask[i]) { continue; }
        if (proj_res.is_better_match(matches[i].trainIdx,
                                     matches[i].distance)) {
            proj_res.erase(matches[i].trainIdx);
            if (ref_frame->is_pt_set(matches[i].queryIdx)) {
                mask[i] = 0;
                m_cur_frame->set_map_pt(
                        matches[i].trainIdx,
                        ref_frame->get_map_pt(matches[i].queryIdx));
            } else {
                mask[i] = 1;
            }
        } else {
            mask[i] = 0;
        }
    }
    matches = filter_by_mask(matches, mask);

    if (!matches.empty()) {
        cv::Mat proj_mat1 =
                get_proj_mat(m_camera.get_intrinsic_mat(), ref_frame->get_Rcw(),
                             ref_frame->get_Tcw());
        cv::Mat proj_mat2 =
                get_proj_mat(m_camera.get_intrinsic_mat(),
                             m_cur_frame->get_Rcw(), m_cur_frame->get_Tcw());
        cv::Mat tri_res;
        std::vector<cv::Point2f> proj_pts1, proj_pts2;
        proj_pts1.reserve(matches.size());
        proj_pts2.reserve(matches.size());
        for (auto &match : matches) {
            proj_pts1.push_back(ref_frame->get_kpt_by_index(match.queryIdx).pt);
            proj_pts2.push_back(
                    m_cur_frame->get_kpt_by_index(match.trainIdx).pt);
        }
        cv::triangulatePoints(proj_mat1, proj_mat2, proj_pts1, proj_pts2,
                              tri_res);
        std::vector<bool> inliers;
        filter_triangulate_points(tri_res, ref_frame->get_Rcw(),
                                  ref_frame->get_Tcw(), m_cur_frame->get_Rcw(),
                                  m_cur_frame->get_Tcw(), proj_pts1, proj_pts2,
                                  inliers);
        total_new_pts =
                set_new_map_points(ref_frame, tri_res, matches, inliers);
    } else {
        total_new_pts = 0;
    }
    return total_new_pts;
}

int Frontend::set_new_map_points(const vo_ptr<Frame> &ref_frame,
                                 const cv::Mat &new_tri_res,
                                 const std::vector<cv::DMatch> &matches,
                                 const std::vector<bool> &inliers) {
    assert(new_tri_res.cols == (int) matches.size());
    assert(inliers.size() == matches.size());
    std::vector<vo_ptr<MapPoint>> new_points;
    bool b_in_local = false;
    for (auto &w_frame : m_window_frame) {
        if (w_frame->get_id() == ref_frame->get_id()) {
            b_in_local = true;
            break;
        }
    }

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

            if (b_in_local) { set_local_map_point(cur_map_pt); }
        }
    }
    insert_map_points(new_points);
    return total_new_pt;
}

void Frontend::select_new_keyframe(const vo_ptr<Frame> &new_keyframe) {
    assert(m_window_frame.size() <= CNT_MAX_WINDOW_FRAMES);
    if (m_window_frame.size() == CNT_MAX_WINDOW_FRAMES) {
        vo_ptr<Frame> old_frame = m_window_frame.front();
        m_window_frame.pop_front();

        for (size_t i = 0; i < old_frame->get_kpt_cnt(); ++i) {
            if (old_frame->is_pt_set((int) i)) {
                vo_id_t pt_id = old_frame->get_map_pt((int) i)->get_id();
                unset_local_map_point(pt_id);
            }
        }
    }

    for (size_t i = 0; i < new_keyframe->get_kpt_cnt(); ++i) {
        if (new_keyframe->is_pt_set((int) i)) {
            vo_ptr<MapPoint> map_pt = new_keyframe->get_map_pt((int) i);
            set_local_map_point(map_pt);
        }
    }

    m_window_frame.push_back(new_keyframe);
    m_keyframe = new_keyframe;
    log_debug_line("New key frame: " << m_keyframe->get_id());
    log_debug_line("Local map points: " << m_local_points.size());
}
}// namespace vo_nono