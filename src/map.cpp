#include "vo_nono/map.h"

#include <memory>

#include "vo_nono/camera.h"
#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/keypoint/match.h"
#include "vo_nono/optimize_graph.h"
#include "vo_nono/util/constants.h"
#include "vo_nono/util/geometry.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
void InvDepthFilter::filter(const cv::Mat &o0_cw, const cv::Mat &o1_cw,
                            const cv::Mat &coord) {
    assert(coord.type() == CV_32F);
    cv::Mat t0 = coord + o0_cw;
    cv::Mat t1 = coord + o1_cw;
    double t0_square = t0.dot(t0);
    double t1_square = t1.dot(t1);
    double cos2 = t0.dot(t1);
    cos2 = (cos2 * cos2) / (t0_square * t1_square);
    double cur_var =
            t1_square / (t0_square * t0_square * (1.0 - cos2)) * basic_var_;
    double cur_d = 1.0 / std::sqrt(t0_square);
    // if (cur_d < mean_ - 2 * var_ || cur_d > mean_ + 2 * var_) { return false; }

    double update_mean = (var_ * cur_d + cur_var * mean_) / (cur_var + var_);
    double update_var = (var_ * cur_var) / (cur_var + var_);
    mean_ = update_mean;
    var_ = update_var;

    dir_ = (dir_ * float(cnt_) + t0 / cv::norm(t0)) / float(cnt_ + 1);
    dir_ /= cv::norm(dir_);
    cnt_ += 1;
}

LocalMap::LocalMap(Map *map) : map_(map), camera_(map->mr_camera) {}
void LocalMap::triangulate_with_keyframe(
        const std::vector<cv::DMatch> &matches) {
    std::vector<cv::DMatch> valid_match;
    std::vector<bool> mask;
    std::vector<cv::Point2f> pts1, pts2;
    cv::Mat R21, t21;
    Geometry::relative_pose(keyframe_->get_Rcw(), keyframe_->get_Tcw(),
                            curframe_->get_Rcw(), curframe_->get_Tcw(), R21,
                            t21);
    cv::Mat ess = Epipolar::compute_essential(R21, t21);
    for (auto &match : matches) {
        pts1.push_back(keyframe_->kpts[match.queryIdx].pt);
        pts2.push_back(curframe_->kpts[match.trainIdx].pt);
    }
    ORBMatcher::filter_match_by_ess(ess, camera_.get_intrinsic_mat(), pts1,
                                    pts2, 0.01, mask);
    valid_match = filter_by_mask(matches, mask);

    std::vector<bool> tri_inliers;
    std::vector<cv::Mat> tri_results;
    Triangulator::triangulate_and_filter_frames(
            keyframe_.get(), curframe_.get(), camera_.get_intrinsic_mat(),
            valid_match, tri_results, tri_inliers, 10000);

    assert(valid_match.size() == tri_inliers.size());
    int cnt_new_tri = 0;
    for (int i = 0; i < int(tri_inliers.size()); ++i) {
        if (tri_inliers[i] &&
            !curframe_->is_index_set(valid_match[i].trainIdx)) {
            vo_ptr<MapPoint> target_pt;

            int keyframe_index = valid_match[i].queryIdx;
            if (keyframe_->is_index_set(keyframe_index)) {
                target_pt = keyframe_->get_map_pt(keyframe_index);
            } else {
                filters_[keyframe_index].filter(keyframe_->get_Tcw(),
                                                curframe_->get_Tcw(),
                                                tri_results[i]);
                if (filters_[keyframe_index].get_variance() < 0.0001) {
                    target_pt = std::make_shared<MapPoint>(
                            MapPoint::create_map_point(
                                    filters_[keyframe_index].get_coord(
                                            -keyframe_->get_Tcw(),
                                            keyframe_->get_Rcw()),
                                    keyframe_->descriptor.row(keyframe_index)));
                    keyframe_->set_map_pt(keyframe_index, target_pt);
                    cnt_new_tri += 1;
                }
            }

            if (target_pt) {
                curframe_->set_map_pt(valid_match[i].trainIdx, target_pt);
            }
        }
    }
    log_debug_line("Triangulated " << cnt_new_tri << " new points.");
}

void LocalMap::insert_keyframe(const vo_ptr<Frame> &keyframe) {
    filters_.clear();
    filters_.resize(keyframe->kpts.size());
    double basic_var = std::min(camera_.fx(), camera_.fy());
    basic_var = 4.0 / (basic_var * basic_var);
    for (size_t i = 0; i < keyframe->kpts.size(); ++i) {
        filters_[i].set_information(camera_, keyframe->kpts[i].pt, basic_var);
    }
    map_->insert_key_frame(keyframe);
}

void LocalMap::insert_frame(const FrameMessage &message, bool is_triangulate) {
    if (map_->m_keyframes.empty()) {
        if (message.is_keyframe) { insert_keyframe(message.frame); }
        return;
    }
    keyframe_ = map_->m_keyframes.back();
    curframe_ = message.frame;
    if (is_triangulate) {
        triangulate_with_keyframe(message.match_with_keyframe);
    }
    map_->m_frames.push_back(message.frame);
    if (message.is_keyframe) { insert_keyframe(message.frame); }
}

void Map::global_bundle_adjustment() {
    std::unique_lock<std::mutex> glb_lock(map_global_mutex);
    while (true) {
        if (mb_shutdown) { break; }
        m_global_ba_cv.wait(glb_lock, [&] { return mb_global_ba; });
        if (mb_shutdown) { break; }
        mb_global_ba = false;
        //if (m_keyframes.size() < 5) { continue; }

        _global_bundle_adjustment(glb_lock);
    }
}

void Map::_global_bundle_adjustment(std::unique_lock<std::mutex> &lock) {
    if (m_keyframes.size() < 2) { return; }

    std::unordered_map<int, vo_ptr<Frame>> id_to_frame;
    std::unordered_map<vo_ptr<MapPoint>, int> point_to_id;
    OptimizeGraph graph(mr_camera);
    int active_frames = 0;
    for (auto iter = m_keyframes.rbegin(); iter != m_keyframes.rend(); ++iter) {
        vo_ptr<Frame> cur_frame = *iter;
        if (active_frames < 5) {
            int cam_id = graph.add_cam_pose(cur_frame->get_Rcw(),
                                            cur_frame->get_Tcw(), false);
            id_to_frame[cam_id] = cur_frame;

            for (int i = 0; i < int(cur_frame->kpts.size()); ++i) {
                if (cur_frame->is_index_set(i)) {
                    vo_ptr<MapPoint> pt = cur_frame->get_map_pt(i);
                    int point_id;
                    if (!point_to_id.count(pt)) {
                        point_id = graph.add_point(pt->get_coord(), true);
                        point_to_id[pt] = point_id;
                    } else {
                        point_id = point_to_id[pt];
                    }
                    graph.add_edge(cam_id, point_id, cur_frame->kpts[i].pt);
                }
            }
            active_frames += 1;
        } else {
            int cam_id;
            bool connected = false;
            for (int i = 0; i < int(cur_frame->kpts.size()); ++i) {
                if (cur_frame->is_index_set(i)) {
                    vo_ptr<MapPoint> pt = cur_frame->get_map_pt(i);
                    if (point_to_id.count(pt)) {
                        if (!connected) {
                            cam_id = graph.add_cam_pose(cur_frame->get_Rcw(),
                                                        cur_frame->get_Tcw(),
                                                        true);
                            connected = true;
                        }
                        int point_id = point_to_id[pt];
                        graph.add_edge(cam_id, point_id, cur_frame->kpts[i].pt);
                    }
                }
            }
        }
    }
    lock.unlock();
    graph.set_loss_kernel(new ceres::HuberLoss(chi2_2_5));
    graph.to_problem();
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    graph.evaluate_solver(options);
    lock.lock();

    for (auto &pair : id_to_frame) {
        int cam_id = pair.first;
        vo_ptr<Frame> cur_frame = pair.second;
        cv::Mat Rcw, Tcw;
        graph.get_cam_pose(cam_id, Rcw, Tcw);
        cur_frame->set_pose(Rcw, Tcw);
    }
    for (auto &pair : point_to_id) {
        int point_id = pair.second;
        vo_ptr<MapPoint> cur_pt = pair.first;
        cv::Mat coord;
        graph.get_point_coord(point_id, coord);
        cur_pt->set_coord(coord);
    }
}
}// namespace vo_nono