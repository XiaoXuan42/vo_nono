#include "vo_nono/map.h"

#include <memory>

#include "vo_nono/camera.h"
#include "vo_nono/keypoint/epipolar.h"
#include "vo_nono/keypoint/match.h"
#include "vo_nono/optimize_graph.h"
#include "vo_nono/util/constants.h"
#include "vo_nono/util/util.h"

namespace vo_nono {
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
    log_debug_line("Before ess check: " << matches.size()
                                        << " after: " << valid_match.size());

    std::vector<bool> tri_inliers;
    std::vector<cv::Mat> tri_results;
    Triangulator::triangulate_and_filter_frames(
            keyframe_.get(), curframe_.get(), camera_.get_intrinsic_mat(),
            valid_match, tri_results, tri_inliers, 10000);

    assert(valid_match.size() == tri_inliers.size());
    int cnt_succ = 0;
    for (int i = 0; i < int(tri_inliers.size()); ++i) {
        if (tri_inliers[i]) {
            if (!curframe_->is_index_set(valid_match[i].trainIdx)) {
                vo_ptr<MapPoint> target_pt;
                if (keyframe_->is_index_set(valid_match[i].queryIdx)) {
                    target_pt = keyframe_->get_map_pt(valid_match[i].queryIdx);
                    if (points_seen_.count(target_pt)) {
                        int seen_times = points_seen_[target_pt];
                        cv::Mat avg_tri =
                                (float(seen_times) * target_pt->get_coord() +
                                 tri_results[i]) /
                                (float(seen_times) + 1.0f);
                        target_pt->set_coord(avg_tri);
                        points_seen_[target_pt] += 1;
                    }
                } else {
                    target_pt = std::make_shared<MapPoint>(
                            MapPoint::create_map_point(
                                    tri_results[i],
                                    keyframe_->descriptor.row(
                                            valid_match[i].queryIdx)));
                    keyframe_->set_map_pt(valid_match[i].queryIdx, target_pt);
                    assert(!points_seen_.count(target_pt));
                    points_seen_[target_pt] = 1;
                    cnt_succ += 1;
                }
                curframe_->set_map_pt(valid_match[i].trainIdx, target_pt);
            }
        }
    }
    log_debug_line("Triangulated " << cnt_succ << " points.");
}

void LocalMap::insert_frame(const FrameMessage &message) {
    keyframe_ = map_->m_keyframes.back();
    curframe_ = message.frame;
    triangulate_with_keyframe(message.match_with_keyframe);
    map_->m_frames.push_back(message.frame);
    if (message.is_keyframe) {
        points_seen_.clear();
        map_->insert_key_frame(message.frame);
    }
}


void Map::insert_frame(const FrameMessage &message) {
    local_map_->insert_frame(message);
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