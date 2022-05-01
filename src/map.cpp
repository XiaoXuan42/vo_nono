#include "vo_nono/map.h"

#include <memory>

#include "vo_nono/camera.h"
#include "vo_nono/keypoint/match.h"
#include "vo_nono/optimize_graph.h"
#include "vo_nono/util/constants.h"

namespace vo_nono {
void Map::global_bundle_adjustment() {
    std::unique_lock<std::mutex> glb_lock(map_global_mutex);
    while (true) {
        if (b_shutdown_) { break; }
        cv_global_ba_.wait(glb_lock, [&] { return b_global_ba_; });
        if (b_shutdown_) { break; }
        b_global_ba_ = false;
        //if (keyframes_.size() < 5) { continue; }

        //_global_bundle_adjustment(glb_lock);
    }
}

void Map::_global_bundle_adjustment(std::unique_lock<std::mutex> &lock) {
    if (keyframes_.size() < 2) { return; }

    std::unordered_map<int, vo_ptr<Frame>> id_to_frame;
    std::unordered_map<vo_ptr<MapPoint>, int> point_to_id;
    OptimizeGraph graph(camera_);
    int active_frames = 0;
    for (auto iter = keyframes_.rbegin(); iter != keyframes_.rend(); ++iter) {
        vo_ptr<Frame> cur_frame = *iter;
        if (active_frames < 5) {
            int cam_id = graph.add_cam_pose(cur_frame->get_Rcw(),
                                            cur_frame->get_Tcw(), false);
            id_to_frame[cam_id] = cur_frame;

            for (int i = 0; i < int(cur_frame->feature_points.size()); ++i) {
                if (cur_frame->is_index_set(i)) {
                    vo_ptr<MapPoint> pt = cur_frame->get_map_pt(i);
                    int point_id;
                    if (!point_to_id.count(pt)) {
                        point_id = graph.add_point(pt->get_coord(), true);
                        point_to_id[pt] = point_id;
                    } else {
                        point_id = point_to_id[pt];
                    }
                    graph.add_edge(cam_id, point_id,
                                   cur_frame->feature_points[i]->keypoint.pt);
                }
            }
            active_frames += 1;
        } else {
            int cam_id;
            bool connected = false;
            for (int i = 0; i < int(cur_frame->feature_points.size()); ++i) {
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
                        graph.add_edge(
                                cam_id, point_id,
                                cur_frame->feature_points[i]->keypoint.pt);
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