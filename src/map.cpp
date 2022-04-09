#include "vo_nono/map.h"

#include "vo_nono/optim.h"

namespace vo_nono {
void Map::global_bundle_adjustment() {
    std::unique_lock<std::mutex> glb_lock(map_global_mutex);
    while (true) {
        if (mb_shutdown) { break; }
        m_global_ba_cv.wait(glb_lock, [&] { return mb_global_ba; });
        if (mb_shutdown) { break; }
        mb_global_ba = false;
        if (m_keyframes.size() < 5) { continue; }

        std::unordered_map<vo_ptr<Frame>, int> frame_to_index;
        std::unordered_map<vo_ptr<MapPoint>, int> point_to_index;
        OptimizeGraph graph(mr_camera);

        for (auto &frame : m_keyframes) {
            assert(frame_to_index.count(frame) == 0);
            int cur_cam_index = graph.add_cam_pose(frame->get_pose());
            for (int i = 0; i < int(frame->get_cnt_kpt()); ++i) {
                if (frame->is_index_set(i)) {
                    auto pt = frame->get_map_pt(i);
                    int cur_pt_index = -1;
                    if (point_to_index.count(pt) == 0) {
                        cur_pt_index = graph.add_point(pt->get_coord());
                        point_to_index[pt] = cur_pt_index;
                    } else {
                        cur_pt_index = point_to_index[pt];
                    }
                    graph.add_edge(cur_cam_index, cur_pt_index,
                                   frame->get_kpt_by_index(i).pt);
                }
            }
        }
        glb_lock.unlock();

        Optimizer::bundle_adjustment(graph, 40);

        glb_lock.lock();
        for (auto &pair : frame_to_index) {
            pair.first->set_pose(graph.get_optim_cam_pose(pair.second));
        }
        for (auto &pair : point_to_index) {
            pair.first->set_coord(graph.get_optim_point(pair.second));
        }
    }
}
}// namespace vo_nono