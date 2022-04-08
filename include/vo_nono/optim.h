#ifndef VO_NONO_OPTIM_H
#define VO_NONO_OPTIM_H

#include <opencv2/core.hpp>
#include <vector>

#include "vo_nono/camera.h"
#include "vo_nono/types.h"

namespace vo_nono {
class OptimizeGraph {
public:
    explicit OptimizeGraph(const Camera& o_camera, int cam_sz, int pt_sz)
        : fx(o_camera.fx()),
          fy(o_camera.fy()),
          cx(o_camera.cx()),
          cy(o_camera.cy()),
          cam_sz(cam_sz),
          pt_sz(pt_sz),
          cam_poses(cam_sz),
          points(pt_sz),
          graph(cam_sz),
          projects(cam_sz) {}

    void add_edge(int cam_graph_id, int point_graph_id, cv::Point2f proj_pt) {
        assert(cam_graph_id < cam_sz);
        assert(cam_sz == int(cam_poses.size()));
        assert(point_graph_id < pt_sz);
        assert(pt_sz == int(points.size()));
        assert(graph[cam_graph_id].size() == projects[cam_graph_id].size());
        graph[cam_graph_id].push_back(point_graph_id);
        projects[cam_graph_id].push_back(proj_pt);
    }

    float fx, fy, cx, cy;
    const int cam_sz, pt_sz;
    std::vector<cv::Mat> cam_poses;
    std::vector<cv::Mat> points;
    std::vector<std::vector<int>> graph;
    std::vector<std::vector<cv::Point2f>> projects;

    std::vector<cv::Mat> optim_cam_poses;
    std::vector<cv::Mat> optim_points;
};
class Optimizer {
public:
    static void bundle_adjustment(OptimizeGraph &graph, int iter_cnt);
};
}// namespace vo_nono

#endif//VO_NONO_OPTIM_H
