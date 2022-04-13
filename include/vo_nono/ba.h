#ifndef VO_NONO_BA_H
#define VO_NONO_BA_H

#include <opencv2/core.hpp>
#include <vector>

#include "vo_nono/camera.h"
#include "vo_nono/types.h"

namespace vo_nono {
class OptimizeGraph {
public:
    explicit OptimizeGraph(const Camera &o_camera)
        : fx(o_camera.fx()),
          fy(o_camera.fy()),
          cx(o_camera.cx()),
          cy(o_camera.cy()) {}

    void add_edge(int cam_graph_id, int point_graph_id, cv::Point2f proj_pt) {
        assert(cam_graph_id < int(cam_poses.size()));
        assert(point_graph_id < int(points.size()));
        assert(graph[cam_graph_id].size() == projects[cam_graph_id].size());
        graph[cam_graph_id].push_back(point_graph_id);
        projects[cam_graph_id].push_back(proj_pt);
    }

    int add_cam_pose(const cv::Mat &pose, bool b_margin) {
        assert(pose.type() == CV_32F);
        assert(pose.rows == 3);
        assert(pose.cols == 4);
        int cur_cam_id = int(cam_poses.size());
        cam_poses.push_back(pose);
        graph.emplace_back(std::vector<int>());
        projects.emplace_back(std::vector<cv::Point2f>());
        b_marginalized_cam.push_back(b_margin);
        return cur_cam_id;
    }

    int add_point(const cv::Mat &point, bool b_margin) {
        assert(point.type() == CV_32F);
        assert(point.rows == 3);
        assert(point.cols == 1);
        int cur_pt_id = int(points.size());
        points.push_back(point);
        b_marginalized_points.push_back(b_margin);
        return cur_pt_id;
    }

    [[nodiscard]] cv::Mat get_optim_cam_pose(int i) const {
        assert(i < int(optim_cam_poses.size()));
        return optim_cam_poses[i];
    }

    [[nodiscard]] cv::Mat get_optim_point(int i) const {
        assert(i < int(optim_points.size()));
        return optim_points[i];
    }

    float fx, fy, cx, cy;
    std::vector<cv::Mat> cam_poses;
    std::vector<cv::Mat> points;
    std::vector<std::vector<int>> graph;
    std::vector<std::vector<cv::Point2f>> projects;
    std::vector<bool> b_marginalized_cam;
    std::vector<bool> b_marginalized_points;

    std::vector<cv::Mat> optim_cam_poses;
    std::vector<cv::Mat> optim_points;
};

class BundleAdjustment {
public:
    static void bundle_adjustment(OptimizeGraph &graph, int iter_cnt);
};
}// namespace vo_nono

#endif//VO_NONO_BA_H
