#ifndef VO_NONO_OPTIMIZE_H
#define VO_NONO_OPTIMIZE_H

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <utility>
#include <vector>

#include "vo_nono/camera.h"
#include "vo_nono/types.h"
#include "vo_nono/util/geometry.h"

namespace vo_nono {
class OptimizeGraph {
public:
    explicit OptimizeGraph(Camera o_camera) : camera(std::move(o_camera)) {}

    void add_edge(int cam_graph_id, int point_id, cv::Point2f proj_pt) {
        assert(cam_graph_id < int(cam_poses.size()));
        assert(point_id < int(points.size()));
        Edge cur_edge{};
        cur_edge.point_id = point_id;
        cur_edge.img_x = proj_pt.x;
        cur_edge.img_y = proj_pt.y;
        edges[cam_graph_id].push_back(cur_edge);
    }

    int add_cam_pose(const cv::Mat &Rcw, const cv::Mat &Tcw, bool b_margin) {
        assert(Rcw.type() == CV_32F);
        assert(Rcw.rows == 3);
        assert(Rcw.cols == 3);
        assert(Tcw.type() == CV_32F);
        assert(Tcw.rows == 3);
        assert(Tcw.cols == 1);
        int cur_cam_id = int(cam_poses.size());
        std::array<double, 3> angle_axis =
                Geometry::rotation_mat_to_angle_axis(Rcw);
        cam_poses.push_back(std::vector<double>{
                angle_axis[0], angle_axis[1], angle_axis[2],
                double(Tcw.at<float>(0)), double(Tcw.at<float>(1)),
                double(Tcw.at<float>(2))});
        edges.emplace_back(std::vector<Edge>());
        b_marginalized_cam.push_back(b_margin);
        return cur_cam_id;
    }

    int add_cam_pose(const cv::Mat &pose, bool b_margin) {
        return add_cam_pose(pose.colRange(0, 3), pose.col(3), b_margin);
    }

    int add_point(const cv::Mat &point, bool b_margin) {
        assert(point.type() == CV_32F);
        assert(point.rows == 3);
        assert(point.cols == 1);
        int cur_pt_id = int(points.size());
        points.push_back(std::vector<double>{
                point.at<float>(0), point.at<float>(1), point.at<float>(2)});
        b_marginalized_points.push_back(b_margin);
        return cur_pt_id;
    }

    void to_problem();
    ceres::Solver::Summary evaluate_solver(
            const ceres::Solver::Options &options) {
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        evaluate_residual();
        return summary;
    }

    double get_loss(int cam_id, int point_id);
    void get_cam_pose(int id, cv::Mat &Rcw, cv::Mat &Tcw) const;
    void get_point_coord(int id, cv::Mat &T) const {
        T = cv::Mat::zeros(3, 1, CV_32F);
        for (int i = 0; i < 3; ++i) { T.at<float>(i) = float(points[id][i]); }
    }
    void set_loss_kernel(ceres::LossFunction *l) { loss = l; }
    void evaluate_residual() {
        auto option = ceres::Problem::EvaluateOptions();
        option.residual_blocks = residual_ids;
        std::vector<double> blk_residuals;
        problem.Evaluate(option, nullptr, &blk_residuals, nullptr, nullptr);
        assert(blk_residuals.size() == residual_ids.size() * 2);
        residual_vals.resize(residual_ids.size());
        for (int i = 0; i < int(blk_residuals.size()); i += 2) {
            double d1 = blk_residuals[i], d2 = blk_residuals[i + 1];
            residual_vals[i / 2] = d1 * d1 + d2 * d2;
        }
    }

private:
    struct Edge {
        int point_id;
        int residual_id;
        double img_x;
        double img_y;
    };

    Camera camera;
    std::vector<std::vector<double>>
            cam_poses;// axis_angle + translation: 6 element
    std::vector<std::vector<double>> points;// coordinates: 3 elements

    std::vector<std::vector<Edge>> edges;

    std::vector<bool> b_marginalized_cam;
    std::vector<bool> b_marginalized_points;

    std::vector<ceres::ResidualBlockId> residual_ids;
    std::vector<double> residual_vals;

    ceres::Problem problem;
    ceres::LossFunction *loss{};
};
}// namespace vo_nono

#endif//VO_NONO_OPTIMIZE_H
