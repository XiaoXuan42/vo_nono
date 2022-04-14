#ifndef VO_NONO_OPTIMIZE_GRAPH_H
#define VO_NONO_OPTIMIZE_GRAPH_H

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <utility>
#include <vector>

#include "vo_nono/camera.h"
#include "vo_nono/types.h"

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

    int add_cam_pose(const cv::Mat &pose, bool b_margin) {
        assert(pose.type() == CV_32F);
        assert(pose.rows == 3);
        assert(pose.cols == 4);
        int cur_cam_id = int(cam_poses.size());
        cv::Mat Rcw = pose.colRange(0, 3);
        Eigen::Matrix3d R;
        Rcw.convertTo(Rcw, CV_64F);
        cv::cv2eigen(Rcw, R);
        Eigen::AngleAxisd angle_axis(R);
        Eigen::Vector3d angle_axis_vec = angle_axis.angle() * angle_axis.axis();

        cam_poses.push_back(std::vector<double>{
                angle_axis_vec(0), angle_axis_vec(1), angle_axis_vec(2),
                double(pose.at<float>(0, 3)), double(pose.at<float>(1, 3)),
                double(pose.at<float>(2, 3))});
        edges.emplace_back(std::vector<Edge>());
        b_marginalized_cam.push_back(b_margin);
        return cur_cam_id;
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
        auto eval_options = ceres::Problem::EvaluateOptions();
        evaluate_residual(eval_options);
        return summary;
    }

    double get_loss(int cam_id, int point_id);
    void get_cam_pose(int id, cv::Mat &Rcw, cv::Mat &Tcw) const;
    void get_point_coord(int id, cv::Mat &T) const {
        T = cv::Mat::zeros(3, 1, CV_32F);
        for (int i = 0; i < 3; ++i) { T.at<float>(i) = float(points[id][i]); }
    }

private:
    void evaluate_residual(ceres::Problem::EvaluateOptions &option) {
        option.residual_blocks = residual_ids;
        problem.Evaluate(option, nullptr, &residual_vals, nullptr, nullptr);
    }

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

#endif//VO_NONO_OPTIMIZE_GRAPH_H
