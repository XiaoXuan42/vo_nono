#include "vo_nono/optim.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <Eigen/Core>

namespace vo_nono {
namespace {
Eigen::Vector3d to_vector3d(const cv::Mat &mat) {
    assert(mat.type() == CV_32F);
    assert(mat.cols == 1);
    assert(mat.rows == 3);
    Eigen::Vector3d res;
    for (int i = 0; i < 3; ++i) {
        res(i) = mat.at<float>(i);
    }
    return res;
}

g2o::SE3Quat to_se3quat(const cv::Mat &mat) {
    assert(mat.type() == CV_32F);
    assert(mat.cols == 4);
    assert(mat.rows == 3);
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 3, 1> t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = mat.at<float>(i, j);
        }
    }
    for (int i = 0; i < 3; ++i) {
        t(i, 0) = mat.at<float>(i, 3);
    }
    return g2o::SE3Quat(R, t);
}

cv::Mat to_cvmat(const g2o::SE3Quat &se3) {
    cv::Mat pose = cv::Mat::zeros(3, 4, CV_32F);
    cv::Mat translation, rotation;
    const Eigen::Vector3d &vec3 = se3.translation();
    Eigen::Matrix3d eigen_rotation = se3.rotation().toRotationMatrix();
    for (int i = 0; i < 3; ++i) { pose.at<float>(i, 3) = float(vec3(i)); }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            pose.at<float>(i, j) = float(eigen_rotation(i, j));
        }
    }
    return pose;
}

cv::Mat to_cvmat(const Eigen::Vector3d &vec) {
    cv::Mat res = cv::Mat::zeros(3, 1, CV_32F);
    for (int i = 0; i < 3; ++i) { res.at<float>(i) = float(vec(i)); }
    return res;
}
}// namespace
void Optimizer::bundle_adjustment(OptimizeGraph &graph, int iter_cnt) {
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver;
    using BaLinearSolver =
            g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>;
    linear_solver = g2o::make_unique<BaLinearSolver>();
    auto *solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));

    optimizer.setAlgorithm(solver);
    for (int i = 0; i < graph.pt_sz; ++i) {
        auto *vp = new g2o::VertexSBAPointXYZ();
        vp->setId(i + graph.cam_sz);
        vp->setMarginalized(true);
        vp->setEstimate(to_vector3d(graph.points[i]));
        optimizer.addVertex(vp);
    }

    for (int i = 0; i < graph.cam_sz; ++i) {
        auto *v_cam = new g2o::VertexSE3Expmap();
        v_cam->setEstimate(to_se3quat(graph.cam_poses[i]));
        v_cam->setId(i);
        optimizer.addVertex(v_cam);
        for (int j = 0; j < int(graph.graph[i].size()); ++j) {
            auto *e = new g2o::EdgeSE3ProjectXYZ();
            cv::Point2f proj_pt = graph.projects[i][j];
            Eigen::Matrix<double, 2, 1> measurement(proj_pt.x, proj_pt.y);
            int p_id = graph.graph[i][j] + graph.cam_sz;
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                                    optimizer.vertex(p_id)));
            e->setVertex(1,
                         dynamic_cast<g2o::OptimizableGraph::Vertex *>(v_cam));
            e->setMeasurement(measurement);
            e->fx = graph.fx;
            e->fy = graph.fy;
            e->cx = graph.cx;
            e->cy = graph.cy;
            e->setInformation(Eigen::Matrix2d::Identity());
            auto *hb_kernel = new g2o::RobustKernelHuber();
            hb_kernel->setDelta(std::sqrt(5.99));// data from orbslam2
            e->setRobustKernel(hb_kernel);
            optimizer.addEdge(e);
        }
    }
    optimizer.initializeOptimization();
    optimizer.optimize(iter_cnt);

    graph.optim_cam_poses.clear();
    graph.optim_points.clear();
    for (int i = 0; i < graph.cam_sz; ++i) {
        auto *v_cam = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
        graph.optim_cam_poses.push_back(to_cvmat(v_cam->estimate()));
    }
    for (int i = 0; i < graph.pt_sz; ++i) {
        auto *v_pt = dynamic_cast<g2o::VertexSBAPointXYZ *>(
                optimizer.vertex(i + graph.cam_sz));
        graph.optim_points.push_back(to_cvmat(v_pt->estimate()));
    }
}
}// namespace vo_nono