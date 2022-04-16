#include "vo_nono/optimize_graph.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "vo_nono/util/geometry.h"

namespace vo_nono {
namespace {
inline Eigen::Matrix3d angle_axis_to_rotation_mat_eigen(
        const double angle_axis[3]) {
    double res[3][3];
    Geometry::angle_axis_to_rotation_mat(angle_axis, res);
    Eigen::Matrix3d m;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { m(i, j) = res[i][j]; }
    }
    return m;
}

// reference: ceres' bundle adjustment example
class ProjectionError {
public:
    explicit ProjectionError(const Camera& camera, double img_x, double img_y)
        : fx(camera.fx()),
          fy(camera.fy()),
          cx(camera.cx()),
          cy(camera.cy()),
          img_x(img_x),
          img_y(img_y) {}

    template<typename T>
    bool operator()(const T* const camera, const T* const point,
                    T* residual) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T x = p[0] / p[2], y = p[1] / p[2];
        T predict_x = fx * x + cx;
        T predict_y = fy * y + cy;
        residual[0] = predict_x - img_x;
        residual[1] = predict_y - img_y;

        return true;
    }

    static ceres::CostFunction* create(const Camera& camera, double img_x,
                                       double img_y) {
        return (new ceres::AutoDiffCostFunction<ProjectionError, 2, 6, 3>(
                new ProjectionError(camera, img_x, img_y)));
    }

private:
    const double fx, fy, cx, cy;
    const double img_x, img_y;
};

// camera is marginalized
class ProjectionErrorCamMargin {
public:
    explicit ProjectionErrorCamMargin(const Camera& camera,
                                      const double cam_pose[6], double img_x,
                                      double img_y)
        : fx(camera.fx()),
          fy(camera.fy()),
          cx(camera.cx()),
          cy(camera.cy()),
          img_x(img_x),
          img_y(img_y),
          Rcw(angle_axis_to_rotation_mat_eigen(cam_pose)),
          Tcw(cam_pose[3], cam_pose[4], cam_pose[5]) {}

    template<typename T>
    bool operator()(const T* const point, T* residual) const {
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> pt(point);
        Eigen::Matrix<T, 3, 1> proj = Rcw * pt + Tcw;

        T predict_x = (proj(0, 0) / proj(2, 0)) * fx + cx;
        T predict_y = (proj(1, 0) / proj(2, 0)) * fy + cy;
        residual[0] = predict_x - img_x;
        residual[1] = predict_y - img_y;
        return true;
    }

    static ceres::CostFunction* create(const Camera& camera,
                                       const double cam_pose[6], double img_x,
                                       double img_y) {
        return (new ceres::AutoDiffCostFunction<ProjectionErrorCamMargin, 2, 3>(
                new ProjectionErrorCamMargin(camera, cam_pose, img_x, img_y)));
    }

private:
    const double fx, fy, cx, cy;
    const double img_x, img_y;
    const Eigen::Matrix3d Rcw;
    const Eigen::Vector3d Tcw;
};

// point is marginalized
class ProjectionErrorPointMargin {
public:
    explicit ProjectionErrorPointMargin(const Camera& camera,
                                        const double coord[3], double img_x,
                                        double img_y)
        : fx(camera.fx()),
          fy(camera.fy()),
          cx(camera.cx()),
          cy(camera.cy()),
          img_x(img_x),
          img_y(img_y),
          coord(coord[0], coord[1], coord[2]) {}

    template<typename T>
    bool operator()(const T* const camera, T* residual) const {
        T p[3];
        Eigen::Matrix<T, 3, 1> coord_t = coord.cast<T>();
        ceres::AngleAxisRotatePoint(camera, coord_t.data(), p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        T predict_x = (p[0] / p[2]) * fx + cx;
        T predict_y = (p[1] / p[2]) * fy + cy;
        residual[0] = predict_x - img_x;
        residual[1] = predict_y - img_y;
        return true;
    }

    static ceres::CostFunction* create(const Camera& camera,
                                       const double coord[3], double img_x,
                                       double img_y) {
        return (new ceres::AutoDiffCostFunction<ProjectionErrorPointMargin, 2,
                                                6>(
                new ProjectionErrorPointMargin(camera, coord, img_x, img_y)));
    }

private:
    const double fx, fy, cx, cy;
    const double img_x, img_y;
    Eigen::Vector3d coord;
};
}// namespace

void OptimizeGraph::to_problem() {
    problem = ceres::Problem();
    for (int i = 0; i < int(cam_poses.size()); ++i) {
        for (int j = 0; j < int(edges[i].size()); ++j) {
            int point_id = edges[i][j].point_id;
            int cur_residual_id = -1;
            ceres::ResidualBlockId blk_id;

            // if camera and point are both marginalized, omit it
            if (b_marginalized_cam[i]) {
                // marginalized
                if (!b_marginalized_points[point_id]) {
                    blk_id = problem.AddResidualBlock(
                            ProjectionErrorCamMargin::create(
                                    camera, cam_poses[i].data(),
                                    edges[i][j].img_x, edges[i][j].img_y),
                            loss, points[point_id].data());
                    cur_residual_id = int(residual_ids.size());
                    residual_ids.push_back(blk_id);
                }
            } else {
                // not marginalized
                if (b_marginalized_points[point_id]) {
                    blk_id = problem.AddResidualBlock(
                            ProjectionErrorPointMargin::create(
                                    camera, points[point_id].data(),
                                    edges[i][j].img_x, edges[i][j].img_y),
                            loss, cam_poses[i].data());
                } else {
                    blk_id = problem.AddResidualBlock(
                            ProjectionError::create(camera, edges[i][j].img_x,
                                                    edges[i][j].img_y),
                            loss, cam_poses[i].data(), points[point_id].data());
                }
                cur_residual_id = int(residual_ids.size());
                residual_ids.push_back(blk_id);
            }
            edges[i][j].residual_id = cur_residual_id;
        }
    }
}

void OptimizeGraph::get_cam_pose(int id, cv::Mat& Rcw, cv::Mat& Tcw) const {
    double r[9];
    ceres::AngleAxisToRotationMatrix(cam_poses[id].data(), r);
    Rcw = cv::Mat::zeros(3, 3, CV_32F);
    // note: r is column major
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rcw.at<float>(j, i) = float(r[i * 3 + j]);
        }
    }
    Tcw = cv::Mat::zeros(3, 1, CV_32F);
    for (int i = 0; i < 3; ++i) {
        Tcw.at<float>(i) = float(cam_poses[id][i + 3]);
    }
}

double OptimizeGraph::get_loss(int cam_id, int point_id) {
    if (edges[cam_id][point_id].residual_id < 0) {
        double* cam = cam_poses[cam_id].data();
        double* pt = points[point_id].data();
        double img_x = edges[cam_id][point_id].img_x;
        double img_y = edges[cam_id][point_id].img_y;
        auto proj_err = ProjectionError(camera, img_x, img_y);
        double diff[2];
        proj_err(cam, pt, diff);
        return 0.5 * (diff[0] * diff[0] + diff[1] * diff[1]);
    } else {
        return residual_vals[edges[cam_id][point_id].residual_id];
    }
}
}// namespace vo_nono