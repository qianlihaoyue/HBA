#ifndef MYPCL_HPP
#define MYPCL_HPP

#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <string>
#include <set>
#include "tools.hpp"

namespace mypcl {
struct pose {
    pose(Eigen::Quaterniond _q = Eigen::Quaterniond(1, 0, 0, 0), Eigen::Vector3d _t = Eigen::Vector3d(0, 0, 0)) : q(_q), t(_t) {}
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
};

std::vector<double> time_vec;

std::vector<pose> read_pose(std::string filename) {
    std::vector<pose> pose_vec;
    std::fstream file;
    file.open(filename);
    double time;
    double tx, ty, tz, w, x, y, z;
    while (file >> time >> tx >> ty >> tz >> x >> y >> z >> w) {
        Eigen::Quaterniond q(w, x, y, z);
        Eigen::Vector3d t(tx, ty, tz);
        pose_vec.push_back(pose(q, t));
        time_vec.push_back(time);
    }
    file.close();
    return pose_vec;
}

void transform_pointcloud(pcl::PointCloud<PointType>& pc_in, pcl::PointCloud<PointType>& pt_out, Eigen::Vector3d t, Eigen::Quaterniond q) {
    size_t size = pc_in.points.size();
    pt_out.points.resize(size);
    for (size_t i = 0; i < size; i++) {
        Eigen::Vector3d pt_cur(pc_in.points[i].x, pc_in.points[i].y, pc_in.points[i].z);
        Eigen::Vector3d pt_to;

        pt_to = q * pt_cur + t;
        pt_out.points[i].x = pt_to.x();
        pt_out.points[i].y = pt_to.y();
        pt_out.points[i].z = pt_to.z();
        pt_out.points[i].intensity = pc_in.points[i].intensity;
    }
}

pcl::PointCloud<PointType>::Ptr append_cloud(pcl::PointCloud<PointType>::Ptr pc1, pcl::PointCloud<PointType> pc2) {
    size_t size1 = pc1->points.size();
    size_t size2 = pc2.points.size();
    pc1->points.resize(size1 + size2);
    for (size_t i = size1; i < size1 + size2; i++) {
        pc1->points[i].x = pc2.points[i - size1].x;
        pc1->points[i].y = pc2.points[i - size1].y;
        pc1->points[i].z = pc2.points[i - size1].z;
    }
    return pc1;
}

// double compute_inlier_ratio(std::vector<double> residuals, double ratio) {
//     std::set<double> dis_vec;
//     for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
//         dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));
//     return *(std::next(dis_vec.begin(), (int)((ratio)*dis_vec.size())));
// }

void write_pose(std::vector<pose>& pose_vec, std::string path) {
    std::ofstream file;
    file.open(path, std::ofstream::trunc);
    file.close();
    Eigen::Quaterniond q0(pose_vec[0].q.w(), pose_vec[0].q.x(), pose_vec[0].q.y(), pose_vec[0].q.z());
    Eigen::Vector3d t0(pose_vec[0].t(0), pose_vec[0].t(1), pose_vec[0].t(2));
    file.open(path, std::ofstream::app);
    std::cout << "save pose to: " << path << std::endl;

    for (size_t i = 0; i < pose_vec.size(); i++) {
        pose_vec[i].t << q0.inverse() * (pose_vec[i].t - t0);
        pose_vec[i].q.w() = (q0.inverse() * pose_vec[i].q).w();
        pose_vec[i].q.x() = (q0.inverse() * pose_vec[i].q).x();
        pose_vec[i].q.y() = (q0.inverse() * pose_vec[i].q).y();
        pose_vec[i].q.z() = (q0.inverse() * pose_vec[i].q).z();
        file << std::fixed << std::setprecision(9) << time_vec[i] << std::setprecision(5) << " " << pose_vec[i].t(0) << " " << pose_vec[i].t(1) << " "
             << pose_vec[i].t(2) << " " << pose_vec[i].q.x() << " " << pose_vec[i].q.y() << " " << pose_vec[i].q.z() << " " << pose_vec[i].q.w();
        if (i < pose_vec.size() - 1) file << "\n";
    }
    file.close();
}

}  // namespace mypcl

#endif