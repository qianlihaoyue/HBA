
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <iostream>

#include "mypcl.hpp"
#include "tools.hpp"
#include <string>
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv) {
    std::string config_path = "../config/savepcd.yaml";
    if (argc == 2) config_path = argv[1];
    YAML::Node config = YAML::LoadFile(config_path);

    std::string data_path = config["data_path"].as<std::string>();
    std::string pose_opt = config["pose_opt"].as<std::string>();

    std::cout << "Data Path: " << data_path << std::endl;
    std::vector<mypcl::pose> pose_vec_opt = mypcl::read_pose(data_path + pose_opt);

    pcl::PointCloud<PointType>::Ptr cloud_opt(new pcl::PointCloud<PointType>());
    for (size_t i = 0; i < pose_vec_opt.size(); i++) {
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(data_path + "Scans/" + std::to_string(i) + ".pcd", *pc);

        pcl::PointCloud<PointType>::Ptr pc_tmp(new pcl::PointCloud<PointType>);
        mypcl::transform_pointcloud(*pc, *pc_tmp, pose_vec_opt[i].t, pose_vec_opt[i].q);
        *cloud_opt += *pc_tmp;

        if (i % 100 == 0) std::cout << "load " << i << std::endl;
    }

    std::cout << "downsample & save_pcd" << std::endl;
    // downsample_voxel(*cloud_opt, 0.05);
    // pcl::io::savePCDFileBinary(data_path + "scans-0.05.pcd", *cloud_opt);

    downsample_voxel(*cloud_opt, 0.125);
    pcl::io::savePCDFileBinary(data_path + "scans-0.125.pcd", *cloud_opt);

    downsample_voxel(*cloud_opt, 0.25);
    pcl::io::savePCDFileBinary(data_path + "scans-0.25.pcd", *cloud_opt);
}