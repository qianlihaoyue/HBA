#include <pcl/visualization/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <iostream>

#include "mypcl.hpp"
#include "tools.hpp"
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv) {
    std::string config_path = "../config/visualize.yaml";
    if (argc == 2) config_path = argv[1];
    YAML::Node config = YAML::LoadFile(config_path);

    std::string data_path = config["data_path"].as<std::string>();
    std::string pose_bfr = config["pose_bfr"].as<std::string>();
    std::string pose_opt = config["pose_opt"].as<std::string>();
    double downsample_size = config["downsample_size"].as<double>();

    std::cout << "Data Path: " << data_path << std::endl;

    std::vector<mypcl::pose> pose_vec_bfr = mypcl::read_pose(data_path + pose_bfr);
    std::vector<mypcl::pose> pose_vec_opt = mypcl::read_pose(data_path + pose_opt);

    // Create point clouds for poses
    pcl::PointCloud<PointType>::Ptr pose_cloud_bfr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr pose_cloud_opt(new pcl::PointCloud<PointType>());
    for (const auto& p : pose_vec_bfr) {
        PointType pt;
        pt.x = p.t.x(), pt.y = p.t.y(), pt.z = p.t.z();
        pose_cloud_bfr->points.emplace_back(pt);
    }
    for (const auto& p : pose_vec_opt) {
        PointType pt;
        pt.x = p.t.x(), pt.y = p.t.y(), pt.z = p.t.z();
        pose_cloud_opt->points.emplace_back(pt);
    }

    pcl::PointCloud<PointType>::Ptr cloud_bfr(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud_opt(new pcl::PointCloud<PointType>());
    for (size_t i = 0; i < pose_vec_bfr.size(); i++) {
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(data_path + "Scans/" + std::to_string(i) + ".pcd", *pc);

        pcl::PointCloud<PointType>::Ptr pc_tmp(new pcl::PointCloud<PointType>);
        mypcl::transform_pointcloud(*pc, *pc_tmp, pose_vec_bfr[i].t, pose_vec_bfr[i].q);
        *cloud_bfr += *pc_tmp;

        mypcl::transform_pointcloud(*pc, *pc_tmp, pose_vec_opt[i].t, pose_vec_opt[i].q);
        *cloud_opt += *pc_tmp;

        if (i % 100 == 0) std::cout << "load " << i << std::endl;
    }

    std::cout << "downsample : " << downsample_size << std::endl;
    downsample_voxel(*cloud_bfr, downsample_size);
    downsample_voxel(*cloud_opt, downsample_size);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    int viewport1(0), viewport2(1);
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, viewport1);
    viewer->createViewPort(0.5, 0.0, 1.0, 1.0, viewport2);

    // 设置背景颜色为白色
    viewer->setBackgroundColor(1.0, 1.0, 1.0, viewport1);
    viewer->setBackgroundColor(1.0, 1.0, 1.0, viewport2); 

    // PointCloud
    pcl::visualization::PointCloudColorHandlerGenericField<PointType> intensity_bfr(cloud_bfr, "intensity");
    pcl::visualization::PointCloudColorHandlerGenericField<PointType> intensity_opt(cloud_opt, "intensity");

    viewer->addPointCloud<PointType>(cloud_bfr, intensity_bfr, "cloud_bfr", viewport1);
    viewer->addPointCloud<PointType>(cloud_opt, intensity_opt, "cloud_opt", viewport2);

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_bfr");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_opt");

    // Poses
    viewer->addPointCloud<PointType>(pose_cloud_bfr, "pose_cloud_bfr", viewport1);
    viewer->addPointCloud<PointType>(pose_cloud_opt, "pose_cloud_opt", viewport2);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pose_cloud_bfr", viewport1);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "pose_cloud_opt", viewport2);

    // Text
    viewer->addText("cloud_bfr", 10, 10, 30, 1.0, 1.0, 1.0, "cloud_bfr", viewport1);
    viewer->addText("cloud_opt", 10, 10, 30, 1.0, 1.0, 1.0, "cloud_opt", viewport2);

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->resetCamera();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}