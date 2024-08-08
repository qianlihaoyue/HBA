#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <iostream>

#include "mypcl.hpp"
#include "tools.hpp"
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv) {
    std::string config_path = "../config/visualizeSeq.yaml";
    if (argc == 2) config_path = argv[1];
    YAML::Node config = YAML::LoadFile(config_path);

    std::string data_path = config["data_path"].as<std::string>();
    std::string pose_opt = config["pose_opt"].as<std::string>();
    double downsample_size = config["downsample_size"].as<double>();
    bool color_inversion = config["color_inversion"].as<bool>();

    std::cout << "Data Path: " << data_path << std::endl;

    std::vector<mypcl::pose> pose_vec_opt = mypcl::read_pose(data_path + pose_opt);
    pcl::PointCloud<PointType>::Ptr cloud_opt(new pcl::PointCloud<PointType>());
    for (size_t i = 0; i < pose_vec_opt.size(); i++) {
        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
        pcl::io::loadPCDFile(data_path + "Scans/" + std::to_string(i) + ".pcd", *pc);
        
        // 颜色映射
        if (color_inversion)
            for (auto& pt : pc->points) pt.intensity = pose_vec_opt.size() - i;
        else
            for (auto& pt : pc->points) pt.intensity = i;

        mypcl::transform_pointcloud(*pc, *pc, pose_vec_opt[i].t, pose_vec_opt[i].q);
        *cloud_opt += *pc;

        if (i % 100 == 0) std::cout << "load " << i << std::endl;
    }

    std::cout << "downsample : " << downsample_size << std::endl;
    downsample_voxel(*cloud_opt, downsample_size);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // PointCloud
    pcl::visualization::PointCloudColorHandlerGenericField<PointType> intensity_opt(cloud_opt, "intensity");
    viewer->addPointCloud<PointType>(cloud_opt, intensity_opt, "cloud_opt");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_opt");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    viewer->resetCamera();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}