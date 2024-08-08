
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

// 将空间分割成体素
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*>& feat_map, pcl::PointCloud<PointType>& feat_pt, Eigen::Quaterniond q, Eigen::Vector3d t, int fnum,
               double voxel_size, int window_size, float eigen_ratio) {
    float loc_xyz[3];
    for (PointType& p_c : feat_pt.points) {
        Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
        Eigen::Vector3d pvec_tran = q * pvec_orig + t;

        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = pvec_tran[j] / voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }

        // 查找feat_map中是否已有对应的体素
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = feat_map.find(position);
        // 如果已有，则将原始和变换后的点向量分别添加到对应的向量队列中，并将它们压入相应的信号队列
        if (iter != feat_map.end()) {
            iter->second->vec_orig[fnum].push_back(pvec_orig);
            iter->second->vec_tran[fnum].push_back(pvec_tran);

            iter->second->sig_orig[fnum].push(pvec_orig);
            iter->second->sig_tran[fnum].push(pvec_tran);
        }
        // 如果没有，则在feat_map中添加一个新体素，同样添加点向量和压入信号队列，并且计算出体素中心坐标和四分之一的体素大小，以及设置体素的层级为0。
        else {
            OCTO_TREE_ROOT* ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
            ot->vec_orig[fnum].push_back(pvec_orig);
            ot->vec_tran[fnum].push_back(pvec_tran);
            ot->sig_orig[fnum].push(pvec_orig);
            ot->sig_tran[fnum].push(pvec_tran);

            ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
            ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
            ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
            ot->quater_length = voxel_size / 4.0;
            ot->layer = 0;
            feat_map[position] = ot;
        }
    }
}

// 多线程调用函数，对LAYER实例计算
void parallel_comp(LAYER& layer, int thread_id, LAYER& next_layer) {
    int& part_length = layer.part_length;
    int& layer_num = layer.layer_num;
    // 按照线程id划分处理区域
    for (int i = thread_id * part_length; i < (thread_id + 1) * part_length; i++) {
        vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
        src_pc.resize(WIN_SIZE);
        raw_pc.resize(WIN_SIZE);

        double residual_cur = 0, residual_pre = 0;
        vector<IMUST> x_buf(WIN_SIZE);
        for (int j = 0; j < WIN_SIZE; j++) {
            x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
            x_buf[j].p = layer.pose_vec[i * GAP + j].t;
        }

        if (layer_num != 1)
            for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();

        size_t mem_cost = 0;
        for (int loop = 0; loop < layer.max_iter; loop++) {
            // 第一次，加载点云
            if (layer_num == 1)
                for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) {
                    if (loop == 0) {
                        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
                        pcl::io::loadPCDFile(layer.data_path + "Scans/" + std::to_string(j) + ".pcd", *pc);
                        raw_pc[j - i * GAP] = pc;
                    }
                    src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
                }
            // 降采样
            unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

            for (size_t j = 0; j < WIN_SIZE; j++) {
                if (layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
                cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
            }
            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->recut();

            // 结果储存在 Vox_HESS
            VOX_HESS voxhess(WIN_SIZE);
            for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) iter->second->tras_opt(voxhess);
            // 异常值移除
            VOX_OPTIMIZER opt_lsv(WIN_SIZE);
            opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
            // 迭代优化
            PLV(6) hess_vec;
            opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

            if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
                if (layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

                for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++) layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

                break;
            }
            residual_pre = residual_cur;
        }

        pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
        for (size_t j = 0; j < WIN_SIZE; j++) {
            Eigen::Quaterniond q_tmp;
            Eigen::Vector3d t_tmp;
            assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R), x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

            pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
            mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
            pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
        }
        downsample_voxel(*pc_keyframe, 0.05);
        next_layer.pcds[i] = pc_keyframe;
    }
}

// 多线程调用，用于最后一个线程
void parallel_tail(LAYER& layer, int thread_id, LAYER& next_layer) {
    int& part_length = layer.part_length;
    int& layer_num = layer.layer_num;
    int& left_gap_num = layer.left_gap_num;

    double load_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0, sol_t = 00;

    if (layer.gap_num - (layer.thread_num - 1) * part_length + 1 != left_gap_num) printf("THIS IS WRONG!\n");

    for (uint i = thread_id * part_length; i < thread_id * part_length + left_gap_num; i++) {
        printf("parallel computing %d\n", i);
        // double t0, t1;
        auto t_begin = GET_TIME();

        vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
        src_pc.resize(WIN_SIZE);
        raw_pc.resize(WIN_SIZE);

        double residual_cur = 0, residual_pre = 0;
        vector<IMUST> x_buf(WIN_SIZE);
        for (int j = 0; j < WIN_SIZE; j++) {
            x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
            x_buf[j].p = layer.pose_vec[i * GAP + j].t;
        }

        if (layer_num != 1) {
            for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
        }

        size_t mem_cost = 0;
        for (int loop = 0; loop < layer.max_iter; loop++) {
            if (layer_num == 1) {
                auto t0 = GET_TIME();
                for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++) {
                    if (loop == 0) {
                        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
                        pcl::io::loadPCDFile(layer.data_path + "Scans/" + std::to_string(j) + ".pcd", *pc);
                        raw_pc[j - i * GAP] = pc;
                    }
                    src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
                }
                load_t += GET_USED(GET_TIME(), t0);
            }

            unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

            for (size_t j = 0; j < WIN_SIZE; j++) {
                auto t0 = GET_TIME();
                if (layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
                dsp_t += GET_USED(GET_TIME(), t0);

                t0 = GET_TIME();
                cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
                cut_t += GET_USED(GET_TIME(), t0);
            }

            auto t0 = GET_TIME();
            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->recut();
            recut_t += GET_USED(GET_TIME(), t0);

            VOX_HESS voxhess(WIN_SIZE);
            for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) iter->second->tras_opt(voxhess);

            VOX_OPTIMIZER opt_lsv(WIN_SIZE);
            t0 = GET_TIME();
            opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
            PLV(6) hess_vec;
            opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
            sol_t += GET_USED(GET_TIME(), t0);

            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

            if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
                if (layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

                if (i < thread_id * part_length + left_gap_num)
                    for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++) layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

                break;
            }
            residual_pre = residual_cur;
        }

        pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
        for (size_t j = 0; j < WIN_SIZE; j++) {
            Eigen::Quaterniond q_tmp;
            Eigen::Vector3d t_tmp;
            assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R), x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

            pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
            mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
            pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
        }
        auto t0 = GET_TIME();
        downsample_voxel(*pc_keyframe, 0.05);
        dsp_t += GET_USED(GET_TIME(), t0);

        next_layer.pcds[i] = pc_keyframe;

        total_t += GET_USED(GET_TIME(), t_begin);
    }
    if (layer.tail > 0) {
        int i = thread_id * part_length + left_gap_num;

        vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
        src_pc.resize(layer.last_win_size);
        raw_pc.resize(layer.last_win_size);

        double residual_cur = 0, residual_pre = 0;
        vector<IMUST> x_buf(layer.last_win_size);
        for (int j = 0; j < layer.last_win_size; j++) {
            x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
            x_buf[j].p = layer.pose_vec[i * GAP + j].t;
        }

        if (layer_num != 1) {
            for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++) src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
        }

        size_t mem_cost = 0;
        for (int loop = 0; loop < layer.max_iter; loop++) {
            if (layer_num == 1)
                for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++) {
                    if (loop == 0) {
                        pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
                        pcl::io::loadPCDFile(layer.data_path + "Scans/" + std::to_string(j) + ".pcd", *pc);
                        raw_pc[j - i * GAP] = pc;
                    }
                    src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
                }

            unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

            for (size_t j = 0; j < layer.last_win_size; j++) {
                if (layer.downsample_size > 0) downsample_voxel(*src_pc[j], layer.downsample_size);
                cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p, j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
            }
            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->recut();

            VOX_HESS voxhess(layer.last_win_size);
            for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++) iter->second->tras_opt(voxhess);

            VOX_OPTIMIZER opt_lsv(layer.last_win_size);
            opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
            PLV(6) hess_vec;
            opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

            for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

            if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
                if (layer.mem_costs[thread_id] < mem_cost) layer.mem_costs[thread_id] = mem_cost;

                for (int j = 0; j < layer.last_win_size * (layer.last_win_size - 1) / 2; j++)
                    layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

                break;
            }
            residual_pre = residual_cur;
        }

        pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
        for (size_t j = 0; j < layer.last_win_size; j++) {
            Eigen::Quaterniond q_tmp;
            Eigen::Vector3d t_tmp;
            assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R), x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

            pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
            mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
            pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
        }
        downsample_voxel(*pc_keyframe, 0.05);
        next_layer.pcds[i] = pc_keyframe;
    }
    printf("total time: %.2fs\n", total_t);
    printf(
        "load pcd %.2fs %.2f%% | downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | solve %.2fs %.2f%%\n",
        load_t, load_t / total_t * 100, dsp_t, dsp_t / total_t * 100, cut_t, cut_t / total_t * 100, recut_t, recut_t / total_t * 100, sol_t, sol_t / total_t * 100);
}

void global_ba(LAYER& layer) {
    int window_size = layer.pose_vec.size();
    vector<IMUST> x_buf(window_size);
    for (int i = 0; i < window_size; i++) {
        x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
        x_buf[i].p = layer.pose_vec[i].t;
    }

    vector<pcl::PointCloud<PointType>::Ptr> src_pc;
    src_pc.resize(window_size);
    for (int i = 0; i < window_size; i++) src_pc[i] = (*layer.pcds[i]).makeShared();

    double residual_cur = 0, residual_pre = 0;
    size_t mem_cost = 0, max_mem = 0;
    double dsp_t = 0, cut_t = 0, recut_t = 0, sol_t = 0, t0;
    for (int loop = 0; loop < layer.max_iter; loop++) {
        std::cout << "---------------------" << std::endl;
        std::cout << "Iteration " << loop << std::endl;

        unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;

        for (int i = 0; i < window_size; i++) {
            auto t0 = GET_TIME();
            if (layer.downsample_size > 0) downsample_voxel(*src_pc[i], layer.downsample_size);
            dsp_t += GET_USED(GET_TIME(), t0);
            t0 = GET_TIME();
            cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i, layer.voxel_size, window_size, layer.eigen_ratio * 2);
            cut_t += GET_USED(GET_TIME(), t0);
        }
        auto t0 = GET_TIME();
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->recut();
        recut_t += GET_USED(GET_TIME(), t0);

        VOX_HESS voxhess(window_size);
        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) iter->second->tras_opt(voxhess);

        t0 = GET_TIME();
        VOX_OPTIMIZER opt_lsv(window_size);
        opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
        PLV(6) hess_vec;
        opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
        sol_t += GET_USED(GET_TIME(), t0);

        for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter) delete iter->second;

        cout << "Residual absolute: " << abs(residual_pre - residual_cur) << " | " << "percentage: " << abs(residual_pre - residual_cur) / abs(residual_cur)
             << endl;

        if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1) {
            if (max_mem < mem_cost) max_mem = mem_cost;
#ifdef FULL_HESS
            for (int i = 0; i < window_size * (window_size - 1) / 2; i++) layer.hessians[i] = hess_vec[i];
#else
            for (int i = 0; i < window_size - 1; i++) {
                Eigen::Matrix<double, 6, 6> hess = Hess_cur.block(6 * i, 6 * i + 6, 6, 6);
                for (int row = 0; row < 6; row++)
                    for (int col = 0; col < 6; col++) hessFile << hess(row, col) << ((row * col == 25) ? "" : " ");
                if (i < window_size - 2) hessFile << "\n";
            }
#endif
            break;
        }
        residual_pre = residual_cur;
    }
    for (int i = 0; i < window_size; i++) {
        layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
        layer.pose_vec[i].t = x_buf[i].p;
    }
    printf("Downsample: %f, Cut: %f, Recut: %f, Sol: %f\n", dsp_t, cut_t, recut_t, sol_t);
}

// 创建并分配线程任务
void distribute_thread(LAYER& layer, LAYER& next_layer) {
    int& thread_num = layer.thread_num;
    
    for (int i = 0; i < thread_num; i++)
        if (i < thread_num - 1)
            layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer));
        else
            layer.mthreads[i] = new thread(parallel_tail, ref(layer), i, ref(next_layer));

    for (int i = 0; i < thread_num; i++) {
        layer.mthreads[i]->join();
        delete layer.mthreads[i];
    }
}

#include <yaml-cpp/yaml.h>
int main(int argc, char** argv) {
    std::string config_path = "../config/hba.yaml";
    if (argc == 2) config_path = argv[1];  // 使用命令行参数提供的路径
    YAML::Node config = YAML::LoadFile(config_path);

    // Access parameters
    HBAConfig cfg;
    cfg.data_path = config["data_path"].as<std::string>();
    cfg.load_pose_name = config["load_pose_name"].as<std::string>();
    cfg.save_pose_name = config["save_pose_name"].as<std::string>();

    cfg.total_layer_num = config["total_layer_num"].as<int>();
    cfg.thread_num = config["thread_num"].as<int>();
    cfg.voxel_size = config["voxel_size"].as<double>();
    cfg.eigen_ratio = config["eigen_ratio"].as<double>();

    HBA hba(cfg);

    printf("initial complete\n");

    for (int i = 0; i < cfg.total_layer_num - 1; i++) {
        std::cout << "---------------------" << std::endl;
        // 在 i 层和 i+1 层之间分配线程进行并行计算
        distribute_thread(hba.layers[i], hba.layers[i + 1]);
        // 更新 i+1 层状态
        hba.update_next_layer_state(i);
    }
    printf("distribute thread complete\n");
    // 全局优化
    global_ba(hba.layers[cfg.total_layer_num - 1]);
    hba.pose_graph_optimization();
    printf("iteration complete\n");
}