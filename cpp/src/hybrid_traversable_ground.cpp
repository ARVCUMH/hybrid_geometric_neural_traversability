#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include "utils.hpp"
#include <vector>
#include <pcl/pcl_config.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/impl/file_io.hpp>
#include <pcl/common/angles.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>


bool myCmp(string s1, string s2)
{
    if (s1.size() == s2.size()) {
        return s1 < s2;
    }
    else {
        return s1.size() < s2.size();
    }
}
// create a function where we can initialise the params into variables
void read_Params(int resolution, Params &params_, vector<double> &min_ranges_, vector<double> &sector_sizes_, vector<double> &ring_sizes_, vector<int> &num_rings_, vector<int> &num_sectors_){
    double min_range_z2_ = 8.25;
    double min_range_z3_ = 18.25;
    double min_range_z4_ = 36;
    min_ranges_ = {params_.min_range, min_range_z2_, min_range_z3_, min_range_z4_};

    if(resolution == 0){
        ring_sizes_ = {(min_range_z2_ - params_.min_range) / params_.num_rings_each_zone.at(0),
                        (min_range_z3_ - min_range_z2_) / params_.num_rings_each_zone.at(1),
                        (min_range_z4_ - min_range_z3_) / params_.num_rings_each_zone.at(2),
                        (params_.max_range - min_range_z4_) / params_.num_rings_each_zone.at(3)};

        sector_sizes_ = {2 * M_PI / params_.num_sectors_each_zone.at(0),
                            2 * M_PI / params_.num_sectors_each_zone.at(1),
                            2 * M_PI / params_.num_sectors_each_zone.at(2),
                            2 * M_PI / params_.num_sectors_each_zone.at(3)};

        num_rings_ = {params_.num_rings_each_zone[0],
                    params_.num_rings_each_zone[1],
                    params_.num_rings_each_zone[2],
                    params_.num_rings_each_zone[3]};

        num_sectors_ = {params_.num_sectors_each_zone[0],
                        params_.num_sectors_each_zone[1],
                        params_.num_sectors_each_zone[2],
                        params_.num_sectors_each_zone[3]};
    }
    else if(resolution == 1){
        ring_sizes_ = {(min_range_z2_ - params_.min_range) / params_.num_rings_each_zone_lvl1.at(0),
                        (min_range_z3_ - min_range_z2_) / params_.num_rings_each_zone_lvl1.at(1),
                        (min_range_z4_ - min_range_z3_) / params_.num_rings_each_zone_lvl1.at(2),
                        (params_.max_range - min_range_z4_) / params_.num_rings_each_zone_lvl1.at(3)};

        sector_sizes_ = {2 * M_PI / params_.num_sectors_each_zone_lvl1.at(0),
                            2 * M_PI / params_.num_sectors_each_zone_lvl1.at(1),
                            2 * M_PI / params_.num_sectors_each_zone_lvl1.at(2),
                            2 * M_PI / params_.num_sectors_each_zone_lvl1.at(3)};

        num_rings_ = {params_.num_rings_each_zone_lvl1[0],
                    params_.num_rings_each_zone_lvl1[1],
                    params_.num_rings_each_zone_lvl1[2],
                    params_.num_rings_each_zone_lvl1[3]};

        num_sectors_ = {params_.num_sectors_each_zone_lvl1[0],
                        params_.num_sectors_each_zone_lvl1[1],
                        params_.num_sectors_each_zone_lvl1[2],
                        params_.num_sectors_each_zone_lvl1[3]};
    }
    else{
        ring_sizes_ = {(min_range_z2_ - params_.min_range) / params_.num_rings_each_zone_lvl2.at(0),
            (min_range_z3_ - min_range_z2_) / params_.num_rings_each_zone_lvl2.at(1),
            (min_range_z4_ - min_range_z3_) / params_.num_rings_each_zone_lvl2.at(2),
            (params_.max_range - min_range_z4_) / params_.num_rings_each_zone_lvl2.at(3)};

        sector_sizes_ = {2 * M_PI / params_.num_sectors_each_zone_lvl2.at(0),
                        2 * M_PI / params_.num_sectors_each_zone_lvl2.at(1),
                        2 * M_PI / params_.num_sectors_each_zone_lvl2.at(2),
                        2 * M_PI / params_.num_sectors_each_zone_lvl2.at(3)};

        num_rings_ = {params_.num_rings_each_zone_lvl2[0],
                params_.num_rings_each_zone_lvl2[1],
                params_.num_rings_each_zone_lvl2[2],
                params_.num_rings_each_zone_lvl2[3]};

        num_sectors_ = {params_.num_sectors_each_zone_lvl2[0],
                    params_.num_sectors_each_zone_lvl2[1],
                    params_.num_sectors_each_zone_lvl2[2],
                    params_.num_sectors_each_zone_lvl2[3]};
    }
}
pcl::PointCloud<pcl::PointXYZL>::Ptr multi_polar_resolution( vector<Zone> &zones, pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_after_stage_1,
    pcl::PointCloud<pcl::PointXYZL>::Ptr &evaluate_sector_multi,pcl::PointCloud<pcl::PointXYZL>::Ptr &init_seeds, 
    pcl::PointIndices::Ptr &idx_obstacles, pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_final)
{

    std::vector<std::pair<int, int>> adjacent_offsets = {{-1, 0}, {1, 0}, {0, 1}, {0, -1}};
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_after_stage_2(new pcl::PointCloud<pcl::PointXYZL>);
    for (int i = 0; i < zones.size(); i++) {
        for (int j = 0; j < zones[i].size(); j++) {
            for (int k = 0; k < zones[i][j].size(); k++) {
                if (zones[i][j][k].size() >= 4) {
                    // Collect points for the current sector
                    for (int l = 0; l < zones[i][j][k].size(); l++) {
                        evaluate_sector_multi->points.push_back(cloud_after_stage_1->points[zones[i][j][k][l].idx]);
                    }
                    for (const auto& offset : adjacent_offsets) {
                        int adj_j = j + offset.first;
                        int adj_k = k + offset.second;
                        if (adj_j >= 0 && adj_j < zones[i].size() && adj_k >= 0 && adj_k < zones[i][j].size()) {
                            if (zones[i][adj_j][adj_k].size() >= 4){
                                for (int l = 0; l < zones[i][adj_j][adj_k].size(); l++) {
                                    evaluate_sector_multi->points.push_back(zones[i][adj_j][adj_k][l].idx);
                                }
                            }
                        }
                    }
                    std::tuple<Eigen::Vector3f, double> plane_normal = utils::extract_initial_seeds_and_compute_plane(evaluate_sector_multi, init_seeds);
                    // std::cout<< "Seeds size: " << init_seeds->points.size() << std::endl;
                    Eigen::Vector3f normal = std::get<0>(plane_normal);
                    double D = std::get<1>(plane_normal);
                    evaluate_sector_multi->clear();
                    std::tuple<double, double> mean_var = utils::calculateMedianAndStdDev(init_seeds);
                    double mean = std::get<0>(mean_var);
                    double var = std::get<1>(mean_var);
                    // std::cout << "Dynamic threshold max: " << dynamic_threshold_max << std::endl;
                    // std::cout << "Dynamic threshold min: " << dynamic_threshold_min << std::endl;
                    for (int t = 0; t < zones[i][j][k].size(); t++) {
                        double distance = utils::computeDistanceToPlane(plane_normal, cloud_after_stage_1->points[zones[i][j][k][t].idx]);
                        double distance_normalized=distance/var;
                        if(std::abs(distance_normalized) < 2 && normal[2] > 0.995){
                            cloud_after_stage_1->points[zones[i][j][k][t].idx].label = 3;
                            cloud_final->points.push_back(cloud_after_stage_1->points[zones[i][j][k][t].idx]);
                            
                        }
                        else{
                            // cloud_after_stage_1->points[zones[i][j][k][t].idx].label = 5;
                            cloud_after_stage_2->points.push_back(cloud_after_stage_1->points[zones[i][j][k][t].idx]);
                            idx_obstacles->indices.push_back(zones[i][j][k][t].idx);
                        }
                    }
                }
            }
        }
    }
    return cloud_after_stage_2;
}
int main() {
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_one(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_infered(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_three(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_final(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr prelabeled_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Nube de Puntos XYZ"));
    viewer->setBackgroundColor(255, 255, 255);
    std::vector<int> nearest_poses;
    std::vector<std::tuple<int, int, int>> confirmed_ground_patches;
    std::vector<Eigen::Matrix4f> transformed_poses;
    pcl::PointCloud<pcl::PointXYZL>::Ptr evaluate_confirmed_sector(new pcl::PointCloud<pcl::PointXYZL>);
    std::vector <std::tuple<Eigen::Vector3f, double, double>> geo_properties;
    std::tuple<double, double> dynamic_height_var;
    std::tuple<double, double> dynamic_surface_var;
    std::tuple<Eigen::Vector3f, Eigen::Vector3f> dynamic_normal_vector;
    pcl::PointCloud<pcl::PointXYZL>::Ptr evaluate_sector (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr evaluate_sector_adj (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr sector_all (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr sector_all_multi (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointIndices::Ptr idx_sector(new pcl::PointIndices);
    pcl::PointIndices::Ptr idx_sector_adj(new pcl::PointIndices);
    pcl::PointIndices::Ptr idx_obstacles(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_gt(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr init_seeds(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr evaluate_sector_multi (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointIndices::Ptr idx_sector_multi(new pcl::PointIndices);
    pcl::PointCloud<pcl::PointXYZL>::Ptr ground_multi (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_four (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_after_stage_1 (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_after_stage_2(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_after_stage_3(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_after_stage_4(new pcl::PointCloud<pcl::PointXYZL>);
    Params params_;
    vector<double> min_ranges_;
    vector<double> sector_sizes_;
    vector<double> ring_sizes_;
    vector<int> num_rings_;
    vector<int> num_sectors_;

    vector<double> sector_sizes_lvl1;
    vector<double> ring_sizes_lvl1;
    vector<int> num_rings_lvl1;
    vector<int> num_sectors_lvl1;

    vector<double> sector_sizes_lvl2;
    vector<double> ring_sizes_lvl2;
    vector<int> num_rings_lvl2;
    vector<int> num_sectors_lvl2;

    std::vector<Zone> zones;
    std::vector<Zone> zones_lvl1; 
    std::vector<Zone> zones_lvl2;
    vector<double> precision;
    vector<double> recall;
    vector<double> f1;
    vector<double> accuracy;
    vector<double> precision_tenext;
    vector<double> recall_tenext;
    vector<double> f1_tenext;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////Module for reading the poses and calibration file to projection////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::string poses_files = "/media/antonio/Extreme SSD/semantic-kitti/mdpi/data_odometry_poses/04.txt";
    std::string calib_file = "/media/antonio/Extreme SSD/semantic-kitti/mdpi/data_odometry_calib/04/calib.txt";
    std::vector<Eigen::Matrix4f> poses = utils::read_poses(poses_files);
    Eigen::Matrix4f Tr = utils::read_calib_file(calib_file);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////Module for reading the pointclouds and create CZM//////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int current_cloud = 1;
    //////////// Read ground truth for evaluation ////////////////////////
    std::vector<std::string> archivos_gt;
    std::string path_cloud_in = "/media/antonio/Extreme SSD/semantic-kitti/mdpi/gt/04/";
    // std::string path_cloud_in = "/media/antonio/Extreme SSD/semantic-kitti/rellis_mdpi/train/00000/os1_cloud_node_color_ply/";
    utils::getAllPLYFilesInDirectory(path_cloud_in, archivos_gt);
    std::sort(archivos_gt.begin(), archivos_gt.end(), myCmp);
    ///////////////// Read inference point clouds////////////////////////////7
    std::vector<std::string> archivos_inf;
    std::string output_path = "/media/antonio/Extreme SSD/semantic-kitti/mdpi/inference-big-voxel/04/";
    // std::string output_path = "/media/antonio/Extreme SSD/semantic-kitti/rellis_mdpi/inferences-big-voxel/00/";
    utils::getAllPLYFilesInDirectory(output_path, archivos_inf);
    std::sort(archivos_inf.begin(), archivos_inf.end(), myCmp);

    read_Params(0,params_, min_ranges_, sector_sizes_, ring_sizes_, num_rings_, num_sectors_);
    utils::init_czm(zones, params_);
    std::cout << "Concetric Zone model Initializated" << std::endl;

    while (current_cloud < archivos_inf.size() )
    {
        if (current_cloud >= archivos_inf.size()) 
        {
            std::cout << "No more point clouds" << std::endl;
            break;
        }
        else 
        {
            utils::flush_patches(zones,params_);
            // flush_patches_multi(zones_multi,params_);
            nearest_poses.clear();
            confirmed_ground_patches.clear();
            transformed_poses.clear();
            viewer->removePointCloud("nube_gt");
            viewer->removePointCloud("nube_pos");
            viewer->removePointCloud("nube_inf");
            cloud_gt = utils::readCloudWithLabel(path_cloud_in + archivos_gt[current_cloud]);
            cloud_infered = utils::readCloudWithLabel(output_path + archivos_inf[current_cloud]);
            prelabeled_cloud = utils::readCloudWithLabel(output_path + archivos_inf[current_cloud]);


            /////////////////// MODULE FOR COMPUTING DYNAMIC VARIANCES //////////////////////////
            nearest_poses = utils::get_future_and_past_poses(current_cloud, poses);
            transformed_poses = utils::transformPoses(current_cloud, Tr, nearest_poses, poses);
            utils::CZM(zones, cloud_infered, min_ranges_, params_.max_range, sector_sizes_, ring_sizes_, num_rings_, num_sectors_);
            confirmed_ground_patches=utils::ground_patches(zones, transformed_poses, cloud_infered, min_ranges_, params_.max_range, sector_sizes_, ring_sizes_, num_rings_, num_sectors_);

            geo_properties=utils::evaluating_patches(confirmed_ground_patches, cloud_infered, zones, evaluate_confirmed_sector);
            // std::cout << "num Nube de puntos:"<< cloud_infered->points.size() << std::endl;

            dynamic_surface_var = utils::calculateMedianAndStdDev(geo_properties,1);
            dynamic_height_var = utils::calculateMedianAndStdDev(geo_properties,2);
            double def_height_threshold_min = std::get<0>(dynamic_height_var) - 2*std::get<1>(dynamic_height_var);
            double def_height_threshold_max = std::get<0>(dynamic_height_var) + 2*std::get<1>(dynamic_height_var);
            double def_surface_threshold_min = std::get<0>(dynamic_surface_var) - 2*std::get<1>(dynamic_surface_var);
            double def_surface_threshold_max = std::get<0>(dynamic_surface_var) + 2*std::get<1>(dynamic_surface_var);
            // std::cout << def_height_threshold_max
            //             << " " << def_height_threshold_min
            //             << " " << def_surface_threshold_max
            //             << " " << def_surface_threshold_min << std::endl;
            
            // //////////////////////////////////////////////////////////////////////////////////////////
            // /////////////// MODULE FOR LABELLING THE POINTS IN THE CLOUD //////////////////////////////
            // //////////////////////////////////////////////////////////////////////////////////////////
            
            std::tuple<Eigen::Vector3f, double, double> geo_properties_actual;
            ////////////////////// Similarity module //////////////////////////
            // #pragma omp parallel for
            for (int i = 0; i < zones.size(); i++) {
                for (int j = 0; j < zones[i].size(); j++) {
                    for (int k = 0; k < zones[i][j].size(); k++) {
                        if (zones[i][j][k].size() > 4) {
                            // Collect points for the current sector
                            for (int l = 0; l < zones[i][j][k].size(); l++) {
                                evaluate_sector->points.push_back(cloud_infered->points[zones[i][j][k][l].idx]);
                                idx_sector->indices.push_back(zones[i][j][k][l].idx);
                            }
                              
                            geo_properties_actual = utils::compute_propierties(evaluate_sector);
                            double height_var_actual = std::get<2>(geo_properties_actual);
                            double surface_var_actual = std::get<1>(geo_properties_actual);
                            if (height_var_actual < def_height_threshold_max && height_var_actual > def_height_threshold_min && surface_var_actual < def_surface_threshold_max && surface_var_actual > def_surface_threshold_min){
                                for (int l = 0; l < zones[i][j][k].size(); l++) {
                                    cloud_infered->points[zones[i][j][k][l].idx].label = 3;
                                    cloud_final->points.push_back(cloud_infered->points[zones[i][j][k][l].idx]);
                                }
                            }
                            else{
                                for (int l = 0; l < zones[i][j][k].size(); l++) {
                                    // cloud_infered->points[zones[i][j][k][l].idx].label = 2;
                                    cloud_after_stage_1->points.push_back(cloud_infered->points[zones[i][j][k][l].idx]);
                                }
                            }
                            evaluate_sector->clear();
                            idx_sector->indices.clear();
                        }
                    }
                }
            }
            ///////////////////////////////////////////////////////////////////////////////
            /////////////////////////MULTI-POLAR RESOLUTION WITH ADJACENTS SECTORS////////////////////////////////////
            utils::flush_patches(zones,params_);

            utils::CZM(zones, cloud_after_stage_1, min_ranges_, params_.max_range, sector_sizes_, ring_sizes_, num_rings_, num_sectors_);
            cloud_after_stage_2=multi_polar_resolution(zones, cloud_after_stage_1, evaluate_sector_multi, init_seeds, idx_obstacles, cloud_final);
            utils::flush_patches(zones,params_);

            utils::init_czm_lvl1(zones_lvl1, params_);
            read_Params(1,params_, min_ranges_, sector_sizes_lvl1, ring_sizes_lvl1, num_rings_lvl1, num_sectors_lvl1);
            utils::CZM(zones_lvl1, cloud_after_stage_2, min_ranges_, params_.max_range, sector_sizes_lvl1, ring_sizes_lvl1, num_rings_lvl1, num_sectors_lvl1);
            cloud_after_stage_3=multi_polar_resolution(zones_lvl1, cloud_after_stage_2, evaluate_sector_multi, init_seeds, idx_obstacles, cloud_final);
            utils::flush_patches_lvl1(zones_lvl1,params_);

            // std::cout << "num Nube de puntos:"<< cloud_after_stage_3->points.size() << std::endl;

            utils::init_czm_lvl2(zones_lvl2, params_);
            read_Params(2, params_, min_ranges_, sector_sizes_lvl2, ring_sizes_lvl2, num_rings_lvl2, num_sectors_lvl2);
            utils::CZM(zones_lvl2, cloud_after_stage_3, min_ranges_, params_.max_range, sector_sizes_lvl2, ring_sizes_lvl2, num_rings_lvl2, num_sectors_lvl2);
            cloud_after_stage_4=multi_polar_resolution(zones_lvl2, cloud_after_stage_3, evaluate_sector_multi, init_seeds, idx_obstacles, cloud_final);
            utils::flush_patches_lvl2(zones_lvl2,params_);

            std::cout << "No participo en "<< cloud_after_stage_4->points.size() << std::endl;

            *cloud_final += *cloud_after_stage_4;
            // utils::flush_patches_lvl2(zones,params_);
            // change label 0 and 1 to 2
            for (int i = 0; i < cloud_final->points.size(); i++) {
                if (cloud_final->points[i].label == 1) {
                    cloud_final->points[i].label = 3;
                }
                else if (cloud_final->points[i].label == 0) {
                    cloud_final->points[i].label = 2;
                }
            }

            // change label 2 for 0 and 3 for 1
            for (int i = 0; i < cloud_final->points.size(); i++) {
                if (cloud_final->points[i].label == 2) {
                    cloud_final->points[i].label = 0;
                }
                else if (cloud_final->points[i].label == 3) {
                    cloud_final->points[i].label = 1;
                }
            }
            
            current_cloud=current_cloud+1;
            // reorder cloud
            pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_def(new pcl::PointCloud<pcl::PointXYZL>);
            cloud_def=utils::reorderPointCloud(cloud_final, cloud_gt);
            // pcl::visualization::PointCloudColorHandlerLabelField<pcl::PointXYZL> color_handler5(cloud_def, "label");
            // viewer->addPointCloud(cloud_def, "nube_inf");
            // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "nube_inf");
            // viewer->spin();
            // viewer->close();
            GroundSegmentationMetrics metrics = utils::evaluateGroundSegmentation(cloud_gt,cloud_def);
            GroundSegmentationMetrics metrics2 = utils::evaluateGroundSegmentation(cloud_gt,prelabeled_cloud);
            std::cout << "Precision paper: " << metrics.precision << " Precision solo red: " << metrics2.precision << std::endl;
            std::cout << "Recall paper: " << metrics.recall << " Recall solo red: " << metrics2.recall << std::endl;
            std::cout << "F1 Score de nuestro metodo: " << metrics.f1_score << " F1 Score solo red " << metrics2.f1_score <<std::endl;
            std::cout << "------------------------------" << std::endl;
            accuracy.push_back(metrics.accuracy);
            precision.push_back(metrics.precision);
            recall.push_back(metrics.recall);
            f1.push_back(metrics.f1_score);
            precision_tenext.push_back(metrics2.precision);
            recall_tenext.push_back(metrics2.recall);
            f1_tenext.push_back(metrics2.f1_score);
            cloud_final->clear();
            cloud_after_stage_1->clear();
            cloud_after_stage_2->clear();
            cloud_after_stage_3->clear();
            cloud_after_stage_4->clear();
            cloud_infered->clear();
        }
    }
    std::cout << "Precision: " << std::accumulate(precision.begin(), precision.end(), 0.0) / precision.size() << std::endl;
    std::cout << "Recall: " << std::accumulate(recall.begin(), recall.end(), 0.0) / recall.size() << std::endl;
    std::cout << "F1 Score: " << std::accumulate(f1.begin(), f1.end(), 0.0) / f1.size() << std::endl;
    std::cout << "Precision tenext: " << std::accumulate(precision_tenext.begin(), precision_tenext.end(), 0.0) / precision_tenext.size() << std::endl;
    std::cout << "Recall tenext: " << std::accumulate(recall_tenext.begin(), recall_tenext.end(), 0.0) / recall_tenext.size() << std::endl;
    std::cout << "F1 Score tenext: " << std::accumulate(f1_tenext.begin(), f1_tenext.end(), 0.0) / f1_tenext.size() << std::endl;
}


                

// else{
//     for (int l = 0; l < zones[i][j][k].size(); l++) {
//         cloud_infered->points[zones[i][j][k][l].idx].label = 2;
//     }
// } 
// }
// evaluate_sector->clear();
// idx_sector->indices.clear();
// }
// }
// }
// visualize the result
// pcl::visualization::PointCloudColorHandlerLabelField<pcl::PointXYZL> color_handler5(cloud_infered, "label");
// viewer->addPointCloud(cloud_infered, "nube_inf");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "nube_inf");
// viewer->spin();
// viewer->close();
// pcl::visualization::PointCloudColorHandlerLabelField<pcl::PointXYZL> color_handler5(cloud_after_stage_1, "label");
// viewer->addPointCloud(cloud_after_stage_1, "nube_inf");
// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "nube_inf");
// viewer->spin();
// viewer->close();