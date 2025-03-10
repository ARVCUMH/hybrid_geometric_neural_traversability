#pragma once

// C++
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <Eigen/Dense>
#include <pcl/io/obj_io.h>
#include <pcl/common/angles.h>
#include <map>
#include <pcl/common/pca.h>

// Visualization
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"  
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

//****************************************************************************//
// TYPE DEFINITIONS ////////////////////////////////////////////////////////////

namespace fs = std::filesystem;
using namespace std;
// typedef pcl::PointXYZ PointT;
// typedef pcl::PointCloud<PointT> PointCloud;
// typedef pcl::PointXYZI PointI;
// typedef pcl::PointCloud<PointI> PointCloudI;
typedef pcl::PointCloud<pcl::PointXYZL> PointCloudL;

struct PointXYZ {
    float x;
    float y;
    float z;
    int idx;

    PointXYZ(float _x, float _y, float _z, int _idx=-1) : x(_x), y(_y), z(_z), idx(_idx) {}
};


typedef std::vector<vector<PointXYZ>> Ring;
typedef std::vector<Ring> Zone;



struct Params 
{
    int num_min_pts;
    int num_zones;
    int num_rings_of_interest;
    double max_range;
    double min_range;

    vector<int> num_sectors_each_zone;
    vector<int> num_rings_each_zone; 
    vector<int> num_sectors_each_zone_lvl1;
    vector<int> num_rings_each_zone_lvl1;
    vector<int> num_sectors_each_zone_lvl2;
    vector<int> num_rings_each_zone_lvl2;

    
Params() {   
        num_min_pts = 10;           // Minimum number of points to be estimated as ground plane in each patch.
        num_zones = 4;              // Setting of Concentric Zone Model(CZM)
        // num_sectors_each_zone = {32, 64, 64, 16};   // Setting of Concentric Zone Model(CZM)
        // num_rings_each_zone = {8, 12, 16, 5};       // Setting of Concentric Zone Model(CZM)

        // num_sectors_each_zone = {64, 96, 96, 32};   // paper
        // num_rings_each_zone = {4, 6, 8, 4}; // paper
        // num_sectors_each_zone_multi = {64, 96, 96, 64};  // paper
        // num_rings_each_zone_multi = {16, 20, 36, 18}; // paper
        // Level 0 GRID
        num_sectors_each_zone = {16, 32, 64, 16};   // paper
        num_rings_each_zone = {4, 6, 8, 4}; // paper
        // Level 1 GRID
        num_sectors_each_zone_lvl1 = {16, 64, 64, 16};   // paper
        num_rings_each_zone_lvl1 = {8, 12, 10, 8}; // paper
        // Level 2 GRID
        num_sectors_each_zone_lvl2 = {64, 96, 96, 32};   // paper
        num_rings_each_zone_lvl2 = {16, 16, 20, 18}; // paper
        // num_sectors_each_zone_multi = {64, 96, 96, 64};  // paper
        // num_rings_each_zone_multi = {16, 20, 36, 18}; // paper

        max_range = 45.0;           // max_range of ground estimation area
        min_range = 2.5;            // min_range of ground estimation area

    }
};

struct ground_idx
{
   pcl::PointIndices::Ptr inliers;
   pcl::PointIndices::Ptr outliers;
};

struct segment_cloud
{
pcl::PointCloud<pcl::PointXYZ>::Ptr ground;
pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground;
pcl::PointIndices::Ptr inliers;
pcl::PointIndices::Ptr outliers;
};

struct GroundSegmentationMetrics {
    int true_positives;   // Correctly identified ground points
    int false_positives;  // Non-ground points incorrectly labeled as ground
    int true_negatives;   // Correctly identified non-ground points
    int false_negatives;  // Ground points incorrectly labeled as non-ground
    float precision;
    float recall;
    float f1_score;
    float accuracy;
};


namespace utils
{

  void 
  getAllPLYFilesInDirectory(const std::string& directory, std::vector<std::string>& file_names)
 {
   boost::filesystem::path p(directory);
   if(boost::filesystem::is_directory(p))
   {
     for(const auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {}))
     {
       if (boost::filesystem::is_regular_file(entry))
       {
         if (entry.path().extension() == ".ply")
           file_names.emplace_back(entry.path().filename().string());
       }
     }
   }
   else
   {
     std::cerr << "Given path is not a directory\n";
     return;
   }
   std::sort(file_names.begin(), file_names.end());
  }

 void 
  getAllPCDFilesInDirectory(const std::string& directory, std::vector<std::string>& file_names)
 {
   boost::filesystem::path p(directory);
   if(boost::filesystem::is_directory(p))
   {
     for(const auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(p), {}))
     {
       if (boost::filesystem::is_regular_file(entry))
       {
         if (entry.path().extension() == ".pcd")
           file_names.emplace_back(entry.path().filename().string());
       }
     }
   }
   else
   {
     std::cerr << "Given path is not a directory\n";
     return;
   }
   std::sort(file_names.begin(), file_names.end());
  }


  /**
   * @brief Lee una nube de puntos en formato .pcd o .ply
   * 
   * @param path Ruta de la nube de puntos
   * @return PointCloudI::Ptr 
    */
  PointCloudL::Ptr 
  readCloudWithLabel (fs::path _path)
  {
    PointCloudL::Ptr _cloud_label (new PointCloudL);
    map<string, int> ext_map = {{".pcd", 0}, {".ply", 1}};

    switch (ext_map[_path.extension().string()])
    {
      case 0: {
        pcl::PCDReader pcd_reader;
        pcd_reader.read(_path.string(), *_cloud_label);
        break;
      }
      case 1: {
        pcl::PLYReader ply_reader;
        ply_reader.read(_path.string(), *_cloud_label);
        break;
      }
      default: {
        std::cout << "Format not compatible, it should be .pcd or .ply" << std::endl;
        break;
      }
    }
    return _cloud_label;
  }

    void 
      init_czm(vector<Zone> &zones, Params &params_)
      { 
        for (int k = 0; k < params_.num_zones; k++) {
          
          Ring empty_ring;
          empty_ring.resize(params_.num_sectors_each_zone[k]);

          Zone z;
          for (int i = 0; i < params_.num_rings_each_zone[k]; i++) {
              z.push_back(empty_ring);
          }
          std::cout << "Zone " << k << " has " << z.size() << " rings" << std::endl;

          zones.push_back(z);
      }
    }

    void 
      init_czm_lvl1(vector<Zone> &zones, Params &params_)
      { 
        for (int k = 0; k < params_.num_zones; k++) {
          
          Ring empty_ring;
          empty_ring.resize(params_.num_sectors_each_zone_lvl1[k]);

          Zone z;
          for (int i = 0; i < params_.num_rings_each_zone_lvl1[k]; i++) {
              z.push_back(empty_ring);
          }
          std::cout << "Zone " << k << " has " << z.size() << " rings" << std::endl;

          zones.push_back(z);
      }
    }

    void 
      init_czm_lvl2(vector<Zone> &zones, Params &params_)
      { 
        for (int k = 0; k < params_.num_zones; k++) {
          
          Ring empty_ring;
          empty_ring.resize(params_.num_sectors_each_zone_lvl2[k]);

          Zone z;
          for (int i = 0; i < params_.num_rings_each_zone_lvl2[k]; i++) {
              z.push_back(empty_ring);
          }
          std::cout << "Zone " << k << " has " << z.size() << " rings" << std::endl;

          zones.push_back(z);
      }
    }
    std::vector<Eigen::Matrix4f> read_poses(string poses_file) {
        // iterate over the file poses_files
        std::ifstream poses_stream(poses_file);
        std::string line;
        std::vector<Eigen::Matrix4f> poses;
        while (std::getline(poses_stream, line)) {
            std::istringstream iss(line);
            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
            for (int i = 0; i < 12; i++) {
                iss >> pose(i / 4, i % 4);
            }
            poses.push_back(pose);
        }
        return poses;
    }
    Eigen::Matrix4f read_calib_file(string calib_file) {
        // Read the calibration file and get the Tr matrix
        std::ifstream calib_stream(calib_file);
        std::string line;
        Eigen::Matrix4f Tr = Eigen::Matrix4f::Identity();
        while (std::getline(calib_stream, line)) {
            if (line.find("Tr") != std::string::npos) {
                std::cout << "line: " << line << std::endl;
                std::istringstream iss(line);
                std::string temp;
                iss >> temp;
                for (int i = 0; i < 12; i++) {
                    iss >> Tr(i / 4, i % 4);
                }
            }
        }
        return Tr;
    }

    vector<Eigen::Matrix4f> transformPoses(
        int current_cloud,
        Eigen::Matrix4f Tr,
        std::vector<int> nearest_poses,
        std::vector<Eigen::Matrix4f> poses  // Original pose
    ) {
        Eigen::Matrix4f transformed_pose = Eigen::Matrix4f::Identity();
        vector<Eigen::Matrix4f> new_poses;
        // Compute inverse of transformation matrix
        for (int i = 0; i < nearest_poses.size(); i++) {
            transformed_pose = poses[current_cloud].inverse()*poses[nearest_poses[i]];
            new_poses.push_back(Tr.inverse()*transformed_pose*Tr);
        }

        return new_poses;
    }




    std::vector<int> get_future_and_past_poses(int current_cloud, std::vector<Eigen::Matrix4f> transformed_poses) {
        std::vector<int> future_and_past_poses;
        Eigen::Vector3f t_current = transformed_poses[current_cloud].block<3, 1>(0, 3);
        // std::cout << "Current pose: " << t_current << std::endl;
        for (int i = 0; i < transformed_poses.size(); i++) {
            Eigen::Vector3f t = transformed_poses[i].block<3, 1>(0, 3);
            // std::cout << "Distance: " << (t - t_current).norm() << std::endl;
            if ((t - t_current).norm() < 40) {
                future_and_past_poses.push_back(i);
            }
        }
        // std::cout << ".................................................."<< std::endl;
        return future_and_past_poses;
    }

    void flush_patches(vector<Zone> &czm, Params &params_) {

        for (int k = 0; k < params_.num_zones; k++) {
            for (int i = 0; i < params_.num_rings_each_zone[k]; i++) {
                for (int j = 0; j < params_.num_sectors_each_zone[k]; j++) {
                    // czm[k][i][j].resize(MAX_POINTS, 3);
                    czm[k][i][j].clear();
                }
            }
        }
    }
    void flush_patches_lvl1(vector<Zone> &czm, Params &params_) {

        for (int k = 0; k < params_.num_zones; k++) {
            for (int i = 0; i < params_.num_rings_each_zone_lvl1[k]; i++) {
                for (int j = 0; j < params_.num_sectors_each_zone_lvl1[k]; j++) {
                    // czm[k][i][j].resize(MAX_POINTS, 3);
                    czm[k][i][j].clear();
                }
            }
        }
    }
    void flush_patches_lvl2(vector<Zone> &czm, Params &params_) {

        for (int k = 0; k < params_.num_zones; k++) {
            for (int i = 0; i < params_.num_rings_each_zone_lvl2[k]; i++) {
                for (int j = 0; j < params_.num_sectors_each_zone_lvl2[k]; j++) {
                    // czm[k][i][j].resize(MAX_POINTS, 3);
                    czm[k][i][j].clear();
                }
            }
        }
    }


    // void flush_patches_multi(vector<Zone> &czm, Params &params_) {

    //     for (int k = 0; k < params_.num_zones; k++) {
    //         for (int i = 0; i < params_.num_rings_each_zone_multi[k]; i++) {
    //             for (int j = 0; j < params_.num_sectors_each_zone_multi[k]; j++) {
    //                 // czm[k][i][j].resize(MAX_POINTS, 3);
    //                 czm[k][i][j].clear();
    //             }
    //         }
    //     }
    // }

double xy2theta(const double &x, const double &y) { // 0 ~ 2 * PI
    double angle = atan2(y, x);
    return angle > 0 ? angle : 2*M_PI+angle;
}

double xy2radius(const double &x, const double &y) {
    // return sqrt(pow(x, 2) + pow(y, 2));
    return sqrt(x*x + y*y);
}

  std::vector<Zone> CZM(vector<Zone> &zones,pcl::PointCloud<pcl::PointXYZL>::Ptr &transformed_cloud, vector<double> min_ranges_, 
    double max_range,vector<double> sector_sizes_, vector<double> ring_sizes_, vector<int> num_rings_,
    vector<int> num_sectors_)
    {
    
      Eigen::MatrixXf src;
      src.resize(transformed_cloud->points.size(), 4); // 3 columns for x,y,z
      for(size_t i = 0; i < transformed_cloud->points.size(); ++i) {
          const auto& point = transformed_cloud->points[i];
          src.row(i) << point.x, point.y, point.z, point.label;
      }
      
      #pragma omp parallel for
      for (int i=0; i<src.rows(); i++) {
          float x = src.row(i)(0), y = src.row(i)(1), z = src.row(i)(2);
        //   std::cout << i << std::endl;
          
          double r = xy2radius(x, y);
          int ring_idx, sector_idx;
          if ((r <= max_range) && (r > min_ranges_[0])) {
              // double theta = xy2theta(pt.x, pt.y);
              double theta = xy2theta(x, y);
              
              if (r < min_ranges_[1]) { // In First rings
                  ring_idx = min(static_cast<int>(((r - min_ranges_[0]) / ring_sizes_[0])), num_rings_[0] - 1);
                  sector_idx = min(static_cast<int>((theta / sector_sizes_[0])), num_sectors_[0] - 1);
                  zones[0][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i));
              } else if (r < min_ranges_[2]) {
                  ring_idx = min(static_cast<int>(((r - min_ranges_[1]) / ring_sizes_[1])), num_rings_[1] - 1);
                  sector_idx = min(static_cast<int>((theta / sector_sizes_[1])), num_sectors_[1] - 1);
                  zones[1][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i));
              } else if (r < min_ranges_[3]) {
                  ring_idx = min(static_cast<int>(((r - min_ranges_[2]) / ring_sizes_[2])), num_rings_[2] - 1);
                  sector_idx = min(static_cast<int>((theta / sector_sizes_[2])), num_sectors_[2] - 1);
                  zones[2][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i));
              } else { // Far!
                  ring_idx = min(static_cast<int>(((r - min_ranges_[3]) / ring_sizes_[3])), num_rings_[3] - 1);
                  sector_idx = min(static_cast<int>((theta / sector_sizes_[3])), num_sectors_[3] - 1);
                  zones[3][ring_idx][sector_idx].emplace_back(PointXYZ(x, y, z, i));
              }
          }
      }

      return zones;}


std::vector<std::tuple<int, int, int>> ground_patches(vector<Zone> &zones,std::vector<Eigen::Matrix4f> transformed_poses, 
                    pcl::PointCloud<pcl::PointXYZL>::Ptr &transformed_cloud,  vector<double> min_ranges_, 
                    double max_range,vector<double> sector_sizes_, vector<double> ring_sizes_, vector<int> num_rings_, 
                    vector<int> num_sectors_){

    std::vector<std::tuple<int, int, int>> confirmed_ground_patches;
    for (int i = 0; i < transformed_poses.size(); i++) {
        float x = transformed_poses[i](0, 3), y = transformed_poses[i](1, 3), z = transformed_poses[i](2, 3);
        // std::cout << "X: " << x << " Y: " << y << " Z: " << z << std::endl;
        double r = utils::xy2radius(x, y);
        int ring_idx, sector_idx, z1;
        if ((r <= max_range) && (r > min_ranges_[0])) {
            // double theta = xy2theta(pt.x, pt.y);
            double theta = utils::xy2theta(x, y);  

            if (r < min_ranges_[1]) { // In First rings
                z1=0;
                ring_idx = min(static_cast<int>(((r - min_ranges_[0]) / ring_sizes_[0])), num_rings_[0] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[0])), num_sectors_[0] - 1);
            } 
            else if (r < min_ranges_[2]) {
                z1=1;
                ring_idx = min(static_cast<int>(((r - min_ranges_[1]) / ring_sizes_[1])), num_rings_[1] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[1])), num_sectors_[1] - 1);
            } 
            else if (r < min_ranges_[3]) {
                z1=2;
                ring_idx = min(static_cast<int>(((r - min_ranges_[2]) / ring_sizes_[2])), num_rings_[2] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[2])), num_sectors_[2] - 1);
            } 
            else { // Far!
                z1=3;
                ring_idx = min(static_cast<int>(((r - min_ranges_[3]) / ring_sizes_[3])), num_rings_[3] - 1);
                sector_idx = min(static_cast<int>((theta / sector_sizes_[3])), num_sectors_[3] - 1);
            }
            if(confirmed_ground_patches.size()==0)
                confirmed_ground_patches.push_back(std::make_tuple(z1, ring_idx, sector_idx));
            else{
                bool flag = false;
                for (const auto& [z, r, s] : confirmed_ground_patches) {
                    if(z==z1 && r==ring_idx && s==sector_idx){
                        flag = true;
                        break;
                    }
                }
                if(!flag)
                    confirmed_ground_patches.push_back(std::make_tuple(z1, ring_idx, sector_idx));
            }
        }
    }
    return confirmed_ground_patches;
    }


std::tuple<Eigen::Vector3f, double, double> compute_propierties(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud) {
    // make a copy of the cloud to avoid modifying the original and without label
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_label(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_no_label);
    // Create PCA object
    pcl::PCA<pcl::PointXYZ> pca;
    
    // Set input cloud
    pca.setInputCloud(cloud_no_label);
    
    // Compute principal directions
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();
    Eigen::Vector3f eigenValues = pca.getEigenValues();
    
    // Select eigenvector corresponding to smallest eigenvalue
    // This represents the plane's normal direction
    Eigen::Vector3f plane_normal = eigenVectors.col(2);
    // Compute variation surface here
    Eigen::Vector3f eigen_values= pca.getEigenValues();

    double surface_var = eigen_values(2) /
                        (eigen_values(0) + eigen_values(1) + eigen_values(2));
    
    // Ensure consistent normal orientation
    if (plane_normal.z() < 0) {
        plane_normal *= -1;
    }

    // Compute height local variance inside the sector
    double sum_height = 0.0;
    for (const auto& point : cloud_no_label->points) {
        sum_height += point.z;
    }
    double mean_height = sum_height / cloud_no_label->points.size();

    // Compute squared differences from mean
    double sum_squared_diff = 0.0;
    for (const auto& point : cloud_no_label->points) {
        double diff = point.z - mean_height;
        sum_squared_diff += diff * diff;
    }

    // Calculate variance
    double variance_height = sum_squared_diff / (cloud_no_label->points.size() - 1);

    return std::make_tuple(plane_normal, surface_var, variance_height);
}


std::tuple<double, double> calculateMedianAndStdDev(const std::vector<std::tuple<Eigen::Vector3f, double, double>>& geo_properties, int index) {
    // Extract the second double values into a separate vector
    std::vector<double> values;
    for (const auto& prop : geo_properties) {
        if (index == 1) {
            values.push_back(std::get<1>(prop));
        } else if (index == 2) {
            values.push_back(std::get<2>(prop));
        } else {
            throw std::invalid_argument("Index must be 1 or 2");
        } // Change to std::get<2>(prop) if you want the third value
    }

    // Calculate Median
    std::sort(values.begin(), values.end());
    double median;
    size_t size = values.size();
    if (size % 2 == 0) {
        median = (values[size / 2 - 1] + values[size / 2]) / 2.0;
    } else {
        median = values[size / 2];
    }

    // Calculate Mean
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    double mean = sum / size;

    // Calculate Standard Deviation
    double sum_squared_diff = 0.0;
    for (double value : values) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }
    double variance = sum_squared_diff / size;
    double std_dev = std::sqrt(variance);

    return std::make_tuple(mean, std_dev);
    // return std::make_tuple(median, std_dev);
}

std::tuple<double, double> calculateMedianAndStdDev(pcl::PointCloud<pcl::PointXYZL>::Ptr init_seeds) {
    // extract the z values into a separate vector
    std::vector<double> values;
    for (const auto& point : init_seeds->points) {
        values.push_back(point.z);
    }
    // compute mean and std deviation
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();
    double sq_sum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / values.size() - mean * mean);
    return std::make_tuple(mean, stdev);
}

std::vector< std::tuple<Eigen::Vector3f, double, double>> evaluating_patches(std::vector<std::tuple<int, int, int>> confirmed_ground_patches,
                                                    pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_two, vector<Zone> &zones,
                                                    pcl::PointCloud<pcl::PointXYZL>::Ptr &evaluate_confirmed_sector)
    {   
        std::vector< std::tuple<Eigen::Vector3f, double, double>> geo_properties;
        std::tuple<Eigen::Vector3f, double, double> geo_properties_actual;
    for (const auto& [z, r, s] : confirmed_ground_patches) {
        for (int j = 0; j < zones[z][r][s].size(); j++) {
            if (cloud_two->points[zones[z][r][s][j].idx].label == 1) {
            evaluate_confirmed_sector->points.push_back(cloud_two->points[zones[z][r][s][j].idx]);
            }
        }
        if (evaluate_confirmed_sector->points.size() > 4) {
            geo_properties_actual = compute_propierties(evaluate_confirmed_sector);
            geo_properties.push_back(geo_properties_actual);
        }    
        evaluate_confirmed_sector->clear();
    }
    return geo_properties;
    }

double computePlaneNormalAngle(const Eigen::Vector3f& plane_normal) {
    // Z unit vector
    Eigen::Vector3f z_unit(0, 0, 1);
    double dot_product = plane_normal.dot(z_unit);
    double magnitude_a = plane_normal.norm();
    double magnitude_b = z_unit.norm();
    
    double cos_angle = dot_product / (magnitude_a * magnitude_b);
    
    // Ensure value is within [-1, 1] to prevent numerical errors
    cos_angle = std::max(-1.0, std::min(cos_angle, 1.0));
    
    double angle_radians = std::acos(cos_angle);
    double angle_degrees = angle_radians * 180.0 / M_PI;
    
    return angle_degrees;
    
    // Convert to degrees
    // float angle_degrees = angle_radians * 180.0 / M_PI;
    
    // return angle_degrees;
}



double computePlaneNormalAngle(const Eigen::Vector3f& plane_normal, const Eigen::Vector3f& plane_normal2) {
    // Z unit vector
    Eigen::Vector3f z_unit(0, 0, 1);
    double dot_product = plane_normal.dot(plane_normal2);
    double magnitude_a = plane_normal.norm();
    double magnitude_b = plane_normal2.norm();
    
    double cos_angle = dot_product / (magnitude_a * magnitude_b);
    
    // Ensure value is within [-1, 1] to prevent numerical errors
    cos_angle = std::max(-1.0, std::min(cos_angle, 1.0));
    
    double angle_radians = std::acos(cos_angle);
    double angle_degrees = angle_radians * 180.0 / M_PI;
    
    return angle_degrees;
}


std::tuple<Eigen::Vector3f, double> extract_initial_seeds_and_compute_plane(const pcl::PointCloud<pcl::PointXYZL>::Ptr& sorted_points, pcl::PointCloud<pcl::PointXYZL>::Ptr& init_seeds) {

        std::sort(sorted_points->points.begin(), sorted_points->points.end(), 
        [](const pcl::PointXYZL& a, const pcl::PointXYZL& b) {
            return a.z < b.z;
        });
        init_seeds->clear();
        int init_idx = 0;
        // std::cout << "Sorted points" << std::endl;
        for (int i = 0; i < sorted_points->points.size(); i++) {
            // std::cout << "Z: " << sorted_points->points[i].z << std::endl;
            if (sorted_points->points[i].z < -1.7) {
                ++init_idx;
            } else {
                break;
            }
        }

        double sum = 0;
        int cnt = 0;
        double lpr_height=0;
        // std::cout << "Init idx: " << init_idx << std::endl;
        // std::cout << "All: " << sorted_points->points.size() << std::endl;


        // if (init_idx == sorted_points->points.size()) {
        //     for (int i = 0; i < sorted_points->points.size(); i++) {
        //         init_seeds->points.push_back(sorted_points->points[i]);
        //     }
        // }
        // else
        // {
           for (int i = init_idx; i < sorted_points->points.size() && cnt < 20; i++) {
                sum += sorted_points->points[i].z;
                cnt++;
            }
            lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0
        // }
        
            // std::cout << "LPR height: " << lpr_height << std::endl;
            int init_seeds_num = 0;

            // iterate pointcloud, filter those height is less than lpr.height+params_.th_seeds
            // std::cout << "LPR height: " << lpr_height << std::endl;
            for (int i = 0; i < sorted_points->points.size(); i++) {
                if (sorted_points->points[i].z < lpr_height+0.125) { // + 0.125
                    init_seeds->points.push_back(sorted_points->points[i]);
                }
            }
            // std::cout << "Seeds selected: " << init_seeds->points.size() << std::endl;
            if (init_seeds->points.size() < 3) {
                // std::cout << "Not enough seeds, using all points" << std::endl;
                init_seeds->clear();
                for (int i = 0; i < sorted_points->points.size(); i++) {
                    init_seeds->points.push_back(sorted_points->points[i]);
                }
            }
            // compute and return the plane coefficients with PCA over the seeds
            Eigen::MatrixXf seeds;
            seeds.resize(init_seeds->points.size(), 3);
            for (size_t i = 0; i < init_seeds->points.size(); ++i) {
                seeds.row(i) << init_seeds->points[i].x, init_seeds->points[i].y, init_seeds->points[i].z;
            }
            // std::cout << "Seeds selected: " << init_seeds->points.size() << std::endl;
            // Create PCA object
            pcl::PCA<pcl::PointXYZL> pca;
            // Set input cloud
            pca.setInputCloud(init_seeds);
            // Compute principal directions
            Eigen::Matrix3f eigenVectors = pca.getEigenVectors();
            Eigen::Vector3f eigenValues = pca.getEigenValues();
            // Select eigenvector corresponding to smallest eigenvalue
            // This represents the plane's normal direction
            Eigen::Vector3f plane_normal = eigenVectors.col(2);
            // Ensure consistent normal orientation
            if (plane_normal.z() < 0) {
                plane_normal *= -1;
            }

            // std::cout << "Plane normal: " << plane_normal << std::endl;
            // std::cout << "Input points: " << sorted_points->points.size() << std::endl;
            // std::cout << "Seeds selected: " << init_seeds->points.size() << std::endl; 
            // // pcl::visualization::PointCloudColorHandlerLabelField<pcl::PointXYZL> color_handler6(sorted_points, "label");
            // To compute the mean of seeds points and then compute the parameter D
            Eigen::Vector3f mean_seeds = seeds.colwise().mean();
            double D = -mean_seeds.dot(plane_normal);

        return std::make_tuple(plane_normal, D);
    }





double computeDistanceToPlane(std::tuple<Eigen::Vector3f, double> plane_normal, const pcl::PointXYZL& point) {
    // Direct point-to-plane distance calculation
    Eigen::Vector3f normal = std::get<0>(plane_normal);
    double D = std::get<1>(plane_normal);
    return std::abs(
        normal.dot(Eigen::Vector3f(point.x, point.y, point.z))+D
    );
}



pcl::PointCloud<pcl::PointXYZL>::Ptr reorderPointCloud(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& cloud_to_reorder,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& reference_cloud,
    float max_distance = 0.01
) {
    // Create output cloud
    pcl::PointCloud<pcl::PointXYZL>::Ptr ordered_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    ordered_cloud->points.resize(reference_cloud->points.size());
    
    // Create KD-tree for fast nearest neighbor search
    pcl::KdTreeFLANN<pcl::PointXYZL> kdtree;
    kdtree.setInputCloud(cloud_to_reorder);

    // For each point in reference cloud, find closest point in cloud_to_reorder
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    
    for (size_t i = 0; i < reference_cloud->size(); ++i) {
        if (kdtree.nearestKSearch(reference_cloud->points[i], 1, 
                                 pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            // Check if the distance is within threshold
            if (std::sqrt(pointNKNSquaredDistance[0]) <= max_distance) {
                ordered_cloud->points[i] = cloud_to_reorder->points[pointIdxNKNSearch[0]];
            } else {
                //   std::cout << "Point " << i << " has no close point" << std::endl;
                // If no close point found, copy reference point but mark as invalid
                ordered_cloud->points[i] = reference_cloud->points[i];
                ordered_cloud->points[i].label = 0;
            }
        }
    }

    return ordered_cloud;
}



GroundSegmentationMetrics evaluateGroundSegmentation(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& ground_truth,
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& prediction
) {
    GroundSegmentationMetrics metrics = {0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f};

    if (ground_truth->points.size() != prediction->points.size()) {
        throw std::runtime_error("Point clouds must have the same size!");
    }

    // Count TP, FP, TN, FN
    for (size_t i = 0; i < ground_truth->points.size(); ++i) {
        bool is_ground_truth = ground_truth->points[i].label == 1;
        bool is_predicted_ground = prediction->points[i].label == 1;

        if (is_ground_truth) {
            if (is_predicted_ground) {
                metrics.true_positives++;  // Correctly identified ground
            } else {
                metrics.false_negatives++; // Missed ground
            }
        } else {
            if (is_predicted_ground) {
                metrics.false_positives++; // Incorrectly labeled as ground
            } else {
                metrics.true_negatives++;  // Correctly identified non-ground
            }
        }
    }

    // Calculate metrics
    if (metrics.true_positives + metrics.false_positives > 0) {
        metrics.precision = static_cast<float>(metrics.true_positives) / 
                          (metrics.true_positives + metrics.false_positives);
    }

    if (metrics.true_positives + metrics.false_negatives > 0) {
        metrics.recall = static_cast<float>(metrics.true_positives) / 
                        (metrics.true_positives + metrics.false_negatives);
    }

    if (metrics.precision + metrics.recall > 0) {
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / 
                          (metrics.precision + metrics.recall);
    }

    metrics.accuracy = static_cast<float>(metrics.true_positives + metrics.true_negatives) / 
                      ground_truth->size();

    return metrics;
}

}   // namespace utils






