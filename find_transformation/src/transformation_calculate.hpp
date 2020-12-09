#include<ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/GetMap.h"
#include "nav_msgs/Odometry.h"
#include<nav_msgs/Path.h>
#include<tf/tf.h>
#include<tf/transform_listener.h>
#include <iostream>
#include <fstream>
#include<string>
#include<math.h>
#include<ctime>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include<thread>

//#include <eigen3/Eigen/Dense>
#include "common.h"
#include "tools_logger.hpp"
#include "tools_random.hpp"
#include<vector>
#include<algorithm>
#include "pcl_tools.hpp"
// PCL specific includes
//#include <pcl/ros/conversions.h>
#include<pcl/conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include<pcl/filters/radius_outlier_removal.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/common/impl/io.hpp>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include"keyframe_process.hpp"
using namespace std;
#define PI 3.1415926
struct odometry_pair
{
  double score;
  int index;
  int points_num;
};
class Local_map
{
    public:
    bool cornericp_continue=true;
    bool surfaceicp_continue=true;
    bool surface_icp_done=false;
    int numbers;
    int iter_num=0;
    int key_frame_num;
    int accumulate_num;
    int keypoints_num_threshold;
    int surface_icp_max_iternum;
    int corner_icp_max_iternum;

    float fit_score_threshold;
    float corner_initial_change_scope;
    float surface_initial_change_scope;

    double surface_icp_corr_threshold;
    double surface_icp_fit_threshold;
    double surface_pass_filter_threshold;
    double corner_icp_corr_threshold;
    double corner_icp_fit_threshold;
    double corner_pass_filter_threshold;
//    float radius=0.3;
    size_t corner_frame_index=0;
    size_t surface_frame_index=0;
    string cornermap_path;
    string surfacemap_path;
    string transformdata_path;
    string vocab_path;
    string descriptors_path;
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> corner_icp;
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> surface_icp;

    Eigen::Matrix4d corner_t_hist = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d surface_t_hist = Eigen::Matrix4d::Identity ();

    vector<double> corner_fitness_score;
    vector<double> surface_fitness_score;

//    vector<double> initial_fitness_score;
//    vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> initial_trans_matrix;
    vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> initial_trans_candidates;

    pcl::PCDWriter writer;
    Eigen::Matrix4d corner_transformation_matrix = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d surface_transformation_matrix = Eigen::Matrix4d::Identity ();

    Eigen::Matrix4d corner_ndttrans = Eigen::Matrix4d::Identity ();
    Eigen::Matrix4d surface_ndttrans = Eigen::Matrix4d::Identity ();

    Eigen::Matrix4f init_ndtguess=Eigen::Matrix4f::Identity ();
    Eigen::Quaterniond quater;

    ros::NodeHandle nh;
    pcl::PointCloud<PointType>::Ptr cornermap=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<PointType>::Ptr current_corner_local_map=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::search::KdTree<PointType>::Ptr cornermap_kdtree=boost::make_shared<pcl::search::KdTree<PointType>>();
    ros::Subscriber corner_sub;

    pcl::PointCloud<PointType>::Ptr surfacemap=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<PointType>::Ptr current_surface_local_map=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<PointType>::Ptr surface_acc_frame=boost::make_shared<pcl::PointCloud<PointType>>();
    pcl::PointCloud<PointType>::Ptr corner_acc_frame=boost::make_shared<pcl::PointCloud<PointType>>();

    pcl::search::KdTree<PointType>::Ptr surfacemap_kdtree=boost::make_shared<pcl::search::KdTree<PointType>>();
    pcl::PassThrough<pcl::PointXYZI> surface_pass;
    pcl::PassThrough<pcl::PointXYZI> corner_pass;
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> surface_r_filter;
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> corner_r_filter;

    ros::Subscriber surface_sub;
    ros::Publisher corner_odometry_publisher,surface_odometry_publisher,path_publisher;
    nav_msgs::Path path;
    DBoW3::Vocabulary surface_vocabulary;
    DBoW3::Database surface_database;
    flann::Matrix<float> data_transform;
    odometry_pair corner_odometry_pair,surface_odometry_pair;

    template<typename T>
    void print4x4Matrix(const T & matrix)
    {
      printf ("Rotation matrix :\n");
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (0, 0), matrix (0, 1), matrix (0, 2));
      printf ("R = | %6.3f %6.3f %6.3f | \n", matrix (1, 0), matrix (1, 1), matrix (1, 2));
      printf ("    | %6.3f %6.3f %6.3f | \n", matrix (2, 0), matrix (2, 1), matrix (2, 2));
      printf ("Translation vector :\n");
      printf ("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix (0, 3), matrix (1, 3), matrix (2, 3));
    }

    int initial_variation(Eigen::Matrix4f & initial,int & iter_num,float range)
    {
      switch (iter_num) {
      case 0:
          initial(0,3)=-range;
          initial(1,3)=0;
          iter_num++;
          break;
      case 1:
          initial(0,3)=-range;
          initial(1,3)=range;
          iter_num++;
          break;
      case 2:
          initial(0,3)=0;
          initial(1,3)=range;
          iter_num++;
          break;
      case 3:
          initial(0,3)=range;
          initial(1,3)=range;
          iter_num++;
          break;
      case 4:
          initial(0,3)=range;
          initial(1,3)=0;
          iter_num++;
          break;
      case 5:
          initial(0,3)=range;
          initial(1,3)=-range;
          iter_num++;
          break;
      case 6:
          initial(0,3)=0;
          initial(1,3)=-range;
          iter_num++;
          break;
      case 7:
          initial(0,3)=-range;
          initial(1,3)=-range;
          iter_num++;
          break;
      default:
          break;
      }
      return 0;
    }
    int cornericp_findtrans(pcl::PointCloud< PointType >::Ptr in_laser_cloud_corner_from_map,
                      pcl::search::KdTree<PointType>::Ptr   kdtree_corner_from_map,
                      pcl::PointCloud< PointType >::Ptr laserCloudCornerStack
                      )
    {
      vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> trans_matrix;
      corner_acc_frame->clear();
      int iter_num=0;
      if(corner_frame_index>100)
      {
        *corner_acc_frame=*corner_acc_frame+*laserCloudCornerStack;
        if((corner_frame_index%accumulate_num==0)&&(corner_acc_frame->size()>keypoints_num_threshold))
        {
          corner_fitness_score.clear();
          trans_matrix.clear();
//          cerr<<"corner_acc_frame size before downsample is "<<corner_acc_frame->size()<<endl;
          pcl::VoxelGrid<PointTypeIO> corner_sample;
          corner_sample.setInputCloud (corner_acc_frame);
          corner_sample.setLeafSize (0.1, 0.1, 0.1);
          corner_sample.filter (*corner_acc_frame);
          cerr<<"corner_acc_frame size after down sampled is: "<<corner_acc_frame->size()<<endl;

//          cerr<<"before corner radius filter: "<<corner_acc_frame->size();
//          corner_r_filter.setInputCloud(corner_acc_frame);
//          corner_r_filter.filter(*corner_acc_frame);
//          cerr<<" after corner radius filter: "<<corner_acc_frame->size()<<endl;
          pcl::PointCloud<PointType>::Ptr corner_trans (new pcl::PointCloud<PointType>);
          pcl::transformPointCloud (*corner_acc_frame, *corner_trans, corner_t_hist);
//          cerr<<"corner_acc_frame size after clear is: "<<corner_acc_frame->size()<<endl;
          corner_icp.setInputSource(corner_trans);
          Eigen::Matrix4f init_icp=Eigen::Matrix4f::Identity ();
          while(cornericp_continue)
          {
              //corner_icp.setRANSACOutlierRejectionThreshold(0.8);
              corner_icp.align(*corner_trans,init_icp);
              if (corner_icp.hasConverged())
              {
                  corner_fitness_score.push_back(corner_icp.getFitnessScore ());
                  Eigen::Matrix4d trans_test=corner_icp.getFinalTransformation().cast<double>();
                  trans_matrix.push_back(trans_test);
                  if((corner_icp.getFitnessScore ()>fit_score_threshold)&&(iter_num<8))
                  {
                    initial_variation(init_icp,iter_num,corner_initial_change_scope);
                    cornericp_continue=true;
                  }
                  else
                  {
                    cornericp_continue=false;
                  }
              }
          }
          if(iter_num!=0)
          {
            cerr<<"corner icp iter num at "<<corner_frame_index<<" is "<<iter_num<<"  best score is: "<<*min_element(begin(corner_fitness_score),end(corner_fitness_score))<<"  worest score is: "<<*max_element(begin(corner_fitness_score),end(corner_fitness_score))<<endl;
          }
          Eigen::Matrix4d corner_t_curr = trans_matrix[distance(begin(corner_fitness_score),min_element(begin(corner_fitness_score),end(corner_fitness_score)))];
          cornericp_continue=true;
          corner_t_hist=corner_t_curr*corner_t_hist;

          //publish odometry
          Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity();
          for(size_t rows=0;rows<3;rows++)
            for(size_t cols=0;cols<3;cols++)
              rotation_matrix(rows,cols)=corner_t_hist(rows,cols);
          Eigen::Quaterniond qua=Eigen::Quaterniond(rotation_matrix);
    //      cout<<"qua: "<<qua.w()<<" "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<endl;
          nav_msgs::Odometry corner_odometry;
          corner_odometry.header.frame_id="camera_init";
          corner_odometry.pose.pose.orientation.x=qua.x();
          corner_odometry.pose.pose.orientation.y=qua.y();
          corner_odometry.pose.pose.orientation.z=qua.z();
          corner_odometry.pose.pose.orientation.w=qua.w();

          corner_odometry.pose.pose.position.x=corner_t_hist(0,3);
          corner_odometry.pose.pose.position.y=corner_t_hist(1,3);
          corner_odometry.pose.pose.position.z=corner_t_hist(2,3);

          corner_odometry_publisher.publish(corner_odometry);
          if(*min_element(begin(corner_fitness_score),end(corner_fitness_score))>0.05)
          {
            //std::cout << "final corner icp score at "<<corner_frame_index<<" is "<<*min_element(begin(corner_fitness_score),end(corner_fitness_score))<<endl;
            //std::cout<<"the transformation is: "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<" "<<qua.w()<<" "<<corner_t_hist(0,3)<<" "<<corner_t_hist(1,3)<<" "<<corner_t_hist(2,3)<<endl;
          }
        }
      }
      corner_frame_index++;
      if((corner_frame_index>(100+accumulate_num))&&(corner_acc_frame->size()>keypoints_num_threshold))
      {
        corner_odometry_pair.index=corner_frame_index;
        corner_odometry_pair.score=*min_element(begin(corner_fitness_score),end(corner_fitness_score));
        corner_odometry_pair.points_num=corner_acc_frame->size();
      }
      else
      {
        corner_odometry_pair.index=corner_frame_index;
        corner_odometry_pair.score=100;//large score means the result is unreliable
        corner_odometry_pair.points_num=corner_acc_frame->size();
      }
      while(corner_frame_index>surface_frame_index)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));//sleep for 1ms
      }
      return 0;
    }
    int surfaceicp_findtrans(pcl::PointCloud< PointType >::Ptr in_laser_cloud_surface_from_map,
                      pcl::search::KdTree<PointType>::Ptr   kdtree_surface_from_map,
                      pcl::PointCloud< PointType >::Ptr laserClouSurfaceStack
                      )
    {
//      cerr<<"surface frame index: "<<surface_frame_index<<endl;
//      vector<double> fitness_score;
      vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> trans_matrix;
      int iter_num=0;
      surface_acc_frame->clear();

      if (surface_frame_index<100)
      {
        *current_surface_local_map=*current_surface_local_map+*laserClouSurfaceStack;
      }
      else if (surface_frame_index==100)
      {
        initial_find(current_surface_local_map,surface_database,data_transform,initial_trans_candidates);
        cerr<<"candidate size: "<<initial_trans_candidates.size()<<" surface local map size is: "<<current_surface_local_map->size()<<endl;
        vector<double> initial_fitness_score;
        vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>> initial_trans_matrix;
        for(size_t i=0;i<initial_trans_candidates.size();i++)
        {
          pcl::PointCloud<PointType>::Ptr surface_trans (new pcl::PointCloud<PointType>);
          surface_pass.setInputCloud(laserClouSurfaceStack);
          surface_pass.filter(*laserClouSurfaceStack);

          pcl::transformPointCloud (*laserClouSurfaceStack, *surface_trans, initial_trans_candidates[i]);
//          pcl::transformPointCloud (*laserClouSurfaceStack, *surface_trans, initial_trans_candidates[i]);
          surface_icp.setInputSource(surface_trans);
          surface_icp.align(*surface_trans);
          if(surface_icp.hasConverged())// && (surface_icp.getFitnessScore()<0.05))
          {
            initial_fitness_score.push_back(surface_icp.getFitnessScore());
            initial_trans_matrix.push_back(surface_icp.getFinalTransformation().cast<double>());
            cerr<<"initial trans is:"<<endl;
            print4x4Matrix(initial_trans_candidates[i]);
            cerr<<"final trans is:"<<endl;
            print4x4Matrix(surface_icp.getFinalTransformation().cast<double>());
            cerr<<"score is: "<<surface_icp.getFitnessScore()<<endl;
          }
        }
        surface_t_hist=initial_trans_matrix[distance(begin(initial_fitness_score),min_element(begin(initial_fitness_score),end(initial_fitness_score)))];
        surface_t_hist=Eigen::Matrix4d::Identity ();
        cerr<<"initial trans is: "<<endl;
        print4x4Matrix(surface_t_hist);
        std::cout<<"size is: "<<initial_fitness_score.size()<<" "<<initial_trans_matrix.size()<<"final initial fitness score is: "<<*min_element(begin(initial_fitness_score),end(initial_fitness_score))<<endl;
      }
      else
      {
        *surface_acc_frame=*surface_acc_frame+*laserClouSurfaceStack;
        if(surface_frame_index%accumulate_num==0)
        {
          surface_fitness_score.clear();
          trans_matrix.clear();
//          cerr<<"before downsample is "<<surface_acc_frame->size();
          pcl::VoxelGrid<PointTypeIO> surf_sample;
          surf_sample.setInputCloud (surface_acc_frame);
          surf_sample.setLeafSize (0.1, 0.1, 0.1);
          surf_sample.filter (*surface_acc_frame);
//          cerr<<" after down sampled is: "<<surface_acc_frame->size()<<endl;

//          cerr<<"before surface radius filter: "<<surface_acc_frame->size();
//          surface_r_filter.setInputCloud(surface_acc_frame);
//          surface_r_filter.filter(*surface_acc_frame);
//          cerr<<" after surface radius filter: "<<surface_acc_frame->size()<<endl;
          pcl::PointCloud<PointType>::Ptr surface_trans (new pcl::PointCloud<PointType>);
          pcl::transformPointCloud (*surface_acc_frame, *surface_trans, surface_t_hist);
//          surface_acc_frame->clear();
//          cerr<<"surface_acc_frame size after clear is: "<<surface_acc_frame->size()<<endl;
          surface_icp.setInputSource(surface_trans);
          Eigen::Matrix4f init_icp=Eigen::Matrix4f::Identity ();
          while(surfaceicp_continue)
          {
              //corner_icp.setRANSACOutlierRejectionThreshold(0.8);
              surface_icp.align(*surface_trans,init_icp);
              if (surface_icp.hasConverged())
              {
                  surface_fitness_score.push_back(surface_icp.getFitnessScore ());
                  Eigen::Matrix4d trans_test=surface_icp.getFinalTransformation().cast<double>();
                  trans_matrix.push_back(trans_test);
                  if((surface_icp.getFitnessScore ()>fit_score_threshold)&&(iter_num<8))
                  {
                    initial_variation(init_icp,iter_num,surface_initial_change_scope);
                    surfaceicp_continue=true;
                  }
                  else
                  {
                    surfaceicp_continue=false;
                  }
              }
          }
          if(iter_num!=0)
          {
            cerr<<"surface icp iter num at "<<surface_frame_index<<" is "<<iter_num<<"  best score is: "<<*min_element(begin(surface_fitness_score),end(surface_fitness_score))<<"  worest score is: "<<*max_element(begin(surface_fitness_score),end(surface_fitness_score))<<endl;
          }
          Eigen::Matrix4d surface_t_curr = trans_matrix[distance(begin(surface_fitness_score),min_element(begin(surface_fitness_score),end(surface_fitness_score)))];
          surfaceicp_continue=true;
          surface_t_hist=surface_t_curr*surface_t_hist;

          //fusion
          if((abs(corner_odometry_pair.index-surface_frame_index)<2)&&(surface_frame_index>(100+accumulate_num))&&(corner_odometry_pair.points_num<keypoints_num_threshold)&&(surface_acc_frame->size()>keypoints_num_threshold))
          {
            corner_t_hist=surface_t_hist;
            //cerr<<"only surface_t_hist"<<endl;
          }
          else if ((abs(corner_odometry_pair.index-surface_frame_index)<2)&&(surface_frame_index>(100+accumulate_num))&&(corner_odometry_pair.points_num>keypoints_num_threshold)&&(surface_acc_frame->size()<keypoints_num_threshold))
          {
            surface_t_hist=corner_t_hist;
            //cerr<<"only corner_t_hist"<<endl;
          }
          else if((abs(corner_odometry_pair.index-surface_frame_index)<2)&&(surface_frame_index>(100+accumulate_num)))
          {
            //cerr<<"fusion at "<<surface_frame_index<<" begin"<<endl;
            if(corner_odometry_pair.score<(*min_element(begin(surface_fitness_score),end(surface_fitness_score))))
            {
              surface_t_hist=corner_t_hist;
              cerr<<"using corner"<<endl;
            }
            else
            {
              corner_t_hist=surface_t_hist;
              cerr<<"using surface"<<endl;
            }
          }
          else
          {
            if(surface_frame_index>(100+accumulate_num))
            {
              cerr<<"fusion failed."<<endl;
            }
            else
            {
              cerr<<"waiting for fusion."<<endl;
            }
          }

          //publish odometry
          Eigen::Matrix3d rotation_matrix=Eigen::Matrix3d::Identity();
          for(size_t rows=0;rows<3;rows++)
            for(size_t cols=0;cols<3;cols++)
              rotation_matrix(rows,cols)=surface_t_hist(rows,cols);
          Eigen::Quaterniond qua=Eigen::Quaterniond(rotation_matrix);
    //      cout<<"qua: "<<qua.w()<<" "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<endl;
          nav_msgs::Odometry surface_odometry;
          surface_odometry.header.frame_id="camera_init";
          surface_odometry.pose.pose.orientation.x=qua.x();
          surface_odometry.pose.pose.orientation.y=qua.y();
          surface_odometry.pose.pose.orientation.z=qua.z();
          surface_odometry.pose.pose.orientation.w=qua.w();

          surface_odometry.pose.pose.position.x=surface_t_hist(0,3);
          surface_odometry.pose.pose.position.y=surface_t_hist(1,3);
          surface_odometry.pose.pose.position.z=surface_t_hist(2,3);

          surface_odometry_publisher.publish(surface_odometry);

          //publish path information
          geometry_msgs::PoseStamped pose;
          pose.header=surface_odometry.header;
          pose.pose=surface_odometry.pose.pose;
          path.header.stamp=surface_odometry.header.stamp;
          path.header.frame_id="camera_init";

          path.poses.push_back(pose);
          path_publisher.publish(path);
          if(*min_element(begin(surface_fitness_score),end(surface_fitness_score))>0.05)
          {
            //std::cout << "final surface icp score at "<<surface_frame_index<<" is "<<*min_element(begin(surface_fitness_score),end(surface_fitness_score))<<endl;
            //std::cout<<"the transformation is: "<<qua.x()<<" "<<qua.y()<<" "<<qua.z()<<" "<<qua.w()<<" "<<surface_t_hist(0,3)<<" "<<surface_t_hist(1,3)<<" "<<surface_t_hist(2,3)<<endl;
          }
        }
      }
      surface_frame_index++;
      while(corner_frame_index<surface_frame_index)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));//sleep for 1ms
      }
      return 0;
    }

    void corner_callback(const sensor_msgs::PointCloud2ConstPtr& input)
    {
      pcl::PointCloud<PointType>::Ptr corner_frame=boost::make_shared<pcl::PointCloud<PointType>>();// (new pcl::PointCloud<PointTypeIO>);
      pcl::fromROSMsg(*input,*corner_frame);
//      std::cerr<<"before filter: "<<corner_frame->size();
//      corner_pass.setInputCloud(corner_frame);
//      corner_pass.filter(*corner_frame);
//      std::cerr<<" after filter: "<<corner_frame->size()<<endl;
      cornericp_findtrans(cornermap,cornermap_kdtree,corner_frame);
      //std::cerr<<"corner frame id is: "<<corner_frame_index<<" corner map size is:"<<cornermap->size()<<endl;
    }
    void surface_callback(const sensor_msgs::PointCloud2ConstPtr& input)
    {
      pcl::PointCloud<PointType>::Ptr surface_frame=boost::make_shared<pcl::PointCloud<PointType>>();// (new pcl::PointCloud<PointTypeIO>);
      pcl::fromROSMsg(*input,*surface_frame);
      if(surface_frame_index>99)
      {
        surface_pass.setInputCloud(surface_frame);
        surface_pass.filter(*surface_frame);
      }
      surfaceicp_findtrans(surfacemap,surfacemap_kdtree,surface_frame);

    }
    Local_map()
    {
      //load parameters
      nh.param<std::string>("/cornermap_path",cornermap_path,"default value");
      nh.param<std::string>("/surfacemap_path",surfacemap_path,"default value");
      nh.param<std::string>("/vocab_path",vocab_path,"default value");
      nh.param<std::string>("/descriptors_path",descriptors_path,"default value");
      nh.param<std::string>("/transformdata_path",transformdata_path,"default value");

      nh.param<int>("relocalization/key_frame_num",key_frame_num,20);
      nh.param<int>("relocalization/accumulate_num",accumulate_num,1);
      nh.param<int>("relocalization/keypoints_num_threshold",keypoints_num_threshold,150);

      nh.param<float>("relocalization/fit_score_threshold",fit_score_threshold,0.05);
      nh.param<float>("relocalization/corner_initial_change_scope",corner_initial_change_scope,0.3);
      nh.param<float>("relocalization/surface_initial_change_scope",surface_initial_change_scope,0.25);

      nh.param<double>("relocalization/surface_icp_corr_threshold",surface_icp_corr_threshold,0.6);
      nh.param<int>("relocalization/surface_icp_max_iternum",surface_icp_max_iternum,50);
      nh.param<double>("relocalization/surface_icp_fit_threshold",surface_icp_fit_threshold,0.05);
      nh.param<double>("relocalization/surface_pass_filter_threshold",surface_pass_filter_threshold,70);
      nh.param<double>("relocalization/corner_icp_corr_threshold",corner_icp_corr_threshold,0.6);
      nh.param<int>("relocalization/corner_icp_max_iternum",corner_icp_max_iternum,50);
      nh.param<double>("relocalization/corner_icp_fit_threshold",corner_icp_fit_threshold,0.05);
      nh.param<double>("relocalization/corner_pass_filter_threshold",corner_pass_filter_threshold,35);

      pcl::io::loadPCDFile<PointType> (cornermap_path, *cornermap);
      cornermap_kdtree->setInputCloud(cornermap);
      corner_sub= nh.subscribe<sensor_msgs::PointCloud2>("cornerstack",10000,&Local_map::corner_callback,this);

      pcl::io::loadPCDFile<PointType> (surfacemap_path, *surfacemap);
      surfacemap_kdtree->setInputCloud(surfacemap);
      surface_sub= nh.subscribe<sensor_msgs::PointCloud2>("surfacestack",10000,&Local_map::surface_callback,this);
      cerr<<"map and kdtree created"<<endl;

      path_publisher=nh.advertise<nav_msgs::Path>("/path",10000);
      surface_odometry_publisher=nh.advertise<nav_msgs::Odometry>("/surface_odometry",10000);
      corner_odometry_publisher=nh.advertise<nav_msgs::Odometry>("/corner_odometry",10000);
      database_initialize(surface_vocabulary,surface_database,key_frame_num,vocab_path,descriptors_path);
      flann::load_from_file(data_transform,transformdata_path,"transform_data");
      cerr<<"vocabulary and transformation loaded"<<endl;
      cout<<"transform: "<<data_transform.rows<<" database size is: "<<surface_database.size()<<" vocab size is: "<<surface_vocabulary.size()<<endl;

      //set surface icp
      surface_icp.setInputTarget(surfacemap);
      surface_icp.setSearchMethodTarget(surfacemap_kdtree,true);
      surface_icp.setMaxCorrespondenceDistance(surface_icp_corr_threshold);//points >0.5m ignored
      surface_icp.setMaximumIterations (surface_icp_max_iternum);
      surface_icp.setTransformationEpsilon (1e-8);
      surface_icp.setEuclideanFitnessEpsilon (surface_icp_fit_threshold);
      std::cerr<<"surface icp has initialized"<<endl;
      //set corner icp
      corner_icp.setInputTarget(cornermap);
      corner_icp.setSearchMethodTarget(cornermap_kdtree,true);
      corner_icp.setMaxCorrespondenceDistance(corner_icp_corr_threshold);//points >0.5m ignored
      corner_icp.setMaximumIterations (corner_icp_max_iternum);
      corner_icp.setTransformationEpsilon (1e-8);
      corner_icp.setEuclideanFitnessEpsilon (corner_icp_fit_threshold);
      std::cerr<<"corner icp has initialized"<<endl;

      corner_pass.setFilterFieldName("x");
      corner_pass.setFilterLimits(0.0,corner_pass_filter_threshold);
      surface_pass.setFilterFieldName("x");
      surface_pass.setFilterLimits(0.0,surface_pass_filter_threshold);
      std::cerr<<"passthrough filter initialized"<<endl;
      surface_r_filter.setRadiusSearch(1);
      surface_r_filter.setMinNeighborsInRadius(3);
      //surface_r_filter.setKeepOrganized(true);
      corner_r_filter.setRadiusSearch(1);
      corner_r_filter.setMinNeighborsInRadius(3);
    }
};
