#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include<vector>
#include<iostream>
#include<fstream>
#include<string>
#include <algorithm>
#include <iomanip>
#include<utility>
#include<flann/flann.h>
#include<flann/io/hdf5.h>
#include<boost/filesystem.hpp>
#include<DBoW3/DBoW3.h>
#include<opencv2/core/core.hpp>
#include<pcl_conversions/pcl_conversions.h>
#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include<pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include<pcl/features/vfh.h>
#include<pcl/visualization/pcl_plotter.h>
using namespace std;
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;
float cluster_distance_threshold=0.5;
class Map_creator
{
  public:
  string corner_frame_path;
  string surf_frame_path;
  string surface_descriptors_path;
  string surface_vocab_path;
  string transformation_path;
  string transformation_saved_path;
  string corner_map_saved_path;
  string surface_map_saved_path;
  string clusters_saved_path;

  size_t cornerframe_index=0;
  size_t surfaceframe_index=0;
  size_t localmap_index=0;
  size_t key_frame_index=0;

  int frame_num;
  int key_frame_num;
  int if_save_clusters;
  int if_map_create;
  int if_transform_create;
  int if_dictionary_generate;

  int mincluster_size;
  int maxcluster_size;

  float mycluster_distance_threshold;

  double angle_threshold;//max difference between model normal and given axis in radians
  double distance_threshold;//distance threshold between inliers and model plane
  double leaf_size;//downsample size
  double cluster_tolerance;//<0.5, evaluated by setConditionFunction

  ros::NodeHandle nh;
  ros::Subscriber corner_aligned,surface_aligned;
  pcl::PCDWriter writer;
  pcl::PointCloud<PointTypeIO>::Ptr cornermap=boost::make_shared<pcl::PointCloud<PointTypeIO>>();
  pcl::PointCloud<PointTypeIO>::Ptr surfacemap=boost::make_shared<pcl::PointCloud<PointTypeIO>>();
  pcl::PointCloud<PointTypeIO>::Ptr local_map=boost::make_shared<pcl::PointCloud<PointTypeIO>>();

  pcl::PassThrough<pcl::PointXYZI> pass;
  vector<cv::Mat> descripors;

  void map_create()
  {
    pcl::PointCloud<PointTypeIO>::Ptr cloud_1 (new pcl::PointCloud<PointTypeIO>), cloud_2 (new pcl::PointCloud<PointTypeIO>), cloud_3 (new pcl::PointCloud<PointTypeIO>), cloud_4 (new pcl::PointCloud<PointTypeIO>);
    pcl::PCDReader reader;
    for(int i=1;i<=frame_num;i++)
    {
        reader.read(surf_frame_path+to_string(i)+".pcd",*cloud_1);
        *cloud_2=*cloud_2+*cloud_1;
        cerr<<i;
    }
    cerr<<"surf map before down sampled is: "<<cloud_2->size()<<endl;
    pcl::VoxelGrid<PointTypeIO> surf_sample;
    surf_sample.setInputCloud (cloud_2);
    surf_sample.setLeafSize (leaf_size, leaf_size, leaf_size);
    surf_sample.setDownsampleAllData (true);
    surf_sample.filter (*cloud_2);
    cerr<<"surf map after down sampled is: "<<cloud_2->size()<<endl;

    writer.write(surface_map_saved_path,*cloud_2);

    for(int i=1;i<=frame_num;i++)
    {
        reader.read(corner_frame_path+to_string(i)+".pcd",*cloud_3);
        *cloud_4=*cloud_4+*cloud_3;
        //cerr<<i;
    }
    cerr<<"corner map before down sampled is: "<<cloud_4->size()<<endl;
    pcl::VoxelGrid<PointTypeIO> corner_sample;
    corner_sample.setInputCloud (cloud_4);
    corner_sample.setLeafSize (leaf_size, leaf_size, leaf_size);
    corner_sample.setDownsampleAllData (true);
    corner_sample.filter (*cloud_4);
    cerr<<"corner map after down sampled is: "<<cloud_4->size()<<endl;
    writer.write(corner_map_saved_path,*cloud_4);
  }
  void transform_creator(){
      vector<double> V;
      vector<double>::iterator it;
      ifstream data(transformation_path);
      double d;
      while (data >> d)
      {
          V.push_back(d);//将数据压入堆栈。//
      }
      data.close();
      size_t rows=V.size()/8;
      it = V.begin();
      flann::Matrix<double> transform_data (new double[V.size()],rows,8);
      for(std::size_t i=0;i<transform_data.rows;++i)
          for(std::size_t j=0;j<transform_data.cols;++j)
          {
              transform_data[i][j]=*it;
              it++;
          }
      //save the model and its transform information
      flann::save_to_file(transform_data,transformation_saved_path,"transform_data");
  }
  void detectObjectsOnCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_filtered)
  {
      if (cloud->size() > 0)
      {
          pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
          pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
          pcl::SACSegmentation<pcl::PointXYZI> seg;
          seg.setOptimizeCoefficients(true);
          seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
          seg.setAxis(Eigen::Vector3f(0,0,1));
          //seg.setModelType(pcl::SACMODEL_PLANE);
          seg.setMethodType(pcl::SAC_RANSAC);
          seg.setEpsAngle(angle_threshold);//max difference between model normal and given axis in radians
          // you can modify the parameter below
          seg.setMaxIterations(10000);
          seg.setDistanceThreshold(distance_threshold);//distance threshold between inliers and model plane
          seg.setInputCloud(cloud);
          seg.segment(*inliers, *coefficients);//inliers:indices support the model
          if (inliers->indices.size() == 0)
          {
              cout<<"error! Could not found any inliers!"<<endl;
          }
          // extract ground
          pcl::ExtractIndices<pcl::PointXYZI> extractor;
          extractor.setInputCloud(cloud);
          extractor.setIndices(inliers);
          extractor.setNegative(true);//true: remove the ground;false: extract the ground
          extractor.filter(*cloud_filtered);
          cout << "filter done."<<endl;
      }
      else
      {
          cout<<"no data!"<<endl;
      }
  }
 static bool
  customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
  {
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < cluster_distance_threshold)//0.5
    {
        return (true);
  //    if (std::abs (point_a.intensity - point_b.intensity) < 8.0f)
  //      return (true);
  //    if (std::abs (point_a_normal.dot (point_b_normal)) < 0.06)
  //      return (true);
    }
    else
    {
        return (false);

  //    if (std::abs (point_a.intensity - point_b.intensity) < 0)//3.0f
  //      return (true);
    }
  //  return (false);
  }
  void dictionary_create(const sensor_msgs::PointCloud2ConstPtr& input)
  {
    localmap_index++;
    pcl::PointCloud<pcl::PointXYZI>::Ptr surface_frame=boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*input,*surface_frame);
    *local_map=*local_map+*surface_frame;
    if(localmap_index==100)
    {
      key_frame_index++;
      cerr<<key_frame_index<<"th key frame begin create"<<endl;
      pcl::PointCloud<PointTypeIO>::Ptr  cloud_out (new pcl::PointCloud<PointTypeIO>);
      pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>());
      pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
      pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
      pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
      cerr<<"local_map size before downsample is : "<<local_map->points.size()<<endl;
      // Downsample the cloud using a Voxel Grid class
      pcl::VoxelGrid<PointTypeIO> vg;
      vg.setInputCloud (local_map);
      vg.setLeafSize (leaf_size, leaf_size, leaf_size);
      //vg.setDownsampleAllData (true);
      vg.filter (*cloud_out);
      local_map->clear();
      cerr<<"local_map size after downsample is: "<<cloud_out->size()<<"local map after clear is: "<<local_map->size()<<endl;
      //ground filter
      detectObjectsOnCloud(cloud_out,cloud_out);
      // Set up a Normal Estimation class and merge data in cloud_with_normals
      pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
      pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
      ne.setInputCloud (cloud_out);
      ne.setSearchMethod (search_tree);
      ne.setRadiusSearch (0.5);
      ne.compute (*cloud_with_normals);

      pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_es;
      normal_es.setInputCloud(cloud_out);
      normal_es.setSearchMethod(search_tree);
      normal_es.setRadiusSearch(0.5);
      normal_es.compute(*normals);
      // Set up a Conditional Euclidean Clustering class
      pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
      cec.setInputCloud (cloud_with_normals);
      cec.setConditionFunction (&customRegionGrowing);

      cec.setClusterTolerance (cluster_tolerance);//<0.5, evaluated by setConditionFunction
      cec.setMinClusterSize (mincluster_size);//min number
      cec.setMaxClusterSize (maxcluster_size);
      cec.segment (*clusters);
      cec.getRemovedClusters (small_clusters, large_clusters);
      //vfh compute
      //cv::Mat descriptor =cv::Mat::zeros(8,308,CV_8U);//clusters->size()
      cout<<"small: "<<small_clusters->size()<<"large: "<<large_clusters->size()<<"normal:"<<clusters->size()<<endl;
      cv::Mat_<float> descriptor(clusters->size(),308);
      flann::Matrix<float> flanndata_descriptor(new float[clusters->size()*308],clusters->size(),308);
      for(size_t k=0;k<clusters->size();++k)
      {
          boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> ((*clusters)[k].indices));
          pcl::VFHEstimation<pcl::PointXYZI,pcl::Normal,pcl::VFHSignature308> vfh;
          //output datasets
          pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_out (new pcl::PointCloud<pcl::VFHSignature308>());
          vfh.setInputCloud(cloud_out);
          vfh.setIndices(indicesptr);
          vfh.setInputNormals(normals);
          vfh.setSearchMethod(search_tree);
          vfh.compute(*vfh_out);
          //cv::Mat_<float> descriptor(1,308);
          //cv::Mat descriptor(1,308,CV_32F);//,vfh_out->points[0].histogram);
          for(size_t l=0;l<308;l++)
          {
              descriptor(k,l)=vfh_out->points[0].histogram[l];
              flanndata_descriptor[k][l]=vfh_out->points[0].histogram[l];
          }
      }
      descripors.push_back(descriptor);
      flann::save_to_file(flanndata_descriptor,surface_descriptors_path,"keyframe"+to_string(key_frame_index));
      // Using the intensity channel for lazy visualization of the output
      for (size_t i = 0; i < small_clusters->size (); ++i)
        for (size_t j = 0; j < (*small_clusters)[i].indices.size (); ++j)
          (*cloud_out)[(*small_clusters)[i].indices[j]].intensity = -2.0;
      for (size_t i = 0; i < large_clusters->size (); ++i)
        for (size_t j = 0; j < (*large_clusters)[i].indices.size (); ++j)
          (*cloud_out)[(*large_clusters)[i].indices[j]].intensity = -1.0;
      for (size_t i = 0; i < clusters->size (); ++i)
      {
        int label =i;
        for (size_t j = 0; j < (*clusters)[i].indices.size (); ++j)
          (*cloud_out)[(*clusters)[i].indices[j]].intensity = label;
      }
      if(if_save_clusters)
        writer.write(clusters_saved_path+to_string(key_frame_index)+".pcd",*cloud_out);

      localmap_index=0;
      if(key_frame_index==key_frame_num)
      {
        cerr<<"key frame num is: "<<key_frame_index<<endl;
        //create vocabulary
        DBoW3::Vocabulary vocab;
        cerr<<descripors.size()<<endl;
        vocab.create(descripors);
        //compare with database
        DBoW3::Database db(vocab,false,0);
        for(size_t i=0;i<descripors.size();i++)
        {
            db.add(descripors[i]);
        }
        for(size_t i=0;i<descripors.size();i++)
        {
            DBoW3::QueryResults ret;
            db.query(descripors[i],ret,4);
            cerr<<ret<<endl<<endl;
        }
        cerr<<vocab.size()<<endl;
        vocab.save(surface_vocab_path);
        cerr<<vocab<<endl;
      }
    }
  }
  void dictionary_generate()
  {
      pcl::PCDReader reader;
      pcl::PCDWriter writer;
      vector<cv::Mat> descripors;
      for(size_t i=1;i<=key_frame_num;i++)
      {
          pcl::PointCloud<PointTypeIO>::Ptr cloud_1 (new pcl::PointCloud<PointTypeIO>), cloud_in (new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
          pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>());
          pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
          pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
          pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);
          for(size_t j=100*(i-1)+1;j<=i*100;j++)
          {
              reader.read(surf_frame_path+to_string(j)+".pcd",*cloud_1);
              *cloud_in=*cloud_in+*cloud_1;
          }
          cerr<<i<<" cloud_in size: "<<cloud_in->size()<<"cloud_1 size: "<<cloud_1->size()<<endl;
          // Downsample the cloud using a Voxel Grid class
          pcl::VoxelGrid<PointTypeIO> vg;
          vg.setInputCloud (cloud_in);
          vg.setLeafSize (leaf_size, leaf_size, leaf_size);
          vg.setDownsampleAllData (true);
          vg.filter (*cloud_out);
          cerr<<"poins of cloud_in is:"<<cloud_in->size()<<"points of cloud_out is:"<<cloud_out->size()<<endl;
          //ground filter
          detectObjectsOnCloud(cloud_out,cloud_out);
          // Set up a Normal Estimation class and merge data in cloud_with_normals
          pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
          pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
          ne.setInputCloud (cloud_out);
          ne.setSearchMethod (search_tree);
          ne.setRadiusSearch (0.5);
          ne.compute (*cloud_with_normals);

          pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_es;
          normal_es.setInputCloud(cloud_out);
          normal_es.setSearchMethod(search_tree);
          normal_es.setRadiusSearch(0.5);
          normal_es.compute(*normals);
          // Set up a Conditional Euclidean Clustering class
          pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
          cec.setInputCloud (cloud_with_normals);
          cec.setConditionFunction (&customRegionGrowing);

          cec.setClusterTolerance (cluster_tolerance);//<0.5, evaluated by setConditionFunction
          cec.setMinClusterSize (mincluster_size);//min number
          cec.setMaxClusterSize (maxcluster_size);
          cec.segment (*clusters);
          cec.getRemovedClusters (small_clusters, large_clusters);
          //vfh compute
          //cv::Mat descriptor =cv::Mat::zeros(8,308,CV_8U);//clusters->size()
          cout<<"small: "<<small_clusters->size()<<"large: "<<large_clusters->size()<<"normal:"<<clusters->size()<<endl;
          cv::Mat_<float> descriptor(clusters->size(),308);
          flann::Matrix<float> flanndata_descriptor(new float[clusters->size()*308],clusters->size(),308);
          for(size_t k=0;k<clusters->size();++k)
          {
              boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> ((*clusters)[k].indices));

              pcl::VFHEstimation<pcl::PointXYZI,pcl::Normal,pcl::VFHSignature308> vfh;
              //output datasets
              pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_out (new pcl::PointCloud<pcl::VFHSignature308>());
              vfh.setInputCloud(cloud_out);
              vfh.setIndices(indicesptr);
              vfh.setInputNormals(normals);
              vfh.setSearchMethod(search_tree);
              vfh.compute(*vfh_out);
              //cv::Mat_<float> descriptor(1,308);

              //cv::Mat descriptor(1,308,CV_32F);//,vfh_out->points[0].histogram);
              for(size_t l=0;l<308;l++)
              {
                  descriptor(k,l)=vfh_out->points[0].histogram[l];
                  flanndata_descriptor[k][l]=vfh_out->points[0].histogram[l];
              }
          }
          descripors.push_back(descriptor);
          flann::save_to_file(flanndata_descriptor,surface_descriptors_path,"keyframe"+to_string(i));
          // Using the intensity channel for lazy visualization of the output
          for (size_t i = 0; i < small_clusters->size (); ++i)
            for (size_t j = 0; j < (*small_clusters)[i].indices.size (); ++j)
              (*cloud_out)[(*small_clusters)[i].indices[j]].intensity = -2.0;
          for (size_t i = 0; i < large_clusters->size (); ++i)
            for (size_t j = 0; j < (*large_clusters)[i].indices.size (); ++j)
              (*cloud_out)[(*large_clusters)[i].indices[j]].intensity = -1.0;
          for (size_t i = 0; i < clusters->size (); ++i)
          {
            int label =i;
            for (size_t j = 0; j < (*clusters)[i].indices.size (); ++j)
              (*cloud_out)[(*clusters)[i].indices[j]].intensity = label;
          }
          if(if_save_clusters)
            writer.write(clusters_saved_path+to_string(i)+".pcd",*cloud_out);
      }

      //create vocabulary
      DBoW3::Vocabulary vocab;
      cerr<<descripors.size()<<endl;
      vocab.create(descripors);
      //compare with database
      DBoW3::Database db(vocab,false,0);
      for(size_t i=0;i<descripors.size();i++)
      {
          db.add(descripors[i]);
      }
      for(size_t i=0;i<descripors.size();i++)
      {
          DBoW3::QueryResults ret;
          db.query(descripors[i],ret,4);
          cerr<<ret<<endl<<endl;
      }
      cerr<<vocab.size()<<endl;
      vocab.save(surface_vocab_path);
      cerr<<vocab<<endl;
  }
  void create()
  {
    cerr<<"creation begin"<<endl;
    if(if_map_create)
    {
      cerr<<"creating map"<<endl;
      map_create();
    }
    else if(if_transform_create)
    {
      cerr<<"creating transformation"<<endl;
      transform_creator();
    }
    else if(if_dictionary_generate)
    {
      cerr<<"creating dictionary"<<endl;
      dictionary_generate();
    }
    else
    {
      cerr<<"nothing create"<<endl;
    }
  }
  void cornermap_creation(const sensor_msgs::PointCloud2ConstPtr& input)
  {
    //cerr<<"corner frame index is: "<<cornerframe_index<<"frame num is: "<<frame_num<<endl;
    if(cornerframe_index<=frame_num)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr corner_frame=boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
      pcl::fromROSMsg(*input,*corner_frame);
      *cornermap=*cornermap+*corner_frame;
      if(cornerframe_index==frame_num)
      {
        cerr<<"corner map before down sampled is: "<<cornermap->size()<<endl;
        pcl::VoxelGrid<PointTypeIO> corner_sample;
        corner_sample.setInputCloud (cornermap);
        corner_sample.setLeafSize (leaf_size, leaf_size, leaf_size);
        //corner_sample.setDownsampleAllData (true);
        corner_sample.filter (*cornermap);
        cerr<<"corner map after down sampled is: "<<cornermap->size()<<endl;
        writer.write(corner_map_saved_path,*cornermap);
      }
    }
  }
  void surfacemap_creation(const sensor_msgs::PointCloud2ConstPtr& input)
  {
    cerr<<"surface frame index is: "<<surfaceframe_index<<"frame num is: "<<frame_num<<endl;
    if(surfaceframe_index<=frame_num)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr surface_frame=boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
      pcl::fromROSMsg(*input,*surface_frame);
      *surfacemap=*surfacemap+*surface_frame;
      if(surfaceframe_index==frame_num)
      {
//        cerr<<"surface map before down sampled is: "<<surfacemap->size()<<endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr part1 (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr part2 (new pcl::PointCloud<pcl::PointXYZI>);
        pass.setInputCloud(surfacemap);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-5,15);
        pass.filter(*part1);
        cerr<<"part1 points num is: "<<part1->points.size()<<endl;
        pass.setFilterLimitsNegative(true);
        pass.filter(*part2);
        cerr<<"part2 points num is: "<<part2->points.size()<<endl;

        pcl::VoxelGrid<PointTypeIO> surface_sample;
        surface_sample.setInputCloud (part1);
        surface_sample.setLeafSize (leaf_size, leaf_size, leaf_size);
        //surface_sample.setDownsampleAllData (true);
        surface_sample.filter (*part1);

        surface_sample.setInputCloud (part2);
        surface_sample.setLeafSize (leaf_size, leaf_size, leaf_size);
        //surface_sample.setDownsampleAllData (true);
        surface_sample.filter (*part2);

        *surfacemap=*part1+*part2;
        cerr<<"surface map after down sampled is: "<<surfacemap->size()<<endl;
        writer.write(surface_map_saved_path,*surfacemap);
      }
    }
  }
  void corner_aligned_callback(const sensor_msgs::PointCloud2ConstPtr& input)
  {
    cornerframe_index++;
    if(if_map_create)
    {
      cornermap_creation(input);
    }
  }
void surface_aligned_callback(const sensor_msgs::PointCloud2ConstPtr& input)
{
  surfaceframe_index++;
  if(if_map_create)
  {
    surfacemap_creation(input);
  }
  if(if_dictionary_generate)
  {
    dictionary_create(input);
  }
}
  Map_creator()
  {
    //init parameters
    nh.param<std::string>("/corner_frame_path",corner_frame_path,"/home/beihai/data/pcd/playground/corner_aligned/");
    nh.param<std::string>("/surf_frame_path",surf_frame_path,"/home/beihai/data/pcd/playground/surf_aligned/");
    nh.param<std::string>("/transformation_path",transformation_path,"/home/beihai/data/pcd/playground/transform.txt");
    nh.param<std::string>("/transformation_saved_path",transformation_saved_path,"/home/beihai/data/vocab/playground/transform_data.h5");
    nh.param<std::string>("/surface_descriptors_path",surface_descriptors_path,"/home/beihai/data/vocab/playground/surface_descriptors.h5");
    nh.param<std::string>("/surface_vocab_path",surface_vocab_path,"/home/beihai/data/vocab/playground/surface_vocabulary.yml.gz");
    nh.param<std::string>("/corner_map_saved_path",corner_map_saved_path,"/home/beihai/data/pcd/playground/map/corner_map.pcd");
    nh.param<std::string>("/surface_map_saved_path",surface_map_saved_path,"/home/beihai/data/pcd/playground/map/surface_map.pcd");
    nh.param<std::string>("/clusters_saved_path",clusters_saved_path,"default value");

    nh.param<int>("map_creation/frame_num",frame_num,2232);
    nh.param<int>("map_creation/key_frame_num",key_frame_num,22);
    nh.param<int>("map_creation/if_save_clusters",if_save_clusters,0);
    nh.param<int>("map_creation/if_map_create",if_map_create,0);
    nh.param<int>("map_creation/if_transform_create",if_transform_create,0);
    nh.param<int>("map_creation/if_dictionary_generate",if_dictionary_generate,0);
    nh.param<int>("map_creation/mincluster_size",mincluster_size,30);
    nh.param<int>("map_creation/maxcluster_size",maxcluster_size,50000);

    nh.param<float>("map_creation/mycluster_distance_threshold",mycluster_distance_threshold,0.5);
    cluster_distance_threshold=mycluster_distance_threshold;
    nh.param<double>("map_creation/angle_threshold",angle_threshold,0.15);
    nh.param<double>("map_creation/distance_threshold",distance_threshold,0.15);
    nh.param<double>("map_creation/leaf_size",leaf_size,0.1);
    nh.param<double>("map_creation/cluster_tolerance",cluster_tolerance,0.5);
    corner_aligned=nh.subscribe<sensor_msgs::PointCloud2>("cornerstack_aligned",10000,&Map_creator::corner_aligned_callback,this);
    surface_aligned=nh.subscribe<sensor_msgs::PointCloud2>("surfacestack_aligned",10000,&Map_creator::surface_aligned_callback,this);

    if(if_transform_create)
    {
      cerr<<"creating transformation"<<endl;
      transform_creator();
      cerr<<"creating transformation complete"<<endl;

    }
  }
};
