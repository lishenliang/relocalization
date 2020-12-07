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
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include<pcl/features/vfh.h>
#include<pcl/visualization/pcl_plotter.h>
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;
using namespace std;

bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance);
void detectObjectsOnCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_filtered);
void get_vfh(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,cv::Mat_<float> & descriptor);
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
        seg.setEpsAngle(0.15);//max angle between z axis
        // you can modify the parameter below
            seg.setMaxIterations(10000);
        seg.setDistanceThreshold(0.15);//distance threshold between inliers and model plane
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
void store_transform(){
    vector<double> V;
    vector<double>::iterator it;
    ifstream data("ground_truth.txt");
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
//    //save the model and its transform information
    flann::save_to_file(transform_data,"transform_data.h5","transform_data");
    //cout << "V[" << i << "]="<< setprecision(16) << *it << endl;
//    for(it = V.begin();it != V.end();it++)
//    {
//    }
}
void test_data_generete()
{
    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    string source_Path = "/home/beihai/catkin_ws/src/loam_livox/pcd/surface/";
    for(size_t i=1;i<=100;i++)
    {
        pcl::PointCloud<PointTypeIO>::Ptr  cloud_in (new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
        for(size_t j=100*(i-1)+50;j<=i*100+49;j++)
        {
            reader.read(source_Path+to_string(j)+".pcd",*cloud_in);
            *cloud_out=*cloud_out+*cloud_in;
        }
        writer.write("./test_data/r50/test_"+to_string(i)+".pcd",*cloud_out);
    }

}
void dictionary_generate(size_t key_frame_num)
{
    string filename="surface_descriptors.h5";
    pcl::PCDReader reader;
    pcl::PCDWriter writer;
    vector<cv::Mat> descripors;    
    string dir_align_surf="/home/beihai/my_workspace/src/livox_aftermaped/pcd/surf_frame_align/";

    for(size_t i=1;i<=key_frame_num;i++)
    {
        pcl::PointCloud<PointTypeIO>::Ptr cloud_1 (new pcl::PointCloud<PointTypeIO>), cloud_in (new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
        pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>());
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
        pcl::search::KdTree<PointTypeIO>::Ptr search_tree (new pcl::search::KdTree<PointTypeIO>);

        for(size_t j=100*(i-1)+1;j<=i*100;j++)
        {
            reader.read(dir_align_surf+to_string(j)+".pcd",*cloud_1);
            *cloud_in=*cloud_in+*cloud_1;
        }
        cerr<<i<<" cloud_in size: "<<cloud_in->size()<<"cloud_1 size: "<<cloud_1->size()<<endl;
        // Downsample the cloud using a Voxel Grid class
        pcl::VoxelGrid<PointTypeIO> vg;
        vg.setInputCloud (cloud_in);
        vg.setLeafSize (0.1, 0.1, 0.1);
        vg.setDownsampleAllData (true);
        vg.filter (*cloud_out);
        cerr<<"poins of cloud_in is:"<<cloud_in->size()<<"points of cloud_out is:"<<cloud_out->size()<<endl;
        //writer.write("/home/beihai/build-qtpro-Desktop_Qt_5_14_0_GCC_64bit-Debug/ground/surface/"+to_string(i)+"_filtered"+".pcd",*cloud_out);
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
        cec.setClusterTolerance (0.5);//<0.5, evaluated by setConditionFunction
        cec.setMinClusterSize (30);//min number
        cec.setMaxClusterSize (50000);
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
        flann::save_to_file(flanndata_descriptor,filename,"keyframe"+to_string(i));
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
        //writer.write("/home/beihai/build-qtpro-Desktop_Qt_5_14_0_GCC_64bit-Debug/segmented/surface/"+to_string(i)+"_filtered"+".pcd",*cloud_out);
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
    vocab.save("surface_vocabulary.yml.gz");
    cerr<<vocab<<endl;
}
void initial_find(pcl::PointCloud< pcl::PointXYZI >::Ptr local_map,
                  DBoW3::Database & database,
                  flann::Matrix<float> data_trans,
                  vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> & trans_vectors
                  )
{
//  cv::Mat_<float> localmap_descriptor;
//  get_vfh(local_map,localmap_descriptor);
  Eigen::Matrix4f initial_trans = Eigen::Matrix4f::Identity ();
  pcl::PointCloud<PointTypeFull>::Ptr cloud_in_with_normals (new pcl::PointCloud<PointTypeFull>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_in_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<PointTypeIO>::Ptr cloud_in_search_tree (new pcl::search::KdTree<PointTypeIO>);
  pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
  // Downsample the cloud using a Voxel Grid class
  pcl::VoxelGrid<PointTypeIO> vg;
  cerr<<"poins of cloud_in is:"<<local_map->size();
  vg.setInputCloud (local_map);
  vg.setLeafSize (0.1, 0.1, 0.1);
  vg.setDownsampleAllData (true);
  vg.filter (*local_map);
  cerr<<" points of cloud_out is:"<<local_map->size()<<endl;
  //ground filter
  detectObjectsOnCloud(local_map,local_map);
  // Set up a Normal Estimation class and merge data in cloud_with_normals
  pcl::copyPointCloud (*local_map, *cloud_in_with_normals);
  pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
  ne.setInputCloud (local_map);
  ne.setSearchMethod (cloud_in_search_tree);
  ne.setRadiusSearch (0.5);
  ne.compute (*cloud_in_with_normals);

  pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_es;
  normal_es.setInputCloud(local_map);
  normal_es.setSearchMethod(cloud_in_search_tree);
  normal_es.setRadiusSearch(0.5);
  normal_es.compute(*cloud_in_normals);
  // Set up a Conditional Euclidean Clustering class
  pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
  cec.setInputCloud (cloud_in_with_normals);
  cec.setConditionFunction (&customRegionGrowing);
  cec.setClusterTolerance (0.5);//<0.5, evaluated by setConditionFunction
  cec.setMinClusterSize (30);//min number
  cec.setMaxClusterSize (50000);
  cec.segment (*clusters);
  cec.getRemovedClusters (small_clusters, large_clusters);
  cerr<<"final localmap size is: "<<local_map->size()<<endl;
  //vfh compute
  cout<<"small: "<<small_clusters->size()<<"large: "<<large_clusters->size()<<"normal:"<<clusters->size()<<endl;
  cv::Mat_<float> descriptor(clusters->size(),308);
  flann::Matrix<float> flanndata_descriptor(new float[clusters->size()*308],clusters->size(),308);
  for(size_t k=0;k<clusters->size();++k)
  {
      boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> ((*clusters)[k].indices));
      pcl::VFHEstimation<pcl::PointXYZI,pcl::Normal,pcl::VFHSignature308> vfh;
      //output datasets
      pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_out (new pcl::PointCloud<pcl::VFHSignature308>());
      vfh.setInputCloud(local_map);
      vfh.setIndices(indicesptr);
      vfh.setInputNormals(cloud_in_normals);
      vfh.setSearchMethod(cloud_in_search_tree);
      vfh.compute(*vfh_out);
      for(size_t l=0;l<308;l++)
      {
          descriptor(k,l)=vfh_out->points[0].histogram[l];
          flanndata_descriptor[k][l]=vfh_out->points[0].histogram[l];
      }
  }
  DBoW3::QueryResults ret;
  cerr<<"database size: "<<database.size()<<" data trans size:"<<data_trans.rows<<endl;
  database.query(descriptor,ret,4);
  cout<<ret<<endl<<endl;
  for(size_t j=0;j<ret.size();j++)
  {
    Eigen::Quaternionf quarter=Eigen::Quaternionf(data_trans[100*(ret[j].Id+1)][7],data_trans[100*(ret[j].Id+1)][4], data_trans[100*(ret[j].Id+1)][5],data_trans[100*(ret[j].Id+1)][6]).normalized();
    Eigen::Translation3f translation(data_trans[100*(ret[j].Id+1)][1], data_trans[100*(ret[j].Id+1)][2], data_trans[100*(ret[j].Id+1)][3]);
    Eigen::Affine3f T=translation*quarter.toRotationMatrix();
    trans_vectors.push_back(T.matrix());
//    cout<<data_trans[100*(ret[j].Id+1)][1]<<" "<<data_trans[100*(ret[j].Id+1)][2]<<" "<<data_trans[100*(ret[j].Id+1)][3]<<endl;
  }
}
void database_initialize(DBoW3::Vocabulary & vocab, DBoW3::Database & database,int key_frame_num,string vocab_path,string descriptors_path)
{
  //vocab.load("/home/beihai/ros_in_qt/src/find_transformation/playground/vocab/playground2_surface_vocabulary.yml.gz");
  vocab.load(vocab_path);
  database.setVocabulary(vocab,false,0);
  for(size_t i=1;i<=key_frame_num;i++)
  {
      flann::Matrix<float> surface_descriptor;
      //flann::load_from_file(surface_descriptor,"/home/beihai/ros_in_qt/src/find_transformation/playground/vocab/playground2_surface_descriptors.h5","keyframe"+to_string(i));
      flann::load_from_file(surface_descriptor,descriptors_path,"keyframe"+to_string(i));
      cv::Mat_<float> descriptor(surface_descriptor.rows,308);
      for(size_t j=0;j<surface_descriptor.rows;j++)
      {
          for(size_t k=0;k<308;k++)
          {
              descriptor[j][k]=surface_descriptor[j][k];
          }
      }
      database.add(descriptor);
  }
}
void test_initial_find()
{
    DBoW3::Vocabulary vocab("/home/beihai/my_workspace/src/livox_aftermaped/vocab/surface_vocabulary.yml.gz");
    DBoW3::Database db(vocab,false,0);
    flann::Matrix<float> data_trans;
    flann::load_from_file(data_trans,"/home/beihai/my_workspace/src/livox_aftermaped/vocab/transform_data.h5","transform_data");
    if(vocab.empty())
        cerr<<"load failed"<<endl;
    for(size_t i=1;i<=136;i++)
    {
        flann::Matrix<float> surface_descriptor;
        flann::load_from_file(surface_descriptor,"/home/beihai/my_workspace/src/livox_aftermaped/vocab/surface_descriptors.h5","keyframe"+to_string(i));
        cv::Mat_<float> descriptor(surface_descriptor.rows,308);
        for(size_t j=0;j<surface_descriptor.rows;j++)
        {
            for(size_t k=0;k<308;k++)
            {
                descriptor[j][k]=surface_descriptor[j][k];
            }
        }
        db.add(descriptor);
    }
    cout<<"db size is: "<<db.size()<<" vocab size is: "<<vocab.size()<<endl;
    DBoW3::QueryResults ret;
    flann::Matrix<float> test_descriptor;
//    flann::load_from_file(test_descriptor,"./vocab/surface_30_0.5/surface_descriptors.h5","keyframe10");
//    cv::Mat_<float> test_surfacedescriptor(test_descriptor.rows,308);
//    for(size_t j=0;j<test_descriptor.rows;j++)
//    {
//        for(size_t k=0;k<308;k++)
//        {
//            test_surfacedescriptor[j][k]=test_descriptor[j][k];
//        }
//    }
    pcl::PointCloud<PointTypeIO>::Ptr cloud_in (new pcl::PointCloud<PointTypeIO>);
    pcl::PCDReader reader;
    for(size_t i=1;i<10;i++)
    {
        reader.read("/home/beihai/my_workspace/src/livox_aftermaped/test_data/r50/test_"+to_string(i)+".pcd",*cloud_in);
        cv::Mat_<float> test_surfacedescriptor;
        get_vfh(cloud_in,test_surfacedescriptor);
        db.query(test_surfacedescriptor,ret,4);
        DBoW3::BowVector vect;
        vocab.transform(test_surfacedescriptor,vect);
        cout<<"num: "<<i<<"vect: "<<vect<<endl;
        cout<<ret<<endl<<endl;
        for(size_t j=0;j<ret.size();j++)
        {
            cout<<data_trans[100*(ret[j].Id+1)][1]<<" "<<data_trans[100*(ret[j].Id+1)][2]<<" "<<data_trans[100*(ret[j].Id+1)][3]<<endl;
        }
    }

}
bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  if (std::abs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  else
    return (false);
}

bool
enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (std::abs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  if (std::abs (point_a_normal.dot (point_b_normal)) < 0.05)
    return (true);
  return (false);
}

bool
customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (squared_distance < 0.5)
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
void get_vfh(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,cv::Mat_<float> & descriptor)
{
    pcl::PointCloud<PointTypeFull>::Ptr cloud_in_with_normals (new pcl::PointCloud<PointTypeFull>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_in_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<PointTypeIO>::Ptr cloud_in_search_tree (new pcl::search::KdTree<PointTypeIO>);
    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
    // Downsample the cloud using a Voxel Grid class
    pcl::VoxelGrid<PointTypeIO> vg;
    cerr<<"poins of cloud_in is:"<<cloud_in->size();
    vg.setInputCloud (cloud_in);
    vg.setLeafSize (0.1, 0.1, 0.1);
    vg.setDownsampleAllData (true);
    vg.filter (*cloud_in);
    cerr<<" points of cloud_out is:"<<cloud_in->size()<<endl;
    //ground filter
    detectObjectsOnCloud(cloud_in,cloud_in);
    // Set up a Normal Estimation class and merge data in cloud_with_normals
    pcl::copyPointCloud (*cloud_in, *cloud_in_with_normals);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    ne.setInputCloud (cloud_in);
    ne.setSearchMethod (cloud_in_search_tree);
    ne.setRadiusSearch (0.5);
    ne.compute (*cloud_in_with_normals);

    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_es;
    normal_es.setInputCloud(cloud_in);
    normal_es.setSearchMethod(cloud_in_search_tree);
    normal_es.setRadiusSearch(0.5);
    normal_es.compute(*cloud_in_normals);
    // Set up a Conditional Euclidean Clustering class
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
    cec.setInputCloud (cloud_in_with_normals);
    cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (0.5);//<0.5, evaluated by setConditionFunction
    cec.setMinClusterSize (30);//min number
    cec.setMaxClusterSize (50000);
    cec.segment (*clusters);
    cec.getRemovedClusters (small_clusters, large_clusters);
    //vfh compute
    cout<<"small: "<<small_clusters->size()<<"large: "<<large_clusters->size()<<"normal:"<<clusters->size()<<endl;
    //cv::Mat_<float> descriptor(clusters->size(),308);
    flann::Matrix<float> flanndata_descriptor(new float[clusters->size()*308],clusters->size(),308);
    for(size_t k=0;k<clusters->size();++k)
    {
        boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> ((*clusters)[k].indices));

        pcl::VFHEstimation<pcl::PointXYZI,pcl::Normal,pcl::VFHSignature308> vfh;
        //output datasets
        pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_out (new pcl::PointCloud<pcl::VFHSignature308>());
        vfh.setInputCloud(cloud_in);
        vfh.setIndices(indicesptr);
        vfh.setInputNormals(cloud_in_normals);
        vfh.setSearchMethod(cloud_in_search_tree);
        vfh.compute(*vfh_out);
        for(size_t l=0;l<308;l++)
        {
            descriptor(k,l)=vfh_out->points[0].histogram[l];
            flanndata_descriptor[k][l]=vfh_out->points[0].histogram[l];
        }
    }
}
