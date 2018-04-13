#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <iostream>
#include <string>
#include <stdlib.h>

using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 
const double camera_factor = 1000;

cv::Mat downSampleDepth(cv::Mat dpt);

int main( int argc, char** argv )
{
    double camera_cx = 326.2449951171875;
    double camera_cy = 239.97451782226562;
    double camera_fx = 617.3065795898438;
    double camera_fy = 617.7140502929688;  

    const float bad_point = std::numeric_limits<float>::quiet_NaN();


    cv::Mat rgb_raw, depth_raw;
    rgb_raw = cv::imread( argv[1] );
    depth_raw = cv::imread( argv[2], -1 );
    


    cv::Mat depth;
    depth = depth_raw.clone();
    // depth = downSampleDepth(depth_raw);
    cv::Mat rgb(depth.size(), CV_8UC3); 

    rgb = rgb_raw.clone();

    PointCloud::Ptr cloud ( new PointCloud );
    int skip = 2;
    int su = 0; int eu = depth.cols; 
    int sv = 0; int ev = depth.rows; 

    for (int m = sv; m < ev; m+=skip){
      for (int n=su; n < eu; n+=skip)
        {
            ushort d = depth.ptr<ushort>(m)[n];


            if (d > 4000 || d < 200 )
                continue;
            if (d == 0)
                continue;
            PointT p;

            p.z = double(d) / camera_factor;
            p.x = (n - camera_cx) * p.z / camera_fx;
            p.y = (m - camera_cy) * p.z / camera_fy;
            
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // if (d > 600 || d < 200 )
            //     p.x = p.y = p.z = bad_point;
            cloud->points.push_back( p );
        }
    }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cout<<"point cloud size = "<<cloud->points.size()<<endl;
    cloud->is_dense = false;

    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ()){}
    pcl::io::savePCDFile( "/home/jin/Desktop/pointcloud.pcd", *cloud );
    
    // 清除数据并退出
    cloud->points.clear();
    cout<<"Point cloud saved."<<endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/home/jin/Desktop/pointcloud.pcd", *cloud1) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    int mnThPoints = 70;
    double mSQ_dis_th = 0.03*0.03;

    std::cerr << "Point cloud data: " << cloud1->points.size () << " points" << std::endl;

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud1);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size () == 0)
    {
    PCL_ERROR ("Could not estimate a planar model for the given dataset.");
    return (-1);
    }

    std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;

    std::cerr << "Model inliers: " << inliers->indices.size () << std::endl;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), cloud_p (new pcl::PointCloud<pcl::PointXYZ>);

    extract.setInputCloud (cloud1);
    extract.setIndices (inliers);
    extract.setNegative (false);
    extract.filter (*cloud_p);

    pcl::visualization::CloudViewer viewer2 ("filtered Cloud Viewer");
    viewer2.showCloud (cloud_p);
    while (!viewer2.wasStopped ()){}

    return 0;
}


cv::Mat downSampleDepth(cv::Mat dpt)
{
  cv::Mat dD = cv::Mat(dpt.rows/2, dpt.cols/2, CV_16UC1); 
  
  int pu, pv;
  int near_th = 200; // < near_th is not valid  
  for(int row = 0; row < dD.rows; ++row)
    for(int col = 0; col < dD.cols; ++col)
    {
      ushort sum = 0; 
      // get meadian value of its neighbors 
      pv = row * 2; 
      pu = col * 2; 
      
      int n = 0; 
      if(dpt.at<ushort>(pv, pu) >= near_th)
      {
        sum += dpt.at<ushort>(pv, pu); 
        ++n; 
      }

      if(pv < dpt.rows - 1)
      {
        if(dpt.at<ushort>(pv+1, pu) >= near_th)
        {
          sum += dpt.at<ushort>(pv+1, pu); 
          ++n; 
        }
      }
      
      if(pu < dpt.cols - 1)
      {
        if(dpt.at<ushort>(pv, pu+1) >= near_th)
        {
          sum += dpt.at<ushort>(pv, pu+1); 
          ++n; 
        }
      }

      if(pv < dpt.rows -1 && pu < dpt.cols -1)
      {
        if(dpt.at<ushort>(pv+1, pu+1) >= near_th)
        {
          sum += dpt.at<ushort>(pv+1, pu+1); 
          ++n; 
        }
      }
      
      // mean value
      if(n == 0)
        dD.at<ushort>(row, col) = 0; 
      else
        dD.at<ushort>(row, col) = sum/n; 
    }
  return dD; 
}


