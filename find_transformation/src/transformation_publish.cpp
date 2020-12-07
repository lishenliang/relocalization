#include <ros/ros.h>
#include"transformation_calculate.hpp"
int main(int argc, char **argv)
{
  ros::init(argc, argv, "transformation_publish");
  Local_map localmap;
  ros::MultiThreadedSpinner spinner(4); // Use 4 threads
  spinner.spin(); // spin() will not return until the node has been shutdown
  return 0;
}
