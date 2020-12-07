#include <ros/ros.h>
#include"map_creator.hpp"
int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_creator");
  Map_creator map_creator;
  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();
}
