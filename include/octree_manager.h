#ifndef OCTREE_MANAGER_H
#define OCTREE_MANAGER_H

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <ros/message_event.h>
#include <pointcloud_roi_msgs/PointcloudWithRoi.h>
#include <octomap_vpp/RoiOcTree.h>
#include <octomap_msgs/Octomap.h>
#include "octomap_vpp/roioctree_utils.h"
#include "observation.capnp.h"

class OctreeManager
{
private:
  tf2_ros::Buffer &tfBuffer;
  std::shared_ptr<octomap_vpp::RoiOcTree> planningTree;
  boost::mutex tree_mtx;
  const std::string map_frame;
  ros::Publisher octomapPub;
  ros::Subscriber roiSub;
  size_t old_rois;

  void registerPointcloudWithRoi(const ros::MessageEvent<pointcloud_roi_msgs::PointcloudWithRoi const> &event);

public:
  OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &map_frame, double tree_resolution);

  std::string saveOctomap(const std::string &name = "planningTree", bool name_is_prefix = true);

  int loadOctomap(const std::string &filename);

  void resetOctomap();

  void publishMap();

  void fillObservation(vpp_msg::Observation::Builder &obs, const octomap::pose6d &viewpoint, size_t theta_steps, size_t phi_steps, size_t layers, double range);

  uint32_t getReward();
};

#endif // OCTREE_MANAGER_H
