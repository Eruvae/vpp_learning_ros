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
#include <roi_viewpoint_planner/gt_octree_loader.h>
#include <roi_viewpoint_planner/evaluator.h>

class OctreeManager
{
private:
  tf2_ros::Buffer &tfBuffer;
  std::shared_ptr<octomap_vpp::RoiOcTree> planningTree;
  std::shared_ptr<roi_viewpoint_planner::GtOctreeLoader> gtLoader;
  std::unique_ptr<roi_viewpoint_planner::Evaluator> evaluator;
  boost::mutex tree_mtx;
  const std::string map_frame;
  ros::Publisher octomapPub;
  ros::Subscriber roiSub;
  size_t old_rois;
  octomap::KeySet encountered_keys;

  // Evaluator variables
  size_t eval_trial_num;
  std::ofstream eval_resultsFile;
  std::ofstream eval_resultsFileOld;
  std::ofstream eval_fruitCellPercFile;
  std::ofstream eval_volumeAccuracyFile;
  std::ofstream eval_distanceFile;
  ros::Time eval_plannerStartTime;
  double eval_accumulatedPlanDuration;
  double eval_accumulatedPlanLength;
  std::string eval_lastStep;

  void registerPointcloudWithRoi(const ros::MessageEvent<pointcloud_roi_msgs::PointcloudWithRoi const> &event);

public:
  OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &map_frame, double tree_resolution, bool initialize_evaluator=false);

  std::string saveOctomap(const std::string &name = "planningTree", bool name_is_prefix = true);

  int loadOctomap(const std::string &filename);

  void resetOctomap();

  void randomizePlants(const geometry_msgs::Point &min, const geometry_msgs::Point &max, double min_dist);

  void publishMap();

  void fillObservation(vpp_msg::Observation::Builder &obs, const octomap::pose6d &viewpoint, size_t theta_steps, size_t phi_steps, size_t layers, double range);

  uint32_t getReward();

  uint32_t getRewardWithGt();

  uint32_t getMaxGtReward();

  bool startEvaluator();
  void setEvaluatorStartParams();
  bool saveEvaluatorData(double plan_length, double traj_duration);
  bool resetEvaluator();
};

#endif // OCTREE_MANAGER_H
