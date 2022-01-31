#ifndef OCTREE_MANAGER_H
#define OCTREE_MANAGER_H

#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <ros/message_event.h>
#include <pointcloud_roi_msgs/PointcloudWithRoi.h>
#include <octomap_vpp/RoiOcTree.h>
#include <octomap_vpp/WorkspaceOcTree.h>
#include <octomap_msgs/Octomap.h>
#include "octomap_vpp/roioctree_utils.h"
#include "observation.capnp.h"
#include "pointcloud.capnp.h"
#include "voxelgrid.capnp.h"
#include <roi_viewpoint_planner/gt_octree_loader.h>
#include <roi_viewpoint_planner/evaluator.h>

class OctreeManager
{
private:
  tf2_ros::Buffer &tfBuffer;
  std::shared_ptr<octomap_vpp::RoiOcTree> planningTree;
  std::shared_ptr<octomap_vpp::WorkspaceOcTree> workspaceTree;
  std::shared_ptr<octomap_vpp::WorkspaceOcTree> samplingTree;
  octomap::point3d wsMin, wsMax;
  octomap::point3d stMin, stMax;
  std::shared_ptr<roi_viewpoint_planner::GtOctreeLoader> gtLoader;
  std::unique_ptr<roi_viewpoint_planner::Evaluator> evaluator;
  boost::mutex own_mtx;
  boost::mutex &tree_mtx;
  const std::string map_frame;
  const std::string ws_frame;
  ros::Publisher octomapPub;
  ros::Publisher workspaceTreePub;
  ros::Publisher samplingTreePub;
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
  // Constructor to store own tree, subscribe to pointcloud roi
  OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &wstree_file, const std::string &sampling_tree_file,
                const std::string &map_frame, const std::string &ws_frame, double tree_resolution, bool initialize_evaluator=false);

  // Constructor to pass existing tree + mutex, e.g. from viewpoint planner
  OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &wstree_file, const std::string &sampling_tree_file,
                const std::string &map_frame, const std::string &ws_frame,
                const std::shared_ptr<octomap_vpp::RoiOcTree> &providedTree, boost::mutex &tree_mtx, bool initialize_evaluator=false);

  void initWorkspace(const std::string &wstree_file, const std::string &sampling_tree_file);

  octomap::point3d transformToMapFrame(const octomap::point3d &p);
  geometry_msgs::Pose transformToMapFrame(const geometry_msgs::Pose &p);

  octomap::point3d transformToWorkspace(const octomap::point3d &p);
  geometry_msgs::Pose transformToWorkspace(const geometry_msgs::Pose &p);

  std::string saveOctomap(const std::string &name = "planningTree", bool name_is_prefix = true);

  int loadOctomap(const std::string &filename);

  void resetOctomap();

  void randomizePlants(const geometry_msgs::Point &min, const geometry_msgs::Point &max, double min_dist);

  void publishMap();

  void fillCountMap(vpp_msg::Observation::Map::CountMap::Builder &cmap, const octomap::pose6d &viewpoint, size_t theta_steps, size_t phi_steps, size_t layers, double range);

  void generatePointcloud(vpp_msg::Pointcloud::Builder &pc);

  void generatePointcloud(vpp_msg::Pointcloud::Builder &pc, const octomap::point3d &center_point, uint16_t vx_cells);

  void generateVoxelgrid(vpp_msg::Voxelgrid::Builder &vx, const octomap::point3d &center_point, uint16_t vx_cells);

  void generateFullVoxelgrid(vpp_msg::Voxelgrid::Builder &vx);

  uint32_t getReward();

  uint32_t getRewardWithGt();

  uint32_t getMaxGtReward();

  std::vector<octomap::point3d> getRoiContours();
  std::vector<octomap::point3d> getOccContours();
  std::vector<octomap::point3d> getFrontiers();

  bool startEvaluator();
  void setEvaluatorStartParams();
  bool saveEvaluatorData(double plan_length, double traj_duration);
  bool resetEvaluator();
};

#endif // OCTREE_MANAGER_H
