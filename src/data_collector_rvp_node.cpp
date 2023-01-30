#include <ros/ros.h>
#include "observation.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include "octree_manager.h"
#include <roi_viewpoint_planner/viewpoint_planner.h>
#include <dynamic_reconfigure/server.h>
#include <roi_viewpoint_planner_msgs/PlannerConfig.h>

roi_viewpoint_planner::ViewpointPlanner *planner;

int main(int argc, char **argv)
{
  ros::init(argc, argv, "data_collector_node");
  ros::NodeHandle nh;
  ros::NodeHandle nhp("~");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  double tree_resolution = nh.param<double>("/roi_viewpoint_planner/tree_resolution", 0.01);
  std::string wstree_default_package = ros::package::getPath("ur_with_cam_gazebo");
  std::string wstree_file = nh.param<std::string>("/roi_viewpoint_planner/workspace_tree", wstree_default_package + "/workspace_trees/ur_with_cam/workspace_map.ot");
  std::string sampling_tree_file = nh.param<std::string>("/roi_viewpoint_planner/sampling_tree", wstree_default_package + "/workspace_trees/ur_with_cam/workspace_map.ot");
  std::string map_frame = nh.param<std::string>("/roi_viewpoint_planner/map_frame", "world");
  std::string ws_frame = nh.param<std::string>("/roi_viewpoint_planner/ws_frame", "arm_base_link");

  bool limit_pointcloud = nhp.param<bool>("limit_pointcloud", true);
  int vx_cells = nhp.param<int>("vx_cells", 128);

  planner = new roi_viewpoint_planner::ViewpointPlanner(nh, nhp, wstree_file, sampling_tree_file, tree_resolution, map_frame, ws_frame, true, false);

  planner->setMode(roi_viewpoint_planner::Planner_SAMPLE_AUTOMATIC);

  tf2_ros::Buffer tfBuffer(ros::Duration(30));
  tf2_ros::TransformListener tfListener(tfBuffer);

  OctreeManager oc_manager(nh, tfBuffer, wstree_file, sampling_tree_file, map_frame, ws_frame, planner->getPlanningTree(), planner->getTreeMutex(), false);

  const size_t NUM_SAMPLES_TO_COLLECT = 10;
  const size_t NUM_TRIES_TO_MOVE = 10;

  for (size_t i=0; i<NUM_SAMPLES_TO_COLLECT && ros::ok(); i++)
  {
    bool moved = false;
    for (size_t move_tries = 0; !moved && move_tries < NUM_TRIES_TO_MOVE && ros::ok(); move_tries++)
    {
      moved = planner->plannerLoopOnce();
    }
    if (!moved)
    {
      ROS_WARN("Robot did not move with RVP planner successfully");
    }
    oc_manager.saveOctomap("octree_" + std::to_string(i), false);

    // Get current camera pose
    geometry_msgs::TransformStamped cur_tf;
    try
    {
      cur_tf = tfBuffer.lookupTransform(map_frame, "camera_link", ros::Time(0));
    }
    catch (const tf2::TransformException &e)
    {
      ROS_ERROR_STREAM("Couldn't find transform to map frame: " << e.what());
    }
    octomap::pose6d cur_pose = octomap_vpp::transformToOctomath(cur_tf.transform);

    // Write Pointcloud
    capnp::MallocMessageBuilder pc_builder;
    vpp_msg::Pointcloud::Builder pc = pc_builder.initRoot<vpp_msg::Pointcloud>();
    if (limit_pointcloud)
      oc_manager.generatePointcloud(pc, cur_pose.trans(), vx_cells);
    else
      oc_manager.generatePointcloud(pc);

    const std::string pc_fname = "pointcloud_" + std::to_string(i) + ".cpc";
    int fd = open(pc_fname.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
    {
      ROS_WARN("Couldn't create pointcloud file");
      continue;
    }
    //void writePackedMessageToFd(int fd, MessageBuilder& builder)
    //void writeMessageToFd(int fd, MessageBuilder& builder)
    capnp::writeMessageToFd(fd, pc_builder);
    close(fd);

    // Write Voxelgrid
    capnp::MallocMessageBuilder vx_builder;
    vpp_msg::Voxelgrid::Builder vx = vx_builder.initRoot<vpp_msg::Voxelgrid>();
    oc_manager.generateFullVoxelgrid(vx);

    const std::string vx_fname = "voxelgrid_" + std::to_string(i) + ".cvx";
    fd = open(vx_fname.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0)
    {
      ROS_WARN("Couldn't create pointcloud file");
      continue;
    }
    //void writePackedMessageToFd(int fd, MessageBuilder& builder)
    //void writeMessageToFd(int fd, MessageBuilder& builder)
    capnp::writeMessageToFd(fd, vx_builder);
    close(fd);
  }

  ROS_INFO("Finished collecting data");
  delete planner;
}
