#include <ros/ros.h>
#include "observation.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include "octree_manager.h"
#include "robot_controller.h"

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

  bool evaluate_results = nhp.param<bool>("evaluate_results", false);

  tf2_ros::Buffer tfBuffer(ros::Duration(30));
  tf2_ros::TransformListener tfListener(tfBuffer);

  OctreeManager oc_manager(nh, tfBuffer, map_frame, tree_resolution, evaluate_results);
  RobotController controller(nh, tfBuffer, map_frame);
  controller.reset();
  oc_manager.resetOctomap();

  const size_t NUM_SAMPLES_TO_COLLECT = 10;
  const ros::Duration MOVE_TIMEOUT = ros::Duration(60);

  for (size_t i=0; i<NUM_SAMPLES_TO_COLLECT; i++)
  {
    bool moved = controller.moveToRandomTarget(false, MOVE_TIMEOUT);
    if (!moved)
    {
      ROS_WARN("Robot did not move to random position successfully");
    }
    oc_manager.saveOctomap("octree_" + std::to_string(i), false);

    capnp::MallocMessageBuilder builder;
    vpp_msg::Pointcloud::Builder pc = builder.initRoot<vpp_msg::Pointcloud>();
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
    capnp::writeMessageToFd(fd, builder);
    close(fd);
  }

  ROS_INFO("Finished collecting data");
}
