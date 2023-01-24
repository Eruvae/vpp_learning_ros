#include <ros/ros.h>
#include "observation.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include "octree_manager.h"
#include <roi_viewpoint_planner/viewpoint_planner.h>
#include <dynamic_reconfigure/server.h>
#include <roi_viewpoint_planner_msgs/PlannerConfig.h>

roi_viewpoint_planner::ViewpointPlanner *planner;
dynamic_reconfigure::Server<roi_viewpoint_planner::PlannerConfig> *config_server;
boost::recursive_mutex config_mutex;
roi_viewpoint_planner::PlannerConfig current_config;

void reconfigureCallback(roi_viewpoint_planner::PlannerConfig &config, uint32_t level)
{
  ROS_INFO_STREAM("Reconfigure callback called");
  if (level & (1 << 0)) // change mode
  {
    planner->mode = static_cast<roi_viewpoint_planner::ViewpointPlanner::PlannerMode>(config.mode);
    planner->roi_sample_mode = static_cast<roi_viewpoint_planner::ViewpointPlanner::PlannerMode>(config.auto_roi_sampling);
    planner->expl_sample_mode = static_cast<roi_viewpoint_planner::ViewpointPlanner::PlannerMode>(config.auto_expl_sampling);

    planner->roiMaxSamples = config.roi_max_samples;
    planner->roiUtil = static_cast<roi_viewpoint_planner::UtilityType>(config.roi_util);
    planner->explMaxSamples = config.expl_max_samples;
    planner->explUtil = static_cast<roi_viewpoint_planner::UtilityType>(config.expl_util);
  }
  if (level & (1 << 1)) // activate execution
  {
    planner->execute_plan = config.activate_execution;
  }
  if (level & (1 << 2)) // request execution confirmation
  {
    planner->require_execution_confirmation = config.require_execution_confirmation;
  }
  if (level & (1 << 3)) // minimum sensor range
  {
    planner->sensor_min_range = config.sensor_min_range;
    if (planner->sensor_max_range < planner->sensor_min_range && !(level & (1 << 4)))
    {
      planner->sensor_max_range = planner->sensor_min_range;
      config.sensor_max_range = planner->sensor_max_range;
    }
  }
  if (level & (1 << 4)) // maximum sensor range
  {
    planner->sensor_max_range = config.sensor_max_range;
    if (planner->sensor_min_range > planner->sensor_max_range)
    {
      planner->sensor_min_range = planner->sensor_max_range;
      config.sensor_min_range = planner->sensor_min_range;
    }
  }
  if (level & (1 << 5)) // insert_scan_if_not_moved
  {
    planner->insert_scan_if_not_moved = config.insert_scan_if_not_moved;
  }
  if (level & (1 << 7)) // insert_scan_while_moving
  {
    planner->insert_scan_while_moving = config.insert_scan_while_moving;
  }
  if (level & (1 << 9)) // wait_for_scan
  {
    planner->wait_for_scan = config.wait_for_scan;
  }
  if (level & (1 << 11)) // publish_planning_state
  {
    planner->publish_planning_state = config.publish_planning_state;
  }
  if (level & (1 << 12)) // planner
  {
    planner->getMotionManager()->setPlannerId(config.planner);
  }
  if (level & (1 << 13)) // planning_time
  {
    planner->getMotionManager()->setPlanningTime(config.planning_time);
  }
  if (level & (1 << 14)) // use_cartesian_motion
  {
    planner->use_cartesian_motion = config.use_cartesian_motion;
  }
  if (level & (1 << 15)) // compute_ik_when_sampling
  {
    planner->compute_ik_when_sampling = config.compute_ik_when_sampling;
  }
  if (level & (1 << 16)) // velocity_scaling
  {
    planner->getMotionManager()->setMaxVelocityScalingFactor(config.velocity_scaling);
  }
  if (level & (1 << 17)) // record_map_updates
  {
    planner->record_map_updates = config.record_map_updates;
  }
  if (level & (1 << 18)) // record_viewpoints
  {
    planner->record_viewpoints = config.record_viewpoints;
  }
  if (level & (1 << 21)) // activate_move_to_see
  {
    planner->activate_move_to_see = config.activate_move_to_see;
    planner->move_to_see_exclusive = config.move_to_see_exclusive;
    planner->m2s_delta_thresh = config.m2s_delta_thresh;
    planner->m2s_max_steps = config.m2s_max_steps;
  }
  if (level & (1 << 22)) // publish_cluster_visualization
  {
    planner->publish_cluster_visualization = config.publish_cluster_visualization;
    planner->minimum_cluster_size = config.minimum_cluster_size;
    planner->cluster_neighborhood = static_cast<octomap_vpp::Neighborhood>(config.cluster_neighborhood);
  }
  current_config = config;
}

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
  config_server = new dynamic_reconfigure::Server<roi_viewpoint_planner::PlannerConfig>(config_mutex, nhp);
  config_server->setCallback(reconfigureCallback);

  config_mutex.lock();
  current_config.mode = roi_viewpoint_planner::Planner_SAMPLE_AUTOMATIC;
  planner->mode = (roi_viewpoint_planner::ViewpointPlanner::PlannerMode) current_config.mode;
  config_server->updateConfig(current_config);
  config_mutex.unlock();

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
  delete config_server;
}
