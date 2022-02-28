#include <random>
#include <ros/ros.h>
#include <ros/package.h>
#include "action.capnp.h"
#include "observation.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <octomap_vpp/octomap_transforms.h>
#include "zmq.hpp"
#include "octree_manager.h"
#include "robot_controller.h"
#include "utils.h"
#include <std_srvs/Empty.h>
/*void fillListWithRandomData(capnp::List<uint32_t>::Builder &list, uint32_t max)
{
  static std::default_random_engine generator;
  std::uniform_int_distribution<uint32_t> dist(0, max);
  for (size_t i = 0; i < list.size(); i++)
  {
    list.set(i, dist(generator));
  }
}*/

int main(int argc, char **argv)
{
  ros::init(argc, argv, "vpp_env_server_node");
  ros::NodeHandle nh;
  ros::NodeHandle nhp("~");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  // initialize the zmq context with a single IO thread
  zmq::context_t context(1);

  // construct a REP (reply) socket and bind to interface
  zmq::socket_t socket(context, zmq::socket_type::rep);
  socket.connect("tcp://localhost:5555");

  double tree_resolution = nh.param<double>("/roi_viewpoint_planner/tree_resolution", 0.01);
  std::string wstree_default_package = ros::package::getPath("ur_with_cam_gazebo");
  std::string wstree_file = nh.param<std::string>("/roi_viewpoint_planner/workspace_tree", wstree_default_package + "/workspace_trees/ur_with_cam/workspace_map.ot");
  std::string sampling_tree_file = nh.param<std::string>("/roi_viewpoint_planner/sampling_tree", wstree_default_package + "/workspace_trees/ur_with_cam/workspace_map.ot");
  std::string map_frame = nh.param<std::string>("/roi_viewpoint_planner/map_frame", "world");
  std::string ws_frame = nh.param<std::string>("/roi_viewpoint_planner/ws_frame", "arm_base_link");

  bool evaluate_results = nhp.param<bool>("evaluate_results", false);

  tf2_ros::Buffer tfBuffer(ros::Duration(30));
  tf2_ros::TransformListener tfListener(tfBuffer);

  ros::ServiceClient resetOctomapClient = nh.serviceClient<std_srvs::Empty>("/clear_octomap");
  std_srvs::Empty resetOctomapSrv;

  OctreeManager oc_manager(nh, tfBuffer, wstree_file, sampling_tree_file, map_frame, ws_frame, tree_resolution, evaluate_results);
  RobotController controller(nh, tfBuffer, map_frame);
  controller.reset();
  oc_manager.resetOctomap();

  if (evaluate_results)
  {
    ROS_INFO_STREAM("Starting evaluator");
    oc_manager.startEvaluator();
  }

  double accumulatedPlanLength = 0, accumulatedTrajectoryDuration = 0;
  vpp_msg::MapType map_type = vpp_msg::MapType::COUNT_MAP;

  while(ros::ok())
  {
      zmq::message_t request;

      // receive a request from client
      zmq::recv_result_t res = socket.recv(request, zmq::recv_flags::none);
      if (!res.has_value())
      {
        ROS_ERROR_STREAM("No message received");
        if (ros::ok())
        {
          ros::Duration(1).sleep();
          continue;
        }
        break;
      }
      ros::Time message_received_time = ros::Time::now();
      size_t word_size = request.size()/sizeof(capnp::word);
      std::unique_ptr<capnp::word> msg_newbuf(new capnp::word[word_size]);
      memcpy(msg_newbuf.get(), request.data(), request.size());
      kj::ArrayPtr dataPtr(msg_newbuf.get(), word_size);
      capnp::FlatArrayMessageReader reader(dataPtr);
      vpp_msg::Action::Reader act = reader.getRoot<vpp_msg::Action>();
      double msg_decode_time = (ros::Time::now() - message_received_time).toSec();
      ROS_INFO_STREAM("Decoding message took " << msg_decode_time << "s");
      double planning_time = 0;
      double plan_length = 0, traj_duration = 0;
      bool success = false;
      switch (act.which())
      {
      case vpp_msg::Action::NONE:
      {
        ROS_INFO_STREAM("Action: None received");
        break;
      }
      case vpp_msg::Action::RESET:
      {
        ROS_INFO_STREAM("Action: Reset received");
        ros::Time reset_start_time = ros::Time::now();
        vpp_msg::Action::Reset::Reader reset_params = act.getReset();

        if (!resetOctomapClient.call(resetOctomapSrv))
        {
            ROS_ERROR("Failed to reset moveit octomap");
        }

        success = controller.reset();

        if (reset_params.getRandomize())
        {
          geometry_msgs::Point min_point = fromActionMsg(reset_params.getRandomizationParameters().getMin());
          geometry_msgs::Point max_point = fromActionMsg(reset_params.getRandomizationParameters().getMax());
          double min_dist = reset_params.getRandomizationParameters().getMinDist();
          oc_manager.randomizePlants(min_point, max_point, min_dist);
        }

        if (reset_params.getMapType() != vpp_msg::MapType::UNCHANGED)
        {
          map_type = reset_params.getMapType();
        }

        oc_manager.resetOctomap();

        if (!resetOctomapClient.call(resetOctomapSrv))
        {
           ROS_ERROR("Failed to reset moveit octomap");
        }
        accumulatedPlanLength = 0;
        accumulatedTrajectoryDuration = 0;

        if (evaluate_results)
          oc_manager.resetEvaluator();

        ROS_INFO_STREAM("About to compute time for reset");
        double reset_time = (ros::Time::now() - reset_start_time).toSec();
        ROS_INFO_STREAM("Reset took " << reset_time << "s");
        break;
      }
      case vpp_msg::Action::RELATIVE_JOINT_TARGET:
      {
        ROS_INFO_STREAM("Action: Relative joint target received");
        ros::Time planning_start = ros::Time::now();
        std::vector<double> relative_joint_values = capnpListToVector<double, capnp::Kind::PRIMITIVE>(act.getRelativeJointTarget());
        success = controller.moveToStateRelative(relative_joint_values, false, &plan_length, &traj_duration);
        accumulatedPlanLength += plan_length;
        accumulatedTrajectoryDuration += traj_duration;
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        ROS_INFO_STREAM("About to compute time for relative joint target");
        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::ABSOLUTE_JOINT_TARGET:
      {
        ROS_INFO_STREAM("Action: Absolute joint target received");
        ros::Time planning_start = ros::Time::now();
        std::vector<double> joint_values = capnpListToVector<double, capnp::Kind::PRIMITIVE>(act.getAbsoluteJointTarget());
        success = controller.moveToState(joint_values, false, &plan_length, &traj_duration);
        accumulatedPlanLength += plan_length;
        accumulatedTrajectoryDuration += traj_duration;
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        ROS_INFO_STREAM("About to compute time for absolute joint target");
        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::GOAL_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getGoalPose());
        ROS_INFO_STREAM("Action: GoalPose received - " << pose);
        ros::Time planning_start = ros::Time::now();
        success = controller.moveToPose(pose, false, &plan_length, &traj_duration);
        accumulatedPlanLength += plan_length;
        accumulatedTrajectoryDuration += traj_duration;
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        ROS_INFO_STREAM("About to compute time for goal pose");
        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::RELATIVE_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getRelativePose());
        ROS_INFO_STREAM("Action: RelativePose received - " << pose);
        ros::Time planning_start = ros::Time::now();
        success = controller.moveToPoseRelative(pose, false, &plan_length, &traj_duration);
        accumulatedPlanLength += plan_length;
        accumulatedTrajectoryDuration += traj_duration;
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        ROS_INFO_STREAM("About to compute time for relative pose");
        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      }

      geometry_msgs::TransformStamped cur_tf;
      controller.getCurrentTransform(cur_tf);
      octomap::pose6d cur_pose = octomap_vpp::transformToOctomath(cur_tf.transform);

      capnp::MallocMessageBuilder builder;
      vpp_msg::Observation::Builder obs = builder.initRoot<vpp_msg::Observation>();

      ros::Time obs_comp_start_time = ros::Time::now();

      if (map_type == vpp_msg::MapType::COUNT_MAP)
      {
        vpp_msg::Observation::Map::CountMap::Builder cmap = obs.initMap().initCountMap();
        const size_t WIDTH = 36;
        const size_t HEIGHT = 18;
        const size_t LAYERS = 5;
        const double RANGE = 2.0;
        oc_manager.fillCountMap(cmap, cur_pose, WIDTH, HEIGHT, LAYERS, RANGE);
        cmap.setWidth(WIDTH);
        cmap.setHeight(HEIGHT);
        cmap.setLayers(LAYERS);
      }
      else if (map_type == vpp_msg::MapType::POINTCLOUD)
      {
        vpp_msg::Pointcloud::Builder pc = obs.initMap().initPointcloud();
        oc_manager.generatePointcloud(pc);
      }
      else if (map_type == vpp_msg::MapType::VOXELGRID)
      {
        vpp_msg::Voxelgrid::Builder vx = obs.initMap().initVoxelgrid();
        oc_manager.generateVoxelgrid(vx, cur_pose.trans(), 128);
      }
      else if (map_type == vpp_msg::MapType::FULL_VOXELGRID)
      {
        vpp_msg::Voxelgrid::Builder vx = obs.initMap().initFullVoxelgrid();
        oc_manager.generateFullVoxelgrid(vx);
      }

      vpp_msg::Pose::Builder pose_msg = obs.initRobotPose();
      toActionMsg(pose_msg, cur_tf.transform);

      //ROS_INFO_STREAM("Current pose: " << cur_tf.transform);

      std::vector<double> cur_joints = controller.getCurrentJointValues();
      kj::ArrayPtr<const double> cur_joints_arr(cur_joints.data(), cur_joints.size());
      obs.setRobotJoints(cur_joints_arr);

      double obs_comp_time = (ros::Time::now() - obs_comp_start_time).toSec();
      ROS_INFO_STREAM("Computing observation took " << obs_comp_time << "s");

      ros::Time reward_comp_start_time = ros::Time::now();
      uint32_t reward = 0;
      if (act.isGoalPose() || act.isRelativePose())
        reward = oc_manager.getRewardWithGt();

      obs.setFoundRois(reward);
      auto [foundFree, foundOcc] = oc_manager.getFoundFreeAndOccupied();
      obs.setFoundFree(foundFree);
      obs.setFoundOcc(foundOcc);
      obs.setTotalRoiCells(oc_manager.getMaxGtReward());
      obs.setPlanningTime(planning_time);

      obs.setEvalTotalTrajectoryDuration(accumulatedTrajectoryDuration);

      obs.setHasMoved(success);

      double reward_comp_time = (ros::Time::now() - reward_comp_start_time).toSec();
      ROS_INFO_STREAM("Computing reward took " << reward_comp_time << "s");

      ros::Time message_encode_start_time = ros::Time::now();
      kj::Array<capnp::word> arr = capnp::messageToFlatArray(builder);
      zmq::const_buffer buf(arr.begin(), arr.size()*sizeof(capnp::word));

      double message_encode_time = (ros::Time::now() - message_encode_start_time).toSec();
      ROS_INFO_STREAM("Encoding message took " << message_encode_time << "s");

      // send the reply to the client
      socket.send(buf, zmq::send_flags::none);
  }
}
