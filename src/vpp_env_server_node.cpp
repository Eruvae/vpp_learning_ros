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
  socket.bind("tcp://*:5555");

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

  if (evaluate_results)
  {
    ROS_INFO_STREAM("Starting evaluator");
    oc_manager.startEvaluator();
  }

  while(ros::ok())
  {
      zmq::message_t request;

      // receive a request from client
      zmq::recv_result_t res = socket.recv(request, zmq::recv_flags::none);
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
        bool success = controller.reset();
        oc_manager.resetOctomap();

        if (evaluate_results)
          oc_manager.resetEvaluator();

        double reset_time = (ros::Time::now() - reset_start_time).toSec();
        ROS_INFO_STREAM("Reset took " << reset_time << "s");
        break;
      }
      case vpp_msg::Action::RELATIVE_JOINT_TARGET:
      {
        ROS_INFO_STREAM("Action: Relative joint target received");
        ros::Time planning_start = ros::Time::now();
        std::vector<double> relative_joint_values = capnpListToVector<double, capnp::Kind::PRIMITIVE>(act.getRelativeJointTarget());
        double plan_length = 0, traj_duration = 0;
        bool success = controller.moveToStateRelative(relative_joint_values, false, &plan_length, &traj_duration);
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::ABSOLUTE_JOINT_TARGET:
      {
        ROS_INFO_STREAM("Action: Absolute joint target received");
        ros::Time planning_start = ros::Time::now();
        std::vector<double> joint_values = capnpListToVector<double, capnp::Kind::PRIMITIVE>(act.getAbsoluteJointTarget());
        double plan_length = 0, traj_duration = 0;
        bool success = controller.moveToState(joint_values, false, &plan_length, &traj_duration);
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::GOAL_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getGoalPose());
        ROS_INFO_STREAM("Action: GoalPose received - " << pose);
        ros::Time planning_start = ros::Time::now();
        double plan_length = 0, traj_duration = 0;
        bool success = controller.moveToPose(pose, false, &plan_length, &traj_duration);
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::RELATIVE_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getRelativePose());
        ROS_INFO_STREAM("Action: RelativePose received - " << pose);
        ros::Time planning_start = ros::Time::now();
        double plan_length = 0, traj_duration = 0;
        bool success = controller.moveToPoseRelative(pose, false, &plan_length, &traj_duration);
        if (evaluate_results)
          oc_manager.saveEvaluatorData(plan_length, traj_duration);

        planning_time = (ros::Time::now() - planning_start).toSec();
        ROS_INFO_STREAM("Moving took " << planning_time << "s");
        break;
      }
      case vpp_msg::Action::RESET_AND_RANDOMIZE:
      {
        ROS_INFO_STREAM("Action: Reset and randomize received");
        ros::Time rand_reset_start_time = ros::Time::now();
        geometry_msgs::Point min_point = fromActionMsg(act.getResetAndRandomize().getMin());
        geometry_msgs::Point max_point = fromActionMsg(act.getResetAndRandomize().getMax());
        double min_dist = act.getResetAndRandomize().getMinDist();

        bool success = controller.reset();

        oc_manager.randomizePlants(min_point, max_point, min_dist);
        oc_manager.resetOctomap();

        if (evaluate_results)
          oc_manager.resetEvaluator();

        double reset_time = (ros::Time::now() - rand_reset_start_time).toSec();
        ROS_INFO_STREAM("Reset and randomize took " << reset_time << "s");
        break;
      }
      }

      geometry_msgs::TransformStamped cur_tf;
      controller.getCurrentTransform(cur_tf);
      octomap::pose6d cur_pose = octomap_vpp::transformToOctomath(cur_tf.transform);

      capnp::MallocMessageBuilder builder;
      vpp_msg::Observation::Builder obs = builder.initRoot<vpp_msg::Observation>();

      ros::Time obs_comp_start_time = ros::Time::now();

      const size_t WIDTH = 36;
      const size_t HEIGHT = 18;
      const size_t LAYERS = 5;
      const double RANGE = 2.0;
      oc_manager.fillObservation(obs, cur_pose, WIDTH, HEIGHT, LAYERS, RANGE);
      obs.setWidth(WIDTH);
      obs.setHeight(HEIGHT);
      obs.setLayers(LAYERS);

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
      obs.setTotalRoiCells(oc_manager.getMaxGtReward());
      obs.setPlanningTime(planning_time);

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
