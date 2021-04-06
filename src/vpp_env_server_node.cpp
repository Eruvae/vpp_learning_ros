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
  //ros::NodeHandle nhp("~");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  // initialize the zmq context with a single IO thread
  zmq::context_t context(1);

  // construct a REP (reply) socket and bind to interface
  zmq::socket_t socket(context, zmq::socket_type::rep);
  socket.bind("tcp://*:5555");

  double tree_resolution = nh.param<double>("/roi_viewpoint_planner/tree_resolution", 0.01);
  std::string wstree_default_package = ros::package::getPath("phenorob_ur5e");
  std::string wstree_file = nh.param<std::string>("/roi_viewpoint_planner/workspace_tree", wstree_default_package + "/workspace_trees/ur_retractable/workspace_map.ot");
  std::string sampling_tree_file = nh.param<std::string>("/roi_viewpoint_planner/sampling_tree", wstree_default_package + "/workspace_trees/ur_retractable/inflated_workspace_map.ot");
  std::string map_frame = nh.param<std::string>("/roi_viewpoint_planner/map_frame", "world");
  std::string ws_frame = nh.param<std::string>("/roi_viewpoint_planner/ws_frame", "arm_base_link");

  tf2_ros::Buffer tfBuffer(ros::Duration(30));
  tf2_ros::TransformListener tfListener(tfBuffer);

  OctreeManager oc_manager(nh, tfBuffer, map_frame, tree_resolution);
  RobotController controller(nh, tfBuffer, map_frame);
  controller.reset();

  while(ros::ok())
  {
      zmq::message_t request;

      // receive a request from client
      zmq::recv_result_t res = socket.recv(request, zmq::recv_flags::none);
      kj::ArrayPtr dataPtr(reinterpret_cast<capnp::word*>(request.data()), request.size()/sizeof(capnp::word));
      capnp::FlatArrayMessageReader reader(dataPtr);
      Action::Reader act = reader.getRoot<Action>();
      switch (act.which())
      {
      case Action::NONE:
      {
        ROS_INFO_STREAM("Action: None received");
        break;
      }
      case Action::RESET:
      {
        ROS_INFO_STREAM("Action: Reset received");
        bool success = controller.reset();
        oc_manager.resetOctomap();
        break;
      }
      case Action::DIRECTION:
      {
        ROS_INFO_STREAM("Action: Direction received");
        break;
      }
      case Action::GOAL_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getGoalPose());
        ROS_INFO_STREAM("Action: GoalPose received - " << pose);
        bool success = controller.moveToPose(pose);
        break;
      }
      case Action::RELATIVE_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getRelativePose());
        ROS_INFO_STREAM("Action: RelativePose received - " << pose);
        bool success = controller.moveToPoseRelative(pose);
        break;
      }
      }

      geometry_msgs::TransformStamped cur_tf;
      controller.getCurrentTransform(cur_tf);
      octomap::pose6d cur_pose = octomap_vpp::transformToOctomath(cur_tf.transform);

      capnp::MallocMessageBuilder builder;
      Observation::Builder obs = builder.initRoot<Observation>();

      oc_manager.fillObservation(obs, cur_pose, 36, 18, 5, 5.0);
      obs.setWidth(36);
      obs.setHeight(18);
      obs.setLayers(5);

      uint32_t reward;
      if (act.isReset())
        reward = 0;
      else
        reward = oc_manager.getReward();

      obs.setFoundRois(reward);

      kj::Array<capnp::word> arr = capnp::messageToFlatArray(builder);
      zmq::const_buffer buf(arr.begin(), arr.size()*sizeof(capnp::word));

      // send the reply to the client
      socket.send(buf, zmq::send_flags::none);
  }
}
