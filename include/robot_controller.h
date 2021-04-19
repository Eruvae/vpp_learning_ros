#ifndef ROBOT_CONTROLLER_H
#define ROBOT_CONTROLLER_H

#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>

using moveit::planning_interface::MoveGroupInterface;
using moveit::planning_interface::MoveItErrorCode;

class RobotController
{
private:
  tf2_ros::Buffer &tfBuffer;
  MoveGroupInterface manipulator_group;
  robot_model_loader::RobotModelLoader robot_model_loader;
  robot_model::RobotModelPtr kinematic_model;
  const robot_state::JointModelGroup* joint_model_group;
  robot_state::RobotStatePtr kinematic_state;

  const std::string pose_reference_frame;
  const std::string end_effector_link;
  std::vector<double> joint_start_values;
  bool start_values_set;

  bool planAndExecute(bool async);

public:
  RobotController(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &pose_reference_frame = "world",
                  const std::string& group_name = "manipulator", const std::string &ee_link_name = "camera_link");

  bool getCurrentTransform(geometry_msgs::TransformStamped &cur_tf);
  std::vector<double> getCurrentJointValues();
  bool reset();
  bool moveToPose(const geometry_msgs::Pose &goal_pose, bool async=false);
  bool moveToPoseRelative(const geometry_msgs::Pose &relative_pose, bool async=false);
  bool moveToState(const robot_state::RobotState &goal_state, bool async=false);
  bool moveToState(const std::vector<double> &joint_values, bool async=false);
  bool moveToStateRelative(const std::vector<double> &relative_joint_values, bool async=false);
};

#endif // ROBOT_CONTROLLER_H
