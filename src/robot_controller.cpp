#include "robot_controller.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <roi_viewpoint_planner/rvp_utils.h>

RobotController::RobotController(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer,
                                 const std::string &pose_reference_frame, const std::string &group_name,
                                 const std::string &ee_link_name)
        : tfBuffer(tfBuffer),
          manipulator_group(group_name),
          robot_model_loader("robot_description"),
          kinematic_model(robot_model_loader.getModel()),
          joint_model_group(kinematic_model->getJointModelGroup(group_name)),
          kinematic_state(new robot_state::RobotState(kinematic_model)),
          pose_reference_frame(pose_reference_frame),
          end_effector_link(ee_link_name) {
    manipulator_group.setPoseReferenceFrame(pose_reference_frame);

    start_values_set = nh.getParam("/roi_viewpoint_planner/initial_joint_values", joint_start_values);

    if (!start_values_set) {
        ROS_WARN("No inital joint values set");
    }
}

bool RobotController::planAndExecute(bool async, double *plan_length, double *traj_duration) {
    MoveGroupInterface::Plan plan;
    ros::Time planStartTime = ros::Time::now();
    MoveItErrorCode res = manipulator_group.plan(plan);
    ROS_INFO_STREAM("Planning duration: " << (ros::Time::now() - planStartTime));
    if (res != MoveItErrorCode::SUCCESS) {
        ROS_INFO("Could not find plan");
        return false;
    }
    try {
        if (async) {
            res = manipulator_group.asyncExecute(plan);
        } else {
            res = manipulator_group.execute(plan);
        }

        if (plan_length) {
            *plan_length = roi_viewpoint_planner::computeTrajectoryLength(plan);
        }
        if (traj_duration) {
            *traj_duration = roi_viewpoint_planner::getTrajectoryDuration(plan);
        }
    }
    catch (std::runtime_error &ex) {
        ROS_ERROR("Exception: [%s]", ex.what());
    }

    /*if (res != MoveItErrorCode::SUCCESS)
    {
      ROS_INFO("Could not execute plan");
      return false;
    }*/
    return true;
}

bool RobotController::getCurrentTransform(geometry_msgs::TransformStamped &cur_tf) {
    try {
        cur_tf = tfBuffer.lookupTransform(pose_reference_frame, end_effector_link, ros::Time(0));
    }
    catch (const tf2::TransformException &e) {
        ROS_ERROR_STREAM("Couldn't find transform to map frame: " << e.what());
        return false;
    }
    return true;
}

std::vector<double> RobotController::getCurrentJointValues() {
    return manipulator_group.getCurrentJointValues();
}

bool RobotController::reset() {
    if (start_values_set)
        return moveToState(joint_start_values);

    return true;
}

bool RobotController::moveToPose(const geometry_msgs::Pose &goal_pose, bool async, double *plan_length,
                                 double *traj_duration) {
    ros::Time setTargetTime = ros::Time::now();
    if (goal_pose.position.z > 1.5) {
        return false;
    }
    if (!manipulator_group.setJointValueTarget(goal_pose, end_effector_link)) {
        ROS_INFO_STREAM("Could not find IK for specified pose (Timeout: " << (ros::Time::now() - setTargetTime) << ")");
        return false;
    }
    ROS_INFO_STREAM("IK solve time: " << (ros::Time::now() - setTargetTime));

    return planAndExecute(async, plan_length, traj_duration);
}

bool RobotController::moveToPoseRelative(const geometry_msgs::Pose &relative_pose, bool async, double *plan_length,
                                         double *traj_duration) {
    geometry_msgs::TransformStamped cur_tf;
    bool success = getCurrentTransform(cur_tf);
    if (!success)
        return false;

    geometry_msgs::Pose goal_pose;
    tf2::doTransform(relative_pose, goal_pose, cur_tf);
    return moveToPose(goal_pose, async, plan_length, traj_duration);
}

bool RobotController::moveToState(const robot_state::RobotState &goal_state, bool async, double *plan_length,
                                  double *traj_duration) {
    manipulator_group.setJointValueTarget(goal_state);

    return planAndExecute(async, plan_length, traj_duration);
}

bool RobotController::moveToState(const std::vector<double> &joint_values, bool async, double *plan_length,
                                  double *traj_duration) {
    kinematic_state->setJointGroupPositions(joint_model_group, joint_values);
    kinematic_state->enforceBounds();
    manipulator_group.setJointValueTarget(*kinematic_state);

    return planAndExecute(async, plan_length, traj_duration);
}

bool
RobotController::moveToStateRelative(const std::vector<double> &relative_joint_values, bool async, double *plan_length,
                                     double *traj_duration) {
    std::vector<double> joint_values = manipulator_group.getCurrentJointValues();
    size_t joint_num = std::min(joint_values.size(), relative_joint_values.size());
    for (size_t i = 0; i < joint_num; i++) {
        joint_values[i] += relative_joint_values[i];
    }
    return moveToState(joint_values, async, plan_length, traj_duration);
}

bool RobotController::moveToRandomTarget(bool async, const ros::Duration &timeout) {
    manipulator_group.setRandomTarget();
    bool success = false;
    ros::Time start_time = ros::Time::now();
    while (!success && ros::ok() && (ros::Time::now() - start_time) < timeout) {
        success = planAndExecute(async);
    }
    return success;
}
