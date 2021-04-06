#ifndef UTILS_H
#define UTILS_H

#include "action.capnp.h"
#include "geometry_msgs/Transform.h"
#include "geometry_msgs/Pose.h"

geometry_msgs::Point fromActionMsg(const Point::Reader &point_msg)
{
  geometry_msgs::Point point;
  point.x = point_msg.getX();
  point.y = point_msg.getY();
  point.z = point_msg.getZ();
  return point;
}

geometry_msgs::Quaternion fromActionMsg(const Quaternion::Reader &quat_msg)
{
  geometry_msgs::Quaternion quat;
  quat.x = quat_msg.getX();
  quat.y = quat_msg.getY();
  quat.z = quat_msg.getZ();
  quat.w = quat_msg.getW();
  return quat;
}

geometry_msgs::Pose fromActionMsg(const Pose::Reader &pose_msg)
{
  geometry_msgs::Pose pose;
  pose.position = fromActionMsg(pose_msg.getPosition());
  pose.orientation = fromActionMsg(pose_msg.getOrientation());
  return pose;
}

std::ostream& operator<<(std::ostream &os, const geometry_msgs::Point& point)
{
    os << "{" << point.x << ", " << point.y << ", " << point.z << "}";
    return os;
}

std::ostream& operator<<(std::ostream &os, const geometry_msgs::Quaternion& quat)
{
    os << "{" << quat.x << ", " << quat.y << ", " << quat.z << ", " << quat.w << "}";
    return os;
}

std::ostream& operator<<(std::ostream &os, const geometry_msgs::Pose& pose)
{
    os << "{" << pose.position << "; " << pose.orientation << "}";
    return os;
}

#endif // UTILS_H
