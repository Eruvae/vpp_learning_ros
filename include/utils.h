#ifndef UTILS_H
#define UTILS_H

#include <capnp/list.h>
#include <vector>
#include "pose.capnp.h"
#include "geometry_msgs/Transform.h"
#include "geometry_msgs/Pose.h"

template<typename T, capnp::Kind K>
std::vector<T> capnpListToVector(const typename capnp::List<T, K>::Reader &list)
{
  std::vector<T> vec;
  vec.reserve(list.size());
  for (const T &el : list)
  {
    vec.push_back(el);
  }
  return vec;
}

geometry_msgs::Point fromActionMsg(const vpp_msg::Point::Reader &point_msg)
{
  geometry_msgs::Point point;
  point.x = point_msg.getX();
  point.y = point_msg.getY();
  point.z = point_msg.getZ();
  return point;
}

void toActionMsg(vpp_msg::Point::Builder &point_msg, const geometry_msgs::Point &point)
{
  point_msg.setX(point.x);
  point_msg.setY(point.y);
  point_msg.setZ(point.z);
}

void toActionMsg(vpp_msg::Point::Builder &point_msg, const geometry_msgs::Vector3 &point)
{
  point_msg.setX(point.x);
  point_msg.setY(point.y);
  point_msg.setZ(point.z);
}

geometry_msgs::Quaternion fromActionMsg(const vpp_msg::Quaternion::Reader &quat_msg)
{
  geometry_msgs::Quaternion quat;
  quat.x = quat_msg.getX();
  quat.y = quat_msg.getY();
  quat.z = quat_msg.getZ();
  quat.w = quat_msg.getW();
  return quat;
}

void toActionMsg(vpp_msg::Quaternion::Builder &quat_msg, const geometry_msgs::Quaternion &quat)
{
  quat_msg.setX(quat.x);
  quat_msg.setY(quat.y);
  quat_msg.setZ(quat.z);
  quat_msg.setW(quat.w);
}

geometry_msgs::Pose fromActionMsg(const vpp_msg::Pose::Reader &pose_msg)
{
  geometry_msgs::Pose pose;
  pose.position = fromActionMsg(pose_msg.getPosition());
  pose.orientation = fromActionMsg(pose_msg.getOrientation());
  return pose;
}

void toActionMsg(vpp_msg::Pose::Builder &pose_msg, const geometry_msgs::Pose &pose)
{
  vpp_msg::Point::Builder point_msg = pose_msg.initPosition();
  toActionMsg(point_msg, pose.position);
  vpp_msg::Quaternion::Builder quat_msg = pose_msg.initOrientation();
  toActionMsg(quat_msg, pose.orientation);
}

void toActionMsg(vpp_msg::Pose::Builder &pose_msg, const geometry_msgs::Transform &tf)
{
  vpp_msg::Point::Builder point_msg = pose_msg.initPosition();
  toActionMsg(point_msg, tf.translation);
  vpp_msg::Quaternion::Builder quat_msg = pose_msg.initOrientation();
  toActionMsg(quat_msg, tf.rotation);
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

std::ostream& operator<<(std::ostream &os, const geometry_msgs::Transform& tf)
{
    os << "{" << tf.translation << "; " << tf.rotation << "}";
    return os;
}

#endif // UTILS_H
