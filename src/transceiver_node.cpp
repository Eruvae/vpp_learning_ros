#include <random>
#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_interface/planning_interface.h>
#include "capnp/action.capnp.h"
#include "capnp/observation.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include "zmq.hpp"

using moveit::planning_interface::MoveGroupInterface;
using moveit::planning_interface::MoveItErrorCode;

class RobotController
{
private:
  MoveGroupInterface manipulator_group;
  const std::string end_effector_link;

public:
  RobotController(const std::string& group_name = "manipulator", const std::string &ee_link_name = "camera_link")
    : manipulator_group(group_name), end_effector_link(ee_link_name)
  {}

  bool moveToPose(const geometry_msgs::Pose &goal_pose)
  {
    ros::Time setTargetTime = ros::Time::now();
    if (!manipulator_group.setJointValueTarget(goal_pose, end_effector_link))
    {
      ROS_INFO_STREAM("Could not find IK for specified pose (Timeout: " << (ros::Time::now() - setTargetTime) << ")");
      return false;
    }
    ROS_INFO_STREAM("IK solve time: " << (ros::Time::now() - setTargetTime));

    MoveGroupInterface::Plan plan;
    ros::Time planStartTime = ros::Time::now();
    MoveItErrorCode res = manipulator_group.plan(plan);
    ROS_INFO_STREAM("Planning duration: " << (ros::Time::now() - planStartTime));
    if (res != MoveItErrorCode::SUCCESS)
    {
      ROS_INFO("Could not find plan");
      return false;
    }
    res = manipulator_group.execute(plan);
    if (res != MoveItErrorCode::SUCCESS)
    {
      ROS_INFO("Could not execute plan");
      return false;
    }
    return true;
  }
};

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

void fillListWithRandomData(capnp::List<uint32_t>::Builder &list, uint32_t max)
{
  static std::default_random_engine generator;
  std::uniform_int_distribution<uint32_t> dist(0, max);
  for (size_t i = 0; i < list.size(); i++)
  {
    list.set(i, dist(generator));
  }
}

void serverFunc(zmq::context_t &context)
{
  // construct a REP (reply) socket and bind to interface
  zmq::socket_t socket(context, zmq::socket_type::rep);
  socket.bind("tcp://*:5555");

  // prepare some static data for responses
  const std::string data = "World";

  RobotController controller;

  for (;;)
  {
      zmq::message_t request;

      // receive a request from client
      zmq::recv_result_t res = socket.recv(request, zmq::recv_flags::none);
      kj::ArrayPtr dataPtr(reinterpret_cast<capnp::word*>(request.data()), request.size()/sizeof(capnp::word));
      capnp::FlatArrayMessageReader reader(dataPtr);
      Action::Reader act = reader.getRoot<Action>();
      switch (act.getData().which())
      {
      case Action::Data::NONE:
      {
        ROS_INFO_STREAM("Action: None received");
        break;
      }
      case Action::Data::DIRECTION:
      {
        ROS_INFO_STREAM("Action: Direction received");
        break;
      }
      case Action::Data::GOAL_POSE:
      {
        geometry_msgs::Pose pose = fromActionMsg(act.getData().getGoalPose());
        ROS_INFO_STREAM("Action: GoalPose received - " << pose);
        bool success = controller.moveToPose(pose);
        break;
      }
      }

      capnp::MallocMessageBuilder builder;
      Observation::Builder obs = builder.initRoot<Observation>();
      capnp::List<uint32_t>::Builder unknownCount = obs.initUnknownCount(100);
      fillListWithRandomData(unknownCount, 100);

      obs.setHeight(10);
      obs.setWidth(10);

      kj::Array<capnp::word> arr = capnp::messageToFlatArray(builder);
      zmq::const_buffer buf(arr.begin(), arr.size()*sizeof(capnp::word));

      // send the reply to the client
      socket.send(buf, zmq::send_flags::none);
  }
}

/*void clientFunc(zmq::context_t &context)
{
  // construct a REQ (request) socket and connect to interface
  zmq::socket_t socket(context, zmq::socket_type::req);
  socket.connect("tcp://localhost:5555");

  // set up some static data to send
  const std::string data{"Hello"};

  for (auto request_num = 0; request_num < 10; ++request_num)
  {
      // send the request message
      std::cout << "Sending Hello " << request_num << "..." << std::endl;
      socket.send(zmq::buffer(data), zmq::send_flags::none);

      // wait for reply from server
      zmq::message_t reply;
      zmq::recv_result_t res = socket.recv(reply, zmq::recv_flags::none);

      std::cout << "Received " << reply.to_string();
      std::cout << " (" << request_num << ")";
      std::cout << std::endl;
  }
}*/

/*void writeAddressBook(int fd)
{
  capnp::MallocMessageBuilder message;

  AddressBook::Builder addressBook = message.initRoot<AddressBook>();
  capnp::List<Person>::Builder people = addressBook.initPeople(2);

  Person::Builder alice = people[0];
  alice.setId(123);
  alice.setName("Alice");
  alice.setEmail("alice@example.com");
  // Type shown for explanation purposes; normally you'd use auto.
  capnp::List<Person::PhoneNumber>::Builder alicePhones =
      alice.initPhones(1);
  alicePhones[0].setNumber("555-1212");
  alicePhones[0].setType(Person::PhoneNumber::Type::MOBILE);
  alice.getEmployment().setSchool("MIT");

  Person::Builder bob = people[1];
  bob.setId(456);
  bob.setName("Bob");
  bob.setEmail("bob@example.com");
  auto bobPhones = bob.initPhones(2);
  bobPhones[0].setNumber("555-4567");
  bobPhones[0].setType(Person::PhoneNumber::Type::HOME);
  bobPhones[1].setNumber("555-7654");
  bobPhones[1].setType(Person::PhoneNumber::Type::WORK);
  bob.getEmployment().setUnemployed();

  writePackedMessageToFd(fd, message);
}*/

/*void printAddressBook(int fd)
{
  capnp::PackedFdMessageReader message(fd);

  AddressBook::Reader addressBook = message.getRoot<AddressBook>();

  for (Person::Reader person : addressBook.getPeople()) {
    std::cout << person.getName().cStr() << ": "
              << person.getEmail().cStr() << std::endl;
    for (Person::PhoneNumber::Reader phone: person.getPhones()) {
      const char* typeName = "UNKNOWN";
      switch (phone.getType()) {
        case Person::PhoneNumber::Type::MOBILE: typeName = "mobile"; break;
        case Person::PhoneNumber::Type::HOME: typeName = "home"; break;
        case Person::PhoneNumber::Type::WORK: typeName = "work"; break;
      }
      std::cout << "  " << typeName << " phone: "
                << phone.getNumber().cStr() << std::endl;
    }
    Person::Employment::Reader employment = person.getEmployment();
    switch (employment.which()) {
      case Person::Employment::UNEMPLOYED:
        std::cout << "  unemployed" << std::endl;
        break;
      case Person::Employment::EMPLOYER:
        std::cout << "  employer: "
                  << employment.getEmployer().cStr() << std::endl;
        break;
      case Person::Employment::SCHOOL:
        std::cout << "  student at: "
                  << employment.getSchool().cStr() << std::endl;
        break;
      case Person::Employment::SELF_EMPLOYED:
        std::cout << "  self-employed" << std::endl;
        break;
    }
  }
}*/

int main(int argc, char **argv)
{
  ros::init(argc, argv, "roi_viewpoint_planner");
  //ros::NodeHandle nh;
  //ros::NodeHandle nhp("~");
  ros::AsyncSpinner spinner(4);
  spinner.start();

  // initialize the zmq context with a single IO thread
  zmq::context_t context(1);

  serverFunc(context);
}
