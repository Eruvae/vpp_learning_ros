#include <ros/ros.h>
#include <boost/algorithm/string.hpp>
#include <octomap_vpp/octomap_transforms.h>
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <fcntl.h>

#include "model_saver.h"

std::string getModelType(const std::string &name)
{
  if (boost::algorithm::starts_with(name, "capsicum_plant_6_no_fruits"))
  {
    return "capsicum_plant_6_no_fruits";
  }
  else if (boost::algorithm::starts_with(name, "capsicum_plant_6_one_fruit"))
  {
    return "capsicum_plant_6_one_fruit";
  }
  else if (boost::algorithm::starts_with(name, "capsicum_plant_6_more_occ"))
  {
    return "capsicum_plant_6_more_occ";
  }
  else if (boost::algorithm::starts_with(name, "capsicum_plant_"))
  {
    return name.substr(0, 16);
  }
  else if (boost::algorithm::starts_with(name, "Floor_room"))
  {
    return "Floor_room";
  }
  else if (boost::algorithm::starts_with(name, "grey_wall"))
  {
    return "grey_wall";
  }
  else
  {
    return "";
  }
}

std::vector<ModelInfo> readModelPoses()
{
  std::vector<ModelInfo> model_list;
  gazebo_msgs::ModelStatesConstPtr model_states = ros::topic::waitForMessage<gazebo_msgs::ModelStates>("/gazebo/model_states");
  if (!model_states)
  {
    ROS_ERROR_STREAM("Model states message not received; could not read poses");
    return model_list;
  }
  for (size_t i=0; i < model_states->name.size(); i++)
  {
    gazebo_msgs::ModelState state;
    state.model_name = model_states->name[i];
    state.pose = model_states->pose[i];
    state.twist = model_states->twist[i];
    ROS_ERROR_STREAM("Model name: " << state.model_name);
    model_list.push_back(ModelInfo(getModelType(state.model_name), state));
  }
  return model_list;
}


RoiOctreeWithBounds generateRoiOctree(double res, const octomap::point3d &min_point, const octomap::point3d &max_point, bool include_walls, bool include_floor, bool check_roi_neighbors)
{
  static constexpr octomap::key_type MINKV(std::numeric_limits<octomap::key_type>::lowest()), MAXKV(std::numeric_limits<octomap::key_type>::max());
  octomap::OcTreeKey min_key(MAXKV, MAXKV, MAXKV), max_key(MINKV, MINKV, MINKV);
  auto model_map = loadModelOctrees(res, include_walls, include_floor);
  auto model_list = readModelPoses();
  std::unique_ptr<octomap_vpp::RoiOcTree> tree(new octomap_vpp::RoiOcTree(res));
  for (const ModelInfo &model : model_list)
  {
    const ModelOctree &model_tree = model_map[model.model];
    octomap::pose6d model_pose = octomap_vpp::poseToOctomath(model.state.pose);
    if (model_tree.occ)
    {
      for(auto it = model_tree.occ->begin_leafs(), end = model_tree.occ->end_leafs(); it != end; it++)
      {
        const octomap::point3d map_point = model_pose.transform(it.getCoordinate());
        if (anyGreater(map_point, max_point) || anySmaller(map_point, min_point))
          continue;

        octomap::OcTreeKey map_key;
        if (!tree->coordToKeyChecked(map_point, map_key))
        {
          ROS_WARN("Key outside of tree bounds");
          continue;
        }
        setMax(max_key, map_key);
        setMin(min_key, map_key);
        octomap::OcTreeKey model_key = it.getKey();
        setTreeValue(tree.get(), model_tree, map_key, model_key, check_roi_neighbors);
      }
    }
  }
  return {std::move(tree), min_key, max_key};
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "world_model_saver");
  ros::NodeHandle nh;

  double resolution = 0.02;
  octomap::point3d min_point(-3.85f, -3.85f, 0.f);
  octomap::point3d max_point(3.85f, 3.85f, 2.9f);
  bool include_walls = true;
  bool include_floor = true;
  bool check_roi_neighbors = true;

  std::cout << "Enter resolution (default 0.02)" << std::endl;
  readInput(resolution);
  std::cout << "Resolution: " << resolution << std::endl;
  std::cout << "Enter minimum bounds (default (-3.85, -3.85, 0))" << std::endl;
  readInput(min_point);
  std::cout << "Min bounds: " << min_point << std::endl;
  std::cout << "Enter maximum bounds (default (3.85, 3.85, 2.9))" << std::endl;
  readInput(max_point);
  std::cout << "Max bounds: " << max_point << std::endl;
  std::cout << "Include walls? (Y/n)" << std::endl;
  readBool(include_walls);
  std::cout << "Answer: " << (include_walls ? "Yes" : "No") << std::endl;
  std::cout << "Include floor? (Y/n)" << std::endl;
  readBool(include_floor);
  std::cout << "Answer: " << (include_floor ? "Yes" : "No") << std::endl;
  std::cout << "Consider voxels neighboring ROI as ROI? (Y/n)" << std::endl;
  readBool(check_roi_neighbors);
  std::cout << "Answer: " << (check_roi_neighbors ? "Yes" : "No") << std::endl;

  auto [tree, min_key, max_key] = generateRoiOctree(resolution, min_point, max_point, include_walls, include_floor, check_roi_neighbors);

  tree->write("saved_world.ot");

  // Write Voxelgrid
  capnp::MallocMessageBuilder vx_builder;
  vpp_msg::Voxelgrid::Builder vx = vx_builder.initRoot<vpp_msg::Voxelgrid>();
  generateVoxelgrid(vx, tree.get(), tree->coordToKey(min_point), tree->coordToKey(max_point));

  int fd = open("saved_world.cvx", O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0)
  {
    ROS_WARN("Couldn't create voxelgrid file");
    return -1;
  }
  //void writePackedMessageToFd(int fd, MessageBuilder& builder)
  //void writeMessageToFd(int fd, MessageBuilder& builder)
  capnp::writeMessageToFd(fd, vx_builder);
  close(fd);

  return 0;
}
