#include <ros/ros.h>
#include <ros/package.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <boost/algorithm/string.hpp>
#include <octomap/OcTree.h>
#include <octomap_vpp/RoiOcTree.h>
#include <octomap_vpp/octomap_transforms.h>
#include "voxelgrid.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <fcntl.h>

using octomap_vpp::nbLut;

struct ModelInfo
{
  ModelInfo(const std::string &model, const gazebo_msgs::ModelState &state)
    : model(model), state(state) {}

  std::string model;
  gazebo_msgs::ModelState state;
};

struct ModelOctree
{
  std::unique_ptr<octomap::OcTree> occ = nullptr;
  std::unique_ptr<octomap::OcTree> roi = nullptr;

  ModelOctree(octomap::OcTree *occ=nullptr, octomap::OcTree *roi=nullptr) : occ(occ), roi(roi)
  {
    if (occ) occ->expand();
    if (roi) roi->expand();
  }
};

std::string getModelType(const std::string &name)
{
  if (boost::algorithm::starts_with(name, "capsicum_plant_6_no_fruits"))
  {
    return "VG07_6_no_fruits";
  }
  else if (boost::algorithm::starts_with(name, "capsicum_plant_6_one_fruit"))
  {
    return "VG07_6_one_fruit";
  }
  else if (boost::algorithm::starts_with(name, "capsicum_plant_6_more_occ"))
  {
    return "VG07_6_more_occ";
  }
  else if (boost::algorithm::starts_with(name, "capsicum_plant_6"))
  {
    return "VG07_6";
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
    model_list.push_back(ModelInfo(getModelType(state.model_name), state));
  }
  return model_list;
}

octomap::OcTree* generateBoxOctree(double res, double sx, double sy, double sz, double ox, double oy, double oz)
{
  octomap::OcTree *octree = new octomap::OcTree(res);
  octomap::OcTreeKey start_key = octree->coordToKey(ox - sx/2, oy - sy/2, oz - sz/2);
  octomap::OcTreeKey end_key = octree->coordToKey(ox + sx/2, oy + sy/2, oz + sz/2);
  for (octomap::key_type x=start_key[0]; x <= end_key[0]; x++)
  {
    for (octomap::key_type y=start_key[1]; y <= end_key[1]; y++)
    {
      for (octomap::key_type z=start_key[2]; z <= end_key[2]; z++)
      {
        octree->setNodeValue(octomap::OcTreeKey(x, y, z), octree->getClampingThresMaxLog());
      }
    }
  }
  return octree;
}

std::unordered_map<std::string, ModelOctree> loadModelOcrees(double res, bool include_walls, bool include_floor)
{
  std::ostringstream oss;
  oss << std::setprecision(8) << std::noshowpoint << res;
  std::string r = oss.str();
  static const std::string model_package = ros::package::getPath("roi_viewpoint_planner");
  std::unordered_map<std::string, ModelOctree> trees;
  //const std::string path = package_path + "/cfg/plant_files/individual_fruits/" + name + "/";
  trees["VG07_6"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6/VG07_6_" + r + ".bt"),
                     new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_fruits_" + r + ".bt")};
  trees["VG07_6_more_occ"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_more_occ/VG07_6_more_occ_" + r + ".bt"),
                              new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_fruits_" + r + ".bt")};
  trees["VG07_6_one_fruit"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_one_fruit/VG07_6_one_fruit_" + r + ".bt"),
                               new octomap::OcTree(model_package + "/cfg/plant_files/individual_fruits/VG07_6/" + r + "/VG07_6_fruit_7_" + r + ".bt")};
  trees["VG07_6_no_fruits"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_no_fruits/VG07_6_no_fruits_" + r + ".bt"), nullptr};
  trees["Floor_room"] = {include_floor ? generateBoxOctree(res, 20, 20, 0.001, 0, 0, 0.01) : nullptr, nullptr};
  trees["grey_wall"] = {include_walls ? generateBoxOctree(res, 7.5, 0.2, 2.8, 0, 0, 1.4) : nullptr, nullptr};
  return trees;
}

void setMax(octomap::OcTreeKey &k1, const octomap::OcTreeKey &k2)
{
  for (unsigned int i=0; i<3; i++) if (k2[i] > k1[i]) k1[i] = k2[i];
}

void setMin(octomap::OcTreeKey &k1, const octomap::OcTreeKey &k2)
{
  for (unsigned int i=0; i<3; i++) if (k2[i] < k1[i]) k1[i] = k2[i];
}

bool anyGreater(const octomap::point3d &p, const octomap::point3d &ref)
{
  return p.x() > ref.x() || p.y() > ref.y() || p.z() > ref.z();
}

bool anySmaller(const octomap::point3d &p, const octomap::point3d &ref)
{
  return p.x() < ref.x() || p.y() < ref.y() || p.z() < ref.z();
}

octomap::OcTreeKey computeGridSize(const octomap::OcTreeKey &min, const octomap::OcTreeKey &max)
{
  return octomap::OcTreeKey(max[0] - min[0] + 1, max[1] - min[1] + 1, max[2] - min[2] + 1);
}

std::ostream& operator<<(std::ostream &os, const octomap::OcTreeKey &k)
{
  os << "(" << k[0] << ", " << k[1] << ", " << k[2] << ")";
  return os;
}

auto generateRoiOctree(double res, const octomap::point3d &min_point, const octomap::point3d &max_point, bool include_walls, bool include_floor, bool check_roi_neighbors) -> std::tuple<std::unique_ptr<octomap_vpp::RoiOcTree>, octomap::OcTreeKey, octomap::OcTreeKey>
{
  static constexpr octomap::key_type MINKV(std::numeric_limits<octomap::key_type>::lowest()), MAXKV(std::numeric_limits<octomap::key_type>::max());
  octomap::OcTreeKey min_key(MAXKV, MAXKV, MAXKV), max_key(MINKV, MINKV, MINKV);
  auto model_map = loadModelOcrees(res, include_walls, include_floor);
  auto model_list = readModelPoses();
  std::unique_ptr<octomap_vpp::RoiOcTree> tree(new octomap_vpp::RoiOcTree(res));
  for (const ModelInfo &model : model_list)
  {
    const ModelOctree &octree = model_map[model.model];
    octomap::pose6d model_pose = octomap_vpp::poseToOctomath(model.state.pose);
    if (octree.occ)
    {
      for(auto it = octree.occ->begin_leafs(), end = octree.occ->end_leafs(); it != end; it++)
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
        octomap_vpp::RoiOcTreeNode *node = tree->setNodeValue(map_key, tree->getClampingThresMaxLog());
        octomap::OcTreeKey k = it.getKey();
        bool is_roi = false;
        if (octree.roi)
        {
          is_roi = octree.roi->search(k) != nullptr;
          if (check_roi_neighbors)
          {
            for (int i = 0; !is_roi && i < octomap_vpp::NB_6; i++)
            {
              octomap::OcTreeKey nbk(k[0] + nbLut[i][0], k[1] + nbLut[i][1], k[2] + nbLut[i][2]);
              if (octree.roi->search(nbk) != nullptr)
                is_roi = true;
            }
          }
        }
        if (is_roi)
          node->setRoiLogOdds(tree->getClampingThresMaxLog());
        else
          node->setRoiLogOdds(tree->getClampingThresMinLog());
      }
    }
  }
  return {std::move(tree), min_key, max_key};
}

std::istream& ignoreUntilNextNumber(std::istream &is)
{
  while(is.peek() != EOF && !std::isdigit(is.peek()) && is.peek() != '-') is.get();
  return is;
}

std::istream& operator>> (std::istream &is, octomap::point3d& val)
{
  ignoreUntilNextNumber(is) >> val.x();
  ignoreUntilNextNumber(is) >> val.y();
  ignoreUntilNextNumber(is) >> val.z();
  return is;
}

template<typename T>
void readInput(T &var)
{
  std::string input;
  std::getline(std::cin, input);
  if ( !input.empty() ) {
    std::istringstream stream(input);
    stream >> var;
  }
}

void readBool(bool &var)
{
  std::string input;
  std::getline(std::cin, input);
  if ( !input.empty() ) {
    char first = input[0];
    if      (std::toupper(first) == 'Y') var = true;
    else if (std::toupper(first) == 'N') var = false;
  }
}

void generateVoxelgrid(vpp_msg::Voxelgrid::Builder &vx, const octomap_vpp::RoiOcTree *tree, const octomap::OcTreeKey &min_key, const octomap::OcTreeKey &max_key)
{
  octomap::OcTreeKey grid_size = computeGridSize(min_key, max_key);
  octomap::point3d center_point = (tree->keyToCoord(min_key) + tree->keyToCoord(max_key))*0.5;
  ROS_INFO_STREAM("Grid size: " << grid_size);
  capnp::List<uint8_t>::Builder labels = vx.initLabels(grid_size[0]*grid_size[1]*grid_size[2]);
  capnp::List<uint16_t>::Builder shape = vx.initShape(3);
  shape.set(0, grid_size[2]);
  shape.set(1, grid_size[1]);
  shape.set(2, grid_size[0]);
  capnp::List<float>::Builder center = vx.initCenter(3);
  center.set(0, center_point.x());
  center.set(1, center_point.y());
  center.set(2, center_point.z());

  vx.setResolution(static_cast<float>(tree->getResolution()));
  unsigned int linear_index = 0;
  for (octomap::key_type z = min_key[2]; z <= max_key[2]; z++)
  {
    for (octomap::key_type y = min_key[1]; y <= max_key[1]; y++)
    {
      for (octomap::key_type x = min_key[0]; x <= max_key[0]; x++)
      {
        octomap::OcTreeKey key(x, y, z);
        octomap_vpp::RoiOcTreeNode *node = tree->search(key);
        if (node == nullptr || !tree->isNodeOccupied(node))
          labels.set(linear_index++, 0); // Free
        else if (tree->isNodeROI(node))
          labels.set(linear_index++, 2); // ROI
        else
          labels.set(linear_index++, 1); // Occupied
      }
    }
  }
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
  std::cout << "Answer: " << (check_roi_neighbors ? "Yes" : "No") << std::endl;
  std::cout << "Include floor? (Y/n)" << std::endl;
  readBool(include_floor);
  std::cout << "Answer: " << (check_roi_neighbors ? "Yes" : "No") << std::endl;
  std::cout << "Consider voxels neighboring ROI as ROI? (Y/n)" << std::endl;
  readBool(check_roi_neighbors);
  std::cout << "Answer: " << (check_roi_neighbors ? "Yes" : "No") << std::endl;

  auto [tree, min_key, max_key] = generateRoiOctree(resolution, min_point, max_point, include_walls, include_floor, check_roi_neighbors);

  tree->write("saved_world.ot");

  // Write Voxelgrid
  capnp::MallocMessageBuilder vx_builder;
  vpp_msg::Voxelgrid::Builder vx = vx_builder.initRoot<vpp_msg::Voxelgrid>();
  generateVoxelgrid(vx, tree.get(), min_key, max_key);

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
