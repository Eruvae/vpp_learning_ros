#include <ros/ros.h>
#include <ros/package.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <boost/algorithm/string.hpp>
#include <octomap/OcTree.h>
#include <octomap_vpp/RoiOcTree.h>
#include <octomap_vpp/octomap_transforms.h>

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
    ROS_ERROR_STREAM("Model states message not received; could not read plant poses");
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

std::unordered_map<std::string, ModelOctree> loadModelOcrees(double res)
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
  trees["Floor_room"] = {generateBoxOctree(res, 20, 20, 0.001, 0, 0, 0.01), nullptr};
  trees["grey_wall"] = {generateBoxOctree(res, 7.5, 0.2, 2.8, 0, 0, 1.4), nullptr};
  return trees;
}

std::unique_ptr<octomap_vpp::RoiOcTree> generateRoiOctree(double res)
{
  auto model_map = loadModelOcrees(res);
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
        octomap_vpp::RoiOcTreeNode *node = tree->setNodeValue(model_pose.transform(it.getCoordinate()), tree->getClampingThresMaxLog());
        octomap::OcTreeKey k = it.getKey();
        bool is_roi = false;
        if (octree.roi)
        {
          is_roi = octree.roi->search(k) != nullptr;
          for (int i = 0; !is_roi && i < octomap_vpp::NB_6; i++)
          {
            octomap::OcTreeKey nbk(k[0] + nbLut[i][0], k[1] + nbLut[i][1], k[2] + nbLut[i][2]);
            if (octree.roi->search(nbk) != nullptr)
              is_roi = true;
          }
        }
        if (is_roi)
          node->setRoiLogOdds(tree->getClampingThresMaxLog());
        else
          node->setRoiLogOdds(tree->getClampingThresMinLog());
      }
    }
  }
  return tree;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "world_model_saver");
  ros::NodeHandle nh;

  std::unique_ptr<octomap_vpp::RoiOcTree> tree = generateRoiOctree(0.01);
  return tree->write("test.ot");
}
