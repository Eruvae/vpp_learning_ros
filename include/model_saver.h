#include <ros/console.h>
#include <ros/package.h>
#include <octomap/OcTree.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/ModelStates.h>
#include <string>
#include <memory>
#include <filesystem>
#include <octomap_vpp/RoiOcTree.h>
#include <open3d/Open3D.h>
#include <tinyxml2.h>
#include "voxelgrid.capnp.h"

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

struct RoiOctreeWithBounds
{
  std::unique_ptr<octomap_vpp::RoiOcTree> tree;
  octomap::OcTreeKey min_key;
  octomap::OcTreeKey max_key;
};

static octomap::OcTree* generateBoxOctree(double res, double sx, double sy, double sz, double ox, double oy, double oz)
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

static octomap::OcTree* readModelO3D(const std::filesystem::path &filepath, double resolution)
{
  ROS_INFO_STREAM("Loading triangle mesh " << filepath.filename());
  std::shared_ptr<open3d::geometry::TriangleMesh> mesh = std::make_shared<open3d::geometry::TriangleMesh>();
  bool success = open3d::io::ReadTriangleMeshUsingASSIMP(filepath, *mesh, open3d::io::ReadTriangleMeshOptions{true, false, nullptr});
  if (!success) return nullptr;

  ROS_INFO_STREAM("Success, removing duplicates...");

  mesh->RemoveDuplicatedVertices();
  mesh->RemoveDuplicatedTriangles();

  ROS_INFO_STREAM("Creating voxelgrid at resolution " << resolution);

  std::shared_ptr<open3d::geometry::VoxelGrid> vx = open3d::geometry::VoxelGrid::CreateFromTriangleMesh(*mesh, resolution);

  ROS_INFO_STREAM("Success, converting to octree...");

  octomap::OcTree *tree(new octomap::OcTree(resolution));
  for (const auto &voxel : vx->GetVoxels())
  {
    const Eigen::Vector3d vc = vx->GetVoxelCenterCoordinate(voxel.grid_index_);
    octomap::OcTreeKey key = tree->coordToKey(octomap::point3d(static_cast<float>(vc(0)), static_cast<float>(-vc(2)), static_cast<float>(vc(1))));
    tree->setNodeValue(key, tree->getClampingThresMaxLog(), true);
  }

  ROS_INFO_STREAM("Success");

  return tree;
}

static inline ModelOctree loadPlantTreesO3D(const std::filesystem::path &plant_path, const std::filesystem::path &fruit_path, double resolution)
{
  return ModelOctree(readModelO3D(plant_path, resolution), readModelO3D(fruit_path, resolution));
}

static inline std::string doubleToString(double d)
{
  std::ostringstream oss;
  oss << std::setprecision(8) << std::noshowpoint << d;
  return oss.str();
}

static std::unordered_map<std::string, ModelOctree> loadModelOctrees(double res, bool include_walls, bool include_floor)
{
  std::string r = doubleToString(res);
  const std::filesystem::path model_package = std::filesystem::path(ros::package::getPath("ur_with_cam_gazebo")) / "models";
  std::unordered_map<std::string, ModelOctree> trees;

  for (auto const& dir_entry : std::filesystem::directory_iterator{model_package})
  {
    if (!dir_entry.is_directory())
      continue;

    std::filesystem::path dirname = dir_entry.path().filename();

    const std::string SDF_NAME = "model.sdf";

    tinyxml2::XMLDocument model_doc;
    tinyxml2::XMLError ret = model_doc.LoadFile((dir_entry.path() / SDF_NAME).c_str());

    if (ret != tinyxml2::XML_SUCCESS)
    {
      ROS_ERROR_STREAM(SDF_NAME << " for " << dirname << " could not be read");
      continue;
    }

    auto readXMLChild = [&](tinyxml2::XMLNode *entry, const char *name = 0) -> tinyxml2::XMLElement*
    {
      if (!entry) return nullptr;
      tinyxml2::XMLElement *child = entry->FirstChildElement(name);
      if (!child)
      {
        ROS_ERROR_STREAM(SDF_NAME << " for " << dirname << " has no element \"" << name << "\"");
      }
      return child;
    };

    tinyxml2::XMLElement *entry = readXMLChild(&model_doc, "sdf");
    entry = readXMLChild(entry, "model");
    entry = readXMLChild(entry, "link");
    entry = readXMLChild(entry, "visual");
    entry = readXMLChild(entry, "geometry");

    if (!entry) continue;

    tinyxml2::XMLElement *mesh = entry->FirstChildElement("mesh");
    if (mesh)
    {
      ROS_INFO_STREAM(SDF_NAME << " for " << dirname << " has element \"mesh\"");
      mesh = readXMLChild(mesh, "uri");
      if (!mesh) continue;
      std::string uri = mesh->GetText();
      std::string find_folder = dirname.string() + "/";
      auto dirname_pos = uri.find(find_folder);
      if (dirname_pos == std::string::npos)
      {
        ROS_ERROR_STREAM(SDF_NAME << ": " << dirname << " not found in URI");
        continue;
      }
      std::filesystem::path model_file = dir_entry / std::filesystem::path(uri.substr(dirname_pos + find_folder.length()));
      std::string model_filename = model_file.filename().string();
      std::string model_extension = model_file.extension().string();
      model_filename = model_filename.substr(0, model_filename.length() - model_extension.length());
      std::filesystem::path fruit_model_file = model_file;
      fruit_model_file.replace_filename(model_filename + "_fruitonly" + model_extension);

      trees[dirname] = loadPlantTreesO3D(model_file, fruit_model_file, res);
      ROS_INFO_STREAM("Path: " << model_file << "; " << fruit_model_file);

      continue;
    }

    tinyxml2::XMLElement *box = entry->FirstChildElement("box");
    if (box)
    {
      ROS_INFO_STREAM("model.sdf for " << dirname << " has element \"box\"");
      continue;
    }

    //if (dir_entry.path().filename().)
    /*if (dir_entry.path().filename() == "Floor_room")
    {
      trees["Floor_room"] = {include_floor ? generateBoxOctree(res, 20, 20, 0.001, 0, 0, 0.01) : nullptr, nullptr};
    }
    else if (dir_entry.path().filename() == "grey_wall")
    {
      trees["grey_wall"] = {include_walls ? generateBoxOctree(res, 7.5, 0.2, 2.8, 0, 0, 1.4) : nullptr, nullptr};
    }*/
  }

  //const std::string path = package_path + "/cfg/plant_files/individual_fruits/" + name + "/";
  /*trees["VG07_6"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6/VG07_6_" + r + ".bt"),
                     new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_fruits_" + r + ".bt")};
  trees["VG07_6_more_occ"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_more_occ/VG07_6_more_occ_" + r + ".bt"),
                              new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_fruits_" + r + ".bt")};
  trees["VG07_6_one_fruit"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_one_fruit/VG07_6_one_fruit_" + r + ".bt"),
                               new octomap::OcTree(model_package + "/cfg/plant_files/individual_fruits/VG07_6/" + r + "/VG07_6_fruit_7_" + r + ".bt")};
  trees["VG07_6_no_fruits"] = {new octomap::OcTree(model_package + "/cfg/plant_files/VG07_6_no_fruits/VG07_6_no_fruits_" + r + ".bt"), nullptr};
  trees["Floor_room"] = {include_floor ? generateBoxOctree(res, 20, 20, 0.001, 0, 0, 0.01) : nullptr, nullptr};
  trees["grey_wall"] = {include_walls ? generateBoxOctree(res, 7.5, 0.2, 2.8, 0, 0, 1.4) : nullptr, nullptr};*/
  return trees;
}


static inline void setMax(octomap::OcTreeKey &k1, const octomap::OcTreeKey &k2)
{
  for (unsigned int i=0; i<3; i++) if (k2[i] > k1[i]) k1[i] = k2[i];
}

static inline void setMin(octomap::OcTreeKey &k1, const octomap::OcTreeKey &k2)
{
  for (unsigned int i=0; i<3; i++) if (k2[i] < k1[i]) k1[i] = k2[i];
}

static inline bool anyGreater(const octomap::point3d &p, const octomap::point3d &ref)
{
  return p.x() > ref.x() || p.y() > ref.y() || p.z() > ref.z();
}

static inline bool anySmaller(const octomap::point3d &p, const octomap::point3d &ref)
{
  return p.x() < ref.x() || p.y() < ref.y() || p.z() < ref.z();
}

static inline octomap::OcTreeKey computeGridSize(const octomap::OcTreeKey &min, const octomap::OcTreeKey &max)
{
  return octomap::OcTreeKey(max[0] - min[0] + 1, max[1] - min[1] + 1, max[2] - min[2] + 1);
}

static inline std::ostream& operator<<(std::ostream &os, const octomap::OcTreeKey &k)
{
  os << "(" << k[0] << ", " << k[1] << ", " << k[2] << ")";
  return os;
}

static void setTreeValue(octomap_vpp::RoiOcTree *tree, const ModelOctree &model_tree, const octomap::OcTreeKey &map_key, const octomap::OcTreeKey &model_key, bool check_roi_neighbors)
{
  octomap_vpp::RoiOcTreeNode *node = tree->setNodeValue(map_key, tree->getClampingThresMaxLog());
  bool is_roi = false;
  if (model_tree.roi)
  {
    is_roi = model_tree.roi->search(model_key) != nullptr;
    if (check_roi_neighbors)
    {
      for (int i = 0; !is_roi && i < octomap_vpp::NB_6; i++)
      {
        octomap::OcTreeKey nbk(model_key[0] + nbLut[i][0], model_key[1] + nbLut[i][1], model_key[2] + nbLut[i][2]);
        if (model_tree.roi->search(nbk) != nullptr)
          is_roi = true;
      }
    }
  }
  if (is_roi)
    node->setRoiLogOdds(tree->getClampingThresMaxLog());
  else
    node->setRoiLogOdds(tree->getClampingThresMinLog());
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

static std::istream& ignoreUntilNextNumber(std::istream &is)
{
  while(is.peek() != EOF && !std::isdigit(is.peek()) && is.peek() != '-') is.get();
  return is;
}

static std::istream& operator>> (std::istream &is, octomap::point3d& val)
{
  ignoreUntilNextNumber(is) >> val.x();
  ignoreUntilNextNumber(is) >> val.y();
  ignoreUntilNextNumber(is) >> val.z();
  return is;
}

template<typename T>
static void readInput(T &var)
{
  std::string input;
  std::getline(std::cin, input);
  if ( !input.empty() ) {
    std::istringstream stream(input);
    stream >> var;
  }
}

static void readBool(bool &var)
{
  std::string input;
  std::getline(std::cin, input);
  if ( !input.empty() ) {
    char first = input[0];
    if      (std::toupper(first) == 'Y') var = true;
    else if (std::toupper(first) == 'N') var = false;
  }
}
