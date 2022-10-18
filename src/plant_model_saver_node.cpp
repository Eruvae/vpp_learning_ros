#include <ros/ros.h>
#include "voxelgrid.capnp.h"
#include <capnp/message.h>
#include <capnp/serialize-packed.h>
#include <fcntl.h>

#include "model_saver.h"

std::map<std::string, RoiOctreeWithBounds> generateModelRoiOctrees(double res, bool check_roi_neighbors)
{
  static constexpr octomap::key_type MINKV(std::numeric_limits<octomap::key_type>::lowest()), MAXKV(std::numeric_limits<octomap::key_type>::max());
  octomap::OcTreeKey min_key(MAXKV, MAXKV, MAXKV), max_key(MINKV, MINKV, MINKV);
  auto model_map = loadModelOctrees(res, false, false);
  std::map<std::string, RoiOctreeWithBounds> result_map;
  for (auto it = model_map.begin(); it != model_map.end(); it++)
  {
    const ModelOctree &model_tree = it->second;
    std::unique_ptr<octomap_vpp::RoiOcTree> tree(new octomap_vpp::RoiOcTree(res));
    if (model_tree.occ)
    {
      for(auto it = model_tree.occ->begin_leafs(), end = model_tree.occ->end_leafs(); it != end; it++)
      {
        octomap::OcTreeKey model_key = it.getKey();
        setMax(max_key, model_key);
        setMin(min_key, model_key);
        setTreeValue(tree.get(), model_tree, model_key, model_key, check_roi_neighbors);
      }
    }
    tree->computeRoiKeys();
    ROS_ERROR_STREAM(it->first << " num ROI keys: " << tree->getRoiKeys().size());
    result_map[it->first] = {std::move(tree), min_key, max_key};
  }
  return result_map;
}

int main()
{
  std::array<double, 3> resolutions = {0.02, 0.01, 0.005};
  std::array<bool, 2> check_roi_neighbors = {false, true};

  for (double res : resolutions)
  {
    for (bool cn : check_roi_neighbors)
    {
      std::map<std::string, RoiOctreeWithBounds> tree_map = generateModelRoiOctrees(res, cn);

      for (auto it = tree_map.begin(); it != tree_map.end(); it++)
      {
        const std::string filename = it->first + "_" + doubleToString(res) + (cn ? "_roi_neighbors" : "_no_roi_neighbors");
        it->second.tree->write(filename + ".ot");
        // Write Voxelgrid
        capnp::MallocMessageBuilder vx_builder;
        vpp_msg::Voxelgrid::Builder vx = vx_builder.initRoot<vpp_msg::Voxelgrid>();
        generateVoxelgrid(vx, it->second.tree.get(), it->second.min_key, it->second.max_key);

        int fd = open((filename + ".cvx").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd < 0)
        {
          ROS_WARN("Couldn't create voxelgrid file");
          return -1;
        }
        //void writePackedMessageToFd(int fd, MessageBuilder& builder)
        //void writeMessageToFd(int fd, MessageBuilder& builder)
        capnp::writeMessageToFd(fd, vx_builder);
        close(fd);
      }
    }
  }

  return 0;
}
