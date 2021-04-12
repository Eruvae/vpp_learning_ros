#include "octree_manager.h"
#include <octomap_msgs/conversions.h>
#include "boost/date_time/posix_time/posix_time.hpp"

OctreeManager::OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &map_frame, double tree_resolution) :
  tfBuffer(tfBuffer), planningTree(new octomap_vpp::RoiOcTree(tree_resolution)), map_frame(map_frame), old_rois(0)
{
  octomapPub = nh.advertise<octomap_msgs::Octomap>("octomap", 1);
  roiSub = nh.subscribe("/detect_roi/results", 1, &OctreeManager::registerPointcloudWithRoi, this);
}

void OctreeManager::registerPointcloudWithRoi(const ros::MessageEvent<pointcloud_roi_msgs::PointcloudWithRoi const> &event)
{
  const pointcloud_roi_msgs::PointcloudWithRoi &roi = *event.getMessage();

  geometry_msgs::TransformStamped pcFrameTf;
  bool cloud_in_map_frame = (roi.cloud.header.frame_id == map_frame);
  if (cloud_in_map_frame)
  {
    ROS_INFO_STREAM("Incoming cloud already in target frame");
    //pcFrameTf.header = roi.cloud.header;
    pcFrameTf.transform = roi.transform;
  }
  else
  {
    ROS_INFO_STREAM("Convert incoming cloud (" << roi.cloud.header.frame_id << ") to map frame (" << map_frame << "), assuming no transform in incoming data");
    try
    {
      pcFrameTf = tfBuffer.lookupTransform(map_frame, roi.cloud.header.frame_id, roi.cloud.header.stamp);
    }
    catch (const tf2::TransformException &e)
    {
      ROS_ERROR_STREAM("Couldn't find transform to map frame in registerRoiOCL: " << e.what());
      return;
    }
  }

  const geometry_msgs::Vector3 &pcfOrig = pcFrameTf.transform.translation;
  octomap::point3d scan_orig(pcfOrig.x, pcfOrig.y, pcfOrig.z);

  octomap::Pointcloud inlierCloud, outlierCloud, fullCloud;
  if (cloud_in_map_frame)
    octomap_vpp::pointCloud2ToOctomapByIndices(roi.cloud, roi.roi_indices, inlierCloud, outlierCloud, fullCloud);
  else
    octomap_vpp::pointCloud2ToOctomapByIndices(roi.cloud, roi.roi_indices, pcFrameTf.transform, inlierCloud, outlierCloud, fullCloud);

  tree_mtx.lock();
  planningTree->insertPointCloud(fullCloud, scan_orig);
  planningTree->insertRegionScan(inlierCloud, outlierCloud);
  tree_mtx.unlock();

  publishMap();
}

std::string OctreeManager::saveOctomap(const std::string &name, bool name_is_prefix)
{
  std::stringstream fName;
  fName << name;
  if (name_is_prefix)
  {
    const boost::posix_time::ptime curDateTime = boost::posix_time::second_clock::local_time();
    boost::posix_time::time_facet *const timeFacet = new boost::posix_time::time_facet("%Y-%m-%d-%H-%M-%S");
    fName.imbue(std::locale(fName.getloc(), timeFacet));
    fName << "_" << curDateTime;
  }
  fName << ".ot";
  tree_mtx.lock();
  bool result = planningTree->write(fName.str());
  tree_mtx.unlock();
  return result ? fName.str() : "";
}

int OctreeManager::loadOctomap(const std::string &filename)
{
  octomap_vpp::RoiOcTree *map = NULL;
  octomap::AbstractOcTree *tree =  octomap::AbstractOcTree::read(filename);
  if (!tree)
    return -1;

  map = dynamic_cast<octomap_vpp::RoiOcTree*>(tree);
  if(!map)
  {
    delete tree;
    return -2;
  }
  tree_mtx.lock();
  planningTree.reset(map);
  planningTree->computeRoiKeys();
  tree_mtx.unlock();
  publishMap();
  return 0;
}

void OctreeManager::resetOctomap()
{
  tree_mtx.lock();
  planningTree->clear();
  planningTree->clearRoiKeys();
  old_rois = 0;
  tree_mtx.unlock();
  publishMap();
}

void OctreeManager::publishMap()
{
  octomap_msgs::Octomap map_msg;
  map_msg.header.frame_id = map_frame;
  map_msg.header.stamp = ros::Time::now();
  tree_mtx.lock();
  bool msg_generated = octomap_msgs::fullMapToMsg(*planningTree, map_msg);
  tree_mtx.unlock();
  if (msg_generated)
  {
    octomapPub.publish(map_msg);
  }
}

void OctreeManager::fillObservation(Observation::Builder &obs, const octomap::pose6d &viewpoint, size_t theta_steps, size_t phi_steps, size_t layers, double range)
{
  size_t list_size = theta_steps * phi_steps * layers;
  capnp::List<uint32_t>::Builder unknownCount = obs.initUnknownCount(list_size);
  capnp::List<uint32_t>::Builder freeCount = obs.initFreeCount(list_size);
  capnp::List<uint32_t>::Builder occupiedCount = obs.initOccupiedCount(list_size);
  capnp::List<uint32_t>::Builder roiCount = obs.initRoiCount(list_size);

  double layer_range = range / layers;

  tree_mtx.lock();
  for (size_t t = 0; t < theta_steps; t++)
  {
    double theta = t*M_PI / theta_steps;
    for(size_t p = 0; p < phi_steps; p++)
    {
      double phi = -M_PI + 2*p*M_PI / phi_steps;
      double x = cos(phi) * sin(theta);
      double y = sin(phi) * sin(theta);
      double z = cos(theta);
      octomap::point3d dir(x, y, z);
      for (size_t layer = 0; layer < layers; layer++)
      {
        size_t flat_index = layer * phi_steps * theta_steps + t * phi_steps + p;
        octomap::point3d start = viewpoint.transform(viewpoint.trans() + dir * (layer * layer_range));
        octomap::point3d end = viewpoint.transform(viewpoint.trans() + dir * ((layer+1) * layer_range));
        octomap::KeyRay ray;
        planningTree->computeRayKeys(start, end, ray);
        uint32_t uc = 0, fc = 0, oc = 0, rc = 0;
        for (const octomap::OcTreeKey &key : ray)
        {
          octomap_vpp::RoiOcTreeNode *node = planningTree->search(key);
          octomap_vpp::NodeState occ_state = planningTree->getNodeState(node, octomap_vpp::NodeProperty::OCCUPANCY);
          octomap_vpp::NodeState roi_state = planningTree->getNodeState(node, octomap_vpp::NodeProperty::ROI);

          if (roi_state == octomap_vpp::NodeState::OCCUPIED_ROI)
            rc++;

          if (occ_state == octomap_vpp::NodeState::OCCUPIED_ROI)
            oc++;
          else if (occ_state == octomap_vpp::NodeState::FREE_NONROI)
            fc++;
          else
            uc++;
        }
        unknownCount.set(flat_index, uc);
        freeCount.set(flat_index, fc);
        occupiedCount.set(flat_index, oc);
        roiCount.set(flat_index, rc);
      }
    }
  }
  tree_mtx.unlock();
}

uint32_t OctreeManager::getReward()
{
  tree_mtx.lock();
  size_t num_rois = planningTree->getRoiSize();
  tree_mtx.unlock();

  uint32_t reward = 0;
  if (num_rois > old_rois)
    reward = num_rois - old_rois;

  old_rois = num_rois;
  return reward;
}