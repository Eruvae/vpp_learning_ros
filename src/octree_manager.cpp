#include "octree_manager.h"
#include <octomap_msgs/conversions.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/range/counting_range.hpp>
#include <execution>
#include <rvp_evaluation/octree_provider_interfaces/provided_tree_interface.h>

#define OBSFILL_USE_PARALLEL_LOOP

OctreeManager::OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &wstree_file, const std::string &sampling_tree_file,
  const std::string &map_frame, const std::string &ws_frame, double tree_resolution, bool initialize_evaluator) :
  tfBuffer(tfBuffer), planningTree(new octomap_vpp::RoiOcTree(tree_resolution)),
  workspaceTree(nullptr), samplingTree(nullptr),
  wsMin(-FLT_MAX, -FLT_MAX, -FLT_MAX),
  wsMax(FLT_MAX, FLT_MAX, FLT_MAX),
  stMin(-FLT_MAX, -FLT_MAX, -FLT_MAX),
  stMax(FLT_MAX, FLT_MAX, FLT_MAX),
  gtLoader(new rvp_evaluation::GtOctreeLoader(tree_resolution)), evaluator(nullptr),
  map_frame(map_frame), ws_frame(ws_frame), old_rois(0), tree_mtx(own_mtx)
{
  planningTree->enableChangeDetection(true);
  octomapPub = nh.advertise<octomap_msgs::Octomap>("octomap", 1);
  workspaceTreePub = nh.advertise<octomap_msgs::Octomap>("workspace_tree", 1, true);
  samplingTreePub = nh.advertise<octomap_msgs::Octomap>("sampling_tree", 1, true);
  initWorkspace(wstree_file, sampling_tree_file);
  roiSub = nh.subscribe("/detect_roi/results", 1, &OctreeManager::registerPointcloudWithRoi, this);

  if (initialize_evaluator)
  {
    ros::NodeHandle nh_eval("evaluator");
    std::shared_ptr<rvp_evaluation::ProvidedTreeInterface> interface(new rvp_evaluation::ProvidedTreeInterface(planningTree, tree_mtx));
    evaluator.reset(new rvp_evaluation::Evaluator(interface, nh, nh_eval, true, false, gtLoader));
    eval_trial_num = 0;
    evaluator->saveGtAsColoredCloud();
  }
}

OctreeManager::OctreeManager(ros::NodeHandle &nh, tf2_ros::Buffer &tfBuffer, const std::string &wstree_file, const std::string &sampling_tree_file,
              const std::string &map_frame, const std::string &ws_frame,
              const std::shared_ptr<octomap_vpp::RoiOcTree> &providedTree, boost::mutex &tree_mtx, bool initialize_evaluator) :
  tfBuffer(tfBuffer), planningTree(providedTree),
  workspaceTree(nullptr), samplingTree(nullptr),
  wsMin(-FLT_MAX, -FLT_MAX, -FLT_MAX),
  wsMax(FLT_MAX, FLT_MAX, FLT_MAX),
  stMin(-FLT_MAX, -FLT_MAX, -FLT_MAX),
  stMax(FLT_MAX, FLT_MAX, FLT_MAX),
  gtLoader(new rvp_evaluation::GtOctreeLoader(providedTree->getResolution())), evaluator(nullptr),
  map_frame(map_frame), ws_frame(ws_frame), old_rois(0), tree_mtx(tree_mtx)
{
  planningTree->enableChangeDetection(true);
  initWorkspace(wstree_file, sampling_tree_file);
  if (initialize_evaluator)
  {
    ros::NodeHandle nh_eval("evaluator");
    std::shared_ptr<rvp_evaluation::ProvidedTreeInterface> interface(new rvp_evaluation::ProvidedTreeInterface(planningTree, tree_mtx));
    evaluator.reset(new rvp_evaluation::Evaluator(interface, nh, nh_eval, true, false, gtLoader));
    eval_trial_num = 0;
    evaluator->saveGtAsColoredCloud();
  }
}

void OctreeManager::initWorkspace(const std::string &wstree_file, const std::string &sampling_tree_file)
{
  // Load workspace

  octomap::AbstractOcTree *tree = octomap::AbstractOcTree::read(wstree_file);
  if (!tree)
  {
    ROS_ERROR_STREAM("Workspace tree file could not be loaded");
  }
  else
  {
    octomap_vpp::CountingOcTree *countingTree = dynamic_cast<octomap_vpp::CountingOcTree*>(tree);

    if (countingTree) // convert to workspace tree if counting tree loaded
    {
      workspaceTree.reset(new octomap_vpp::WorkspaceOcTree(*countingTree));
      delete countingTree;
    }
    else
    {
      workspaceTree.reset(dynamic_cast<octomap_vpp::WorkspaceOcTree*>(tree));
    }

    if (!workspaceTree)
    {
      ROS_ERROR("Workspace tree type not recognized; please load either CountingOcTree or WorkspaceOcTree");
      delete tree;
    }
    else
    {
      wsMin = octomap::point3d(FLT_MAX, FLT_MAX, FLT_MAX);
      wsMax = octomap::point3d(-FLT_MAX, -FLT_MAX, -FLT_MAX);
      for (auto it = workspaceTree->begin_leafs(), end = workspaceTree->end_leafs(); it != end; it++)
      {
        octomap::point3d coord = it.getCoordinate();
        if (coord.x() < wsMin.x()) wsMin.x() = coord.x();
        if (coord.y() < wsMin.y()) wsMin.y() = coord.y();
        if (coord.z() < wsMin.z()) wsMin.z() = coord.z();
        if (coord.x() > wsMax.x()) wsMax.x() = coord.x();
        if (coord.y() > wsMax.y()) wsMax.y() = coord.y();
        if (coord.z() > wsMax.z()) wsMax.z() = coord.z();
      }

      if (workspaceTreePub)
      {
        octomap_msgs::Octomap ws_msg;
        ws_msg.header.frame_id = ws_frame;
        ws_msg.header.stamp = ros::Time(0);
        bool msg_generated = octomap_msgs::fullMapToMsg(*workspaceTree, ws_msg);
        if (msg_generated)
        {
          workspaceTreePub.publish(ws_msg);
        }
      }
    }
  }

  // Load sampling tree

  tree = octomap::AbstractOcTree::read(sampling_tree_file);
  if (!tree)
  {
    ROS_ERROR_STREAM("Sampling tree file could not be loaded");
  }
  else
  {
    samplingTree.reset(dynamic_cast<octomap_vpp::WorkspaceOcTree*>(tree));
    if (!samplingTree)
    {
      ROS_ERROR("Sampling tree must be of type WorkspaceOcTree");
      delete tree;
    }
  }

  if (!samplingTree) // if sampling tree not specified, use workspace octree
  {
    samplingTree = workspaceTree;
  }

  if (samplingTree)
  {
    stMin = octomap::point3d(FLT_MAX, FLT_MAX, FLT_MAX);
    stMax = octomap::point3d(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (auto it = samplingTree->begin_leafs(), end = samplingTree->end_leafs(); it != end; it++)
    {
      octomap::point3d coord = it.getCoordinate();
      if (coord.x() < stMin.x()) stMin.x() = coord.x();
      if (coord.y() < stMin.y()) stMin.y() = coord.y();
      if (coord.z() < stMin.z()) stMin.z() = coord.z();
      if (coord.x() > stMax.x()) stMax.x() = coord.x();
      if (coord.y() > stMax.y()) stMax.y() = coord.y();
      if (coord.z() > stMax.z()) stMax.z() = coord.z();
    }

    if (samplingTreePub)
    {
      octomap_msgs::Octomap st_msg;
      st_msg.header.frame_id = ws_frame;
      st_msg.header.stamp = ros::Time(0);
      bool msg_generated = octomap_msgs::fullMapToMsg(*samplingTree, st_msg);
      if (msg_generated)
      {
        samplingTreePub.publish(st_msg);
      }
    }
  }
}

void OctreeManager::registerPointcloudWithRoi(const ros::MessageEvent<pointcloud_roi_msgs::PointcloudWithRoi const> &event)
{
  const pointcloud_roi_msgs::PointcloudWithRoi &roi = *event.getMessage();

  geometry_msgs::TransformStamped pcFrameTf;
  bool cloud_in_map_frame = (roi.cloud.header.frame_id == map_frame);
  if (cloud_in_map_frame)
  {
    //ROS_INFO_STREAM("Incoming cloud already in target frame");
    //pcFrameTf.header = roi.cloud.header;
    pcFrameTf.transform = roi.transform;
  }
  else
  {
    //ROS_INFO_STREAM("Convert incoming cloud (" << roi.cloud.header.frame_id << ") to map frame (" << map_frame << "), assuming no transform in incoming data");
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

octomap::point3d OctreeManager::transformToMapFrame(const octomap::point3d &p)
{
  if (map_frame == ws_frame)
    return p;

  geometry_msgs::TransformStamped trans;
  try
  {
    trans = tfBuffer.lookupTransform(map_frame, ws_frame, ros::Time(0));
  }
  catch (const tf2::TransformException &e)
  {
    ROS_ERROR_STREAM("Couldn't find transform to ws frame in transformToWorkspace: " << e.what());
    return p;
  }

  octomap::point3d pt;
  tf2::doTransform(p, pt, trans);
  return pt;
}

geometry_msgs::Pose OctreeManager::transformToMapFrame(const geometry_msgs::Pose &p)
{
  if (map_frame == ws_frame)
    return p;

  geometry_msgs::TransformStamped trans;
  try
  {
    trans = tfBuffer.lookupTransform(map_frame, ws_frame, ros::Time(0));
  }
  catch (const tf2::TransformException &e)
  {
    ROS_ERROR_STREAM("Couldn't find transform to ws frame in transformToWorkspace: " << e.what());
    return p;
  }

  geometry_msgs::Pose pt;
  tf2::doTransform(p, pt, trans);
  return pt;
}

octomap::point3d OctreeManager::transformToWorkspace(const octomap::point3d &p)
{
  if (map_frame == ws_frame)
    return p;

  geometry_msgs::TransformStamped trans;
  try
  {
    trans = tfBuffer.lookupTransform(ws_frame, map_frame, ros::Time(0));
  }
  catch (const tf2::TransformException &e)
  {
    ROS_ERROR_STREAM("Couldn't find transform to ws frame in transformToWorkspace: " << e.what());
    return p;
  }

  octomap::point3d pt;
  tf2::doTransform(p, pt, trans);
  return pt;
}

geometry_msgs::Pose OctreeManager::transformToWorkspace(const geometry_msgs::Pose &p)
{
  if (map_frame == ws_frame)
    return p;

  geometry_msgs::TransformStamped trans;
  try
  {
    trans = tfBuffer.lookupTransform(ws_frame, map_frame, ros::Time(0));
  }
  catch (const tf2::TransformException &e)
  {
    ROS_ERROR_STREAM("Couldn't find transform to ws frame in transformToWorkspace: " << e.what());
    return p;
  }

  geometry_msgs::Pose pt;
  tf2::doTransform(p, pt, trans);
  return pt;
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
  planningTree->resetChangeDetection();
  old_rois = 0;
  tree_mtx.unlock();
  encountered_keys.clear();
  publishMap();
}

void OctreeManager::randomizePlants(const geometry_msgs::Point &min, const geometry_msgs::Point &max, double min_dist)
{
  if (evaluator)
    evaluator->randomizePlantPositions(octomap_vpp::pointToOctomath(min), octomap_vpp::pointToOctomath(max), min_dist);
  else
    gtLoader->randomizePlantPositions(octomap_vpp::pointToOctomath(min), octomap_vpp::pointToOctomath(max), min_dist);
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

void OctreeManager::fillCountMap(vpp_msg::Observation::Map::CountMap::Builder &cmap, const octomap::pose6d &viewpoint, size_t theta_steps, size_t phi_steps, size_t layers, double range)
{
  size_t list_size = theta_steps * phi_steps * layers;
  capnp::List<uint32_t>::Builder unknownCount = cmap.initUnknownCount(list_size);
  capnp::List<uint32_t>::Builder freeCount = cmap.initFreeCount(list_size);
  capnp::List<uint32_t>::Builder occupiedCount = cmap.initOccupiedCount(list_size);
  capnp::List<uint32_t>::Builder roiCount = cmap.initRoiCount(list_size);

  double layer_range = range / layers;

  tree_mtx.lock();
  auto it1 = boost::counting_range<size_t>(0, theta_steps);
  #ifdef OBSFILL_USE_PARALLEL_LOOP
  auto loop_policy = std::execution::par;
  #else
  auto loop_policy = std::execution::seq;
  #endif

  std::for_each(loop_policy, it1.begin(), it1.end(), [&](size_t t)
  {
    double theta = t*M_PI / theta_steps;
    auto it2 = boost::counting_range<size_t>(0, phi_steps);
    std::for_each(loop_policy, it2.begin(), it2.end(), [&](size_t p)
    {
      double phi = -M_PI + 2*p*M_PI / phi_steps;
      double x = cos(phi) * sin(theta);
      double y = sin(phi) * sin(theta);
      double z = cos(theta);
      octomap::point3d dir(x, y, z);
      auto it3 = boost::counting_range<size_t>(0, layers);
      std::for_each(loop_policy, it3.begin(), it3.end(), [&](size_t layer)
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
      });
    });
  });
  /*for (size_t t = 0; t < theta_steps; t++)
  {

    for(size_t p = 0; p < phi_steps; p++)
    {
      for (size_t layer = 0; layer < layers; layer++)
      {

      }
    }
  }*/
  tree_mtx.unlock();
}

void OctreeManager::generatePointcloud(vpp_msg::Pointcloud::Builder &pc)
{
  tree_mtx.lock();
  std::vector<octomap::point3d> points_vec;
  std::vector<uint8_t> labels_vec;
  for (auto it = planningTree->begin_leafs(), end = planningTree->end_leafs(); it != end; it++)
  {
    points_vec.push_back(it.getCoordinate());
    if (planningTree->isNodeROI(*it))
      labels_vec.push_back(2); // ROI
    else if (planningTree->isNodeOccupied(*it))
      labels_vec.push_back(1); // Occupied
    else
      labels_vec.push_back(0); // Free

  }
  tree_mtx.unlock();
  capnp::List<float>::Builder points = pc.initPoints(points_vec.size() * 3);
  capnp::List<uint8_t>::Builder labels = pc.initLabels(labels_vec.size());
  for (size_t i = 0; i < points_vec.size(); i++)
  {
    points.set(i*3 + 0, points_vec[i].x());
    points.set(i*3 + 1, points_vec[i].y());
    points.set(i*3 + 2, points_vec[i].z());
    labels.set(i, labels_vec[i]);
  }
}

void OctreeManager::generatePointcloud(vpp_msg::Pointcloud::Builder &pc, const octomap::point3d &center_point, uint16_t vx_cells)
{
  tree_mtx.lock();
  octomap::OcTreeKey center_key = planningTree->coordToKey(center_point);
  octomap::OcTreeKey start_key(center_key[0] - vx_cells/2, center_key[1] - vx_cells/2, center_key[2] - vx_cells/2);
  octomap::OcTreeKey end_key(start_key[0] + vx_cells - 1, start_key[1] + vx_cells - 1, start_key[2] + vx_cells - 1);
  std::vector<octomap::point3d> points_vec;
  std::vector<uint8_t> labels_vec;
  for (auto it = planningTree->begin_leafs_bbx(start_key, end_key), end = planningTree->end_leafs_bbx(); it != end; it++)
  {
    points_vec.push_back(it.getCoordinate());
    if (planningTree->isNodeROI(*it))
      labels_vec.push_back(2); // ROI
    else if (planningTree->isNodeOccupied(*it))
      labels_vec.push_back(1); // Occupied
    else
      labels_vec.push_back(0); // Free

  }
  tree_mtx.unlock();
  capnp::List<float>::Builder points = pc.initPoints(points_vec.size() * 3);
  capnp::List<uint8_t>::Builder labels = pc.initLabels(labels_vec.size());
  for (size_t i = 0; i < points_vec.size(); i++)
  {
    points.set(i*3 + 0, points_vec[i].x());
    points.set(i*3 + 1, points_vec[i].y());
    points.set(i*3 + 2, points_vec[i].z());
    labels.set(i, labels_vec[i]);
  }
}

void OctreeManager::generateVoxelgrid(vpp_msg::Voxelgrid::Builder &vx, const octomap::point3d &center_point, uint16_t vx_cells)
{
  capnp::List<uint8_t>::Builder labels = vx.initLabels(vx_cells*vx_cells*vx_cells);
  capnp::List<uint16_t>::Builder shape = vx.initShape(3);
  shape.set(0, vx_cells);
  shape.set(1, vx_cells);
  shape.set(2, vx_cells);
  capnp::List<float>::Builder center = vx.initCenter(3);
  center.set(0, center_point.x());
  center.set(1, center_point.y());
  center.set(2, center_point.z());

  tree_mtx.lock();
  vx.setResolution(planningTree->getResolution());
  octomap::OcTreeKey center_key = planningTree->coordToKey(center_point);
  octomap::OcTreeKey start_key(center_key[0] - vx_cells/2, center_key[1] - vx_cells/2, center_key[2] - vx_cells/2);
  size_t linear_index = 0;
  for (octomap::key_type z = start_key[2]; z < start_key[2] + vx_cells; z++)
  {
    for (octomap::key_type y = start_key[1]; y < start_key[1] + vx_cells; y++)
    {
      for (octomap::key_type x = start_key[0]; x < start_key[0] + vx_cells; x++)
      {
        octomap::OcTreeKey key(x, y, z);
        octomap_vpp::RoiOcTreeNode *node = planningTree->search(key);
        if (node == nullptr)
          labels.set(linear_index++, 3); // unknown
        else if (planningTree->isNodeROI(node))
          labels.set(linear_index++, 2); // ROI
        else if (planningTree->isNodeOccupied(node))
          labels.set(linear_index++, 1); // Occupied
        else
          labels.set(linear_index++, 0); // Free
      }
    }
  }
  tree_mtx.unlock();
}

void OctreeManager::generateFullVoxelgrid(vpp_msg::Voxelgrid::Builder &vx)
{
  octomap::point3d center = (wsMin + wsMax)*0.5;
  octomap::point3d size = (wsMax - wsMin) * (1.0/planningTree->getResolution());
  uint16_t vx_cells = 0;
  for (unsigned int i=0; i<3; i++)
  {
    uint16_t cells = static_cast<uint16_t>(size(i));
    if (cells > vx_cells) vx_cells = cells;
  }
  generateVoxelgrid(vx, center, vx_cells);
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

uint32_t OctreeManager::getRewardWithGt()
{
  tree_mtx.lock();
  octomap::KeySet roi_keys = planningTree->getRoiKeys();
  tree_mtx.unlock();

  uint32_t reward = 0;
  for (const octomap::OcTreeKey &key : roi_keys)
  {
    if (encountered_keys.find(key) != encountered_keys.end())
      continue; // Key already encountered
    else if (gtLoader->getFruitIndex(key) == 0)
      continue; // Key not in GT
    else
    {
      encountered_keys.insert(key);
      reward++;
    }
  }
  return reward;
}

uint32_t OctreeManager::getMaxGtReward()
{
  return gtLoader->getTotalFruitCellCount();
}

std::tuple<uint32_t, uint32_t> OctreeManager::getFoundFreeAndOccupied()
{
  uint32_t found_free = 0, found_occ = 0;
  tree_mtx.lock();
  for (auto it = planningTree->changedKeysBegin(), end = planningTree->changedKeysEnd(); it != end; it++)
  {
    const octomap::OcTreeKey &key = it->first;
    octomap_vpp::RoiOcTreeNode *node = planningTree->search(key);
    octomap_vpp::NodeState occ = planningTree->getNodeState(node, octomap_vpp::NodeProperty::OCCUPANCY);
    octomap_vpp::NodeState roi = planningTree->getNodeState(node, octomap_vpp::NodeProperty::ROI);
    if (roi == octomap_vpp::NodeState::OCCUPIED_ROI) continue; // ROI nodes counted by separate function
    else if (occ == octomap_vpp::NodeState::OCCUPIED_ROI) found_occ++;
    else if (occ == octomap_vpp::NodeState::FREE_NONROI) found_free++;
  }
  planningTree->resetChangeDetection();
  tree_mtx.unlock();
  return {found_free, found_occ};
}

bool OctreeManager::startEvaluator()
{
  eval_trial_num = 0;
  setEvaluatorStartParams();
  return true;
}

std::vector<octomap::point3d> OctreeManager::getRoiContours()
{
  std::vector<octomap::point3d> roi_contours;

  octomap::KeySet roi = planningTree->getRoiKeys();
  octomap::KeySet freeNeighbours;
  for (const octomap::OcTreeKey &key : roi)
  {
    planningTree->getNeighborsInState(key, freeNeighbours, octomap_vpp::NodeProperty::OCCUPANCY, octomap_vpp::NodeState::FREE_NONROI, octomap_vpp::NB_18);
  }
  for (const octomap::OcTreeKey &key : freeNeighbours)
  {
    if (planningTree->hasNeighborInState(key, octomap_vpp::NodeProperty::OCCUPANCY, octomap_vpp::NodeState::UNKNOWN, octomap_vpp::NB_18))
    {
      roi_contours.push_back(planningTree->keyToCoord(key));
    }
  }
  return roi_contours;
}

std::vector<octomap::point3d> OctreeManager::getOccContours()
{
  std::vector<octomap::point3d> occ_contours;
  octomap::point3d stMin_tf = transformToMapFrame(stMin), stMax_tf = transformToMapFrame(stMax);
  for (unsigned int i = 0; i < 3; i++)
  {
    if (stMin_tf(i) > stMax_tf(i))
      std::swap(stMin_tf(i), stMax_tf(i));
  }
  for (auto it = planningTree->begin_leafs_bbx(stMin_tf, stMax_tf), end = planningTree->end_leafs_bbx(); it != end; it++)
  {
    if (samplingTree != nullptr && samplingTree->search(transformToWorkspace(it.getCoordinate())) == nullptr)
    {
      continue; // sampling tree specified and sampled point not in sampling tree
    }
    if (it->getLogOdds() < 0) // is node free; TODO: replace with bounds later
    {
      if (planningTree->hasNeighborInState(it.getKey(), octomap_vpp::NodeProperty::OCCUPANCY, octomap_vpp::NodeState::UNKNOWN, octomap_vpp::NB_6) &&
          planningTree->hasNeighborInState(it.getKey(), octomap_vpp::NodeProperty::OCCUPANCY, octomap_vpp::NodeState::OCCUPIED_ROI, octomap_vpp::NB_6))
      {
        occ_contours.push_back(it.getCoordinate());
      }
    }
  }
  return occ_contours;
}

std::vector<octomap::point3d> OctreeManager::getFrontiers()
{
  std::vector<octomap::point3d> frontiers;
  octomap::point3d wsMin_tf = transformToMapFrame(wsMin), wsMax_tf = transformToMapFrame(wsMax);
  for (unsigned int i = 0; i < 3; i++)
  {
    if (wsMin_tf(i) > wsMax_tf(i))
      std::swap(wsMin_tf(i), wsMax_tf(i));
  }
  for (auto it = planningTree->begin_leafs_bbx(wsMin_tf, wsMax_tf), end = planningTree->end_leafs_bbx(); it != end; it++)
  {
    if (workspaceTree != nullptr && workspaceTree->search(transformToWorkspace(it.getCoordinate())) == nullptr)
    {
      continue; // sampling tree specified and sampled point not in sampling tree
    }
    if (it->getLogOdds() < 0) // is node free; TODO: replace with bounds later
    {
      if (planningTree->hasNeighborInState(it.getKey(), octomap_vpp::NodeProperty::OCCUPANCY, octomap_vpp::NodeState::UNKNOWN, octomap_vpp::NB_6))
      {
        frontiers.push_back(it.getCoordinate());
      }
    }
  }
  return frontiers;
}

void OctreeManager::setEvaluatorStartParams()
{
  std::string file_index_str = std::to_string(eval_trial_num);
  eval_resultsFile = std::ofstream("planner_results_" + file_index_str + ".csv");
  eval_resultsFileOld = std::ofstream("planner_results_old" + file_index_str + ".csv");
  eval_fruitCellPercFile = std::ofstream("results_fruit_cells_" + file_index_str + ".csv");
  eval_volumeAccuracyFile = std::ofstream("results_volume_accuracy_" + file_index_str + ".csv");
  eval_distanceFile = std::ofstream("results_distances_" + file_index_str + ".csv");
  eval_resultsFile << "Time (s),Plan duration (s),Plan Length,";
  evaluator->writeHeader(eval_resultsFile) << ",Step" << std::endl;
  eval_resultsFileOld << "Time (s),Plan duration (s),Plan Length,";
  evaluator->writeHeaderOld(eval_resultsFileOld) << ",Step" << std::endl;
  eval_plannerStartTime = ros::Time::now();
  eval_accumulatedPlanDuration = 0;
  eval_accumulatedPlanLength = 0;
}

template<typename T>
std::ostream& writeVector(std::ostream &os, double passed_time, const std::vector<T> &vec)
{
  os << passed_time << ",";
  for (size_t i = 0; i < vec.size(); i++)
  {
    os << vec[i];
    if (i < vec.size() - 1)
      os << ",";
  }
  return os;
}

bool OctreeManager::saveEvaluatorData(double plan_length, double traj_duration)
{
  ros::Time currentTime = ros::Time::now();

  double passed_time = (currentTime - eval_plannerStartTime).toSec();

  eval_accumulatedPlanDuration += traj_duration;
  eval_accumulatedPlanLength += plan_length;

  rvp_evaluation::EvaluationParameters res = evaluator->processDetectedRois(true, eval_trial_num, static_cast<size_t>(passed_time));
  rvp_evaluation::EvaluationParametersOld resOld = evaluator->processDetectedRoisOld();

  eval_resultsFile << passed_time << "," << eval_accumulatedPlanDuration << "," << eval_accumulatedPlanLength << ",";
  evaluator->writeParams(eval_resultsFile, res) << "," << eval_lastStep << std::endl;

  eval_resultsFileOld << passed_time << "," << eval_accumulatedPlanDuration << "," << eval_accumulatedPlanLength << ",";
  evaluator->writeParamsOld(eval_resultsFileOld, resOld) << "," << eval_lastStep << std::endl;

  writeVector(eval_fruitCellPercFile, passed_time, res.fruit_cell_percentages) << std::endl;
  writeVector(eval_volumeAccuracyFile, passed_time, res.volume_accuracies) << std::endl;
  writeVector(eval_distanceFile, passed_time, res.distances) << std::endl;

  return true;
}

bool OctreeManager::resetEvaluator()
{
  eval_resultsFile.close();
  eval_resultsFileOld.close();
  eval_fruitCellPercFile.close();
  eval_volumeAccuracyFile.close();
  eval_distanceFile.close();
  eval_trial_num++;
  setEvaluatorStartParams();
  return true;
}
