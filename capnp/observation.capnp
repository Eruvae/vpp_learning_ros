@0xb6359c08aa155a42;  # unique file ID, generated by `capnp id`

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("vpp_msg");

using import "pose.capnp".Pose;
using import "pointcloud.capnp".Pointcloud;
using import "voxelgrid.capnp".Voxelgrid;

struct Observation {
  map :union {
    countMap :group{
      unknownCount @0 :List(UInt32);
      freeCount @1 :List(UInt32);
      occupiedCount @2 :List(UInt32);
      roiCount @3 :List(UInt32);
      width @4 :UInt32;
      height @5 :UInt32;
      layers @6 :UInt32;
    }
    pointcloud @13 :Pointcloud;
    voxelgrid @14 :Voxelgrid;
    fullVoxelgrid @15 :Voxelgrid;
  }

  foundRois @7 :UInt32;
  planningTime @8 :Float64;

  robotPose @9 :Pose;
  robotJoints @10 :List(Float64);

  totalRoiCells @11 :UInt32;
  evalTotalTrajectoryDuration @12 :Float64;

  foundFree @16 :UInt32;
  foundOcc @17 :UInt32;

  hasMoved @18 :Bool;
}
