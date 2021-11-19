import os
import sys
import capnp
import glob

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', "capnp"))
import pointcloud_capnp

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
    return pcd


SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.)
}

parent_dir = "/home/zeng/catkin_ws/data"
fnames = glob.glob(os.path.join(parent_dir, f"*.cpc"))
pointclouds = []
resolution = 128
for path in fnames:
    print(path)
    f = open(path, 'rb')
    pointcloud = pointcloud_capnp.Pointcloud.read(f, traversal_limit_in_words=2 ** 63)
    points = np.reshape(np.array(pointcloud.points), (-1, 3))
    labels = np.array(pointcloud.labels).tolist()
    truth_color = np.array([SCANNET_COLOR_MAP[l] for l in labels])
    opcd = PointCloud(points, truth_color)
    opcd.translate([-0.6 * resolution, 0, 0])
    opcd.estimate_normals()
    opcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))

    o3d.visualization.draw_geometries([opcd])
    pointclouds.append(pointcloud.points)
print()
