import numpy as np

# Must be imported before large libs
from autoencoder.ae_dataset import PointCloud

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.)
}


def visualize_coords_features(pred_coords, truth_coords, resolution=128, pred_feats=None, truth_feats=None):
    """
    visualize coords and features
    Args:
        pred_coords:
        truth_coords:
        resolution:
        pred_feats:
        truth_feats:

    Returns:

    """
    # Map color
    pred_color = None
    truth_color = None
    if pred_feats is not None:
        pred_feats = pred_feats.squeeze().numpy()
        pred_color = np.array([SCANNET_COLOR_MAP[l] for l in pred_feats])
    if truth_feats is not None:
        truth_feats = truth_feats.squeeze().numpy()
        truth_color = np.array([SCANNET_COLOR_MAP[l] for l in truth_feats])

    pcd = PointCloud(pred_coords, pred_color)
    pcd.estimate_normals()
    pcd.translate([0.6 * resolution, 0, 0])
    pcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))

    opcd = PointCloud(truth_coords, truth_color)
    opcd.translate([-0.6 * resolution, 0, 0])
    opcd.estimate_normals()
    opcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
    o3d.visualization.draw_geometries([pcd, opcd])
