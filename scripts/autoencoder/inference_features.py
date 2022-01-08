import os
import argparse
import logging
import time

import numpy as np

from autoencoder.dataset.ae_dataset_with_features import make_data_loader_with_features
from autoencoder.dataset.data_reader import PointCloud
from autoencoder.network.mink_unet34 import MinkUNet34
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch.utils.data

import MinkowskiEngine as ME

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=32)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
# parser.add_argument("--weights", type=str, default="modelnet_completion_vae_Nov_30.pth")
parser.add_argument("--weights", type=str, default="modelnet_features_DEC13.pth")

parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)

# colors = np.array([[1, 1, 0], [1, 0, 0]])
SCANNET_COLOR_MAP = {
    0: (255., 0., 0.),  # red
    1: (0., 255., 0.),  # green
    2: (0., 0., 255.),  # blue
    3: (31., 119., 180.)
}

#{0,0,255},
# {255,0,0},
# {0,255,0},
# {0,0,51}

def visualize_truth(data_dict, batch_ind):
    point = data_dict["tensor_batch_truth_coordinates"][batch_ind]
    label = data_dict["tensor_batch_truth_feats"][batch_ind].numpy().squeeze()
    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in label]))
    opcd.translate([-2 * config.resolution, 0, 0])
    # opcd.estimate_normals()
    # opcd.rotate(M)
    return opcd


def visualize_truth2(data_dict, batch_ind):
    point = data_dict["tensor_batch_truth_coordinates"][batch_ind]
    label = data_dict["tensor_batch_truth_feats"][batch_ind].numpy().squeeze()
    point = point[label.argmax(1) != 0]
    label = label[label.argmax(1) != 0]

    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in label]))
    opcd.translate([-4 * config.resolution, 0, 0])
    # opcd.estimate_normals()
    # opcd.rotate(M)
    return opcd


def visualize_input(data_dict, batch_ind):
    point = data_dict["tensor_batch_crop_coordinates"][batch_ind]
    label = data_dict["tensor_batch_crop_feats"][batch_ind].numpy().squeeze()
    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l.argmax()] for l in label]))
    opcd.translate([0 * config.resolution, 0, 0])
    opcd.estimate_normals()
    # opcd.rotate(M)
    return opcd


def visualize_prediction(coords, feats=None):
    point = coords
    if feats is not None:
        label = np.around(feats.numpy()).squeeze()
        pcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l] for l in label]))
    else:
        pcd = PointCloud(point)
    pcd.translate([2 * config.resolution, 0, 0])
    pcd.estimate_normals()
    # pcd.rotate(M)
    return pcd


def visualize_hidden_code(batch_code_coord, batch_code_feats):
    codes = []
    for coords, feats in zip(batch_code_coord, batch_code_feats):
        coords = coords.numpy()
        feats = feats.numpy()
        code = np.zeros(shape=(config.resolution, config.resolution, config.resolution, 256))
        for coord, feat in zip(coords, feats):
            code[coord] = feat
            codes.append(code)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        feat = feats[:, 0]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(x, y, z, feat)
        plt.show()
        print("")

    return np.array(codes)


def visualize(net, dataloader, device, config):
    net.eval()
    n_vis = 0

    for data_dict in dataloader:
        start_time = time.time()
        sin = ME.SparseTensor(
            features=data_dict["crop_feats"],
            coordinates=ME.utils.batched_coordinates(data_dict["tensor_batch_crop_coordinates"]),
            device=device,
        )

        # Generate target sparse tensor
        cm = sin.coordinate_manager
        target_key, _ = cm.insert_and_map(
            coordinates=ME.utils.batched_coordinates(data_dict["tensor_batch_truth_coordinates"]).to(device),
            string_id="target",
        )

        code, sout = net(sin)
        print("inference time:{}".format(time.time() - start_time))
        batch_code_coord, batch_code_feats = code.decomposed_coordinates_and_features
        # visualize_hidden_code(batch_code_coord, batch_code_feats)
        batch_coords, batch_feats = sout.decomposed_coordinates_and_features

        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd_input = visualize_input(data_dict, b)
            pcd_truth = visualize_truth(data_dict, b)
            pcd_truth2 = visualize_truth2(data_dict, b)
            _, feats = feats.max(1)
            pcd = visualize_prediction(coords, feats)
            o3d.visualization.draw_geometries([pcd_input, pcd_truth, pcd_truth2, pcd])

            n_vis += 1
            if n_vis > config.max_visualization:
                return


if __name__ == '__main__':
    config = parser.parse_args()
    # config.weights = "modelnet_completion_vae_Nov_26.pth"
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = MinkUNet34(4, 4).to(device)

    logging.info(net)

    logging.info(f"Loading weights from {config.weights}")
    checkpoint = torch.load(config.weights)
    net.load_state_dict(checkpoint["state_dict"])

    # paths_to_data = ["/media/zeng/Data/dataset/Pheno4D/*/*.pcd",
    #                  "/media/zeng/Data/dataset/ModelNet40/chair/train/*.off"]

    paths_to_data = ["/home/zeng/catkin_ws/data/data_cvx_test2/*.cvx"]
    dataloader = make_data_loader_with_features(
        paths_to_data,
        "train",
        augment_data=True,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config,
        crop=True,
    )

    with torch.no_grad():
        visualize(net, dataloader, device, config)
