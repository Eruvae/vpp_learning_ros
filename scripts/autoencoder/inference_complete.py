import os
import sys
import subprocess
import argparse
import logging
import numpy as np
from time import time
import urllib

# Must be imported before large libs
from autoencoder.ae_dataset import make_data_loader, make_data_loader_with_features, PointCloud
from autoencoder.network_vae import VAE
from network_complete import CompletionShadowNet

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import MinkowskiEngine as ME

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_completion_Nov_22.pth")
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


def visualize_truth(data_dict, batch_ind):
    point = data_dict["tensor_batch_truth_coordinates"][batch_ind]
    label = data_dict["tensor_batch_truth_feats"][batch_ind].numpy().squeeze()
    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l] for l in label]))
    opcd.translate([-2 * config.resolution, 0, 0])
    # opcd.estimate_normals()
    # opcd.rotate(M)
    return opcd


def visualize_input(data_dict, batch_ind):
    point = data_dict["tensor_batch_crop_coordinates"][batch_ind]
    label = data_dict["tensor_batch_crop_feats"][batch_ind].numpy().squeeze()
    opcd = PointCloud(point, np.array([SCANNET_COLOR_MAP[l] for l in label]))
    opcd.translate([2 * config.resolution, 0, 0])
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
    pcd.translate([3 * config.resolution, 0, 0])
    pcd.estimate_normals()
    # pcd.rotate(M)
    return pcd


def visualize(net, dataloader, device, config):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:

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

        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            a1 = out_cl.F.squeeze()
            a2 = target.type(out_cl.F.dtype).to(device)
            curr_loss = crit(a1, a2)
            losses.append(curr_loss.item())
            loss += curr_loss / num_layers

        print(loss)

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        # batch_coords2, cls = out_cls.decomposed_coordinates_and_features
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd_input = visualize_input(data_dict, b)
            pcd_truth = visualize_truth(data_dict, b)
            pcd = visualize_prediction(coords)
            o3d.visualization.draw_geometries([pcd_input, pcd_truth, pcd])

            n_vis += 1
            if n_vis > config.max_visualization:
                return


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = CompletionShadowNet().to(device)

    logging.info(net)

    if not os.path.exists(config.weights):
        logging.info(f"Downloaing pretrained weights. This might take a while...")
        urllib.request.urlretrieve(
            "https://bit.ly/39TvWys", filename=config.weights
        )

    logging.info(f"Loading weights from {config.weights}")
    checkpoint = torch.load(config.weights)
    net.load_state_dict(checkpoint["state_dict"])

    # paths_to_data = ["/media/zeng/Data/dataset/Pheno4D/*/*.pcd",
    #                  "/media/zeng/Data/dataset/ModelNet40/chair/train/*.off"]

    paths_to_data = ["/home/zeng/catkin_ws/data/test/*.cpc"]
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
