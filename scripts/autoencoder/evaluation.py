# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import random
import sys
import subprocess
import argparse
import logging
import numpy as np
from time import time
import urllib

# Must be imported before large libs
from autoencoder.ae_dataset import load_mesh_file, quantize_coordinates, load_pcd_file, PointCloud, \
    construct_data_batch, quantize_coordinates_with_feats, construct_input_and_target_key
from autoencoder.visualize import visualize_coords_features
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

assert (
        int(o3d.__version__.split(".")[1]) >= 8
), f"Requires open3d version >= 0.8, the current version is {o3d.__version__}"

if not os.path.exists("ModelNet40"):
    logging.info("Downloading the pruned ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=200)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_completion.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)


###############################################################################
# End of utility functions
###############################################################################


def visualize_one(net, file, device, config, transform=None):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0
    if file.endswith(".off"):
        xyz = load_mesh_file(mesh_file_path=file)
    else:
        xyz = load_pcd_file(pcd_file_path=file)

    truth_feats = torch.ones_like(xyz)

    quantized_coords, original_coords = quantize_coordinates_with_feats(xyz, feats=truth_feats,
                                                                        resolution=config.resolution)
    data_batch_dict = construct_data_batch([quantized_coords], [original_coords], [truth_feats])

    sparse_input, target_key = construct_input_and_target_key(data_batch_dict, device)

    # Generate from a dense tensor
    out_cls, targets, sout = net(sparse_input, target_key)
    num_layers, loss = len(out_cls), 0
    for out_cl, target in zip(out_cls, targets):
        loss += (
                crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
                / num_layers
        )

    pred_batch_coords, pred_batch_feats = sout.decomposed_coordinates_and_features
    truth_batch_coords, truth_batch_feats = sparse_input.decomposed_coordinates_and_features

    for b, (pred_coords, pred_feats, truth_coords, truth_feats) in enumerate(
            zip(pred_batch_coords, pred_batch_feats, truth_batch_coords, truth_batch_feats)):

        visualize_coords_features(pred_coords, truth_coords, config.resolution, pred_feats, truth_feats)

        n_vis += 1
        if n_vis > config.max_visualization:
            return


def visualize(net, dataloader, device, config):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:
        in_feat = torch.ones((len(data_dict["coords"]), 1))

        sin = ME.SparseTensor(
            features=in_feat,
            coordinates=data_dict["coords"],
            device=device,
        )

        # Generate target sparse tensor
        cm = sin.coordinate_manager
        target_key, _ = cm.insert_and_map(
            ME.utils.batched_coordinates(data_dict["xyzs"]).to(device),
            string_id="target",
        )

        # Generate from a dense tensor
        out_cls, targets, sout = net(sin, target_key)
        num_layers, loss = len(out_cls), 0
        for out_cl, target in zip(out_cls, targets):
            loss += (
                    crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
                    / num_layers
            )

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd = PointCloud(coords)
            pcd.estimate_normals()
            pcd.translate([0.6 * config.resolution, 0, 0])
            pcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            opcd = PointCloud(data_dict["cropped_coords"][b])
            opcd.translate([-0.6 * config.resolution, 0, 0])
            opcd.estimate_normals()
            opcd.rotate(M, np.array([[0.0], [0.0], [0.0]]))
            o3d.visualization.draw_geometries([pcd, opcd])

            n_vis += 1
            if n_vis > config.max_visualization:
                return


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")
    # path_to_data = "/media/zeng/Data/dataset/ModelNet40"
    path_to_data = "/media/zeng/Data/dataset/Pheno4D"

    net = CompletionShadowNet(config.resolution)
    net.to(device)

    logging.info(net)

    if not os.path.exists(config.weights):
        logging.info(f"Downloaing pretrained weights. This might take a while...")
        urllib.request.urlretrieve(
            "https://bit.ly/36d9m1n", filename=config.weights
        )

    logging.info(f"Loading weights from {config.weights}")
    checkpoint = torch.load(config.weights)
    net.load_state_dict(checkpoint["state_dict"])

    # visualize(net, dataloader, device, config)
    pcd_path = "/media/zeng/Data/dataset/Pheno4D/Maize01/M01_0313_a_label1.pcd"
    mesh_path = "/media/zeng/Data/dataset/ModelNet40/airplane/test/airplane_0627.off"
    visualize(net, pcd_path, device, config)
