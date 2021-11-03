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
import pickle
import sys
import subprocess
import argparse
import logging

import numpy as np

from autoencoder.ae_dataset import load_batches_points_labels_from_pickle
from autoencoder.inference import inference_one_live
from autoencoder.network_complete import CompletionShadowNet
from vpp_env_client import EnvironmentClient

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import MinkowskiEngine as ME

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
parser.add_argument("--weights", type=str, default="autoencoder/trained_models/modelnet_completion.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)


###############################################################################
# End of utility functions
###############################################################################
def connect_data(num, file_path):
    """
    connect data
    Args:
        num: the size of the data
        file_path: path to save data

    Returns:

    """
    client = EnvironmentClient(handle_simulation=False)
    dataset = []
    for i in range(num):
        print("The {}-th observation".format(i))
        points, labels, robotPose, robotJoints, reward = client.sendReset(map_type='pointcloud')
        dataset.append([points, labels])
    pickle.dump(dataset, open(file_path, 'wb'))


if __name__ == "__main__":
    # connect_data(num=48, file_path="autoencoder/temp_data/observation_48.obj")
    points_batch, labels_batch = load_batches_points_labels_from_pickle(
        pickle_file_path="autoencoder/temp_data/observation_48.obj")

    config = parser.parse_args()
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    net = CompletionShadowNet(config.resolution)
    net.to(device)

    logging.info(net)

    logging.info(f"Loading weights from {config.weights}")
    checkpoint = torch.load(config.weights)
    net.load_state_dict(checkpoint["state_dict"])

    # client = EnvironmentClient(handle_simulation=False)
    # points, labels, robotPose, robotJoints, reward = client.sendReset(map_type='pointcloud')
    points_batch, labels_batch = load_batches_points_labels_from_pickle(
        pickle_file_path="autoencoder/temp_data/observation_48.obj")
    points, labels = points_batch[0], labels_batch[0]
    inference_one_live(net, points, labels, device, config)
