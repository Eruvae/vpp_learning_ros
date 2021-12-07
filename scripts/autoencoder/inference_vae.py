import os
import argparse
import logging
import numpy as np
import urllib

# Must be imported before large libs
from autoencoder.dataset.ae_dataset import make_data_loader, PointCloud
from autoencoder.network.network_vae import VAE

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
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
parser.add_argument("--resolution", type=int, default=128)
parser.add_argument("--max_iter", type=int, default=30000)
parser.add_argument("--val_freq", type=int, default=100)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_vae_no_global.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)


def visualize(net, dataloader, device, config):
    net.eval()
    crit = nn.BCEWithLogitsLoss()
    n_vis = 0

    for data_dict in dataloader:

        sin = ME.SparseTensor(
            features=data_dict["feats"],
            coordinates=data_dict["coords"].int(),
            device=device,
        )

        # Generate target sparse tensor
        target_key = sin.coordinate_map_key

        out_cls, targets, sout, means, log_vars, zs = net(sin, target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        KLD = -0.5 * torch.mean(
            torch.sum(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1)
        )
        loss = KLD + BCE

        print(loss)

        batch_coords, batch_feats = sout.decomposed_coordinates_and_features
        for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
            pcd = PointCloud(coords)
            pcd.estimate_normals()
            pcd.translate([0.6 * config.resolution, 0, 0])
            pcd.rotate(M)
            opcd = PointCloud(data_dict["xyzs"][b])
            opcd.translate([-0.6 * config.resolution, 0, 0])
            opcd.estimate_normals()
            opcd.rotate(M)
            o3d.visualization.draw_geometries([pcd, opcd])

            n_vis += 1
            if n_vis > config.max_visualization:
                return


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = VAE()
    net.to(device)

    logging.info(net)

    if not os.path.exists(config.weights):
        logging.info(f"Downloaing pretrained weights. This might take a while...")
        urllib.request.urlretrieve(
            "https://bit.ly/39TvWys", filename=config.weights
        )

    logging.info(f"Loading weights from {config.weights}")
    checkpoint = torch.load(config.weights)
    net.load_state_dict(checkpoint["state_dict"])

    paths_to_data = ["/media/zeng/Data/dataset/Pheno4D/*/*.pcd",
                     "/media/zeng/Data/dataset/ModelNet40/chair/train/*.off"]

    # paths_to_data = ["/home/zeng/catkin_ws/data/*.cpc", "/home/zeng/catkin_ws/data/*/*.cpc"]
    dataloader = make_data_loader(
        paths_to_data,
        "train",
        augment_data=True,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        repeat=True,
        config=config,
        train=False
    )

    with torch.no_grad():
        visualize(net, dataloader, device, config)
