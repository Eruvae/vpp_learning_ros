import os
import sys
import subprocess
import argparse
import logging
import numpy as np
from time import time
import urllib

# Must be imported before large libs
from autoencoder.ae_dataset import make_data_loader, make_data_loader_with_features
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
parser.add_argument("--val_freq", type=int, default=50)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--stat_freq", type=int, default=50)
parser.add_argument("--weights", type=str, default="modelnet_vae.pth")
parser.add_argument("--load_optimizer", type=str, default="true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--max_visualization", type=int, default=4)


def train_vae(dataloader, device, config):
    net = VAE().to(device)
    logging.info(net)

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    crit = nn.BCEWithLogitsLoss()

    start_iter = 0

    net.train()
    train_iter = iter(dataloader)
    logging.info(f"LR: {scheduler.get_lr()}")

    for i in range(start_iter, config.max_iter):

        s = time()
        data_dict = train_iter.next()
        d = time() - s

        optimizer.zero_grad()
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
            # BCE += curr_loss * weight[count]

        KLD = -0.5 * torch.mean(
            torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1)
        )
        loss = KLD + BCE

        loss.backward()
        optimizer.step()
        t = time() - s

        if i % config.stat_freq == 0:
            logging.info(
                f"Iter: {i}, Loss: {loss.item():.3e}, Depths: {len(out_cls)} Data Loading Time: {d:.3e}, Tot Time: {t:.3e}"
            )

        if i % config.val_freq == 0 and i > 0:
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                },
                config.weights,
            )

            scheduler.step()
            logging.info(f"LR: {scheduler.get_lr()}")

            net.train()


if __name__ == "__main__":
    config = parser.parse_args()
    logging.info(config)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # path_to_data = "/media/zeng/Data/dataset/ModelNet40"
    # paths_to_data = ["/media/zeng/Data/dataset/Pheno4D/*/*.pcd"]
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
        train=True
    )

    train_vae(dataloader, device, config)
