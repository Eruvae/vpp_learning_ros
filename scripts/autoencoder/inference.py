import argparse
import logging
import os

import numpy as np

from autoencoder.ae_dataset import load_mesh_file, quantize_coordinates, load_pcd_file, PointCloud, \
    quantize_coordinates_with_feats, construct_data_batch, construct_input_and_target_key, \
    load_batches_points_labels_from_pickle
from autoencoder.network_complete import CompletionShadowNet
from autoencoder.visualize import visualize_coords_features

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")

import torch
import torch.nn as nn
import torch.utils.data


def inference(net, file_paths, device, resolution):
    for file_path in file_paths:
        if file_path.endswith(".off"):
            xyz = load_mesh_file(mesh_file_path=file_path)
        else:
            xyz = load_pcd_file(pcd_file_path=file_path)

        in_feats = torch.ones_like(xyz)

        _inference_coords_feats_in_one_object(net, xyz, in_feats, device, resolution)


def _inference_coords_feats_in_one_object(net, xyz, feats, device, resolution):
    net.eval()
    crit = nn.BCEWithLogitsLoss()

    quantized_coords, original_coords, feats_at_inds = quantize_coordinates_with_feats(xyz, feats=feats,
                                                                                       resolution=resolution)

    data_batch_dict = construct_data_batch([quantized_coords], [original_coords], [feats_at_inds])

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
        visualize_coords_features(pred_coords, truth_coords, resolution, pred_feats=None, truth_feats=truth_feats)


def load_weight(net, weights):
    if not os.path.exists(weights):
        logging.info(f"Weight file does not exist!")

    logging.info(f"Loading weights from {weights}")
    checkpoint = torch.load(weights)
    net.load_state_dict(checkpoint["state_dict"])


if __name__ == '__main__':
    resolution = 128
    weights = "trained_models/modelnet_completion.pth"
    device = torch.device("cpu")

    net = CompletionShadowNet(resolution).to(device)
    logging.info(net)

    load_weight(net, weights)

    points_batch, labels_batch = load_batches_points_labels_from_pickle(
        pickle_file_path="temp_data/observation_48.obj")

    for points, labels in zip(points_batch, labels_batch):
        feats = labels
        _inference_coords_feats_in_one_object(net, points, feats, device, resolution)
