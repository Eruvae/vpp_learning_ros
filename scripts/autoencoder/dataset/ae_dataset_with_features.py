import os
import sys
import glob
import numpy as np

from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data
import MinkowskiEngine as ME
import capnp

from autoencoder.dataset.ae_dataset import CollationAndTransformation, random_crop
from autoencoder.dataset.data_reader import InfSampler, normalize

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', "capnp"))
import pointcloud_capnp


class AEDatasetWithFeatures(torch.utils.data.Dataset):
    # TODO 改这个地方
    def __init__(self, paths_to_data, phase, transform=None, config=None, crop=False, return_cropped_original=True):
        self.phase = phase
        self.cache = {}
        self.transform = transform
        self.resolution = config.resolution
        self.crop = crop
        # load from path
        fnames = []
        for paths_to_data in paths_to_data:
            fnames.extend(glob.glob(paths_to_data))
        self.files = fnames
        self.return_cropped_original = return_cropped_original

        # loading into cache
        for file_path in self.files:
            file = open(file_path, 'rb')
            pointcloud = pointcloud_capnp.Pointcloud.read(file, traversal_limit_in_words=2 ** 63)

            # TODO normalize
            # points = np.reshape(np.array(pointcloud.points), (-1, 3))
            # labels = np.array(pointcloud.labels).tolist()
            points = np.reshape(np.array(pointcloud.points), (-1, 3))
            labels = np.array(pointcloud.labels)
            # points_roi = points[np.where(labels == 2)]
            # labels_roi = labels[np.where(labels == 2)]
            # points = normalize(points)
            points_roi = points[np.where(labels != 0)]
            labels_roi = labels[np.where(labels != 0)]

            labels_roi_one_hot = to_one_hot(labels_roi - 1)

            if len(points_roi) >= 100:
                # points_roi = normalize(points_roi)
                self.cache[len(self.cache)] = [points_roi, labels_roi_one_hot]
                print("Add file:{} into cache!".format(file))
            else:
                print("File:{} too small!".format(file))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        points, labels = self.cache[idx]
        # labels_one_hot = np.eye(2, dtype=np.int8)[labels - 1]
        coords, _, feats = quantize_coordinates_with_feats(points, feats=labels,
                                                           resolution=self.resolution)
        # TODO rotate the pointcloud
        coords_crop, feats_crop, coords_truth, feats_truth = random_crop(coords, feats, self.resolution,
                                                                         partial_rate=0.5)

        return coords_crop, feats_crop, coords_truth, feats_truth, idx


def quantize_coordinates_with_feats(xyz, feats, resolution):
    """

    Args:
        xyz:
        feats:
        resolution:

    Returns: quantized coordinates,
             original coordinates at inds of quantized coordinates,
             features at inds of quantized coordinates

    """
    # Use labels (free, occupied, ROI) as features
    if feats.ndim < 2:
        feats = np.expand_dims(feats, axis=1)
    # feats = np.ones((len(xyz), 1))

    # Get coords
    xyz = xyz * resolution
    quantized_coords, feats_at_inds, inds = ME.utils.sparse_quantize(xyz, features=feats, return_index=True)

    original_coords_at_inds = xyz[inds]

    return quantized_coords, original_coords_at_inds, feats_at_inds


def to_one_hot(indexes):
    one_hot_ = []
    for ind in indexes:
        one_hot_.append(np.eye(2, dtype=np.int8)[ind])
    return np.array(one_hot_)


def make_data_loader_with_features(paths_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config,
                                   crop):
    dset = AEDatasetWithFeatures(paths_to_data, phase, config=config, crop=crop)
    print("dataset size:{}".format(len(dset)))

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformation(config.resolution),
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader
