from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data

import MinkowskiEngine as ME

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )


###############################################################################
# Utility functions
###############################################################################



def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def collate_pointcloud_fn(list_data):
    coords, feats, labels = list(zip(*list_data))

    # Concatenate all lists
    return {
        "coords": coords,
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "labels": torch.LongTensor(labels),
    }


def collate_pointcloud_fn_ae(list_data):
    coords, feats, labels = list(zip(*list_data))
    item = {
        "coords": ME.utils.batched_coordinates(coords),
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "labels": torch.LongTensor(labels),
    }
    # Concatenate all lists
    return item


class CollationAndTransformation:
    def __init__(self, resolution):
        self.resolution = resolution

    def random_crop(self, coords_list):
        crop_coords_list = []
        for coords in coords_list:
            sel = coords[:, 0] < self.resolution / 3
            crop_coords_list.append(coords[sel])
        return crop_coords_list

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        # coords : { list : 16 } : [Tensor:(4556,3),Tensor(3390,3) ... ] 一个batch中所有物体的所有points的坐标
        coords = self.random_crop(coords)
        # 裁掉了部分坐标
        item = {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "cropped_coords": coords,
            "labels": torch.LongTensor(labels),
        }
        # Concatenate all lists
        return item


class CollationAndTransformationAE:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        # coords : { list : 16 } : [Tensor:(4556,3),Tensor(3390,3) ... ] 一个batch中所有物体的所有points的坐标
        # 裁掉了部分坐标
        item = {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "cropped_coords": coords,
            "labels": torch.LongTensor(labels),
        }
        # Concatenate all lists
        return item


def make_data_loader(path_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config):
    dset = ModelNet40Dataset(path_to_data, phase, config=config)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


def make_data_loader_completion(path_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config):
    dset = ModelNet40Dataset(path_to_data, phase, config=config)

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


def make_pheno4d_data_loader_ae(path_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config):
    dset = Pheno4DDataset(path_to_data, phase, config=config)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformationAE(config.resolution),
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


def make_pheno4d_data_loader(path_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config):
    dset = Pheno4DDataset(path_to_data, phase, config=config)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader
