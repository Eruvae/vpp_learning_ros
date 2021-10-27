import os
import random
import sys
import subprocess
import argparse
import logging
import glob
import numpy as np
from time import time
import urllib

from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import MinkowskiEngine as ME

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )


def make_data_loader(paths_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config):
    dset = AEDataset(paths_to_data, phase, config=config)

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


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, paths_to_data, phase, transform=None, config=None):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        # self.root = path_to_data
        # glob.glob(os.path.join(self.root, "*/*.pcd"))
        # fnames = glob.glob(os.path.join(self.root, "chair/train/*.off"))
        fnames = []
        for path_to_data in paths_to_data:
            fnames.extend(glob.glob(path_to_data))
        # fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} : {len(self.files)} files"
        )

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            xyz = self.cache[idx]
        else:
            assert os.path.exists(self.files[idx])
            if self.files[idx].endswith(".off"):
                xyz = load_mesh_file(mesh_file=self.files[idx])
            else:
                xyz = load_pcd_file(pcd_file=self.files[idx])

            self.cache[idx] = xyz
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if (
                    cache_percent > 0
                    and cache_percent % 10 == 0
                    and cache_percent != self.last_cache_percent
            ):
                logging.info(
                    f"Cached {self.phase}: {len(self.cache)} / {len(self)}: {cache_percent}%"
                )
                self.last_cache_percent = cache_percent

        if len(xyz) < 1000:
            logging.info(
                f"Skipping {self.files[idx]}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
            )
            return None

        coords, xyz_inds = quantize(xyz, self.resolution, transform=None)

        return (coords, xyz_inds, idx)


def load_pcd_file(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
    vertices = np.asarray(pcd.points)
    vmax = vertices.max(0, keepdims=True)
    vmin = vertices.min(0, keepdims=True)
    pcd.points = o3d.utility.Vector3dVector(
        (vertices - vmin) / (vmax - vmin).max()
    )
    xyz = np.asarray(pcd.points)

    return xyz


def load_mesh_file(mesh_file):
    density = 30000
    pcd = o3d.io.read_triangle_mesh(mesh_file)
    # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
    vertices = np.asarray(pcd.vertices)

    vmax = vertices.max(0, keepdims=True)
    vmin = vertices.min(0, keepdims=True)
    pcd.vertices = o3d.utility.Vector3dVector(
        (vertices - vmin) / (vmax - vmin).max()
    )
    # xyz = np.asarray(pcd.points)
    xyz = resample_mesh(pcd, density=density)

    return xyz


def quantize(xyz, resolution, transform):
    # Use color or other features if available
    feats = np.ones((len(xyz), 1))

    if transform:
        xyz, feats = transform(xyz, feats)

    # Get coords
    xyz = xyz * resolution
    coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)
    return coords, xyz[inds]


def quantize_with_feats(xyz, feats, resolution, transform):
    # Use labels (free, occupied, ROI) as features
    feats = np.expand_dims(feats, axis=1)

    if transform:
        xyz, feats = transform(xyz, feats)

    # Get coords
    xyz = xyz * resolution
    coords, feats, inds = ME.utils.sparse_quantize(xyz, features=feats, return_index=True)
    return coords, xyz[inds], feats


def resample_mesh(mesh_cad, density=1):
    """
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    Samples point cloud on the surface of the model defined as vectices and
    faces. This function uses vectorized operations so fast at the cost of some
    memory.

    param mesh_cad: low-polygon triangle mesh in o3d.geometry.TriangleMesh
    param density: density of the point cloud per unit area
    param return_numpy: return numpy format or open3d pointcloud format
    return resampled point cloud

    Reference :
      [1] Barycentric coordinate system
      \begin{align}
        P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
      \end{align}
    """
    faces = np.array(mesh_cad.triangles).astype(int)
    vertices = np.array(mesh_cad.vertices)

    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))

    n_samples = (np.sum(face_areas) * density).astype(int)
    # face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Bug fix by Yangyan (yangyan.lee@gmail.com)
    n_samples_per_face = np.ceil(density * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]

    P = (
            (1 - np.sqrt(r[:, 0:1])) * A
            + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
            + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )

    return P


class CollationAndTransformation:
    def __init__(self, resolution):
        self.resolution = resolution

    def random_crop(self, coords_list, feats_list):
        crop_coords_list = []
        crop_feats_list = []
        # ranges = [self.resolution / 3, self.resolution / 2, self.resolution * 2 / 3, self.resolution]
        for coords, feats in zip(coords_list, feats_list):
            rand_idx = random.randint(0, int(self.resolution * 0.66))
            # range_ = np.random.randint(0, len(ranges))

            sel0 = coords[:, 0] > rand_idx
            # max_range = min(self.resolution, rand_idx + range_)
            max_range = self.resolution / 3 + rand_idx
            sel1 = coords[:, 0] < max_range
            sel = sel0 * sel1
            crop_coords_list.append(coords[sel])
            crop_feats_list.append(feats[sel])
        return crop_coords_list, crop_feats_list

    def __call__(self, list_data):
        coords, feats, labels = list(zip(*list_data))
        # coords : { list : 16 } : [Tensor:(4556,3),Tensor(3390,3) ... ] 一个batch中所有物体的所有points的坐标
        coords, feats = self.random_crop(coords, feats)
        # 裁掉了部分坐标
        item = {
            "coords": ME.utils.batched_coordinates(coords),
            "xyzs": [torch.from_numpy(feat).float() for feat in feats],
            "cropped_coords": coords,
            "labels": torch.LongTensor(labels),
        }
        # Concatenate all lists
        return item


def random_crop(coords_list, resolution):
    crop_coords_list = []
    for coords in coords_list:
        sel = coords[:, 0] < resolution / 3
        crop_coords_list.append(coords[sel])
    return crop_coords_list


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)
