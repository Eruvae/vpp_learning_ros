import os
import sys

import pickle
import random
import logging
import glob
import numpy as np

from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data
import MinkowskiEngine as ME
import capnp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', "capnp"))
import pointcloud_capnp

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )


def filter_0_size_point_clouds(points_batch, labels_batch):
    for i in range(len(points_batch) - 1, -1, -1):
        temp = points_batch[i].shape[0]
        if temp == 0:
            points_batch = np.delete(points_batch, i)
            labels_batch = np.delete(labels_batch, i)
    return points_batch, labels_batch


def load_batches_points_labels_from_pickle(pickle_file_path):
    dataset = pickle.load(open(pickle_file_path, 'rb'))
    dataset = np.array(dataset)
    points_batch = dataset[:, 0]
    labels_batch = dataset[:, 1]

    points_batch, labels_batch = filter_0_size_point_clouds(points_batch, labels_batch)

    # normalize the coordinates
    xyzs = []
    for i in range(len(points_batch)):
        points = normalize(points_batch[i])
        xyzs.append(points)

    # list to array
    xyzs = np.array(xyzs)
    labels_batch = np.array(labels_batch)

    return xyzs, labels_batch


def load_pcd_file(pcd_file_path):
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = normalize(pcd.points)

    xyz = np.asarray(points)

    return xyz


def load_mesh_file(mesh_file_path):
    density = 30000
    pcd = o3d.io.read_triangle_mesh(mesh_file_path)
    # Normalize to fit the mesh inside a unit cube while preserving aspect ratio
    pcd.vertices = o3d.utility.Vector3dVector(normalize(pcd.vertices))
    xyz = resample_mesh(pcd, density=density)
    return xyz


def construct_data_batch(coords_crop, feats_crop, coords_truth, feats_truth):
    coords_crop_batch = ME.utils.batched_coordinates(coords_crop)
    feats_crop_batch = ME.utils.batched_coordinates(feats_crop)

    data_batch_dict = {
        # "batched_crop_coordinates": coords_crop_batch,
        "crop_feats": torch.cat([torch.Tensor(feats).float() for feats in feats_crop]),
        "tensor_batch_crop_coordinates": [coord.float() for coord in coords_crop],
        "tensor_batch_crop_feats": [torch.from_numpy(feats).float() for feats in feats_crop],
        "tensor_batch_truth_coordinates": [coord_truth for coord_truth in coords_truth],
        "tensor_batch_truth_feats": [torch.from_numpy(feats).float() for feats in feats_truth]
    }
    return data_batch_dict


def construct_input_and_target_key(data_batch_dict, device):
    sparse_input = ME.SparseTensor(
        features=data_batch_dict["feats"],
        coordinates=data_batch_dict["coords"],
        device=device,
    )

    # Generate target sparse tensor, the input and the target share the same coordinate_manager
    cm = sparse_input.coordinate_manager
    target_key, _ = cm.insert_and_map(
        ME.utils.batched_coordinates(data_batch_dict["xyzs"]).to(device),
        string_id="target",
    )
    return sparse_input, target_key


def quantize_coordinates(xyz, resolution):
    """

    Args:
        xyz:
        resolution:

    Returns: quantized coordinates, original coordinates

    """
    # Get coords
    xyz = xyz * resolution
    quantized_coords, inds = ME.utils.sparse_quantize(xyz, return_index=True)
    original_coords = xyz[inds]
    return quantized_coords, original_coords


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


def normalize(vertices):
    """
    normalize vertices
    Args:
        vertices:

    Returns:

    """
    # Normalize to a unit cube while preserving aspect ratio
    vertices = np.asarray(vertices)
    vmax = vertices.max(0, keepdims=True)
    vmin = vertices.min(0, keepdims=True)
    normalized_vertices = (vertices - vmin) / (vmax - vmin).max()
    return normalized_vertices


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


def make_data_loader(paths_to_data, phase, augment_data, batch_size, shuffle, num_workers, repeat, config, train,
                     return_cropped_original):
    dset = AEDataset(paths_to_data, phase, config=config, train=train,
                     return_cropped_original=return_cropped_original)

    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": CollationAndTransformation(config.resolution),
        "pin_memory": False,
        "drop_last": True,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader


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
            # points = np.reshape(np.array(pointcloud.points), (-1, 3))
            # labels = np.array(pointcloud.labels).tolist()
            points = np.reshape(np.array(pointcloud.points), (-1, 3))
            labels = np.array(pointcloud.labels)
            points_occ = points[np.where(labels != 0)]
            labels_occ = labels[np.where(labels != 0)]
            points_roi = points[np.where(labels != 0)]
            labels_roi = labels[np.where(labels != 0)]
            if len(points_occ) >= 1000:
                self.cache[len(self.cache)] = [points_occ, labels_occ]
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


class AEDataset(torch.utils.data.Dataset):
    def __init__(self, paths_to_data, phase, transform=None, config=None, train=True, return_cropped_original=True):
        self.phase = phase
        self.files = []
        self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0
        self.train = train
        fnames = []
        for path_to_data in paths_to_data:
            fnames.extend(glob.glob(path_to_data))
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} : {len(self.files)} files"
        )
        self.return_cropped_original = return_cropped_original
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
                xyz = load_mesh_file(mesh_file_path=self.files[idx])
            else:
                xyz = load_pcd_file(pcd_file_path=self.files[idx])

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

        feats = torch.ones(len(xyz), 1)
        quantized_coords, original_coords, feats = quantize_coordinates_with_feats(xyz, feats=feats,
                                                                                   resolution=self.resolution)
        if self.train:
            coords_crop, feats_crop, coords_truth, feats_truth = random_crop(quantized_coords, feats, self.resolution)
            # print(coords.size())
            return (coords, ori_coords, feats, idx)
        else:
            return (quantized_coords, original_coords, feats, idx)


def random_crop(coords, feats, resolution, partial_rate=0.8):
    """
    take 1/2 part of the object
    Args:
        coords:
        feats:
        resolution:

    Returns:

    """
    # partial_rate = 0.8
    rand_idx = random.randint(0, int(resolution * (1 - partial_rate)))
    rand_idy = random.randint(0, int(resolution * (1 - partial_rate)))
    rand_idz = random.randint(0, int(resolution * (1 - partial_rate)))
    max_range_x = int(resolution * partial_rate + rand_idx)
    max_range_y = int(resolution * partial_rate + rand_idy)
    max_range_z = int(resolution * partial_rate + rand_idz)

    sel0_x = coords[:, 0] > rand_idx
    sel1_x = coords[:, 0] < max_range_x
    sel_x = sel0_x * sel1_x

    sel0_y = coords[:, 1] > rand_idy
    sel1_y = coords[:, 1] < max_range_y
    sel_y = sel0_y * sel1_y

    sel0_z = coords[:, 2] > rand_idz
    sel1_z = coords[:, 2] < max_range_z
    sel_z = sel0_z * sel1_z

    sel = sel_x * sel_y * sel_z
    coords_crop = coords[sel]
    feats_crop = feats[sel]
    coords_truth = coords
    feats_truth = feats

    if coords_crop.shape[0] < 100:
        return random_crop(coords, feats, resolution, partial_rate)
    return coords_crop, feats_crop, coords_truth, feats_truth


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

    def random_crop(self, data_list):
        # coords_list, ori_coords_list, feats_list = data_list
        # coords_list, ori_coords_list = data_list
        crop_coords_list = []
        crop_ori_coords_list = []
        crop_feats_list = []
        if len(data_list) == 3:
            for coords, ori_coords, feats in zip(*data_list):
                rand_idx = random.randint(0, int(self.resolution * 0.66))
                sel0 = coords[:, 0] > rand_idx
                max_range = self.resolution / 3 + rand_idx
                sel1 = coords[:, 0] < max_range
                sel = sel0 * sel1
                crop_coords_list.append(coords[sel])
                crop_ori_coords_list.append(ori_coords[sel])
                crop_feats_list.append(feats[sel])
            return crop_coords_list, crop_ori_coords_list, crop_feats_list
        else:
            for coords, ori_coords in zip(*data_list):
                rand_idx = random.randint(0, int(self.resolution * 0.66))
                sel0 = coords[:, 0] > rand_idx
                max_range = self.resolution / 3 + rand_idx
                sel1 = coords[:, 0] < max_range
                sel = sel0 * sel1
                crop_coords_list.append(coords[sel])
                crop_ori_coords_list.append(ori_coords[sel])
            return crop_coords_list, crop_ori_coords_list

    def __call__(self, list_data):
        coords_crop, feats_crop, coords_truth, feats_truth, indx = list(zip(*list_data))
        item = construct_data_batch(coords_crop, feats_crop, coords_truth, feats_truth)
        # else:
        #     coords, ori_coords, indx = list(zip(*list_data))
        #     item = construct_data_batch(coords, ori_coords)
        return item


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.)
    return pcd


# def random_crop(coords_list, resolution):
#     crop_coords_list = []
#     for coords in coords_list:
#         sel = coords[:, 0] < resolution / 3
#         crop_coords_list.append(coords[sel])
#     return crop_coords_list


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
