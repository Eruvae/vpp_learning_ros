import os
import sys
import random
import logging
import glob
from torch.utils.data.sampler import Sampler
import torch
import torch.utils.data
import MinkowskiEngine as ME

from autoencoder.dataset.data_reader import load_mesh_file, load_pcd_file, InfSampler

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..', "capnp"))
import pointcloud_capnp

try:
    import open3d as o3d
except ImportError:
    raise ImportError(
        "Please install open3d and scipy with `pip install open3d scipy`."
    )


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
