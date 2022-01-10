import argparse
import logging
import os.path

import torch

from autoencoder.utility import setup_logger
from autoencoder.learner import AELearner
from vpp_env_client import EnvironmentClient

setup_logger()


def get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--max_iter", type=int, default=30000)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--stat_freq", type=int, default=50)
    parser.add_argument("--weights", type=str, default="model/modelnet_features.pth")
    parser.add_argument("--load_optimizer", type=str, default="true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--use_gpu", type=bool, default= False)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--repeat", type=bool, default=False)
    parser.add_argument("--phase", type=str, default="inference", help="choose from train or eval or inference")
    parser.add_argument("--dir_to_data", type=str, default="/home/zeng/catkin_ws/data/data_cvx_test2",
                        help="path to the data directory")
    # "/home/zeng/catkin_ws/data/data_cvx/*.cvx"
    # "/home/zeng/catkin_ws/data/data_cvx_test2/*.cvx"
    config = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = torch.device("cpu")

    # data dir
    config.paths_to_data = [config.dir_to_data + "/*.cvx"]

    # create dir to save the model.
    if not os.path.exists(config.weights):
        os.makedirs(config.weights[:config.weights.rindex("/")])
    return config


if __name__ == "__main__":
    config = get_parse_args()
    logging.info(config)

    learner = AELearner(config)
    if config.phase == "train":
        learner.training()
    elif config.phase == "eval":
        learner.evaluation()
    else:
        client = EnvironmentClient(handle_simulation=False)
        voxelgrid, robotPose, robotJoints, reward = client.sendReset(map_type='voxelgrid')
        # points, labels, robotPose, robotJoints, reward = client.sendReset(map_type='pointcloud')
        learner.inference(voxelgrid)
