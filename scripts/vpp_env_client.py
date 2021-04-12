import sys
import os
import time
import zmq
import capnp
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "capnp"))
import action_capnp
import observation_capnp
import numpy as np

class EnvironmentClient:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:5555")

    def decodeObservation(self, obs_msg):
        shape = (obs_msg.layers, obs_msg.height, obs_msg.width)
        unknownCount = np.reshape(np.array(obs_msg.unknownCount), shape)
        freeCount = np.reshape(np.array(obs_msg.freeCount), shape)
        occupiedCount = np.reshape(np.array(obs_msg.occupiedCount), shape)
        roiCount = np.reshape(np.array(obs_msg.roiCount), shape)
        if obs_msg.planningTime > 0:
            reward = obs_msg.foundRois / obs_msg.planningTime
        else:
            reward = 0
        return unknownCount, freeCount, occupiedCount, roiCount, reward

    def encodeGoalPose(self, action_msg, data):
        action_msg.init("goalPose")
        action_msg.goalPose.position.x = data[0]
        action_msg.goalPose.position.y = data[1]
        action_msg.goalPose.position.z = data[2]
        action_msg.goalPose.orientation.x = data[3]
        action_msg.goalPose.orientation.y = data[4]
        action_msg.goalPose.orientation.z = data[5]
        action_msg.goalPose.orientation.w = data[6]

    def encodeRelativePose(self, action_msg, data):
        action_msg.init("relativePose")
        action_msg.relativePose.position.x = data[0]
        action_msg.relativePose.position.y = data[1]
        action_msg.relativePose.position.z = data[2]
        action_msg.relativePose.orientation.x = data[3]
        action_msg.relativePose.orientation.y = data[4]
        action_msg.relativePose.orientation.z = data[5]
        action_msg.relativePose.orientation.w = data[6]

    def sendAction(self, action_msg):
        self.socket.send(action_msg.to_bytes())

        #  Get the reply.
        message = self.socket.recv()
        obs_msg = observation_capnp.Observation.from_bytes(message)
        return self.decodeObservation(obs_msg)

    def sendGoalPose(self, goal_pose):
        action_msg = action_capnp.Action.new_message()
        self.encodeGoalPose(action_msg, goal_pose)
        return self.sendAction(action_msg)

    def sendRelativePose(self, relative_pose):
        action_msg = action_capnp.Action.new_message()
        self.encodeRelativePose(action_msg, relative_pose)
        return self.sendAction(action_msg)

    def sendReset(self):
        action_msg = action_capnp.Action.new_message()
        action_msg.reset = None
        return self.sendAction(action_msg)


def main(args):
    client = EnvironmentClient()
    unknownCount, freeCount, occupiedCount, roiCount, reward = client.sendRelativePose([0.1, 0, 0, 0, 0, 0, 1])
    print("unknownCount")
    print(unknownCount)
    print("freeCount")
    print(freeCount)
    print("occupiedCount")
    print(occupiedCount)
    print("roiCount")
    print(roiCount)
    print("Reward", reward)

if __name__ == '__main__':
    main(sys.argv)