import sys
import os
import time
import zmq
import capnp
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capnp'))
import action_capnp
import observation_capnp
import numpy as np

#def serverFunc(context):
#    socket = context.socket(zmq.REP)
#    socket.bind("tcp://*:5555")
#
#    while True:
#        #  Wait for next request from client
#        message = socket.recv()
#        print(f"Received request: {message}")
#
#        #  Do some 'work'
#        time.sleep(1)
#
#        #  Send reply back to client
#        socket.send(b"World")

def decodeObservation(obs_mgs):
    arr_size = len(obs_mgs.unknownCount)
    print(f"Unknown count len: {arr_size}")
    arr = np.reshape(np.array(obs_mgs.unknownCount), (10, 10))
    print(arr)

def encodeGoalPose(action_msg, data):
    action_msg.data.init('goalPose')
    action_msg.data.goalPose.position.x = data[0]
    action_msg.data.goalPose.position.y = data[1]
    action_msg.data.goalPose.position.z = data[2]
    action_msg.data.goalPose.orientation.x = data[3]
    action_msg.data.goalPose.orientation.y = data[4]
    action_msg.data.goalPose.orientation.z = data[5]
    action_msg.data.goalPose.orientation.w = data[6]

def clientFunc(context):
    #  Socket to talk to server
    print("Connecting to hello world server…")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    action_msg = action_capnp.Action.new_message()
    encodeGoalPose(action_msg, [0, 0, 0, 0, 0, 0, 1])

    #  Do 10 requests, waiting each time for a response
    for request in range(10):
        print(f"Sending request {request} …")
        socket.send(action_msg.to_bytes())

        #  Get the reply.
        message = socket.recv()
        obs_mgs = observation_capnp.Observation.from_bytes(message)
        decodeObservation(obs_mgs)

context = zmq.Context()
clientFunc(context)
