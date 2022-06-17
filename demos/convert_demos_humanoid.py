import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import math
import random
import copy
from collections import OrderedDict
import torch
import pickle


def get_demo_humanoid( demo_path):
    """
    Extract demo from pickled file
    """
    L_observations = []
    L_sim_states = []

    assert os.path.isfile(demo_path)

    with open(demo_path, "rb") as f:
        demo = pickle.load(f)
        print("demo.keys() = ", demo.keys())
    print("len(demo['obs']) = ", len(demo["obs"]))

    for obs, sim_state in zip(demo["obs"][1:], demo["checkpoints"]): ## shift of one between obs & checkpoints
        # print("obs = ", obs[:10])
        # print("sim_state = ", sim_state[3][:10])
        # print("obs == sim_state ", obs == sim_state[3])

        print("obs.shape = ", obs.shape)

        L_observations.append(obs)
        L_sim_states.append(sim_state)

    return L_observations, L_sim_states

def save_demo_humanoid(save_path, L_observations, L_sim_states):
    """
    save converted demo
    """
    demo = {}
    demo["observations"] = L_observations
    demo["sim_states"] = L_sim_states

    print("length observations = ", len(demo["observations"]))
    print("length full states = ", len(demo["sim_states"]))

    # print("observations[0] = ", demo["observations"][0])
    # print("sim_states[0] = ", demo["sim_states"][0])

    with open(save_path, 'wb') as handle:
        pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

if __name__ == '__main__':
    demo_path = "./humanoid/1.demo"
    save_path = "./humanoid_convert/1.demo"

    L_observations, L_sim_states = get_demo_humanoid(demo_path)

    save_demo_humanoid(save_path, L_observations, L_sim_states)
